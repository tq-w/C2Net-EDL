"""
Main training script for C2Net EDL (Evidential Deep Learning).

This implementation extends the original C2Net with:
1. EDL head that outputs alpha and beta parameters
2. EDL loss function with KL annealing
3. Uncertainty-aware evaluation metrics
4. Long-tail class handling without merging tail classes
"""

import argparse
import datetime
import os
import sys
import time
import math

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
import yaml
from torch.utils.tensorboard import SummaryWriter

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config, get_args_parser
from data.dataset import ODIRDatasetEDL, split_dataset, analyze_dataset_distribution
from data.transforms import get_train_transform, get_eval_transform, get_normalization_stats
from losses.edl_loss import EDLLoss, SupConLoss
from models.head import EDLHead
from utils import (
    adjust_learning_rate, MetricLogger, NativeScalerWithGradNormCount, all_reduce_mean,
    init_distributed_mode, is_dist_avail_and_initialized, is_main_process,
    load_pretrained_weights, param_groups_lrd, save_model
)
from utils.metrics import compute_edl_metrics, log_uncertainty_analysis


def main(args):
    """Main training function."""
    # Load configuration
    config = Config()
    if args.config is not None:
        config.update_from_yaml(args.config)

    # Override config with command line arguments
    config.update_from_args(args)

    # Set device
    device = torch.device(config.device)

    # Initialize distributed training
    if config.distributed:
        init_distributed_mode(args)

    # Fix seed
    seed = config.training['seed'] + (args.rank if hasattr(args, 'rank') else 0)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(os.path.join(config.output_dir, config.task), exist_ok=True)

    # Log file
    if is_main_process():
        log_file = open(os.path.join(config.output_dir, config.task, 'log.txt'), 'w')
        print("Logging to:", log_file.name)
    else:
        log_file = None

    # Print training configuration
    print("\n" + "="*80)
    print("C2Net EDL Training Configuration")
    print("="*80)
    print(f"Model type: {config.model['type']}")
    print(f"Loss type: {config.loss['loss_type']}")
    print(f"KL annealing epochs: {config.edl['kl_annealing_epochs']}")
    print(f"Min samples threshold: {config.data['min_samples']}")
    print(f"Output directory: {os.path.join(config.output_dir, config.task)}")
    print("="*80 + "\n")

    # Analyze dataset distribution
    print("Analyzing dataset distribution...")
    analysis = analyze_dataset_distribution(
        config.data['csv_file'],
        min_samples=config.data['min_samples']
    )

    if analysis is None:
        raise ValueError("Failed to analyze dataset distribution!")

    # Update number of classes based on analysis
    config.model['num_classes'] = analysis['num_valid_classes']
    print(f"Updated number of classes: {config.model['num_classes']}")

    # Create model with EDL head
    print(f"Creating model: {config.model['type']} with EDL head")
    from types import SimpleNamespace
    model_args = SimpleNamespace(**config.model)
    
    if config.model['type'] == 'vit_tiny':
        from models.vision_transformer import vit_tiny_vfm_multitaskv2
        model = vit_tiny_vfm_multitaskv2(model_args, num_classes=config.model['num_classes'])
    elif config.model['type'] == 'vit_small':
        from models.vision_transformer import vit_small_vfm_multitaskv2
        model = vit_small_vfm_multitaskv2(model_args, num_classes=config.model['num_classes'])
    elif config.model['type'] == 'vit_base':
        from models.vision_transformer import vit_base_vfm_multitaskv2_2
        model = vit_base_vfm_multitaskv2_2(model_args, num_classes=config.model['num_classes'])
    elif config.model['type'] == 'vit_large':
        from models.vision_transformer import vit_large_vfm_multitaskv2
        model = vit_large_vfm_multitaskv2(model_args, num_classes=config.model['num_classes'])
    else:
        raise ValueError(f"Unknown model: {config.model['type']}")

    # Replace classification head with EDL head
    from models.head import EDLHead
    if hasattr(model, 'classifier'):
        embed_dim = model.classifier.embed_dim
        model.classifier = EDLHead(
            embed_dim, 
            config.model['num_classes'],
            evidence_scale=config.edl['evidence_scale']
        )
    elif hasattr(model, 'head'):
        embed_dim = model.head.embed_dim
        model.head = EDLHead(
            embed_dim, 
            config.model['num_classes'],
            evidence_scale=config.edl['evidence_scale']
        )
    else:
        raise ValueError("Could not find classifier/head to replace with EDL head")

    model.to(device)

    # Load pretrained weights
    if config.model['finetune'] and config.model['finetune'] != '':
        load_pretrained_weights(
            model, config.model['finetune'], None, config.model['type'], config.model['patch_size']
        )

    # Create DDP model if distributed
    if config.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    else:
        model_without_ddp = model

    # Print model info
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_parameters / 1e6:.2f}M")

    # Create loss function
    criterion = EDLLoss(
        kl_annealing_epochs=config.edl['kl_annealing_epochs'],
        uncertainty_gamma=config.edl['uncertainty_gamma'],
        neg_weight=config.edl['neg_weight'],
        annealing_type=config.edl['annealing_type']
    )
    
    criterion_contra = SupConLoss(temperature=0.07) if config.model['contra_learning'] else None

    # Create optimizer
    print(f"Creating optimizer: {config.training['opt'] if 'opt' in config.training else 'adamw'}")
    param_groups = param_groups_lrd(
        model_without_ddp,
        weight_decay=config.training['weight_decay'],
        no_weight_decay_list=[],
        layer_decay=config.training['layer_decay']
    )
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=config.training['lr'],
        betas=(0.9, 0.999)
    )

    # Cosine annealing LR scheduler with linear warmup
    warmup_epochs = config.training['warmup_epochs']
    total_epochs = config.training['epochs']
    min_lr = config.training.get('min_lr', 0.0)

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return max(min_lr / config.training['lr'], 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Create loss scaler for AMP
    loss_scaler = NativeScalerWithGradNormCount()

    # Get normalization stats (ImageNet per paper)
    norm_type = config.data.get('normalization', 'imagenet')
    mean, std = get_normalization_stats(norm_type)

    # Create transforms
    train_transform = get_train_transform(
        resize_size=config.model['input_size'],
        mean=mean,
        std=std
    )
    eval_transform = get_eval_transform(
        resize_size=config.model['input_size'],
        mean=mean,
        std=std
    )

    # Load dataset
    print("Loading dataset...")
    dataset_type = getattr(args, 'dataset', 'odir')
    fold = getattr(args, 'fold', None)

    if dataset_type == 'mured':
        # MuReD dataset
        from data.dataset import MuReDDatasetEDL
        df = pd.read_csv('datasets/MuReD_20class.csv')
        mured_root = getattr(args, 'data_path', config.data.get('mured_root_dir', '/home/daocp01/wtq/C2Net_EDL/data/MuReD'))

        train_dataset = MuReDDatasetEDL(df, mured_root, mode='train', transform=train_transform, fold=fold)
        val_dataset = MuReDDatasetEDL(df, mured_root, mode='val', transform=eval_transform, fold=fold)
        train_dataset.get_class_map = lambda: {i: i for i in range(20)}
        train_dataset.get_num_classes = lambda: 20
    else:
        # ODIR dataset
        df = pd.read_csv(config.data['csv_file'])

        if fold is not None and 'fold' in df.columns:
            # Use fold-based splitting
            train_dataset = ODIRDatasetEDL(
                df,
                root_dir=config.data['root_dir'],
                mode='train',
                transform=train_transform,
                enhance_dir=config.data['enhance_dir'],
                grade_csv=config.data['grade_csv'] if os.path.exists(config.data['grade_csv']) else None,
                min_samples=config.data['min_samples'],
                split_column='fold',
                fold=fold
            )
            val_dataset = ODIRDatasetEDL(
                df,
                root_dir=config.data['root_dir'],
                mode='val',
                transform=eval_transform,
                min_samples=config.data['min_samples'],
                split_column='fold',
                fold=fold
            )
        else:
            # Original random_mode splitting
            if 'random_mode' not in df.columns:
                df = split_dataset(df, seed=config.training['seed'])
            train_dataset = ODIRDatasetEDL(
                df,
                root_dir=config.data['root_dir'],
                mode='train',
                transform=train_transform,
                enhance_dir=config.data['enhance_dir'],
                grade_csv=config.data['grade_csv'] if os.path.exists(config.data['grade_csv']) else None,
                min_samples=config.data['min_samples']
            )
            val_dataset = ODIRDatasetEDL(
                df,
                root_dir=config.data['root_dir'],
                mode='val',
                transform=eval_transform,
                min_samples=config.data['min_samples'],
                class_map=train_dataset.get_class_map()
            )

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
    print(f"Number of classes: {train_dataset.get_num_classes()}")

    # Create data loaders
    if config.distributed:
        sampler_train = torch.utils.data.DistributedSampler(
            train_dataset, num_replicas=args.world_size, rank=args.rank, shuffle=True
        )
        sampler_val = torch.utils.data.DistributedSampler(
            val_dataset, num_replicas=args.world_size, rank=args.rank, shuffle=False
        )
    else:
        sampler_train = torch.utils.data.RandomSampler(train_dataset)
        sampler_val = torch.utils.data.SequentialSampler(val_dataset)

    data_loader_train = torch.utils.data.DataLoader(
        train_dataset,
        sampler=sampler_train,
        batch_size=config.training['batch_size'],
        num_workers=config.training['num_workers'],
        pin_memory=True,
        drop_last=True
    )
    data_loader_val = torch.utils.data.DataLoader(
        val_dataset,
        sampler=sampler_val,
        batch_size=config.training['batch_size'],
        num_workers=config.training['num_workers'],
        pin_memory=True,
        drop_last=False
    )

    # Create TensorBoard writer
    if is_main_process():
        log_writer = SummaryWriter(log_dir=os.path.join(config.output_dir, config.task, 'tensorboard'))
    else:
        log_writer = None

    # Training loop
    print(f"Start training for {config.training['epochs']} epochs")
    start_time = time.time()
    best_score = 0.0
    patience = config.training.get('early_stop_patience', 15)
    patience_counter = 0

    for epoch in range(config.training['epochs']):
        if config.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        # Train for one epoch
        train_stats = train_one_epoch_edl(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            clip_grad=config.training['clip_grad'], log_writer=log_writer,
            config=config, criterion_contra=criterion_contra
        )

        # Step the LR scheduler (warmup + cosine annealing)
        scheduler.step()

        # Evaluate
        val_stats, score = evaluate_edl(
            data_loader_val, model, criterion, device,
            config, epoch, 'val', log_writer
        )

        # Save best checkpoint + early stopping
        if is_main_process():
            if score > best_score:
                best_score = score
                patience_counter = 0
                print(f"New best score: {best_score:.4f}")
                save_model(config, epoch, model, model_without_ddp, optimizer, loss_scaler, 'best')
            else:
                patience_counter += 1

            # Save latest checkpoint
            save_model(config, epoch, model, model_without_ddp, optimizer, loss_scaler, 'latest')

            # Log to file
            log_str = (
                f"Epoch [{epoch}/{config.training['epochs']}] | "
                f"Train Loss: {train_stats['loss']:.3f} | "
                f"Val Loss: {val_stats['loss']:.3f} | "
                f"Val Score: {score:.3f} | "
                f"Best Score: {best_score:.3f}\n"
            )
            log_file.write(log_str)
            log_file.flush()

            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time: {total_time_str}")

    if is_main_process():
        log_file.close()
        if log_writer:
            log_writer.close()

    print("Training completed!")
    print(f"Best score: {best_score:.4f}")


def train_one_epoch_edl(model, criterion, data_loader, optimizer, device, epoch,
                       loss_scaler, clip_grad=0.0, log_writer=None, config=None, criterion_contra=None):
    """
    Train for one epoch with EDL loss and optional contrastive loss.
    """
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    header = f'Epoch: [{epoch}]'

    for batch_idx, (samples, targets) in enumerate(metric_logger.log_every(data_loader, config.log_interval, header)):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=config.amp):
            # Forward pass - model returns a dictionary
            outputs = model(samples)
            alpha, beta = outputs['d']

            # Compute EDL loss
            loss, loss_components = criterion(
                alpha, beta, targets, 
                epoch=epoch, 
                total_epochs=config.training['epochs']
            )
            
            # Compute contrastive loss if enabled
            if criterion_contra is not None and 'emb' in outputs:
                loss_contra = criterion_contra(outputs['emb'], targets)
                # Weighted total loss using weight from config
                contra_weight = config.edl['contra_weight']
                loss = loss + contra_weight * loss_contra
                loss_components['loss_contra'] = loss_contra.item()

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        optimizer.zero_grad()
        loss_scaler(loss, optimizer, clip_grad=clip_grad, parameters=model.parameters())

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # Log to tensorboard
        if log_writer is not None and batch_idx % config.log_interval == 0:
            log_writer.add_scalar('train_loss', loss_value, epoch * len(data_loader) + batch_idx)
            log_writer.add_scalar('likelihood_loss', loss_components['likelihood_loss'], epoch * len(data_loader) + batch_idx)
            log_writer.add_scalar('kl_loss', loss_components['kl_loss'], epoch * len(data_loader) + batch_idx)
            log_writer.add_scalar('annealing_coef', loss_components['annealing_coef'], epoch * len(data_loader) + batch_idx)
            if 'loss_contra' in loss_components:
                log_writer.add_scalar('loss_contra', loss_components['loss_contra'], epoch * len(data_loader) + batch_idx)

    # Gather stats from all processes
    metric_logger.synchronize_between_processes()
    print(f"Averaged stats: {metric_logger}")
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_edl(data_loader, model, criterion, device, config, epoch, prefix='val', log_writer=None):
    """
    Evaluate model with EDL metrics.
    """
    model.eval()
    all_alpha = []
    all_beta = []
    all_targets = []

    for samples, targets in data_loader:
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=config.amp):
            outputs = model(samples)
            alpha, beta = outputs['d']

            # Store for later analysis
            all_alpha.append(alpha.cpu())
            all_beta.append(beta.cpu())
            all_targets.append(targets.cpu())

    # Concatenate all batches
    alpha_all = torch.cat(all_alpha, dim=0)
    beta_all = torch.cat(all_beta, dim=0)
    targets_all = torch.cat(all_targets, dim=0)

    # Compute comprehensive EDL metrics
    metrics = compute_edl_metrics(alpha_all.numpy(), beta_all.numpy(), targets_all.numpy())
    
    avg_prob = np.mean(metrics['probabilities'])
    max_prob = np.max(metrics['probabilities'])

    # Compute score (same as original C2Net: (F1 + AUC + Kappa) / 3)
    # For simplicity, we'll use F1 + AUC as proxy since Kappa requires binary predictions
    if np.isnan(metrics['roc_auc']):
        score = metrics['f1_score']
    else:
        score = (metrics['f1_score'] + metrics['roc_auc']) / 2.0

    # Log uncertainty analysis
    if log_writer is not None:
        log_writer.add_scalar(f'{prefix}_f1', metrics['f1_score'], epoch)
        log_writer.add_scalar(f'{prefix}_auc', metrics['roc_auc'], epoch)
        log_writer.add_scalar(f'{prefix}_ece', metrics['ece'], epoch)
        log_writer.add_scalar(f'{prefix}_brier', metrics['brier_score'], epoch)
        log_writer.add_scalar(f'{prefix}_score', score, epoch)
        log_writer.add_scalar(f'{prefix}_avg_prob', avg_prob, epoch)

        # Log uncertainty metrics
        uncertainty_results = metrics['uncertainty_analysis']
        overall = uncertainty_results['overall']
        log_writer.add_scalar(f'{prefix}_uncertainty_mean_correct', overall['mean_uncertainty_correct'], epoch)
        log_writer.add_scalar(f'{prefix}_uncertainty_mean_incorrect', overall['mean_uncertainty_incorrect'], epoch)

    # Print results
    print(f"{prefix.capitalize()} Results:")
    print(f"  F1 Score: {metrics['f1_score']:.3f}")
    print(f"  ROC AUC: {metrics['roc_auc']:.3f}")
    print(f"  ECE: {metrics['ece']:.3f}")
    print(f"  Brier Score: {metrics['brier_score']:.3f}")
    print(f"  Score: {score:.3f}")

    return {'loss': 0.0}, score  # Loss is not computed in evaluation for simplicity


if __name__ == '__main__':
    parser = argparse.ArgumentParser('C2Net EDL training script', parents=[get_args_parser()])
    args = parser.parse_args()

    if args.output_dir:
        from pathlib import Path
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(args.output_dir, args.task)).mkdir(parents=True, exist_ok=True)

    main(args)