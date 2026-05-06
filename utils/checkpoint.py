"""
Checkpoint management utilities.

Contains:
- save_model: Save model checkpoint
- load_model: Load model checkpoint
"""

import os
from pathlib import Path

import torch

from .distributed import is_main_process


def save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler, mode):
    """
    Save model checkpoint.

    Args:
        args: Arguments namespace
        epoch: Current epoch number
        model: Model (potentially with DDP wrapper)
        model_without_ddp: Model without DDP wrapper
        optimizer: Optimizer
        loss_scaler: Native scaler for AMP
        mode: Save mode ('best' or 'latest')
    """
    output_dir = Path(args.output_dir)
    epoch_name = str(epoch)
    os.makedirs(os.path.join(args.output_dir, args.task), exist_ok=True)

    if loss_scaler is not None:
        if mode == 'best':
            checkpoint_paths = [os.path.join(args.output_dir, args.task, 'checkpoint-best.pth')]
        else:
            checkpoint_paths = [os.path.join(args.output_dir, args.task, 'checkpoint-latest.pth')]

        for checkpoint_path in checkpoint_paths:
            if mode == 'best':
                to_save = {
                    'model': model_without_ddp.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }
            else:
                if epoch == args.training['epochs'] - 1:
                    to_save = {
                        'model': model_without_ddp.state_dict(),
                        'args': args,
                    }
                else:
                    to_save = {
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch,
                        'scaler': loss_scaler.state_dict(),
                        'args': args,
                    }

            if is_main_process():
                torch.save(to_save, checkpoint_path)
    else:
        if mode == 'best':
            to_save = {
                'model': model_without_ddp.state_dict(),
                'epoch': epoch,
            }
            if is_main_process():
                torch.save(to_save, os.path.join(args.output_dir, args.task, "checkpoint-best.pth"))
        else:
            if epoch == args.training['epochs'] - 1:
                to_save = {
                    'model': model_without_ddp.state_dict(),
                }
            else:
                to_save = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }
            if is_main_process():
                torch.save(to_save, os.path.join(args.output_dir, args.task, "checkpoint-latest.pth"))


def load_pretrained_weights(model, checkpoint_path, prefix=None, model_type='vit_base', patch_size=16):
    """
    Load pretrained weights for ViT model. Supports local paths and URLs.
    
    Args:
        model: Model instance
        checkpoint_path: Path or URL to pretrained weights
        prefix: Optional prefix in state dict
        model_type: Model architecture type
        patch_size: ViT patch size
    """
    if not checkpoint_path:
        print("No pretrained weights specified.")
        return

    print(f"Loading pretrained weights from {checkpoint_path}")
    
    if checkpoint_path.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            checkpoint_path, map_location='cpu', check_hash=True
        )
    else:
        if not os.path.exists(checkpoint_path):
            print(f"Error: Checkpoint path {checkpoint_path} does not exist.")
            # If the user wanted auto-download but didn't provide a URL, 
            # we could potentially fallback to a known URL or timm, 
            # but for now we'll just log the error.
            return
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Remove prefix if specified
    if prefix:
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith(prefix):
                new_state_dict[k[len(prefix):]] = v
            else:
                new_state_dict[k] = v
        state_dict = new_state_dict

    # Interpolate position embedding if needed
    if 'pos_embed' in state_dict:
        pos_embed_checkpoint = state_dict['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new model
        new_size = int(num_patches ** 0.5)
        
        if orig_size != new_size:
            print(f"Interpolating position embeddings from {orig_size}x{orig_size} to {new_size}x{new_size}")
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            state_dict['pos_embed'] = new_pos_embed

    # Load state dict
    msg = model.load_state_dict(state_dict, strict=False)
    print(f"Pretrained weights loaded with message: {msg}")


def load_model(args, model_without_ddp, optimizer, loss_scaler):
    """
    Load model checkpoint.

    Args:
        args: Arguments namespace
        model_without_ddp: Model without DDP wrapper
        optimizer: Optimizer
        loss_scaler: Native scaler for AMP
    """
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True
            )
        else:
            checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)

        if 'model' in checkpoint:
            checkpoint_model = checkpoint['model']
        else:
            checkpoint_model = checkpoint

        model_without_ddp.load_state_dict(checkpoint_model, strict=False)
        print("Resume checkpoint %s" % args.resume)

        if 'optimizer' in checkpoint and 'epoch' in checkpoint and not (
            hasattr(args, 'eval') and args.eval
        ):
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("Optimizer params loaded from checkpoint!")
            args.start_epoch = checkpoint['epoch'] + 1
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
                print("Loss scaler loaded from checkpoint!")
            print("With optim & sched!")