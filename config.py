"""
Configuration for C2Net EDL (Evidential Deep Learning) project.

This configuration extends the original C2Net with EDL-specific parameters
for handling long-tail distributions using Beta distribution evidence modeling.
"""

import yaml
import argparse
from pathlib import Path


class Config:
    """Configuration class for C2Net EDL."""

    def __init__(self):
        # Model configuration
        self.model = {
            'type': 'vit_base',  # Options: vit_tiny, vit_small, vit_base, vit_large
            'patch_size': 16,
            'input_size': 256,
            'num_classes': 20,
            'embed_dim': 768,
            'depth': 12,
            'num_heads': 12,
            'mlp_ratio': 4.0,
            'drop_path': 0.1,
            'attn_drop_rate': 0.0,
            'drop_rate': 0.0,

            # Pretrained weights path (Supports local path or URL for auto-download)
            'finetune': 'https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth',

            # Contrastive learning settings
            'contra_learning': True,
            'contra_features_head': 'linear',  # Options: linear, mlp, identity
            'contra_feat_dim': 128,
            'contra_feature': 'cls',  # Options: cls, mean, concat

            # Multi-task learning
            'multi_task': True,
            'global_pool': True
        }

        # Data configuration
        self.data = {
            'root_dir': '/root/C2Net_EDL/data/ODIR_crop',
            'csv_file': 'datasets/ODIR_20class.csv',
            'grade_csv': 'datasets/ODIR_eyeQ_new.csv',
            'enhance_dir': None,  # Optional: path to enhanced images

            # Dataset split columns in CSV
            'split_column': 'random_mode',  # train/val/test
            'unusable_column': 'unusable',

            # Data normalization (ImageNet as per paper)
            'normalization': 'imagenet',

            # Long-tail handling
            'min_samples': 10,  # Minimum samples threshold for class inclusion
            'merge_tail_classes': False  # Set to False to preserve long-tail classes
        }

        # Training configuration
        self.training = {
            'epochs': 100,
            'batch_size': 32,
            'num_workers': 8,
            'accum_iter': 1,

            # Learning rate
            'lr': 1e-4,
            'min_lr': 0.0,
            'blr': 1e-4,  # Base learning rate for layer decay
            'warmup_epochs': 10,

            # Optimizer
            'opt': 'adamw',
            'weight_decay': 0.05,
            'layer_decay': 0.65,

            # Gradient clipping
            'clip_grad': 0.0,  # 0 means no clipping

            # Early stopping
            'early_stop_patience': 15,

            # Other
            'seed': 0,
            'resume': None,  # Path to checkpoint to resume from
            'auto_resume': True
        }

        # Loss configuration - Updated for EDL
        self.loss = {
            'loss_type': 'edl_beta',  # Default changed to edl_beta

            # Default values for 20 classes
            'alpha': [0.25] * 20,
            'gamma': [2.0] * 20,
            'pos_weight': [1.0] * 20,
            'notuse_posweight': False,

            # Constraint loss weights
            'lambda_list': [6.0, 1.5, 1.5],  # [focal, rule1, rule2]
            'loss_mode': None,

            # Contrastive loss parameters
            'margin': 2.5
        }

        # EDL-specific hyperparameters
        self.edl = {
            'kl_annealing_epochs': 20,  # KL divergence annealing period
            'uncertainty_gamma': 1.0,   # Coefficient for uncertainty-based weighting
            'contra_weight': 0.1,       # Weight for supervised contrastive loss
            'evidence_activation': 'softplus', # Activation function for evidence
            'evidence_scale': 10.0,     # Scale factor for evidence parameters
            'neg_weight': 0.2,          # Weight for negative samples in likelihood loss
            'annealing_type': 'cyclical' # 'linear' or 'cyclical'
        }

        # Augmentation
        self.augmentation = {
            'train': {
                'resize_size': 256,
                'random_crop': True,
                'random_horizontal_flip': True,
                'random_vertical_flip': True,
                'random_affine': {
                    'enabled': True,
                    'degrees': 45
                },
                'color_jitter': {
                    'enabled': False,
                    'brightness': 0.4,
                    'contrast': 0.4,
                    'saturation': 0.4,
                    'hue': 0.1
                }
            },
            'eval': {
                'resize_size': 256
            }
        }

        # Distributed training
        self.distributed = {
            'world_size': 1,
            'dist_url': 'env://',
            'dist_backend': 'nccl'
        }

        # Output configuration
        self.output_dir = 'outputs'
        self.task = 'c2net_edl'

        # Logging
        self.log_interval = 20
        self.save_interval = 1

        # Mixed precision
        self.amp = True  # Automatic mixed precision

        # Device
        self.device = 'cuda'  # cuda or cpu

    def update_from_args(self, args):
        """Update configuration from command line arguments."""
        for key, value in vars(args).items():
            if hasattr(self, key):
                setattr(self, key, value)
            elif '.' in key:
                # Handle nested attributes like model.type
                parts = key.split('.')
                obj = self
                for part in parts[:-1]:
                    if hasattr(obj, part):
                        obj = getattr(obj, part)
                    else:
                        break
                else:
                    if hasattr(obj, parts[-1]):
                        setattr(obj, parts[-1], value)

    def update_from_yaml(self, config_path):
        """Update configuration from YAML file."""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        def update_dict(obj, d):
            for key, value in d.items():
                if isinstance(obj, dict):
                    if key in obj:
                        if isinstance(value, dict) and isinstance(obj[key], dict):
                            update_dict(obj[key], value)
                        else:
                            obj[key] = value
                elif hasattr(obj, key):
                    attr_value = getattr(obj, key)
                    if isinstance(value, dict) and isinstance(attr_value, dict):
                        update_dict(attr_value, value)
                    else:
                        setattr(obj, key, value)

        update_dict(self, config_dict)

    def save_to_yaml(self, output_path):
        """Save configuration to YAML file."""
        def serialize_config(obj):
            if hasattr(obj, '__dict__'):
                result = {}
                for key, value in obj.__dict__.items():
                    if not key.startswith('_'):
                        result[key] = serialize_config(value)
                return result
            elif isinstance(obj, (list, tuple)):
                return [serialize_config(item) for item in obj]
            else:
                return obj

        config_dict = serialize_config(self)
        with open(output_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)


def get_args_parser():
    """Create argument parser for training configuration."""
    parser = argparse.ArgumentParser('C2Net EDL training script', add_help=False)

    # Model parameters
    parser.add_argument('--model_type', default='vit_base', type=str,
                        choices=['vit_tiny', 'vit_small', 'vit_base', 'vit_large'],
                        help='Name of model to train')
    parser.add_argument('--patch_size', default=16, type=int,
                        help='Patch size of ViT')
    parser.add_argument('--input_size', default=256, type=int,
                        help='Input image size')
    parser.add_argument('--drop_path', type=float, default=0.1,
                        help='Drop path rate')
    parser.add_argument('--finetune', default='', help='Path to pretrained weights')

    # Data parameters
    parser.add_argument('--data_path', default='', type=str,
                        help='Path to dataset')
    parser.add_argument('--csv_file', default='', type=str,
                        help='Path to CSV file')
    parser.add_argument('--grade_csv', default='datasets/ODIR_eyeQ_new.csv', type=str,
                        help='Path to grade CSV file')
    parser.add_argument('--enhance_dir', default=None, type=str,
                        help='Path to enhanced images')
    parser.add_argument('--min_samples', default=10, type=int,
                        help='Minimum samples threshold for class inclusion')

    # Training parameters
    parser.add_argument('--epochs', default=300, type=int,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Batch size per GPU')
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations')
    parser.add_argument('--lr', default=1e-4, type=float,
                        help='Base learning rate')
    parser.add_argument('--blr', default=1e-4, type=float,
                        help='Base learning rate for layer decay')
    parser.add_argument('--min_lr', type=float, default=1e-6,
                        help='Minimum learning rate')
    parser.add_argument('--warmup_epochs', type=int, default=10,
                        help='Number of warmup epochs')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='Weight decay')
    parser.add_argument('--layer_decay', type=float, default=0.65,
                        help='Layer-wise learning rate decay')
    parser.add_argument('--clip_grad', type=float, default=0.0,
                        help='Gradient clipping norm (0 to disable)')

    # EDL parameters
    parser.add_argument('--kl_annealing_epochs', default=20, type=int,
                        help='KL divergence annealing period')
    parser.add_argument('--uncertainty_gamma', default=1.0, type=float,
                        help='Coefficient for uncertainty-based weighting')
    parser.add_argument('--evidence_scale', default=10.0, type=float,
                        help='Scale factor for evidence parameters')
    parser.add_argument('--neg_weight', default=0.2, type=float,
                        help='Weight for negative samples in likelihood loss')
    parser.add_argument('--annealing_type', default='cyclical', type=str,
                        choices=['linear', 'cyclical'],
                        help='Type of KL annealing')

    # Loss parameters
    parser.add_argument('--loss_type', default='edl_beta', type=str,
                        choices=['edl_beta', 'focal', 'bce'],
                        help='Loss type to use')

    # Distributed training
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--dist_backend', default='nccl',
                        help='distributed backend')
    parser.add_argument('--distributed', default=False, action='store_true')

    # Misc
    parser.add_argument('--output_dir', default='./outputs',
                        help='Path where to save outputs')
    parser.add_argument('--task', default='c2net_edl',
                        help='Task name for saving outputs')
    parser.add_argument('--seed', default=0, type=int,
                        help='Random seed')
    parser.add_argument('--resume', default='', type=str,
                        help='Resume from checkpoint')
    parser.add_argument('--auto_resume', default=True, action='store_true',
                        help='Auto resume from latest checkpoint')
    parser.add_argument('--num_workers', default=8, type=int,
                        help='Number of data loading workers per GPU')
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader')

    # Device
    parser.add_argument('--device', default='cuda', type=str,
                        help='Device to use')
    parser.add_argument('--config', default=None, type=str,
                        help='Path to config YAML file')

    return parser


if __name__ == '__main__':
    # Example usage
    config = Config()
    print("Default configuration created successfully!")
    print(f"Model type: {config.model['type']}")
    print(f"Number of classes: {config.model['num_classes']}")
    print(f"KL annealing epochs: {config.edl['kl_annealing_epochs']}")
    print(f"Loss type: {config.loss['loss_type']}")
