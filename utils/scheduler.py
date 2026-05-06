"""
Learning rate scheduler utilities.

Contains:
- adjust_learning_rate: Cosine decay with warmup
"""

import math


def adjust_learning_rate(optimizer, epoch, args):
    """
    Decay the learning rate with half-cycle cosine after warmup.

    Args:
        optimizer: PyTorch optimizer
        epoch: Current epoch number
        args: Arguments namespace containing scheduler parameters

    Returns:
        lr: Current learning rate
    """
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * (
            1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs))
        )
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr