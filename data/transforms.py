"""
Data augmentation and transformation functions.

Paper-specified augmentation:
- Random horizontal flip (p=0.5)
- Random rotation (±15°)
- Random color jitter (brightness/contrast/saturation ±10%)
- Random Gaussian noise (std=0.01)
- ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
"""

import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image


# ImageNet normalization (as specified in paper)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Legacy ODIR stats (not used in paper experiments)
ODIR_MEAN = [0.423737496, 0.260946065, 0.128403902]
ODIR_STD = [0.294825345, 0.201673657, 0.136680201]


class AddGaussianNoise:
    """Add Gaussian noise with given std to a tensor."""

    def __init__(self, std=0.01):
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn_like(tensor) * self.std

    def __repr__(self):
        return f'{self.__class__.__name__}(std={self.std})'


def get_train_transform(
    resize_size=256,
    mean=IMAGENET_MEAN,
    std=IMAGENET_STD
):
    """
    Training augmentation per paper:
    - Resize to 256×256
    - Random horizontal flip (p=0.5)
    - Random rotation (±15°)
    - ToTensor
    - Color jitter (brightness/contrast/saturation ±10%)
    - Gaussian noise (std=0.01)
    - ImageNet normalization
    """
    return transforms.Compose([
        transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        AddGaussianNoise(std=0.01),
        transforms.Normalize(mean=mean, std=std),
    ])


def get_eval_transform(
    resize_size=256,
    mean=IMAGENET_MEAN,
    std=IMAGENET_STD
):
    """Eval: resize + normalize only."""
    return transforms.Compose([
        transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def get_normalization_stats(normalization='imagenet'):
    normalization = normalization.lower()
    if normalization in ('imagenet', 'vfm'):
        return IMAGENET_MEAN, IMAGENET_STD
    elif normalization == 'odir':
        return ODIR_MEAN, ODIR_STD
    else:
        raise ValueError(f"Unknown normalization type: {normalization}")
