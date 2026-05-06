"""
Data module for C2Net EDL
"""
from .dataset import ODIRDatasetEDL
from .transforms import get_train_transform, get_eval_transform, get_normalization_stats

__all__ = ['ODIRDatasetEDL', 'get_train_transform', 'get_eval_transform', 'get_normalization_stats']