"""
Losses module for C2Net EDL
"""
from .edl_loss import EDLLoss, EDLMultiLabelLoss, kl_divergence, log_likelihood_loss

__all__ = ['EDLLoss', 'EDLMultiLabelLoss', 'kl_divergence', 'log_likelihood_loss']