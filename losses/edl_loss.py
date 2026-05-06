"""
Evidential Deep Learning (EDL) loss functions for multi-label classification.

Implements:
1. KL Divergence between Beta(α,β) and Beta(1,1) with full Digamma terms
2. Asymmetric log-likelihood loss with neg_weight for long-tail balancing
3. Cyclical KL annealing (4 cycles, warmup for first half of each cycle)
4. Supervised contrastive loss for multi-label scenarios

Reference: NeurIPS 2018 "Evidential Deep Learning to Quantify Classification Uncertainty"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def kl_divergence(alpha, beta):
    """
    KL(Beta(α,β) || Beta(1,1)) with full Digamma terms (Eq. in paper).

    KL = -log B(α,β) + (α-1)ψ(α) + (β-1)ψ(β) - (α+β-2)ψ(α+β)

    where B(α,β) = Γ(α)Γ(β)/Γ(α+β) and ψ is the Digamma function.

    Args:
        alpha: (B, num_classes), α > 0
        beta:  (B, num_classes), β > 0
    Returns:
        kl: (B, num_classes)
    """
    S = alpha + beta
    kl = (
        torch.lgamma(alpha)
        + torch.lgamma(beta)
        - torch.lgamma(S)
        + (alpha - 1.0) * torch.digamma(alpha)
        + (beta - 1.0) * torch.digamma(beta)
        - (S - 2.0) * torch.digamma(S)
    )
    return kl


def log_likelihood_loss(alpha, beta, target, neg_weight=0.2):
    """
    Asymmetric evidence risk loss (Eq. in paper).

    L_risk = y·(ψ(S) - ψ(α)) + (1-y)·(ψ(S) - ψ(β))·w_neg

    Args:
        alpha: (B, num_classes)
        beta:  (B, num_classes)
        target: (B, num_classes), binary labels
        neg_weight: w_neg, default 0.2
    Returns:
        loss: (B, num_classes)
    """
    S = alpha + beta
    digamma_S = torch.digamma(S)
    digamma_alpha = torch.digamma(alpha)
    digamma_beta = torch.digamma(beta)

    loss_pos = target * (digamma_S - digamma_alpha)
    loss_neg = (1 - target) * (digamma_S - digamma_beta) * neg_weight

    return loss_pos + loss_neg


class EDLLoss(nn.Module):
    """
    EDL loss: L = L_risk + λ(t)·L_KL

    Args:
        kl_annealing_epochs: total annealing period reference
        uncertainty_gamma: unused (kept for config compatibility)
        neg_weight: w_neg for asymmetric risk loss
        annealing_type: 'linear' or 'cyclical'
    """

    def __init__(self, kl_annealing_epochs=20, uncertainty_gamma=1.0,
                 neg_weight=0.2, annealing_type='cyclical'):
        super(EDLLoss, self).__init__()
        self.kl_annealing_epochs = kl_annealing_epochs
        self.uncertainty_gamma = uncertainty_gamma
        self.neg_weight = neg_weight
        self.annealing_type = annealing_type

    def get_annealing_coef(self, epoch, total_epochs):
        """
        Cyclical KL annealing: 4 cycles, warmup for first half of each cycle.
        λ(t) = min(1, (t mod T_cycle) / T_warmup) where T_warmup = T_cycle / 2
        """
        if epoch is None:
            return 1.0

        if self.annealing_type == 'linear':
            return min(1.0, float(epoch) / float(self.kl_annealing_epochs))

        elif self.annealing_type == 'cyclical':
            cycles = 4
            if total_epochs is None:
                total_epochs = 100

            period = max(1, total_epochs // cycles)
            step_in_cycle = epoch % period
            warmup = max(1, period // 2)
            coef = min(1.0, float(step_in_cycle) / float(warmup))
            return coef

        return 1.0

    def forward(self, alpha, beta, target, epoch=None, total_epochs=None):
        """
        Total loss = L_risk + λ(t)·L_KL

        Args:
            alpha: (B, num_classes)
            beta:  (B, num_classes)
            target: (B, num_classes)
            epoch: current epoch
            total_epochs: total training epochs
        """
        # Asymmetric evidence risk loss
        risk_loss = log_likelihood_loss(alpha, beta, target, neg_weight=self.neg_weight)
        mean_risk_loss = risk_loss.mean()

        # KL divergence
        kl_loss = kl_divergence(alpha, beta)
        mean_kl_loss = kl_loss.mean()

        # Annealing coefficient
        annealing_coef = self.get_annealing_coef(epoch, total_epochs)

        # Total loss
        total_loss = mean_risk_loss + annealing_coef * mean_kl_loss

        loss_components = {
            'total_loss': total_loss.item(),
            'likelihood_loss': mean_risk_loss.item(),
            'kl_loss': mean_kl_loss.item(),
            'annealing_coef': annealing_coef
        }

        return total_loss, loss_components


class EDLMultiLabelLoss(nn.Module):
    """Multi-label EDL loss with optional class weighting."""

    def __init__(self, kl_annealing_epochs=20, uncertainty_gamma=1.0, class_weights=None):
        super(EDLMultiLabelLoss, self).__init__()
        self.kl_annealing_epochs = kl_annealing_epochs
        self.uncertainty_gamma = uncertainty_gamma
        self.class_weights = class_weights

    def forward(self, alpha, beta, target, epoch=None, total_epochs=None):
        if alpha.shape != beta.shape or alpha.shape != target.shape:
            raise ValueError(f"Shape mismatch: alpha{alpha.shape}, beta{beta.shape}, target{target.shape}")

        risk_loss = log_likelihood_loss(alpha, beta, target)

        if self.class_weights is not None:
            if isinstance(self.class_weights, torch.Tensor):
                weights = self.class_weights.to(target.device)
            else:
                weights = torch.tensor(self.class_weights, device=target.device, dtype=target.dtype)
            weights = weights.unsqueeze(0).expand_as(target)
            risk_loss = risk_loss * weights

        mean_risk_loss = risk_loss.mean()
        kl_loss = kl_divergence(alpha, beta)
        mean_kl_loss = kl_loss.mean()

        if epoch is not None:
            annealing_coef = min(1.0, float(epoch) / float(self.kl_annealing_epochs))
        else:
            annealing_coef = 1.0

        total_loss = mean_risk_loss + annealing_coef * mean_kl_loss

        loss_components = {
            'total_loss': total_loss.item(),
            'likelihood_loss': mean_risk_loss.item(),
            'kl_loss': mean_kl_loss.item(),
            'annealing_coef': annealing_coef
        }

        return total_loss, loss_components


def compute_evidence_regularization(alpha, beta, reg_type='l2'):
    if reg_type == 'l1':
        reg_loss = torch.mean(alpha + beta)
    elif reg_type == 'l2':
        reg_loss = torch.mean((alpha + beta) ** 2)
    elif reg_type == 'none':
        reg_loss = 0.0
    else:
        raise ValueError(f"Unknown regularization type: {reg_type}")
    return reg_loss


def compute_predictive_uncertainty(alpha, beta):
    """u = 2 / (α + β)"""
    return 2.0 / (alpha + beta)


def compute_probability_from_evidence(alpha, beta):
    """p = α / (α + β)"""
    return alpha / (alpha + beta)


def get_binary_predictions(alpha, beta, threshold=0.5):
    prob = compute_probability_from_evidence(alpha, beta)
    return (prob >= threshold).float()


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Learning Loss for multi-label classification.
    Samples sharing at least one positive label are treated as positive pairs.
    """

    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            features = features.unsqueeze(1)

        batch_size = features.shape[0]
        if labels is not None and mask is None:
            mask = torch.matmul(labels, labels.T)
            mask = (mask > 0).float().to(device)
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError(f'Unknown mode: {self.contrast_mode}')

        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs > 0, mask_pos_pairs, torch.ones_like(mask_pos_pairs))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
