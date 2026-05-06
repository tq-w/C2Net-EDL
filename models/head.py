"""
Evidential Deep Learning (EDL) head for multi-label classification.

Dual-branch Beta evidence prediction head with independent α and β branches.
Each branch outputs evidence parameters for Beta distributions.

Key features:
- Independent α and β branches (no weight sharing)
- Softplus activation + 1.0 to ensure positive evidence parameters
- Evidence scaling factor σ for dynamic range control
- Differential initialization: α bias=-2.0, β bias=0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_


class EDLHead(nn.Module):
    """
    Dual-branch Beta evidence prediction head.

    α branch predicts positive evidence, β branch predicts negative evidence.
    Both branches are structurally symmetric but have fully independent parameters.

    Args:
        embed_dim: Input embedding dimension from backbone
        num_classes: Number of output classes
        layers: Number of MLP layers per branch (1, 2, or 3)
        evidence_scale: Scaling factor σ for evidence parameters
    """

    def __init__(self, embed_dim, num_classes, layers=3, evidence_scale=10.0):
        super(EDLHead, self).__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.layers = layers
        self.evidence_scale = evidence_scale

        if self.layers == 3:
            channels = [self.embed_dim, self.embed_dim // 2, self.embed_dim // 4, self.num_classes]
            self.alpha_head = nn.Sequential(
                nn.Linear(channels[0], channels[1]),
                nn.GELU(),
                nn.Dropout(p=0.1),
                nn.Linear(channels[1], channels[2]),
                nn.GELU(),
                nn.Dropout(p=0.1),
                nn.Linear(channels[2], channels[3])
            )
            self.beta_head = nn.Sequential(
                nn.Linear(channels[0], channels[1]),
                nn.GELU(),
                nn.Dropout(p=0.1),
                nn.Linear(channels[1], channels[2]),
                nn.GELU(),
                nn.Dropout(p=0.1),
                nn.Linear(channels[2], channels[3])
            )
        elif self.layers == 2:
            channels = [self.embed_dim, self.embed_dim // 4, self.num_classes]
            self.alpha_head = nn.Sequential(
                nn.Linear(channels[0], channels[1]),
                nn.GELU(),
                nn.Dropout(p=0.1),
                nn.Linear(channels[1], channels[2])
            )
            self.beta_head = nn.Sequential(
                nn.Linear(channels[0], channels[1]),
                nn.GELU(),
                nn.Dropout(p=0.1),
                nn.Linear(channels[1], channels[2])
            )
        elif self.layers == 1:
            channels = [self.embed_dim, self.num_classes]
            self.alpha_head = nn.Sequential(nn.Linear(channels[0], channels[1]))
            self.beta_head = nn.Sequential(nn.Linear(channels[0], channels[1]))

        self.channel_bn = nn.BatchNorm2d(
            self.embed_dim,
            eps=1e-6,
            momentum=0.99,
        )
        self.init_weights()

    def init_weights(self):
        """Xavier initialization for weights; differential bias initialization."""
        for branch in [self.alpha_head, self.beta_head]:
            for m in branch:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0.0)
        # α branch last layer bias = -2.0 (weaker initial positive evidence)
        # β branch last layer bias = 0 (uniform prior)
        last_alpha = None
        last_beta = None
        for m in self.alpha_head:
            if isinstance(m, nn.Linear):
                last_alpha = m
        for m in self.beta_head:
            if isinstance(m, nn.Linear):
                last_beta = m
        if last_alpha is not None:
            nn.init.constant_(last_alpha.bias, -2.0)
        if last_beta is not None:
            nn.init.constant_(last_beta.bias, 0.0)

    def forward(self, x):
        """
        Forward pass for dual-branch EDL head.

        Args:
            x: Input tensor, shape (B, embed_dim) or (B, embed_dim, 1, 1)

        Returns:
            alpha: Positive evidence, shape (B, num_classes), α = σ·Softplus(z_α) + 1
            beta: Negative evidence, shape (B, num_classes), β = σ·Softplus(z_β) + 1
        """
        if len(x.shape) == 2:
            x = x.unsqueeze(2).unsqueeze(3)

        x = self.channel_bn(x)
        x = x.view(x.size(0), -1)

        # Independent dual-branch forward
        z_alpha = self.alpha_head(x)
        z_beta = self.beta_head(x)

        # Evidence activation: α = σ · Softplus(z_α) + 1, β = σ · Softplus(z_β) + 1
        alpha = F.softplus(z_alpha) * self.evidence_scale + 1.0
        beta = F.softplus(z_beta) * self.evidence_scale + 1.0

        return alpha, beta

    def predict_proba(self, alpha, beta):
        """p = α / (α + β)"""
        return alpha / (alpha + beta)

    def compute_uncertainty(self, alpha, beta):
        """u = 2 / (α + β)"""
        return 2.0 / (alpha + beta)

    def get_predictions(self, alpha, beta, threshold=0.5):
        """Binary predictions at given threshold."""
        probs = self.predict_proba(alpha, beta)
        return (probs >= threshold).float()


class RegHead(nn.Module):
    """Regression head for continuous value prediction (e.g., age prediction)."""
    def __init__(self, embed_dim, num_classes):
        super(RegHead, self).__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        channels = [self.embed_dim, self.embed_dim // 2, self.embed_dim // 4, self.num_classes]
        self.classifier = nn.Sequential(
            nn.Linear(channels[0], channels[2]),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(channels[2], channels[3]),
        )
        self.channel_bn = nn.BatchNorm2d(self.embed_dim, eps=1e-6, momentum=0.99)
        self.init_weights()

    def init_weights(self):
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.bias.data, 0.0)
                nn.init.normal_(m.weight.data, mean=0.0, std=2e-5)

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(2).unsqueeze(3)
        x = self.channel_bn(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class ClsHead(nn.Module):
    """Classification head for discrete value prediction (e.g., gender classification)."""
    def __init__(self, embed_dim, num_classes):
        super(ClsHead, self).__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        channels = [self.embed_dim, self.embed_dim // 2, self.embed_dim // 4, self.num_classes]
        self.classifier = nn.Sequential(
            nn.Linear(channels[0], channels[2]),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(channels[2], channels[3]),
        )
        self.channel_bn = nn.BatchNorm2d(self.embed_dim, eps=1e-6, momentum=0.99)
        self.init_weights()

    def init_weights(self):
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.bias.data, 0.0)
                nn.init.normal_(m.weight.data, mean=0.0, std=2e-5)

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(2).unsqueeze(3)
        x = self.channel_bn(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
