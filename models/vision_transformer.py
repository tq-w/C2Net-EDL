"""
Vision Transformer (ViT) implementation with multi-task and contrastive learning support.
Modified for Evidential Deep Learning (EDL) with Beta distribution evidence modeling.

Implements:
- Vision Transformer backbone
- Multi-task heads (disease, age, gender)
- Contrastive learning feature extraction
- Pretrained weight loading with position embedding interpolation
- EDL head for uncertainty-aware multi-label classification
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import trunc_normal_

from .layers import DropPath, Mlp, FlashAttention, Attention, Block, PatchEmbed, PatchSampler
from .head import EDLHead, RegHead, ClsHead


class VisionTransformer(nn.Module):
    """
    Vision Transformer with multi-task and contrastive learning support.
    Modified to support EDL head for evidential deep learning.

    Args:
        img_size: Input image size
        patch_size: Patch size for embedding
        in_chans: Number of input channels
        num_classes: Number of output classes
        embed_dim: Embedding dimension
        depth: Number of transformer blocks
        num_heads: Number of attention heads
        mlp_ratio: MLP expansion ratio
        qkv_bias: Whether to use bias in QKV projection
        qk_scale: Custom scale for QK computation
        drop_rate: Dropout rate
        attn_drop_rate: Attention dropout rate
        drop_path_rate: Drop path rate
        norm_layer: Normalization layer
        return_all_tokens: Whether to return all tokens
        init_values: Layer scale initialization values
        use_mean_pooling: Whether to use mean pooling
        masked_im_modeling: Whether to use masked image modeling
        feat_concat: Whether to concatenate CLS and mean features
        use_flash: Whether to use flash attention
        avg_pool: Whether to use average pooling for feature extraction
        global_pool: Whether to use global pooling
        contra_learning: Whether to enable contrastive learning
        contra_features_head: Type of feature head (linear, mlp, identity)
        contra_feat_dim: Dimension of contrastive features
        contra_feature: Type of contrastive feature (cls, mean, concat)
        multi_task: Whether to enable multi-task learning (age, gender)
        edl_mode: Whether to use EDL head instead of standard classification head
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=0,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        return_all_tokens=False,
        init_values=0,
        use_mean_pooling=False,
        masked_im_modeling=False,
        feat_concat=False,
        use_flash=False,
        avg_pool=False,
        global_pool=True,
        contra_learning=True,
        contra_features_head='linear',
        contra_feat_dim=128,
        contra_feature='cls',
        multi_task=True,
        edl_mode=True  # New parameter for EDL mode
    ):
        super().__init__()
        self.feature_head = None
        self.avg_pool = avg_pool
        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim
        self.return_all_tokens = return_all_tokens
        self.feat_concat = feat_concat
        self.contra_learning = contra_learning
        self.features_head = contra_features_head
        self.feat_dim = contra_feat_dim
        self.feature = contra_feature
        self.multi_task = multi_task
        self.edl_mode = edl_mode

        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches

        # Positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic depth decay
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[i], norm_layer=norm_layer, init_values=init_values, use_flash=use_flash
            )
            for i in range(depth)
        ])

        # Normalization layers
        self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None

        # Classification head - now supports EDL mode
        final_chs = embed_dim * 2 if self.feat_concat else embed_dim
        if self.edl_mode and num_classes > 0:
            # Use EDL head for evidential deep learning
            self.head = EDLHead(final_chs, num_classes)
        else:
            # Use standard classification head
            self.head = nn.Linear(final_chs, num_classes) if num_classes > 0 else nn.Identity()

        # Multi-task heads
        self.gender_head = ClsHead(final_chs, 1)
        self.age_head = RegHead(final_chs, 1)

        # Patch sampling for masked image modeling
        self.mask_ratio = 0.0
        self.patch_sampler = PatchSampler(mask_size=img_size, patch_size=patch_size, mask_ratio=self.mask_ratio)

        # Pooling layer
        pool_mode = 'avg'
        if pool_mode == 'avg':
            self.pool = nn.AvgPool2d(kernel_size=patch_size, stride=patch_size)
        elif pool_mode == 'max':
            self.pool = nn.MaxPool2d(kernel_size=patch_size, stride=patch_size)
        else:
            self.pool = None

        # Masked image modeling
        self.masked_im_modeling = masked_im_modeling
        if masked_im_modeling:
            self.masked_embed = nn.Parameter(torch.zeros(1, embed_dim))

        # Initialize weights
        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initialize weights for linear and layer norm layers."""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x, w, h):
        """
        Interpolate positional embeddings for different image sizes.

        Args:
            x: Input tokens
            w: Image width
            h: Image height

        Returns:
            Interpolated positional embeddings
        """
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def prepare_tokens(self, x, mask=None):
        """
        Prepare tokens by patch embedding and adding positional encoding.

        Args:
            x: Input images, shape (B, C, H, W)
            mask: Optional mask for masked image modeling

        Returns:
            tokens: Embedded tokens with positional encoding
        """
        B, nc, w, h = x.shape
        x = self.patch_embed(x)

        # Masked image modeling
        if mask is not None:
            x = self.mask_model(x, mask)
        x = x.flatten(2).transpose(1, 2)

        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Add positional encoding
        x = x + self.interpolate_pos_encoding(x, w, h)
        return self.pos_drop(x)

    def forward_head(self, x, pre_logits: bool = False):
        """
        Forward through the classification head.

        Args:
            x: Input tokens
            pre_logits: Whether to return pre-logits

        Returns:
            Output logits or pre-logits
        """
        if self.feat_concat:
            feats = x[:, 1:].mean(dim=1)
            x = torch.cat((x[:, 0], feats), dim=1)
        elif self.global_pool:
            x = x[:, 1:].mean(dim=1) if self.global_pool else x[:, 0]

        if self.edl_mode:
            # Return alpha and beta parameters for EDL
            return self.head(x)
        else:
            # Return standard logits
            return x if pre_logits else self.head(x)

    def forward_age(self, x):
        """
        Forward through the age prediction head.

        Args:
            x: Input tokens

        Returns:
            Age prediction
        """
        if self.feat_concat:
            feats = x[:, 1:].mean(dim=1)
            x = torch.cat((x[:, 0], feats), dim=1)
        elif self.global_pool:
            x = x[:, 1:].mean(dim=1) if self.global_pool else x[:, 0]
        return self.age_head(x)

    def forward_gender(self, x):
        """
        Forward through the gender classification head.

        Args:
            x: Input tokens

        Returns:
            Gender prediction
        """
        if self.feat_concat:
            feats = x[:, 1:].mean(dim=1)
            x = torch.cat((x[:, 0], feats), dim=1)
        elif self.global_pool:
            x = x[:, 1:].mean(dim=1) if self.global_pool else x[:, 0]
        return self.gender_head(x)

    def forward_first(self, x, return_all_tokens=None, mask=None, maps=None):
        """
        First forward pass through transformer blocks.

        Args:
            x: Input images
            return_all_tokens: Whether to return all tokens
            mask: Optional mask for masked image modeling
            maps: Optional saliency maps for patch selection

        Returns:
            features: Output features
        """
        if self.masked_im_modeling:
            assert mask is not None
            x = self.prepare_tokens(x, mask=mask)
        else:
            x = self.prepare_tokens(x)

        # Select patches based on saliency maps
        if maps is not None and self.mask_ratio < 1:
            maps = torch.cat(maps)
            maps = None if self.pool is None else self.pool(maps)
            active_idx = self.patch_sampler(maps)
            active_idx = active_idx.unsqueeze(-1).repeat(1, 1, self.embed_dim)
            x = torch.gather(x, dim=1, index=active_idx)

        # Forward through transformer blocks
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        if self.fc_norm is not None:
            x[:, 0] = self.fc_norm(x[:, 1:, :].mean(1))

        return_all_tokens = self.return_all_tokens if return_all_tokens is None else return_all_tokens
        if return_all_tokens:
            return x
        return x[:, 0]

    def forward_secend(self, f, features_head='linear', feat_dim=None, feature='cls'):
        """
        Second forward pass for contrastive learning feature extraction.

        Args:
            f: Transformer output features
            features_head: Type of feature head (linear, mlp, identity)
            feat_dim: Dimension of output features
            feature: Type of feature to extract (cls, mean, concat)

        Returns:
            Normalized contrastive features
        """
        if features_head == 'linear':
            self.feature_head = nn.Linear(
                self.embed_dim * 2 if feature == 'concat' else self.embed_dim, feat_dim
            ).to(f.device)
        elif features_head == 'mlp':
            self.feature_head = nn.Sequential(
                nn.Linear(self.embed_dim * 2 if feature == 'concat' else self.embed_dim, self.embed_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.embed_dim, feat_dim)
            ).to(f.device)
        elif features_head == 'identity':
            self.feature_head = nn.Identity().to(f.device)
        else:
            raise NotImplementedError(f'head not supported: {features_head}')

        if feature == 'cls':
            return F.normalize(self.feature_head(f[:, 0]), dim=1)
        elif feature == 'mean':
            return F.normalize(self.feature_head(f[:, 1:].mean(dim=1)), dim=1)
        elif feature == 'concat':
            return F.normalize(self.feature_head(torch.cat((f[:, 0], f[:, 1:].mean(dim=1)), dim=1)), dim=1)
        else:
            raise NotImplementedError(f'mode not supported: {feature}')

    def forward(self, x):
        """
        Forward pass for multi-task prediction.

        Args:
            x: Input images, shape (B, C, H, W)

        Returns:
            outputs: Dictionary containing predictions
                - 'd': Disease logits or (alpha, beta) for EDL
                - 'emb': Contrastive features (if contra_learning enabled)
                - 'a': Age prediction (if multi_task enabled)
                - 'g': Gender prediction (if multi_task enabled)
        """
        f = self.forward_first(x, return_all_tokens=True).to(x.device)
        cls_token = f[:, 0]

        # Disease prediction
        if hasattr(self, 'gnn_reasoning') and self.gnn_reasoning is not None:
            gnn_logits = self.gnn_reasoning(cls_token)
        else:
            if self.edl_mode:
                # For EDL mode, head returns (alpha, beta)
                alpha, beta = self.head(cls_token)
                gnn_logits = (alpha, beta)
            else:
                gnn_logits = self.head(cls_token)
        outputs = {'d': gnn_logits}

        # Contrastive learning features
        if getattr(self, 'contra_learning', False):
            emb = self.forward_secend(
                f, features_head=self.features_head, feat_dim=self.feat_dim, feature=self.feature
            )
            outputs['emb'] = emb

        # Multi-task outputs
        if getattr(self, 'multi_task', False):
            age = self.forward_age(f)
            gender = self.forward_gender(f)
            outputs['a'] = age
            outputs['g'] = gender

        return outputs

    def get_last_selfattention(self, x):
        """Get self-attention from the last transformer block."""
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                return blk(x, return_attention=True)

    def get_intermediate_layers(self, x, n=1):
        """
        Get output from the last n transformer blocks.

        Args:
            x: Input images
            n: Number of layers to return

        Returns:
            output: Concatenated intermediate outputs
            intermediate_output: List of intermediate outputs
        """
        x = self.prepare_tokens(x)
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        intermediate_output = output
        if not self.avg_pool:
            output = [x[:, 0] for x in intermediate_output]
        elif self.avg_pool:
            output = [torch.mean(intermediate_output[-1][:, 1:], dim=1)]
        output = torch.cat(output, dim=-1)
        return output, intermediate_output

    def get_num_layers(self):
        """Get number of transformer blocks."""
        return len(self.blocks)

    def mask_model(self, x, mask):
        """Apply masking to the model."""
        x.permute(0, 2, 3, 1)[mask, :] = self.masked_embed.to(x.dtype)
        return x


def vit_tiny_vfm_multitaskv2(args, **kwargs):
    """Vision Transformer Tiny model."""
    model = VisionTransformer(
        patch_size=args.patch_size, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
        img_size=args.input_size, qkv_bias=True, avg_pool=args.global_pool, edl_mode=True, **kwargs
    )
    return model


def vit_small_vfm_multitaskv2(args, **kwargs):
    """Vision Transformer Small model."""
    model = VisionTransformer(
        patch_size=args.patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        img_size=args.input_size, qkv_bias=True, avg_pool=args.global_pool, edl_mode=True, **kwargs
    )
    return model


def vit_base_vfm_multitaskv2_2(args, **kwargs):
    """Vision Transformer Base model."""
    contra_features_head = getattr(args, 'contra_features_head', 'linear')
    contra_feat_dim = getattr(args, 'contra_feat_dim', 128)
    contra_feature = getattr(args, 'contra_feature', 'cls')
    model = VisionTransformer(
        patch_size=args.patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        img_size=args.input_size, qkv_bias=True, avg_pool=args.global_pool,
        contra_learning=args.contra_learning, contra_features_head=contra_features_head,
        contra_feat_dim=contra_feat_dim, contra_feature=contra_feature,
        multi_task=args.multi_task, edl_mode=True, **kwargs
    )
    return model


def vit_large_vfm_multitaskv2(args, **kwargs):
    """Vision Transformer Large model."""
    model = VisionTransformer(
        patch_size=args.patch_size, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4,
        img_size=args.input_size, qkv_bias=True, avg_pool=args.global_pool, edl_mode=True, **kwargs
    )
    return model