"""
Basic layers for Vision Transformer implementation.

Contains the fundamental building blocks:
- DropPath: Stochastic depth for regularization
- Mlp: Feed-forward network
- FlashAttention: Flash attention for efficiency (PyTorch 2.0+)
- Attention: Multi-head self-attention
- Block: Transformer block with attention and MLP
- PatchEmbed: Image to patch embedding
- PatchSampler: Select patches based on saliency maps
"""

import math
import torch
import torch.nn as nn
from functools import partial
from packaging import version
from collections import namedtuple

from timm.models.layers import trunc_normal_


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample.

    Args:
        x: Input tensor
        drop_prob: Probability of dropping path
        training: Whether in training mode

    Returns:
        Output tensor with dropped paths
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    """
    MLP block for Transformer.

    Args:
        in_features: Input feature dimension
        hidden_features: Hidden feature dimension
        out_features: Output feature dimension
        act_layer: Activation function
        drop: Dropout rate
    """

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


Config = namedtuple('FlashAttentionConfig', ['enable_flash', 'enable_math', 'enable_mem_efficient'])


class FlashAttention(nn.Module):
    """
    Flash Attention implementation for PyTorch 2.0+.

    Uses scaled_dot_product_attention with optimized kernel selection.
    """

    def __init__(self):
        super().__init__()

        self.cpu_config = Config(True, True, True)
        self.cuda_config = None

        if torch.cuda.is_available():
            device_properties = torch.cuda.get_device_properties(torch.device('cuda'))

            if device_properties.major == 8 and device_properties.minor == 0:
                self.cuda_config = Config(True, False, False)
            else:
                self.cuda_config = Config(False, True, True)
        else:
            self.cuda_config = Config(False, True, True)

    def forward(self, q, k, v):
        config = self.cuda_config if q.is_cuda else self.cpu_config

        # Flash attention - https://arxiv.org/abs/2205.14135
        with torch.backends.cuda.sdp_kernel(**config._asdict()):
            out = nn.functional.scaled_dot_product_attention(q, k, v)

        return out


class Attention(nn.Module):
    """
    Multi-head self-attention module.

    Args:
        dim: Input feature dimension
        num_heads: Number of attention heads
        qkv_bias: Whether to use bias in QKV projection
        qk_scale: Custom scale for QK computation
        attn_drop: Attention dropout rate
        proj_drop: Output projection dropout rate
        use_flash: Whether to use flash attention
    """

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.,
        proj_drop=0.,
        use_flash=False
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.use_flash = use_flash

        if self.use_flash and (version.parse(torch.__version__) < version.parse('2.0.0')):
            print(
                f'in order to use flash attention, you must be using pytorch 2.0 or above, '
                f'but current version is: {version.parse(torch.__version__)}'
            )
            print(f'will disable the flash attention')
            self.use_flash = False

        if self.use_flash:
            self.flash_attn = FlashAttention()
            print(f"will use the Flash Attention.")

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        """
        Forward pass for attention.

        Args:
            x: Input tensor, shape (B, N, C)

        Returns:
            out: Output tensor, shape (B, N, C)
            attn: Attention weights (if not using flash attention)
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.use_flash:
            out = self.flash_attn(q, k, v)
            attn = None
            x = out.permute(0, 2, 1, 3).reshape(B, N, C)
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(nn.Module):
    """
    Transformer block with attention and MLP.

    Args:
        dim: Input feature dimension
        num_heads: Number of attention heads
        mlp_ratio: MLP expansion ratio
        qkv_bias: Whether to use bias in QKV projection
        qk_scale: Custom scale for QK computation
        drop: Dropout rate
        attn_drop: Attention dropout rate
        drop_path: Drop path rate
        act_layer: Activation function
        norm_layer: Normalization layer
        init_values: Layer scale initialization values
        use_flash: Whether to use flash attention
    """

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.,
        qkv_bias=False,
        qk_scale=None,
        drop=0.,
        attn_drop=0.,
        drop_path=0.,
        act_layer=nn.GELU,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_values=0,
        use_flash=False
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, use_flash=use_flash
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x, return_attention=False):
        """
        Forward pass for transformer block.

        Args:
            x: Input tensor, shape (B, N, C)
            return_attention: Whether to return attention weights

        Returns:
            x: Output tensor
            attn: Attention weights (if return_attention=True)
        """
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        if self.gamma_1 is None:
            x = x + self.drop_path(y)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * y)
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.

    Args:
        img_size: Input image size
        patch_size: Patch size
        in_chans: Number of input channels
        embed_dim: Embedding dimension
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.grid_size = (img_size // patch_size, img_size // patch_size)
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        """
        Forward pass for patch embedding.

        Args:
            x: Input tensor, shape (B, C, H, W)

        Returns:
            Embedded patches, shape (B, embed_dim, H/patch_size, W/patch_size)
        """
        B, C, H, W = x.shape
        return self.proj(x)


class PatchSampler(object):
    """
    Select patches based on saliency maps.

    Args:
        mask_size: Size of the mask
        patch_size: Size of each patch
        mask_ratio: Ratio of patches to mask (default 0.25)
    """

    def __init__(self, mask_size, patch_size, mask_ratio=0.25):
        self.mask_size = mask_size
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio

    def __call__(self, pmap):
        """
        Sample patches based on saliency map.

        Args:
            pmap: Saliency probability map, shape (B, 1, H, W)

        Returns:
            active_idx: Indices of selected patches, shape (B, num_patches)
        """
        B, C, H, W = pmap.shape
        num_sample = int((1 - self.mask_ratio) * H * W)

        # Get indices of top-k patches
        feat_idx = pmap.flatten(1).argsort(descending=True)[:, :num_sample]
        feat_idx += 1  # class embedding concat before the image embedding
        cls_idx = torch.zeros((B, 1), dtype=torch.int64, device=pmap.device)
        active_idx = torch.cat([cls_idx, feat_idx], dim=1)
        return active_idx