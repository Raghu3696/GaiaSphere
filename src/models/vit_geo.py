"""
GaiaSphere — Vision Transformer (Geo-Aware)
============================================
A patch-based Vision Transformer adapted for multi-spectral satellite imagery.

Key differences from standard ViT:
  - Accepts arbitrary number of input channels (Sentinel-2: 13 bands)
  - Geo-positional encoding: Fourier features over (lat, lon) fused with patch PE
  - Optional temporal conditioning for sequential satellite passes

Author: GaiaSphere Project
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Geo-aware Positional Encoding
# ---------------------------------------------------------------------------

class GeoPositionalEncoding(nn.Module):
    """
    Encodes geographic coordinates (latitude, longitude) as Fourier features
    and fuses them with the standard learned patch position embedding.

    Fourier features project (lat, lon) into a high-dimensional sine/cosine
    space, giving the model a continuous, smooth representation of location
    rather than a discrete grid.
    """

    def __init__(self, embed_dim: int, num_fourier_bands: int = 64):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_fourier_bands = num_fourier_bands

        # Learnable frequency matrix for Fourier projection
        self.freq_matrix = nn.Parameter(
            torch.randn(2, num_fourier_bands) * 0.1
        )

        # Project Fourier features → embed_dim
        fourier_out_dim = num_fourier_bands * 2  # sin + cos
        self.geo_proj = nn.Sequential(
            nn.Linear(fourier_out_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: [B, 2] tensor of (lat, lon) in degrees, normalized to [-1, 1]
        Returns:
            geo_encoding: [B, embed_dim]
        """
        # Project to Fourier space
        proj = coords @ self.freq_matrix  # [B, num_fourier_bands]
        fourier_features = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
        return self.geo_proj(fourier_features)  # [B, embed_dim]


# ---------------------------------------------------------------------------
# Multi-Head Self-Attention
# ---------------------------------------------------------------------------

class Attention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = Attention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Patch Embedding (multi-spectral)
# ---------------------------------------------------------------------------

class MultiSpectralPatchEmbed(nn.Module):
    """
    Embeds multi-spectral image patches into a flat token sequence.
    Accepts in_channels != 3 (e.g., 13 for Sentinel-2).
    """

    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 16,
        in_channels: int = 13,
        embed_dim: int = 768,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]  e.g. [B, 13, 256, 256]
        Returns:
            tokens: [B, num_patches, embed_dim]
        """
        x = self.proj(x)                        # [B, embed_dim, H/P, W/P]
        x = rearrange(x, 'b c h w -> b (h w) c')
        return self.norm(x)


# ---------------------------------------------------------------------------
# ViT-Geo: Main Model
# ---------------------------------------------------------------------------

class ViTGeo(nn.Module):
    """
    Vision Transformer with geographic positional encoding.

    Designed for multi-spectral satellite imagery. Produces a rich spatial
    feature map that can be consumed by the ST-Former or task-specific heads.

    Args:
        img_size:       Spatial resolution of input tiles (default 256×256)
        patch_size:     Size of each patch token (default 16)
        in_channels:    Number of spectral bands (13 for Sentinel-2)
        embed_dim:      Transformer hidden dimension
        depth:          Number of transformer blocks
        num_heads:      Number of attention heads
        mlp_ratio:      MLP expansion factor
        dropout:        Dropout probability
        num_fourier_bands: Bands for geo-positional Fourier encoding
    """

    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 16,
        in_channels: int = 13,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        num_fourier_bands: int = 64,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # Patch embedding
        self.patch_embed = MultiSpectralPatchEmbed(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        # [CLS] token and positional embedding (learned, for patch positions)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Geographic positional encoding
        self.geo_encoder = GeoPositionalEncoding(embed_dim, num_fourier_bands)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        images: torch.Tensor,
        coords: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            images: [B, C, H, W]  Multi-spectral satellite tile
            coords: [B, 2]        (lat, lon) of tile center, normalized to [-1, 1]

        Returns:
            cls_token:     [B, embed_dim]           Global scene representation
            patch_tokens:  [B, num_patches, embed_dim]  Spatial feature map
        """
        B = images.shape[0]

        # Tokenize
        x = self.patch_embed(images)               # [B, N, D]

        # Prepend CLS token
        cls = repeat(self.cls_token, '1 1 d -> b 1 d', b=B)
        x = torch.cat([cls, x], dim=1)             # [B, N+1, D]

        # Add patch positional embedding
        x = x + self.pos_embed

        # Optionally add geographic encoding to CLS token
        if coords is not None:
            geo_enc = self.geo_encoder(coords)     # [B, D]
            x[:, 0] = x[:, 0] + geo_enc

        x = self.dropout(x)

        # Transformer
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        cls_token = x[:, 0]                        # [B, D]
        patch_tokens = x[:, 1:]                    # [B, N, D]

        return cls_token, patch_tokens


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model = ViTGeo(
        img_size=256,
        patch_size=16,
        in_channels=13,
        embed_dim=768,
        depth=12,
        num_heads=12,
    )

    dummy_images = torch.randn(2, 13, 256, 256)
    dummy_coords = torch.tensor([[0.3, 0.7], [-0.1, 0.4]])  # normalized lat/lon

    cls, patches = model(dummy_images, dummy_coords)
    print(f"CLS token:    {cls.shape}")       # [2, 768]
    print(f"Patch tokens: {patches.shape}")   # [2, 256, 768]
    print("ViT-Geo: OK ✓")
