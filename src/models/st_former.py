"""
GaiaSphere — Spatio-Temporal Transformer (ST-Former)
======================================================
Processes sequences of spatial feature maps across time to model how
environmental conditions evolve. Uses factorized attention:
  - Spatial attention: each position attends to its neighbors within a frame
  - Temporal attention: each position attends to itself across all timesteps

This approach scales to long sequences while preserving both where and when
information, critical for detecting flood onset, crop stress progression,
or wildfire spread.

Architecture inspired by:
  TimeSformer (Bertasius et al., 2021)
  SatMAE (Cong et al., 2022)

Author: GaiaSphere Project
"""

import torch
import torch.nn as nn
from einops import rearrange
from typing import Optional


# ---------------------------------------------------------------------------
# Factorized Attention Block
# ---------------------------------------------------------------------------

class SpatialAttention(nn.Module):
    """Self-attention across spatial positions within a single timestep."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, N, D]  (batch, time, spatial_tokens, embed_dim)
        Returns:
            x: [B, T, N, D]
        """
        B, T, N, D = x.shape
        residual = x

        # Merge batch and time → treat each frame independently
        x = rearrange(x, 'b t n d -> (b t) n d')
        x = self.norm(x)

        qkv = self.qkv(x).reshape(B * T, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B * T, N, D)
        x = self.proj(x)
        x = rearrange(x, '(b t) n d -> b t n d', b=B, t=T)

        return residual + x


class TemporalAttention(nn.Module):
    """Self-attention across timesteps for each spatial position."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, N, D]
        Returns:
            x: [B, T, N, D]
        """
        B, T, N, D = x.shape
        residual = x

        # Merge batch and spatial → each position attends across time
        x = rearrange(x, 'b t n d -> (b n) t d')
        x = self.norm(x)

        qkv = self.qkv(x).reshape(B * N, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B * N, T, D)
        x = self.proj(x)
        x = rearrange(x, '(b n) t d -> b t n d', b=B, n=N)

        return residual + x


class STFormerBlock(nn.Module):
    """
    One block of the Spatio-Temporal Transformer.
    Applies: Spatial Attn → Temporal Attn → MLP
    """

    def __init__(
        self,
        embed_dim: int,
        spatial_heads: int,
        temporal_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.spatial_attn = SpatialAttention(embed_dim, spatial_heads, dropout)
        self.temporal_attn = TemporalAttention(embed_dim, temporal_heads, dropout)

        self.norm = nn.LayerNorm(embed_dim)
        mlp_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.spatial_attn(x)
        x = self.temporal_attn(x)
        x = x + self.mlp(self.norm(x))
        return x


# ---------------------------------------------------------------------------
# Temporal Positional Encoding
# ---------------------------------------------------------------------------

class TemporalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding over timesteps.
    Supports non-uniform time intervals (e.g., irregular satellite revisit).
    """

    def __init__(self, embed_dim: int, max_len: int = 365):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, t: int) -> torch.Tensor:
        """Return positional encoding for t timesteps: [1, t, 1, embed_dim]"""
        return self.pe[:t].unsqueeze(0).unsqueeze(2)


# ---------------------------------------------------------------------------
# ST-Former: Main Model
# ---------------------------------------------------------------------------

class STFormer(nn.Module):
    """
    Spatio-Temporal Transformer for environmental time-series analysis.

    Takes a sequence of spatial feature maps (output of ViT-Geo over T days)
    and produces a temporally-fused representation encoding what changed,
    where, and when.

    Args:
        embed_dim:       Feature dimension (must match ViT-Geo output)
        depth:           Number of ST-Former blocks
        spatial_heads:   Heads for spatial attention
        temporal_heads:  Heads for temporal attention
        mlp_ratio:       MLP expansion factor
        max_seq_len:     Maximum number of timesteps supported
        dropout:         Dropout probability
    """

    def __init__(
        self,
        embed_dim: int = 768,
        depth: int = 6,
        spatial_heads: int = 8,
        temporal_heads: int = 4,
        mlp_ratio: float = 4.0,
        max_seq_len: int = 60,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        self.temporal_pe = TemporalPositionalEncoding(embed_dim, max_seq_len)

        self.blocks = nn.ModuleList([
            STFormerBlock(embed_dim, spatial_heads, temporal_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # Temporal aggregation: learned weighted pooling over timesteps
        self.temporal_pool = nn.Sequential(
            nn.Linear(embed_dim, 1),
            nn.Softmax(dim=1),
        )

    def forward(
        self,
        x: torch.Tensor,
        weather: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:       [B, T, N, D]   Spatial features over T timesteps
                     (ViT-Geo patch tokens, stacked in time)
            weather: [B, T, W]      Optional weather time-series per timestep
                     (ERA5 variables: temperature, precip, humidity, wind, ...)

        Returns:
            fused:   [B, N, D]      Spatio-temporally fused feature map
        """
        B, T, N, D = x.shape

        # Add temporal positional encoding
        tpe = self.temporal_pe(T)          # [1, T, 1, D]
        x = x + tpe

        # Optionally fuse weather signal into spatial tokens (broadcast)
        if weather is not None:
            # weather: [B, T, W] → project to [B, T, 1, D] → broadcast to [B, T, N, D]
            # (weather projection must be defined in subclass or passed pre-projected)
            weather = weather.unsqueeze(2).expand(-1, -1, N, -1)
            x = x + weather[..., :D]       # simple additive fusion (truncate if needed)

        # ST-Former blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)   # [B, T, N, D]

        # Temporal aggregation: weighted pool over T → [B, N, D]
        weights = self.temporal_pool(x.mean(dim=2))  # [B, T, 1]
        x = (x * weights.unsqueeze(-1)).sum(dim=1)   # [B, N, D]

        return x


# ---------------------------------------------------------------------------
# Weather Projection Helper
# ---------------------------------------------------------------------------

class WeatherProjection(nn.Module):
    """Projects ERA5 weather variables into embed_dim space."""

    def __init__(self, weather_dim: int, embed_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(weather_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim),
        )

    def forward(self, weather: torch.Tensor) -> torch.Tensor:
        """
        Args:
            weather: [B, T, weather_dim]
        Returns:
            [B, T, embed_dim]
        """
        return self.proj(weather)


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model = STFormer(
        embed_dim=768,
        depth=6,
        spatial_heads=8,
        temporal_heads=4,
        max_seq_len=30,
    )

    # Simulate 30-day sequence, 256 spatial patches, 768-dim features
    x = torch.randn(2, 30, 256, 768)   # [B, T, N, D]

    fused = model(x)
    print(f"Input:  {x.shape}")        # [2, 30, 256, 768]
    print(f"Output: {fused.shape}")    # [2, 256, 768]
    print("ST-Former: OK ✓")
