"""
GaiaSphere — GaiaNet (Full Multi-Modal Fusion Model)
======================================================
Combines ViT-Geo + ST-Former + GeoGraph into one unified model.
Produces task-specific predictions for:
  - FloodNet:      Binary flood / non-flood segmentation map
  - CropHealthNet: Continuous crop health score (NDVI proxy)
  - WildfireNet:   Fire spread probability per cell

Input modalities:
  images:   [B, T, C, H, W]   Multi-spectral satellite time-series
  weather:  [B, T, W]         ERA5 weather variables
  dem:      [B, 1, H, W]      Digital Elevation Model (static)
  coords:   [B, 2]            Tile center (lat, lon)

Author: GaiaSphere Project
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Optional, Dict

from .vit_geo import ViTGeo
from .st_former import STFormer, WeatherProjection
from .geo_graph import GeoGraph, build_grid_graph


# ---------------------------------------------------------------------------
# DEM Encoder (terrain)
# ---------------------------------------------------------------------------

class DEMEncoder(nn.Module):
    """
    Lightweight CNN encoder for Digital Elevation Model data.
    Extracts terrain features at the same spatial resolution as ViT-Geo patches.
    """

    def __init__(self, out_dim: int = 256, patch_size: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, out_dim, kernel_size=patch_size, stride=patch_size),
        )
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, dem: torch.Tensor) -> torch.Tensor:
        """
        Args:
            dem: [B, 1, H, W]
        Returns:
            terrain_tokens: [B, N, out_dim]  (N = H/P * W/P patches)
        """
        x = self.encoder(dem)                          # [B, out_dim, H/P, W/P]
        x = rearrange(x, 'b c h w -> b (h w) c')
        return self.norm(x)


# ---------------------------------------------------------------------------
# Task-Specific Prediction Heads
# ---------------------------------------------------------------------------

class FloodHead(nn.Module):
    """
    Outputs a per-patch flood probability → reshaped to spatial map.
    Binary segmentation: flood / no-flood.
    """

    def __init__(self, in_dim: int, patch_grid: int):
        super().__init__()
        self.patch_grid = patch_grid
        self.head = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.GELU(),
            nn.Linear(in_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, in_dim]  Per-patch features
        Returns:
            flood_map: [B, 1, patch_grid, patch_grid]  Probability [0,1]
        """
        B, N, _ = x.shape
        out = self.head(x).squeeze(-1)                         # [B, N]
        out = out.view(B, 1, self.patch_grid, self.patch_grid) # [B, 1, G, G]
        return torch.sigmoid(out)


class CropHealthHead(nn.Module):
    """
    Outputs a continuous crop health index per patch.
    Regression target: NDVI or similar vegetation index in [-1, 1].
    """

    def __init__(self, in_dim: int, patch_grid: int):
        super().__init__()
        self.patch_grid = patch_grid
        self.head = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.GELU(),
            nn.Linear(in_dim // 2, 1),
            nn.Tanh(),   # Output in [-1, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        out = self.head(x).squeeze(-1)
        return out.view(B, 1, self.patch_grid, self.patch_grid)


class WildfireHead(nn.Module):
    """
    Outputs fire spread probability per patch.
    Binary classification: fire / no-fire.
    Also predicts fire spread direction (auxiliary).
    """

    def __init__(self, in_dim: int, patch_grid: int):
        super().__init__()
        self.patch_grid = patch_grid
        self.fire_prob = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.GELU(),
            nn.Linear(in_dim // 2, 1),
        )
        # Auxiliary: wind-adjusted spread direction (8 directions)
        self.spread_dir = nn.Sequential(
            nn.Linear(in_dim, in_dim // 4),
            nn.GELU(),
            nn.Linear(in_dim // 4, 8),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, N, _ = x.shape
        prob = torch.sigmoid(self.fire_prob(x).squeeze(-1))   # [B, N]
        prob = prob.view(B, 1, self.patch_grid, self.patch_grid)
        direction = self.spread_dir(x)                         # [B, N, 8]
        direction = direction.view(B, self.patch_grid, self.patch_grid, 8)
        return {"fire_prob": prob, "spread_direction": direction}


# ---------------------------------------------------------------------------
# GaiaNet: Full Model
# ---------------------------------------------------------------------------

class GaiaNet(nn.Module):
    """
    GaiaNet — multi-modal, multi-task environmental threat detector.

    Architecture:
        ViT-Geo  →  spatial features per timestep
        ST-Former →  fuse across time + weather
        DEMEncoder →  terrain features
        Fusion    →  concat satellite + terrain
        GeoGraph  →  propagate across geographic graph
        Heads     →  task-specific predictions

    Args:
        img_size:        Spatial resolution of satellite tiles
        patch_size:      Patch size for ViT-Geo
        in_channels:     Satellite bands (13 for Sentinel-2)
        vit_embed_dim:   ViT-Geo hidden dimension
        vit_depth:       ViT-Geo transformer layers
        st_depth:        ST-Former layers
        weather_dim:     Number of ERA5 weather variables
        gnn_hidden:      GeoGraph hidden dimension
        gnn_layers:      GeoGraph layers
        task:            One of 'flood', 'crop', 'wildfire', 'all'
    """

    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 16,
        in_channels: int = 13,
        vit_embed_dim: int = 768,
        vit_depth: int = 12,
        vit_heads: int = 12,
        st_depth: int = 6,
        weather_dim: int = 16,
        dem_embed_dim: int = 256,
        gnn_hidden: int = 256,
        gnn_layers: int = 3,
        task: str = 'flood',
        dropout: float = 0.1,
    ):
        super().__init__()
        self.task = task
        self.patch_grid = img_size // patch_size  # e.g., 16

        # Backbone: Vision Transformer
        self.vit = ViTGeo(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=vit_embed_dim,
            depth=vit_depth,
            num_heads=vit_heads,
            dropout=dropout,
        )

        # Weather encoder
        self.weather_proj = WeatherProjection(weather_dim, vit_embed_dim)

        # Spatio-Temporal Transformer
        self.st_former = STFormer(
            embed_dim=vit_embed_dim,
            depth=st_depth,
            dropout=dropout,
        )

        # Terrain encoder
        self.dem_encoder = DEMEncoder(dem_embed_dim, patch_size)

        # Fusion: satellite (vit_embed_dim) + terrain (dem_embed_dim) → gnn_hidden
        self.fusion = nn.Sequential(
            nn.Linear(vit_embed_dim + dem_embed_dim, gnn_hidden),
            nn.GELU(),
            nn.LayerNorm(gnn_hidden),
        )

        # Graph Neural Network
        self.gnn = GeoGraph(
            node_input_dim=gnn_hidden,
            edge_input_dim=8,
            hidden_dim=gnn_hidden,
            num_layers=gnn_layers,
            dropout=dropout,
        )

        # Task heads
        if task in ('flood', 'all'):
            self.flood_head = FloodHead(gnn_hidden, self.patch_grid)
        if task in ('crop', 'all'):
            self.crop_head = CropHealthHead(gnn_hidden, self.patch_grid)
        if task in ('wildfire', 'all'):
            self.wildfire_head = WildfireHead(gnn_hidden, self.patch_grid)

    def forward(
        self,
        images: torch.Tensor,              # [B, T, C, H, W]
        weather: Optional[torch.Tensor],   # [B, T, weather_dim]
        dem: torch.Tensor,                 # [B, 1, H, W]
        coords: Optional[torch.Tensor],    # [B, 2]
        edge_index: Optional[torch.Tensor] = None,   # [2, E]
        edge_features: Optional[torch.Tensor] = None,# [E, 8]
    ) -> Dict[str, torch.Tensor]:

        B, T, C, H, W = images.shape

        # 1. Extract visual features per timestep with ViT-Geo
        patch_seq = []
        for t in range(T):
            _, patches = self.vit(images[:, t], coords)   # [B, N, D]
            patch_seq.append(patches)
        patch_seq = torch.stack(patch_seq, dim=1)          # [B, T, N, D]

        # 2. Project weather & fuse with ST-Former
        weather_embed = None
        if weather is not None:
            weather_embed = self.weather_proj(weather)     # [B, T, D]

        fused = self.st_former(patch_seq, weather_embed)   # [B, N, D]

        # 3. Encode terrain
        terrain = self.dem_encoder(dem)                    # [B, N, dem_dim]

        # 4. Fuse satellite + terrain features
        x = self.fusion(torch.cat([fused, terrain], dim=-1))  # [B, N, gnn_hidden]

        # 5. Graph propagation
        N = x.shape[1]
        if edge_index is None:
            # Default: build grid graph over patch grid
            edge_index = build_grid_graph(
                self.patch_grid, self.patch_grid, device=x.device
            )
        if edge_features is None:
            E = edge_index.shape[1]
            edge_features = torch.zeros(E, 8, device=x.device)

        # Process each sample in batch through GNN (nodes differ per sample in general)
        gnn_out = []
        for b in range(B):
            node_b = x[b]                              # [N, gnn_hidden]
            out_b = self.gnn(node_b, edge_index, edge_features)  # [N, gnn_hidden]
            gnn_out.append(out_b)
        x = torch.stack(gnn_out, dim=0)               # [B, N, gnn_hidden]

        # 6. Task heads
        outputs = {}
        if self.task in ('flood', 'all'):
            outputs['flood'] = self.flood_head(x)
        if self.task in ('crop', 'all'):
            outputs['crop'] = self.crop_head(x)
        if self.task in ('wildfire', 'all'):
            outputs.update(self.wildfire_head(x))

        return outputs


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model = GaiaNet(
        img_size=256,
        patch_size=16,
        in_channels=13,
        vit_embed_dim=768,
        vit_depth=6,      # smaller for quick test
        st_depth=3,
        weather_dim=16,
        gnn_hidden=256,
        gnn_layers=2,
        task='all',
    )

    B, T = 1, 5
    images  = torch.randn(B, T, 13, 256, 256)
    weather = torch.randn(B, T, 16)
    dem     = torch.randn(B, 1, 256, 256)
    coords  = torch.tensor([[0.2, 0.5]])

    outputs = model(images, weather, dem, coords)
    for k, v in outputs.items():
        print(f"  {k}: {v.shape}")
    print("GaiaNet: OK ✓")
