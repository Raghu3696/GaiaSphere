"""
GaiaSphere — GeoGraph (Geographic Graph Neural Network)
========================================================
Models the Earth's surface as a dynamic graph where nodes represent
geographic units (watersheds, crop parcels, fire risk cells) and edges
encode physical connectivity and similarity.

GNN message passing propagates environmental signals across the graph,
allowing the model to reason about how floods drain through watersheds,
how fire spreads downwind, or how drought stress diffuses across a region.

Architecture:
  - GraphSAGE-style neighborhood aggregation
  - GAT-style attention-weighted edges
  - Edge features: elevation delta, flow direction, distance, wind alignment

Author: GaiaSphere Project
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Edge Feature Computation
# ---------------------------------------------------------------------------

class EdgeFeatureEncoder(nn.Module):
    """
    Encodes physical properties of graph edges into a learned representation.

    Edge features can include:
      - Elevation difference between nodes (flow potential)
      - Great-circle distance
      - Wind alignment (dot product of wind vector with edge direction)
      - Soil type similarity
      - Hydrological connectivity (upstream/downstream)
    """

    def __init__(self, edge_input_dim: int, edge_embed_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(edge_input_dim, edge_embed_dim),
            nn.GELU(),
            nn.Linear(edge_embed_dim, edge_embed_dim),
        )

    def forward(self, edge_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            edge_features: [E, edge_input_dim]  Raw physical edge attributes
        Returns:
            [E, edge_embed_dim]
        """
        return self.encoder(edge_features)


# ---------------------------------------------------------------------------
# Graph Attention Convolution Layer
# ---------------------------------------------------------------------------

class GeoGraphConv(nn.Module):
    """
    A single graph convolution layer combining:
      - GraphSAGE-style neighbor aggregation (mean pooling)
      - GAT-style learned edge attention weights

    For each node v:
      h_v' = MLP( concat(h_v,  Σ_{u∈N(v)} α(v,u) · W · h_u) )

    where α(v,u) = softmax( LeakyReLU( a^T [Wh_v || Wh_u || e_{vu}] ) )
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        out_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        assert out_dim % num_heads == 0

        # Node transformation
        self.W_node = nn.Linear(node_dim, out_dim, bias=False)

        # Attention: a^T [Wh_src || Wh_dst || edge_embed]
        self.att = nn.Linear(out_dim * 2 + edge_dim, num_heads, bias=False)

        # Output projection
        self.proj = nn.Linear(out_dim * 2, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(
        self,
        node_features: torch.Tensor,        # [N, node_dim]
        edge_index: torch.Tensor,           # [2, E]  (src, dst) pairs
        edge_features: torch.Tensor,        # [E, edge_dim]
    ) -> torch.Tensor:
        """
        Args:
            node_features:  [N, node_dim]
            edge_index:     [2, E]
            edge_features:  [E, edge_dim]
        Returns:
            updated_nodes:  [N, out_dim]
        """
        N = node_features.shape[0]
        src, dst = edge_index[0], edge_index[1]

        # Transform nodes
        h = self.W_node(node_features)     # [N, out_dim]

        # Compute attention scores for each edge
        h_src = h[src]                     # [E, out_dim]
        h_dst = h[dst]                     # [E, out_dim]
        att_input = torch.cat([h_src, h_dst, edge_features], dim=-1)  # [E, 2D+edge_dim]
        att_scores = self.att(att_input)   # [E, num_heads]
        att_scores = F.leaky_relu(att_scores, 0.2)

        # Softmax over neighbors (per destination node, per head)
        att_weights = torch.zeros(N, self.num_heads, device=node_features.device)
        att_weights.scatter_add_(
            0,
            dst.unsqueeze(1).expand(-1, self.num_heads),
            att_scores.exp()
        )
        # Normalize: divide each edge score by its dst node's sum
        norm = att_weights[dst] + 1e-8     # [E, num_heads]
        alpha = att_scores.exp() / norm    # [E, num_heads]
        alpha = self.dropout(alpha)

        # Weighted aggregation: Σ α_{vu} · h_u → [N, out_dim]
        # Sum contributions of all neighbors for each node
        agg = torch.zeros(N, self.num_heads, self.head_dim, device=node_features.device)
        h_src_heads = h_src.view(-1, self.num_heads, self.head_dim)  # [E, heads, head_dim]
        alpha_expanded = alpha.unsqueeze(-1)                          # [E, heads, 1]

        for head_idx in range(self.num_heads):
            contrib = alpha[:, head_idx:head_idx+1] * h_src          # [E, out_dim]
            agg[:, head_idx].scatter_add_(0, dst.unsqueeze(1).expand(-1, self.head_dim), contrib.view(-1, self.num_heads, self.head_dim)[:, head_idx])

        agg = agg.view(N, -1)              # [N, out_dim]

        # Combine self + aggregated neighbors
        out = self.proj(torch.cat([h, agg], dim=-1))  # [N, out_dim]
        out = self.act(self.norm(out))

        return out


# ---------------------------------------------------------------------------
# GeoGraph: Full GNN
# ---------------------------------------------------------------------------

class GeoGraph(nn.Module):
    """
    Multi-layer Geographic Graph Neural Network.

    Takes the spatio-temporally fused node features from ST-Former and
    propagates them across the geographic graph, enabling the model to
    reason about connected regions (watersheds, wind corridors, etc.)

    Args:
        node_input_dim:   Input node feature dimension (from ST-Former)
        edge_input_dim:   Raw edge feature dimension (physical attributes)
        hidden_dim:       Hidden dimension for GNN layers
        edge_embed_dim:   Dimension for encoded edge features
        num_layers:       Number of GNN layers
        num_heads:        Attention heads per layer
        dropout:          Dropout probability
    """

    def __init__(
        self,
        node_input_dim: int = 768,
        edge_input_dim: int = 8,
        hidden_dim: int = 256,
        edge_embed_dim: int = 64,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Node input projection
        self.node_proj = nn.Sequential(
            nn.Linear(node_input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        # Edge feature encoder
        self.edge_encoder = EdgeFeatureEncoder(edge_input_dim, edge_embed_dim)

        # GNN layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                GeoGraphConv(
                    node_dim=hidden_dim,
                    edge_dim=edge_embed_dim,
                    out_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                )
            )

        self.out_dim = hidden_dim

    def forward(
        self,
        node_features: torch.Tensor,       # [N, node_input_dim]
        edge_index: torch.Tensor,          # [2, E]
        edge_features: torch.Tensor,       # [E, edge_input_dim]
    ) -> torch.Tensor:
        """
        Args:
            node_features:  [N, node_input_dim]  Per-node ST-Former features
            edge_index:     [2, E]               (src, dst) connectivity
            edge_features:  [E, edge_input_dim]  Physical edge attributes

        Returns:
            node_embeddings: [N, hidden_dim]  Graph-propagated node features
        """
        # Project nodes to hidden dim
        h = self.node_proj(node_features)       # [N, hidden_dim]

        # Encode edge features
        edge_embed = self.edge_encoder(edge_features)  # [E, edge_embed_dim]

        # Multi-layer GNN
        for layer in self.layers:
            h = layer(h, edge_index, edge_embed)

        return h


# ---------------------------------------------------------------------------
# Graph Construction Utilities
# ---------------------------------------------------------------------------

def build_grid_graph(
    height: int,
    width: int,
    device: torch.device = torch.device('cpu'),
    include_diagonal: bool = True,
) -> torch.Tensor:
    """
    Build a grid graph (nodes = spatial grid cells, edges = adjacency).
    Used when geographic unit boundaries are unavailable.

    Args:
        height, width:     Grid dimensions
        include_diagonal:  Whether to include diagonal neighbors

    Returns:
        edge_index: [2, E]  src/dst pairs
    """
    edges_src, edges_dst = [], []
    N = height * width

    def idx(r, c):
        return r * width + c

    offsets = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    if include_diagonal:
        offsets += [(1, 1), (1, -1), (-1, 1), (-1, -1)]

    for r in range(height):
        for c in range(width):
            for dr, dc in offsets:
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width:
                    edges_src.append(idx(r, c))
                    edges_dst.append(idx(nr, nc))

    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long, device=device)
    return edge_index


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    N = 64    # 64 geographic nodes (e.g., 8x8 grid of watersheds)
    E = 200   # number of edges

    model = GeoGraph(
        node_input_dim=768,
        edge_input_dim=8,
        hidden_dim=256,
        num_layers=3,
    )

    node_feats = torch.randn(N, 768)
    edge_index = torch.randint(0, N, (2, E))
    edge_feats = torch.randn(E, 8)  # elevation delta, distance, wind alignment, ...

    out = model(node_feats, edge_index, edge_feats)
    print(f"Node input:  {node_feats.shape}")   # [64, 768]
    print(f"Node output: {out.shape}")           # [64, 256]
    print("GeoGraph: OK ✓")
