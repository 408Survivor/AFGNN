"""
Facial Landmark Graph Neural Network for AFGNN.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import AttentionalAggregation

from models.weighted_gat_conv import WeightedGATConv


class FaceGNN(nn.Module):
    """Graph Attention Network encoding a spatio-temporal facial landmark graph.

    Changes for training stability:
    - LayerNorm replaces BatchNorm (more stable for graph batches and small batch size).
    - AttentionalAggregation readout from PyG (learns a gate function for stable attention).
    - Xavier / Kaiming weight initialization.
    - Supports optional edge_weight to down-weight dynamic edges.
    - Supports optional edge_type embedding to distinguish static/temporal/dynamic edges.

    Args:
        in_channels: input node feature dimension (4 for [x, y, dx, dy]).
        hidden_channels: hidden dimension.
        out_channels: output graph-level embedding dimension.
        num_layers: number of GAT layers.
        heads: number of attention heads.
        dropout: dropout probability.
        num_edge_types: number of edge types. If > 0, use edge type embeddings.
        edge_emb_dim: dimension of edge type embedding per head.
    """

    def __init__(
        self,
        in_channels: int = 4,
        hidden_channels: int = 128,
        out_channels: int = 128,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.3,
        num_edge_types: int = None,
        edge_emb_dim: int = 1,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.heads = heads
        self.num_edge_types = num_edge_types
        self.edge_emb_dim = edge_emb_dim

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # First GAT layer
        self.convs.append(
            WeightedGATConv(
                in_channels=in_channels,
                out_channels=hidden_channels // heads,
                heads=heads,
                concat=True,
                dropout=dropout,
                num_edge_types=num_edge_types,
                edge_emb_dim=edge_emb_dim,
            )
        )
        self.norms.append(nn.LayerNorm(hidden_channels))

        # Intermediate GAT layers
        for _ in range(num_layers - 2):
            self.convs.append(
                WeightedGATConv(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels // heads,
                    heads=heads,
                    concat=True,
                    dropout=dropout,
                    num_edge_types=num_edge_types,
                    edge_emb_dim=edge_emb_dim,
                )
            )
            self.norms.append(nn.LayerNorm(hidden_channels))

        # Last GAT layer
        self.convs.append(
            WeightedGATConv(
                in_channels=hidden_channels,
                out_channels=out_channels,
                heads=1,
                dropout=dropout,
                concat=False,
                num_edge_types=num_edge_types,
                edge_emb_dim=edge_emb_dim,
            )
        )

        # AttentionalAggregation readout: a small MLP gate function
        self.readout_gate = nn.Sequential(
            nn.Linear(out_channels, out_channels // 2),
            nn.ReLU(),
            nn.Linear(out_channels // 2, 1),
        )
        self.readout = AttentionalAggregation(gate_nn=self.readout_gate)

        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize weights for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, edge_index, batch, edge_weight=None, edge_type=None):
        """
        Args:
            x: node features, shape (N_V_total, in_channels)
            edge_index: graph edges, shape (2, E)
            batch: batch vector assigning nodes to graphs, shape (N_V_total,)
            edge_weight: optional edge weights, shape (E,). Lower values for
                         dynamic edges reduce their attention influence.
            edge_type: optional edge type indices, shape (E,). 0=static, 1=temporal, 2=dynamic.

        Returns:
            graph_embedding: shape (batch_size, out_channels)
        """
        # GNN layers
        for i, conv in enumerate(self.convs):
            if i < self.num_layers - 1:
                x = conv(x, edge_index, edge_weight=edge_weight, edge_type=edge_type)
                x = self.norms[i](x)
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            else:
                x = conv(x, edge_index, edge_weight=edge_weight, edge_type=edge_type)

        # Attention readout
        out = self.readout(x, batch)  # (batch_size, out_channels)
        return out
