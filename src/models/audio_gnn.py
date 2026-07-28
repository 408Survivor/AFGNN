"""
Audio Graph Neural Network for AFGNN.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import AttentionalAggregation

from models.weighted_gat_conv import WeightedGATConv


class AudioGNN(nn.Module):
    """Graph Attention Network encoding a temporal audio chain graph.

    The input is a sequence of frame-level acoustic features. We treat each
    frame as a node in a simple bidirectional chain graph and apply GAT layers.
    The graph-level embedding is obtained via AttentionalAggregation.

    Args:
        in_channels: input node feature dimension (25 for D-Vlog acoustic features).
        hidden_channels: hidden dimension.
        out_channels: output graph-level embedding dimension.
        num_layers: number of GAT layers.
        heads: number of attention heads.
        dropout: dropout probability.
    """

    def __init__(
        self,
        in_channels: int = 25,
        hidden_channels: int = 64,
        out_channels: int = 64,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.heads = heads

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

    @staticmethod
    def build_chain_edges(num_frames: int, device: torch.device) -> torch.Tensor:
        """Build bidirectional chain edges for a single graph."""
        edges = []
        for t in range(num_frames - 1):
            edges.append((t, t + 1))
            edges.append((t + 1, t))
        edge_index = torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()
        return edge_index

    @staticmethod
    def build_batched_chain_edges(num_frames: int, batch: torch.Tensor) -> torch.Tensor:
        """Build batched bidirectional chain edges for a batch of audio graphs.

        Args:
            num_frames: number of frames per graph.
            batch: batch vector assigning audio nodes to graphs.

        Returns:
            edge_index: shape (2, num_graphs * 2 * (num_frames - 1)).
        """
        num_graphs = batch.max().item() + 1
        base_edges = AudioGNN.build_chain_edges(num_frames, batch.device)
        # Replicate base edges for each graph and offset node indices.
        base_edges = base_edges.unsqueeze(0).repeat(num_graphs, 1, 1)
        offsets = torch.arange(num_graphs, device=batch.device).view(-1, 1, 1) * num_frames
        batched_edges = base_edges + offsets
        batched_edges = batched_edges.permute(1, 0, 2).reshape(2, -1)
        return batched_edges

    def forward(self, audio_x, audio_edge_index=None, batch=None):
        """
        Args:
            audio_x: node features, shape (N_A_total, in_channels).
            audio_edge_index: optional graph edges, shape (2, E). If None,
                a chain is built automatically from `batch`.
            batch: batch vector assigning nodes to graphs, shape (N_A_total,).

        Returns:
            graph_embedding: shape (batch_size, out_channels).
        """
        if audio_edge_index is None:
            if batch is None:
                raise ValueError("Either audio_edge_index or batch must be provided.")
            # Infer number of frames from the first graph in the batch.
            # All graphs in a batch have the same num_frames by construction.
            num_frames = (batch == 0).sum().item()
            audio_edge_index = self.build_batched_chain_edges(num_frames, batch)

        x = audio_x

        # GNN layers
        for i, conv in enumerate(self.convs):
            if i < self.num_layers - 1:
                x = conv(x, audio_edge_index)
                x = self.norms[i](x)
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            else:
                x = conv(x, audio_edge_index)

        # Attention readout
        out = self.readout(x, batch)  # (batch_size, out_channels)
        return out
