"""
WeightedGATConv: GAT convolution that supports edge weights and edge types.

This is a simplified multi-head GAT implementation based on PyG MessagePassing.
- Attention coefficients can be multiplied by edge_weight (scalar).
- Edge type embeddings are added to attention logits (learned).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax


class WeightedGATConv(MessagePassing):
    """Graph Attention Convolution with edge weight and edge type support.

    Args:
        in_channels: input feature dimension.
        out_channels: output feature dimension per head.
        heads: number of attention heads.
        concat: if True, concatenate heads (output dim = heads * out_channels).
                if False, average heads (output dim = out_channels).
        dropout: dropout on attention coefficients.
        negative_slope: negative slope for LeakyReLU.
        bias: whether to use bias.
        num_edge_types: number of edge types (e.g., 3 for static/temporal/dynamic).
                        If None, no edge type embedding is used.
        edge_emb_dim: dimension of edge type embedding per head.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        dropout: float = 0.0,
        negative_slope: float = 0.2,
        bias: bool = True,
        num_edge_types: int = None,
        edge_emb_dim: int = 1,
    ):
        super().__init__(aggr="add", node_dim=0)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        self.negative_slope = negative_slope
        self.num_edge_types = num_edge_types

        self.lin = nn.Linear(in_channels, heads * out_channels, bias=False)

        # Attention parameters
        self.att_src = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_dst = nn.Parameter(torch.Tensor(1, heads, out_channels))

        # Edge type embedding (optional)
        if num_edge_types is not None and num_edge_types > 0:
            self.edge_emb = nn.Embedding(num_edge_types, heads * edge_emb_dim)
            # Project edge embedding to attention space (per head)
            self.edge_emb_proj = nn.Linear(edge_emb_dim, 1, bias=False)
        else:
            self.register_parameter("edge_emb", None)
            self.register_parameter("edge_emb_proj", None)

        if bias:
            self.bias = nn.Parameter(
                torch.Tensor(heads * out_channels if concat else out_channels)
            )
        else:
            self.register_parameter("bias", None)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        if self.edge_emb is not None:
            nn.init.xavier_uniform_(self.edge_emb.weight)
            nn.init.xavier_uniform_(self.edge_emb_proj.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor = None,
        edge_type: torch.Tensor = None,
    ):
        """
        Args:
            x: node features, shape (N, in_channels)
            edge_index: graph edges, shape (2, E)
            edge_weight: optional edge weights, shape (E,)
            edge_type: optional edge types as integer indices, shape (E,)

        Returns:
            out: updated node features
        """
        # Linear transformation and reshape to (N, heads, out_channels)
        x = self.lin(x).view(-1, self.heads, self.out_channels)

        # Compute attention logits for each node
        alpha_src = (x * self.att_src).sum(dim=-1)  # (N, heads)
        alpha_dst = (x * self.att_dst).sum(dim=-1)  # (N, heads)

        # Propagate messages
        out = self.propagate(
            edge_index,
            x=x,
            alpha_src=alpha_src,
            alpha_dst=alpha_dst,
            edge_weight=edge_weight,
            edge_type=edge_type,
        )

        # Combine heads
        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias

        return out

    def message(
        self,
        x_j: torch.Tensor,
        alpha_src_j: torch.Tensor,
        alpha_dst_i: torch.Tensor,
        edge_weight: torch.Tensor,
        edge_type: torch.Tensor,
        index: torch.Tensor,
        ptr: torch.Tensor,
        size_i: int,
    ):
        """Compute messages for each edge."""
        # Attention score: sum of source and destination logits
        alpha = alpha_src_j + alpha_dst_i

        # Add edge type embedding to attention logits
        if edge_type is not None and self.edge_emb is not None:
            # edge_emb: (E, heads * edge_emb_dim) -> (E, heads, edge_emb_dim)
            e_emb = self.edge_emb(edge_type)
            e_emb = e_emb.view(-1, self.heads, e_emb.size(-1) // self.heads)
            # Project each head's embedding to scalar and squeeze
            e_score = self.edge_emb_proj(e_emb).squeeze(-1)  # (E, heads)
            alpha = alpha + e_score

        alpha = F.leaky_relu(alpha, self.negative_slope)

        # Apply edge weight to attention scores
        if edge_weight is not None:
            alpha = alpha * edge_weight.view(-1, 1)

        # Softmax over neighbors
        alpha = softmax(alpha, index, ptr, size_i)

        # Dropout on attention
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # Weighted message
        return x_j * alpha.unsqueeze(-1)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"{self.in_channels}, {self.out_channels}, heads={self.heads})"
        )
