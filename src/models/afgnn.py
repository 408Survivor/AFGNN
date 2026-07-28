"""
AFGNN: Audio-Facial Graph Neural Network.
"""

import torch
import torch.nn as nn

from models.audio_gnn import AudioGNN
from models.face_gnn import FaceGNN
from models.fusion import build_fusion


class AFGNN(nn.Module):
    """AFGNN model for depression recognition.

    Supports three operating modes via `use_face` and `use_audio` flags:
    - Face-only (`use_face=True`, `use_audio=False`).
    - Audio-only (`use_face=False`, `use_audio=True`).
    - Multimodal fusion (`use_face=True`, `use_audio=True`).

    Args:
        use_face: whether to use the facial landmark graph branch.
        use_audio: whether to use the temporal audio graph branch.
        face_in_channels: input node feature dim for face graph.
        face_hidden_channels: hidden dim for Face GNN.
        face_out_channels: output embedding dim for Face GNN.
        face_num_layers: number of GAT layers in Face GNN.
        face_heads: number of GAT heads.
        audio_in_channels: input node feature dim for audio graph.
        audio_hidden_channels: hidden dim for Audio GNN.
        audio_out_channels: output embedding dim for Audio GNN.
        audio_num_layers: number of GAT layers in Audio GNN.
        audio_heads: number of GAT heads.
        mlp_hidden: hidden dim of classification MLP.
        dropout: dropout probability.
        num_edge_types: number of edge types for edge type embedding.
        edge_emb_dim: dimension of edge type embedding per head.
    """

    def __init__(
        self,
        use_face: bool = True,
        use_audio: bool = True,
        face_in_channels: int = 4,
        face_hidden_channels: int = 128,
        face_out_channels: int = 128,
        face_num_layers: int = 3,
        face_heads: int = 4,
        audio_in_channels: int = 25,
        audio_hidden_channels: int = 64,
        audio_out_channels: int = 64,
        audio_num_layers: int = 2,
        audio_heads: int = 4,
        mlp_hidden: int = 128,
        dropout: float = 0.3,
        fusion_type: str = "concat",
        fusion_hidden_dim: int = 64,
        num_edge_types: int = None,
        edge_emb_dim: int = 1,
    ):
        super().__init__()
        self.use_face = use_face
        self.use_audio = use_audio
        self.fusion_type = fusion_type

        if not use_face and not use_audio:
            raise ValueError("At least one of use_face or use_audio must be True.")

        if use_face:
            self.face_gnn = FaceGNN(
                in_channels=face_in_channels,
                hidden_channels=face_hidden_channels,
                out_channels=face_out_channels,
                num_layers=face_num_layers,
                heads=face_heads,
                dropout=dropout,
                num_edge_types=num_edge_types,
                edge_emb_dim=edge_emb_dim,
            )
        else:
            self.face_gnn = None

        if use_audio:
            self.audio_gnn = AudioGNN(
                in_channels=audio_in_channels,
                hidden_channels=audio_hidden_channels,
                out_channels=audio_out_channels,
                num_layers=audio_num_layers,
                heads=audio_heads,
                dropout=dropout,
            )
        else:
            self.audio_gnn = None

        # Fusion module for multimodal case.
        if use_face and use_audio:
            self.fusion = build_fusion(
                fusion_type, face_out_channels, audio_out_channels, fusion_hidden_dim
            )
            classifier_in = self.fusion.out_dim
        else:
            self.fusion = None
            classifier_in = 0
            if use_face:
                classifier_in += face_out_channels
            if use_audio:
                classifier_in += audio_out_channels

        self.classifier = nn.Sequential(
            nn.Linear(classifier_in, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(mlp_hidden, 1),
        )

        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize classifier weights."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, data):
        """
        Args:
            data: torch_geometric.data.Data with face graph attributes
                  (x, edge_index, batch, optional edge_weight / edge_type)
                  and optional audio attribute audio_x.

        Returns:
            logit: shape (batch_size, 1).
        """
        embeddings = []

        if self.use_face:
            x = data.x
            edge_index = data.edge_index
            batch = data.batch
            edge_weight = getattr(data, "edge_weight", None)
            edge_type = getattr(data, "edge_type", None)
            h_face = self.face_gnn(
                x, edge_index, batch, edge_weight=edge_weight, edge_type=edge_type
            )
            embeddings.append(h_face)

        if self.use_audio:
            audio_x = getattr(data, "audio_x", None)
            if audio_x is None:
                if self.training:
                    raise ValueError(
                        "use_audio=True but input Data has no audio_x. "
                        "Make sure the dataset returns audio graphs."
                    )
                # During inference, gracefully fall back to zeros if audio is missing.
                batch_size = data.y.size(0)
                h_audio = torch.zeros(
                    batch_size,
                    self.audio_gnn.out_channels,
                    device=data.y.device,
                    dtype=data.y.dtype,
                )
            else:
                batch = getattr(data, "audio_batch", None)
                if batch is None:
                    # Fallback: derive audio batch from face batch if PyG did not
                    # create an explicit audio_batch (audio_x and x have different
                    # numbers of nodes, so they cannot share the default batch).
                    face_batch = data.batch
                    num_audio_nodes = audio_x.size(0)
                    num_graphs = face_batch.max().item() + 1
                    # Each graph has the same number of audio frames by dataset
                    # construction.
                    num_frames = num_audio_nodes // num_graphs
                    batch = torch.arange(
                        num_graphs,
                        device=audio_x.device,
                        dtype=torch.long,
                    ).repeat_interleave(num_frames)
                    batch = batch[:num_audio_nodes]
                h_audio = self.audio_gnn(audio_x, batch=batch)
            embeddings.append(h_audio)

        if self.fusion is not None:
            h = self.fusion(embeddings[0], embeddings[1])
        else:
            h = torch.cat(embeddings, dim=-1)
        logit = self.classifier(h)
        return logit

    def extract_face_embedding(self, data):
        """Extract face graph embedding (for visualization / analysis)."""
        if not self.use_face:
            raise ValueError("use_face=False; no face embedding available.")
        x = data.x
        edge_index = data.edge_index
        batch = data.batch
        edge_weight = getattr(data, "edge_weight", None)
        edge_type = getattr(data, "edge_type", None)
        return self.face_gnn(
            x, edge_index, batch, edge_weight=edge_weight, edge_type=edge_type
        )

    def extract_audio_embedding(self, data):
        """Extract audio graph embedding (for visualization / analysis)."""
        if not self.use_audio:
            raise ValueError("use_audio=False; no audio embedding available.")
        audio_x = getattr(data, "audio_x", None)
        if audio_x is None:
            raise ValueError("Input Data has no audio_x.")
        batch = getattr(data, "audio_batch", None)
        if batch is None:
            face_batch = data.batch
            num_audio_nodes = audio_x.size(0)
            num_graphs = face_batch.max().item() + 1
            num_frames = num_audio_nodes // num_graphs
            batch = torch.arange(
                num_graphs,
                device=audio_x.device,
                dtype=torch.long,
            ).repeat_interleave(num_frames)
        return self.audio_gnn(audio_x, batch=batch)
