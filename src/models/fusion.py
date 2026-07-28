"""
Cross-modal fusion modules for AFGNN.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConcatFusion(nn.Module):
    """Simple concatenation of face and audio embeddings."""

    def __init__(self, face_dim: int, audio_dim: int):
        super().__init__()
        self.face_dim = face_dim
        self.audio_dim = audio_dim
        self.out_dim = face_dim + audio_dim

    def forward(self, h_face: torch.Tensor, h_audio: torch.Tensor) -> torch.Tensor:
        return torch.cat([h_face, h_audio], dim=-1)


class CrossModalAttentionFusion(nn.Module):
    """Cross-modal attention fusion between face and audio embeddings.

    Each modality attends to the other via a learned query/key/value projection.
    The attended context is added back as a residual, and the enhanced embeddings
    are concatenated for classification.
    """

    def __init__(self, face_dim: int, audio_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.face_dim = face_dim
        self.audio_dim = audio_dim
        self.hidden_dim = hidden_dim
        self.out_dim = face_dim + audio_dim

        # Face attends to audio
        self.face_query = nn.Linear(face_dim, hidden_dim)
        self.audio_key = nn.Linear(audio_dim, hidden_dim)
        self.audio_value = nn.Linear(audio_dim, face_dim)

        # Audio attends to face
        self.audio_query = nn.Linear(audio_dim, hidden_dim)
        self.face_key = nn.Linear(face_dim, hidden_dim)
        self.face_value = nn.Linear(face_dim, audio_dim)

        self.scale = hidden_dim ** -0.5

        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, h_face: torch.Tensor, h_audio: torch.Tensor) -> torch.Tensor:
        # Face -> Audio attention
        q_face = self.face_query(h_face)          # (B, hidden)
        k_audio = self.audio_key(h_audio)         # (B, hidden)
        v_audio = self.audio_value(h_audio)       # (B, face_dim)

        scores_fa = torch.sum(q_face * k_audio, dim=-1, keepdim=True) * self.scale
        attn_fa = torch.sigmoid(scores_fa)        # scalar gate per sample
        h_face_enhanced = h_face + attn_fa * v_audio

        # Audio -> Face attention
        q_audio = self.audio_query(h_audio)       # (B, hidden)
        k_face = self.face_key(h_face)            # (B, hidden)
        v_face = self.face_value(h_face)          # (B, audio_dim)

        scores_af = torch.sum(q_audio * k_face, dim=-1, keepdim=True) * self.scale
        attn_af = torch.sigmoid(scores_af)
        h_audio_enhanced = h_audio + attn_af * v_face

        return torch.cat([h_face_enhanced, h_audio_enhanced], dim=-1)


def build_fusion(fusion_type: str, face_dim: int, audio_dim: int, hidden_dim: int = 64):
    """Factory function for fusion modules."""
    if fusion_type == "concat":
        return ConcatFusion(face_dim, audio_dim)
    elif fusion_type == "cross_attention":
        return CrossModalAttentionFusion(face_dim, audio_dim, hidden_dim)
    else:
        raise ValueError(f"Unsupported fusion type: {fusion_type}")
