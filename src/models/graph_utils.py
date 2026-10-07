"""
Graph construction utilities for AFGNN.

This module builds spatio-temporal facial landmark graphs from sequences of
68 2D landmarks (shape: (T, 136)).
"""

from typing import List, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import Data

from data.landmark_layout import flat_to_coords


# ---------------------------------------------------------------------------
# 68-point facial landmark topology (Dlib / OpenFace convention, 0-based index)
# ---------------------------------------------------------------------------

LANDMARK_GROUPS_68 = {
    "face_contour": list(range(0, 17)),       # 0-16
    "left_eyebrow": list(range(17, 22)),      # 17-21
    "right_eyebrow": list(range(22, 27)),     # 22-26
    "nose_bridge": list(range(27, 31)),       # 27-30
    "nose_bottom": list(range(31, 36)),       # 31-35
    "left_eye": list(range(36, 42)),          # 36-41
    "right_eye": list(range(42, 48)),         # 42-47
    "outer_mouth": list(range(48, 60)),       # 48-59
    "inner_mouth": list(range(60, 68)),       # 60-67
}

# Map each landmark to its facial region id (for region-restricted dynamic edges)
LANDMARK_REGION_68 = [0] * 68
for region_id, indices in enumerate(LANDMARK_GROUPS_68.values()):
    for idx in indices:
        LANDMARK_REGION_68[idx] = region_id

# Edge type indices
EDGE_TYPE_STATIC = 0
EDGE_TYPE_TEMPORAL = 1
EDGE_TYPE_DYNAMIC = 2

# Number of facial regions defined in LANDMARK_GROUPS_68
NUM_FACE_REGIONS = len(LANDMARK_GROUPS_68)


def build_region_onehot(num_landmarks: int = 68) -> np.ndarray:
    """Return a (num_landmarks, NUM_FACE_REGIONS) one-hot region matrix."""
    assert num_landmarks == 68, "Only 68-landmark topology is supported."
    region_ids = np.array(LANDMARK_REGION_68, dtype=np.int64)
    return np.eye(NUM_FACE_REGIONS, dtype=np.float32)[region_ids]


def _chain_edges(indices: List[int], closed: bool = False) -> List[Tuple[int, int]]:
    """Generate chain edges from a list of indices."""
    edges = []
    for i in range(len(indices) - 1):
        edges.append((indices[i], indices[i + 1]))
        edges.append((indices[i + 1], indices[i]))
    if closed and len(indices) > 1:
        edges.append((indices[-1], indices[0]))
        edges.append((indices[0], indices[-1]))
    return edges


def build_static_edges(num_landmarks: int = 68) -> torch.Tensor:
    """Build intra-frame anatomical edges for 68 facial landmarks."""
    assert num_landmarks == 68, "Only 68-landmark topology is supported."

    edges = []
    edges.extend(_chain_edges(LANDMARK_GROUPS_68["face_contour"], closed=False))
    edges.extend(_chain_edges(LANDMARK_GROUPS_68["left_eyebrow"], closed=False))
    edges.extend(_chain_edges(LANDMARK_GROUPS_68["right_eyebrow"], closed=False))
    edges.extend(_chain_edges(LANDMARK_GROUPS_68["nose_bridge"], closed=False))
    edges.extend(_chain_edges(LANDMARK_GROUPS_68["nose_bottom"], closed=True))
    edges.extend(_chain_edges(LANDMARK_GROUPS_68["left_eye"], closed=True))
    edges.extend(_chain_edges(LANDMARK_GROUPS_68["right_eye"], closed=True))
    edges.extend(_chain_edges(LANDMARK_GROUPS_68["outer_mouth"], closed=True))
    edges.extend(_chain_edges(LANDMARK_GROUPS_68["inner_mouth"], closed=True))

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index


def build_random_static_edges(
    num_landmarks: int = 68,
    num_edges: Optional[int] = None,
    seed: int = 0,
) -> torch.Tensor:
    """Build an edge-matched random intra-frame graph (control baseline).

    Generates a single fixed random topology with the SAME number of edges as
    the anatomical ``build_static_edges()``. It samples ``num_edges // 2``
    distinct undirected landmark pairs (no self-loops, no duplicates) and adds
    both directions, so the directed edge count matches exactly.

    The topology is deterministic given ``seed`` and is reused for every frame
    and every sample (mirroring how anatomical edges are identical everywhere),
    so the only variable versus the full model is *which* landmarks are wired.

    Args:
        num_landmarks: number of landmarks (must be 68).
        num_edges: number of DIRECTED edges to match. Defaults to
            ``build_static_edges().size(1)``.
        seed: RNG seed controlling the random topology.

    Returns:
        edge_index: shape (2, num_edges).
    """
    assert num_landmarks == 68, "Only 68-landmark topology is supported."
    if num_edges is None:
        num_edges = build_static_edges(num_landmarks).size(1)
    num_undirected = num_edges // 2

    rng = np.random.default_rng(seed)
    possible = [
        (i, j)
        for i in range(num_landmarks)
        for j in range(i + 1, num_landmarks)
    ]
    chosen = rng.choice(len(possible), size=num_undirected, replace=False)

    edges = []
    for k in chosen:
        i, j = possible[k]
        edges.append((i, j))
        edges.append((j, i))

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index


def build_temporal_edges(num_frames: int, num_landmarks: int = 68) -> torch.Tensor:
    """Build temporal edges connecting the same landmark across consecutive frames."""
    edges = []
    for t in range(num_frames - 1):
        for i in range(num_landmarks):
            src = t * num_landmarks + i
            dst = (t + 1) * num_landmarks + i
            edges.append((src, dst))
            edges.append((dst, src))
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index


def build_dynamic_edges(
    num_frames: int,
    num_landmarks: int,
    node_features: torch.Tensor,
    k: int = 3,
    existing_edges: Optional[torch.Tensor] = None,
    metric: str = "cosine",
    feature_mode: str = "full",
    region_restricted: bool = False,
) -> torch.Tensor:
    """Build dynamic edges within each frame based on node feature similarity.

    Args:
        feature_mode: "full" uses [x, y, dx, dy]; "motion" uses only [dx, dy]
                      to connect landmarks with similar movement patterns.
        region_restricted: if True, only connect landmarks within the same
                           facial region (e.g., mouth, eye, eyebrow).
    """
    k = min(k, num_landmarks - 1)
    if k <= 0:
        return torch.zeros((2, 0), dtype=torch.long)

    existing_set = set()
    if existing_edges is not None and existing_edges.numel() > 0:
        existing_set = {
            (int(existing_edges[0, i]), int(existing_edges[1, i]))
            for i in range(existing_edges.size(1))
            if int(existing_edges[0, i]) // num_landmarks
            == int(existing_edges[1, i]) // num_landmarks
        }

    node_features = node_features.detach().cpu().numpy()
    dynamic_edges = []
    eps = 1e-8
    regions = np.array(LANDMARK_REGION_68) if region_restricted else None

    for t in range(num_frames):
        offset = t * num_landmarks
        frame_feats = node_features[offset : offset + num_landmarks]

        if feature_mode == "motion":
            # Use only velocity [dx, dy]
            frame_feats = frame_feats[:, 2:4]
        elif feature_mode != "full":
            raise ValueError(f"Unsupported feature_mode: {feature_mode}")

        if metric == "cosine":
            norms = np.linalg.norm(frame_feats, axis=1, keepdims=True)
            valid = norms.squeeze() >= eps
            normalized = np.zeros_like(frame_feats)
            normalized[valid] = frame_feats[valid] / norms[valid]
            sim = normalized @ normalized.T
        elif metric == "euclidean":
            dist = np.linalg.norm(
                frame_feats[:, None, :] - frame_feats[None, :, :], axis=2
            )
            sim = -dist
            norms = np.linalg.norm(frame_feats, axis=1)
            valid = norms >= eps
        else:
            raise ValueError(f"Unsupported similarity metric: {metric}")

        # Do not connect landmarks with near-zero motion when using motion mode
        if feature_mode == "motion":
            invalid = ~valid
            sim[invalid, :] = -np.inf
            sim[:, invalid] = -np.inf

        for i in range(num_landmarks):
            sim[i, i] = -np.inf
            if region_restricted:
                for j in range(num_landmarks):
                    if regions[j] != regions[i]:
                        sim[i, j] = -np.inf
            if existing_edges is not None:
                for j in range(num_landmarks):
                    if i != j and (offset + i, offset + j) in existing_set:
                        sim[i, j] = -np.inf

            top_k = np.argpartition(sim[i], -k)[-k:]
            for j in top_k:
                if sim[i, j] > -np.inf:
                    dynamic_edges.append((offset + i, offset + j))
                    dynamic_edges.append((offset + j, offset + i))

    if len(dynamic_edges) == 0:
        return torch.zeros((2, 0), dtype=torch.long)

    edge_index = torch.tensor(dynamic_edges, dtype=torch.long).t().contiguous()
    return edge_index


def sample_frames(visual_seq: np.ndarray, num_frames: int) -> np.ndarray:
    """Uniformly sample frames from a landmark sequence."""
    T = visual_seq.shape[0]
    indices = np.linspace(0, T - 1, num_frames, dtype=int)
    return visual_seq[indices]


def compute_velocity(coords: np.ndarray) -> np.ndarray:
    """Compute velocity features from landmark coordinates."""
    velocity = np.zeros_like(coords)
    velocity[1:] = coords[1:] - coords[:-1]
    return velocity


def remove_duplicate_edges(edge_index: torch.Tensor) -> torch.Tensor:
    """Remove duplicate edges from an edge_index tensor."""
    edges = edge_index.t().numpy()
    edges = np.unique(edges, axis=0)
    return torch.from_numpy(edges).t().contiguous().long()


def aggregate_edges_with_type_and_weight(
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    edge_weight: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Remove duplicate edges and keep the highest-priority type/weight.

    Priority: static (0) > temporal (1) > dynamic (2).
    If a duplicate has a higher-priority type, use its type and weight.
    Otherwise keep the first occurrence.
    """
    edges = edge_index.t().numpy()
    types = edge_type.numpy()
    weights = edge_weight.numpy()

    edge_dict = {}
    for i in range(edges.shape[0]):
        e = (edges[i, 0], edges[i, 1])
        if e not in edge_dict:
            edge_dict[e] = (types[i], weights[i])
        else:
            # Lower type index = higher priority
            if types[i] < edge_dict[e][0]:
                edge_dict[e] = (types[i], weights[i])

    unique_edges = np.array(list(edge_dict.keys()), dtype=np.int64)
    unique_types = np.array([v[0] for v in edge_dict.values()], dtype=np.int64)
    unique_weights = np.array([v[1] for v in edge_dict.values()], dtype=np.float32)

    edge_index = torch.from_numpy(unique_edges).t().contiguous().long()
    edge_type = torch.from_numpy(unique_types).contiguous().long()
    edge_weight = torch.from_numpy(unique_weights).contiguous().float()

    return edge_index, edge_type, edge_weight


# ---------------------------------------------------------------------------
# Audio graph construction
# ---------------------------------------------------------------------------


def build_audio_chain_edges(num_frames: int) -> torch.Tensor:
    """Build bidirectional chain edges for a temporal audio graph.

    Args:
        num_frames: number of frames (audio nodes) in the chain.

    Returns:
        edge_index: shape (2, 2 * (num_frames - 1)).
    """
    edges = []
    for t in range(num_frames - 1):
        edges.append((t, t + 1))
        edges.append((t + 1, t))
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index


def build_audio_edges(
    num_frames: int,
    self_loops: bool = False,
    skip: int = 0,
) -> torch.Tensor:
    """Build audio graph edges: chain + optional self-loops + optional skip edges.

    Args:
        num_frames: number of frames (audio nodes).
        self_loops: whether to add self-loop edges (t, t).
        skip: if > 0, add bidirectional edges between frames t and t + skip + 1.
              For example, skip=1 connects t to t+2 (skip-one neighbor).

    Returns:
        edge_index: shape (2, E).
    """
    edges = []
    # Chain edges
    for t in range(num_frames - 1):
        edges.append((t, t + 1))
        edges.append((t + 1, t))
    # Skip edges
    if skip > 0:
        for t in range(num_frames - skip - 1):
            edges.append((t, t + skip + 1))
            edges.append((t + skip + 1, t))
    # Self-loops
    if self_loops:
        for t in range(num_frames):
            edges.append((t, t))

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index


def _compute_delta(seq: np.ndarray) -> np.ndarray:
    """Compute first-order delta features with simple padding.

    Args:
        seq: np.ndarray of shape (T, D).

    Returns:
        delta: np.ndarray of shape (T, D).
    """
    T = seq.shape[0]
    delta = np.zeros_like(seq)
    if T > 2:
        delta[1:-1] = (seq[2:] - seq[:-2]) / 2.0
    if T > 1:
        delta[0] = seq[1] - seq[0]
        delta[-1] = seq[-1] - seq[-2]
    return delta


def compute_audio_norm_stats(acoustic_seqs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute training-set mean and std for acoustic features.

    Args:
        acoustic_seqs: np.ndarray of shape (N, T, 25).

    Returns:
        mean: shape (25,).
        std: shape (25,).
    """
    # Flatten time and batch dimensions.
    feats = acoustic_seqs.reshape(-1, acoustic_seqs.shape[-1])
    mean = np.mean(feats, axis=0)
    std = np.std(feats, axis=0)
    return mean, std


def build_audio_graph(
    acoustic_seq: np.ndarray,
    label: int,
    num_frames: int = 16,
    norm_mean: Optional[np.ndarray] = None,
    norm_std: Optional[np.ndarray] = None,
    use_delta: bool = False,
    self_loops: bool = False,
    skip: int = 0,
) -> Data:
    """Build a temporal audio graph from frame-level acoustic features.

    Args:
        acoustic_seq: np.ndarray of shape (T, 25).
        label: integer label (0 or 1).
        num_frames: number of frames to sample.
        norm_mean: training-set mean per acoustic dim, shape (25,).
        norm_std: training-set std per acoustic dim, shape (25,).
        use_delta: whether to append delta and delta-delta features
                   (resulting audio_x dim = 75).
        self_loops: whether to add self-loop edges.
        skip: number of frames to skip for long-range edges (0 = disabled).

    Returns:
        torch_geometric.data.Data with audio_x, audio_edge_index, y.
    """
    sampled = sample_frames(acoustic_seq, num_frames)

    if norm_mean is not None and norm_std is not None:
        std = norm_std.copy()
        std[std < 1e-8] = 1.0
        sampled = (sampled - norm_mean) / std

    if use_delta:
        delta = _compute_delta(sampled)
        delta2 = _compute_delta(delta)
        sampled = np.concatenate([sampled, delta, delta2], axis=-1)

    audio_x = torch.from_numpy(sampled).float()
    audio_edge_index = build_audio_edges(
        num_frames, self_loops=self_loops, skip=skip
    )
    y = torch.tensor([label], dtype=torch.float)

    return Data(audio_x=audio_x, audio_edge_index=audio_edge_index, y=y)


def build_face_graph(
    visual_seq: np.ndarray,
    label: int,
    num_frames: int = 16,
    add_static_edges: bool = True,
    add_temporal_edges: bool = True,
    add_dynamic_edges: bool = False,
    dynamic_k: int = 3,
    dynamic_metric: str = "cosine",
    dynamic_edge_weight: float = 1.0,
    dynamic_feature: str = "full",
    dynamic_region_restricted: bool = False,
    use_edge_type: bool = False,
    add_region_onehot: bool = False,
    add_self_loops: bool = False,
    use_velocity: bool = True,
    region_ablation: Optional[int] = None,
    random_static_edges: bool = False,
    random_edge_seed: int = 0,
) -> Data:
    """Build a spatio-temporal facial landmark graph from a visual sequence.

    Args:
        visual_seq: np.ndarray of shape (T, 136)
        label: integer label (0 or 1).
        num_frames: number of frames to sample.
        add_static_edges: whether to add static anatomical edges.
        add_temporal_edges: whether to add temporal edges.
        add_dynamic_edges: whether to add feature-similarity dynamic edges.
        dynamic_k: number of dynamic neighbors per landmark per frame.
        dynamic_metric: similarity metric for dynamic edges.
        dynamic_edge_weight: weight applied to dynamic edges.
        dynamic_feature: "full" for [x,y,dx,dy] similarity; "motion" for [dx,dy].
        use_edge_type: whether to return edge_type (0=static, 1=temporal, 2=dynamic).
        add_region_onehot: whether to append a facial-region one-hot vector
            to each node feature (increases feature dim by NUM_FACE_REGIONS).
        add_self_loops: whether to add self-loop edges for every node.
        use_velocity: whether to append velocity [dx, dy] to node features.
            If False, only [x, y] is used.
        region_ablation: if not None, zero out the one-hot vector of this
            facial region id (for region leave-one-out analysis).
        random_static_edges: if True, replace the anatomical static edges with
            an edge-matched random intra-frame graph (control baseline). Only the
            static edges are randomized; temporal/region/self-loop are untouched.
        random_edge_seed: RNG seed for the random static topology.

    Returns:
        torch_geometric.data.Data with x, edge_index, edge_weight, edge_type, y.
    """
    sampled = sample_frames(visual_seq, num_frames)
    # D-Vlog visual features use the OpenFace block layout
    # [x_0..x_67, y_0..y_67]; parse via the single shared entry point.
    coords = flat_to_coords(sampled)

    if use_velocity:
        velocity = compute_velocity(coords)
        node_feats = np.concatenate([coords, velocity], axis=-1)
    else:
        node_feats = coords

    if add_region_onehot:
        region_onehot = build_region_onehot(68)
        if region_ablation is not None:
            region_onehot[:, region_ablation] = 0.0
        region_onehot = np.tile(region_onehot[np.newaxis, :, :], (num_frames, 1, 1))
        node_feats = np.concatenate([node_feats, region_onehot], axis=-1)

    node_feats = node_feats.reshape(num_frames * 68, -1)
    x = torch.from_numpy(node_feats).float()

    edge_list = []
    type_list = []
    weight_list = []

    if add_static_edges:
        if random_static_edges:
            static_intra = build_random_static_edges(68, seed=random_edge_seed)
        else:
            static_intra = build_static_edges()
        for t in range(num_frames):
            offset = t * 68
            frame_edges = static_intra + offset
            edge_list.append(frame_edges)
            type_list.append(
                torch.full((frame_edges.size(1),), EDGE_TYPE_STATIC, dtype=torch.long)
            )
            weight_list.append(torch.ones(frame_edges.size(1), dtype=torch.float))

    if add_temporal_edges:
        temporal_edges = build_temporal_edges(num_frames, 68)
        edge_list.append(temporal_edges)
        type_list.append(
            torch.full((temporal_edges.size(1),), EDGE_TYPE_TEMPORAL, dtype=torch.long)
        )
        weight_list.append(torch.ones(temporal_edges.size(1), dtype=torch.float))

    if edge_list:
        base_edge_index = torch.cat(edge_list, dim=1)
    else:
        base_edge_index = torch.zeros((2, 0), dtype=torch.long)

    if add_dynamic_edges:
        dynamic_edges = build_dynamic_edges(
            num_frames=num_frames,
            num_landmarks=68,
            node_features=x,
            k=dynamic_k,
            existing_edges=base_edge_index,
            metric=dynamic_metric,
            feature_mode=dynamic_feature,
            region_restricted=dynamic_region_restricted,
        )
        if dynamic_edges.numel() > 0:
            edge_list.append(dynamic_edges)
            type_list.append(
                torch.full((dynamic_edges.size(1),), EDGE_TYPE_DYNAMIC, dtype=torch.long)
            )
            weight_list.append(
                torch.full((dynamic_edges.size(1),), dynamic_edge_weight, dtype=torch.float)
            )

    if add_self_loops:
        num_nodes = num_frames * 68
        self_loop_edges = torch.arange(num_nodes, dtype=torch.long).repeat(2, 1)
        edge_list.append(self_loop_edges)
        type_list.append(
            torch.full((num_nodes,), EDGE_TYPE_STATIC, dtype=torch.long)
        )
        weight_list.append(torch.ones(num_nodes, dtype=torch.float))

    if edge_list:
        edge_index = torch.cat(edge_list, dim=1)
        edge_type = torch.cat(type_list, dim=0)
        edge_weight = torch.cat(weight_list, dim=0)

        edge_index, edge_type, edge_weight = aggregate_edges_with_type_and_weight(
            edge_index, edge_type, edge_weight
        )
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_type = torch.zeros((0,), dtype=torch.long)
        edge_weight = torch.zeros((0,), dtype=torch.float)

    y = torch.tensor([label], dtype=torch.float)
    data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight, y=y)
    if use_edge_type:
        data.edge_type = edge_type
    return data
