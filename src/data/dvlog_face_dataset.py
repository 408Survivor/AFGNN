"""
PyTorch Dataset for D-Vlog facial landmark and audio graphs.
"""

import os
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from data.augmentation import augment_landmark_sequence
from models.graph_utils import build_audio_graph, build_face_graph, compute_audio_norm_stats


class DVlogFaceDataset(Dataset):
    """Dataset that loads preprocessed D-Vlog visual/acoustic features and builds graphs.

    Args:
        visual_path: path to `{split}_visual.npy`.
        labels_path: path to `{split}_labels.npy`.
        acoustic_path: optional path to `{split}_acoustic.npy`. If provided,
            the returned Data object also contains `audio_x`.
        num_frames: number of frames to sample for the face graph (T_v).
        audio_num_frames: number of frames to sample for the audio graph.
                          Defaults to num_frames if not provided.
        add_static_edges: whether to add static anatomical edges.
        add_temporal_edges: whether to add temporal edges.
        add_dynamic_edges: whether to add feature-similarity dynamic edges.
        dynamic_k: number of dynamic neighbors per landmark per frame.
        dynamic_metric: similarity metric for dynamic edges.
        dynamic_edge_weight: weight applied to dynamic edges.
        dynamic_feature: "full" or "motion" feature for dynamic edge similarity.
        dynamic_region_restricted: restrict dynamic edges to same facial region.
        use_edge_type: whether to include edge_type in the graph.
        add_region_onehot: whether to append a facial-region one-hot vector.
        add_self_loops: whether to add self-loop edges for every node.
        use_velocity: whether to append velocity [dx, dy] to node features.
        region_ablation: if not None, zero out the one-hot of this region id.
        audio_norm_mean: training-set mean for acoustic normalization.
        audio_norm_std: training-set std for acoustic normalization.
    """

    def __init__(
        self,
        visual_path: str,
        labels_path: str,
        acoustic_path: str = None,
        num_frames: int = 16,
        audio_num_frames: int = None,
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
        augment: bool = False,
        rotation_range: Tuple[float, float] = (-10.0, 10.0),
        scale_range: Tuple[float, float] = (0.95, 1.05),
        translate_range: Tuple[float, float] = (-0.05, 0.05),
        noise_std: float = 0.02,
        temporal_mask_prob: float = 0.2,
        temporal_mask_max_ratio: float = 0.15,
        audio_use_delta: bool = False,
        audio_self_loops: bool = False,
        audio_skip: int = 0,
        audio_norm_mean: np.ndarray = None,
        audio_norm_std: np.ndarray = None,
    ):
        if not os.path.exists(visual_path):
            raise FileNotFoundError(f"Visual features not found: {visual_path}")
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"Labels not found: {labels_path}")

        self.visual = np.load(visual_path).astype(np.float32)
        self.labels = np.load(labels_path).astype(np.int64)
        self.num_frames = num_frames
        self.audio_num_frames = audio_num_frames if audio_num_frames is not None else num_frames
        self.add_static_edges = add_static_edges
        self.add_temporal_edges = add_temporal_edges
        self.add_dynamic_edges = add_dynamic_edges
        self.dynamic_k = dynamic_k
        self.dynamic_metric = dynamic_metric
        self.dynamic_edge_weight = dynamic_edge_weight
        self.dynamic_feature = dynamic_feature
        self.dynamic_region_restricted = dynamic_region_restricted
        self.use_edge_type = use_edge_type
        self.add_region_onehot = add_region_onehot
        self.add_self_loops = add_self_loops
        self.use_velocity = use_velocity
        self.region_ablation = region_ablation
        self.augment = augment
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.translate_range = translate_range
        self.noise_std = noise_std
        self.temporal_mask_prob = temporal_mask_prob
        self.temporal_mask_max_ratio = temporal_mask_max_ratio
        self.audio_use_delta = audio_use_delta
        self.audio_self_loops = audio_self_loops
        self.audio_skip = audio_skip
        self.audio_norm_mean = audio_norm_mean
        self.audio_norm_std = audio_norm_std
        self.random_static_edges = random_static_edges
        self.random_edge_seed = random_edge_seed

        self.acoustic = None
        self.use_audio = False
        if acoustic_path is not None and os.path.exists(acoustic_path):
            self.acoustic = np.load(acoustic_path).astype(np.float32)
            self.use_audio = True

        assert len(self.visual) == len(self.labels), (
            f"Mismatch: {len(self.visual)} visual samples vs {len(self.labels)} labels"
        )
        if self.acoustic is not None:
            assert len(self.acoustic) == len(self.labels), (
                f"Mismatch: {len(self.acoustic)} acoustic samples vs {len(self.labels)} labels"
            )

    def __len__(self):
        return len(self.visual)

    def __getitem__(self, idx):
        visual_seq = self.visual[idx]
        label = int(self.labels[idx])

        if self.augment:
            visual_seq = augment_landmark_sequence(
                visual_seq,
                rotation_range=self.rotation_range,
                scale_range=self.scale_range,
                translate_range=self.translate_range,
                noise_std=self.noise_std,
                temporal_mask_prob=self.temporal_mask_prob,
                temporal_mask_max_ratio=self.temporal_mask_max_ratio,
            )

        data = build_face_graph(
            visual_seq,
            label=label,
            num_frames=self.num_frames,
            add_static_edges=self.add_static_edges,
            add_temporal_edges=self.add_temporal_edges,
            add_dynamic_edges=self.add_dynamic_edges,
            dynamic_k=self.dynamic_k,
            dynamic_metric=self.dynamic_metric,
            dynamic_edge_weight=self.dynamic_edge_weight,
            dynamic_feature=self.dynamic_feature,
            dynamic_region_restricted=self.dynamic_region_restricted,
            use_edge_type=self.use_edge_type,
            add_region_onehot=self.add_region_onehot,
            add_self_loops=self.add_self_loops,
            use_velocity=self.use_velocity,
            region_ablation=self.region_ablation,
            random_static_edges=self.random_static_edges,
            random_edge_seed=self.random_edge_seed,
        )

        if self.use_audio and self.acoustic is not None:
            audio_data = build_audio_graph(
                self.acoustic[idx],
                label=label,
                num_frames=self.audio_num_frames,
                norm_mean=self.audio_norm_mean,
                norm_std=self.audio_norm_std,
                use_delta=self.audio_use_delta,
                self_loops=self.audio_self_loops,
                skip=self.audio_skip,
            )
            data.audio_x = audio_data.audio_x
            # audio_edge_index is intentionally not attached; AudioGNN builds
            # the chain internally because the audio graph has a different
            # number of nodes than the face graph.

        return data


class DVlogFaceArrayDataset(Dataset):
    """Same as DVlogFaceDataset but takes in-memory arrays instead of file paths.

    Useful for combining train/valid splits or for cross-validation.
    """

    def __init__(
        self,
        visual: np.ndarray,
        labels: np.ndarray,
        acoustic: np.ndarray = None,
        num_frames: int = 16,
        audio_num_frames: int = None,
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
        augment: bool = False,
        rotation_range: Tuple[float, float] = (-10.0, 10.0),
        scale_range: Tuple[float, float] = (0.95, 1.05),
        translate_range: Tuple[float, float] = (-0.05, 0.05),
        noise_std: float = 0.02,
        temporal_mask_prob: float = 0.2,
        temporal_mask_max_ratio: float = 0.15,
        audio_use_delta: bool = False,
        audio_self_loops: bool = False,
        audio_skip: int = 0,
        audio_norm_mean: np.ndarray = None,
        audio_norm_std: np.ndarray = None,
    ):
        self.visual = np.asarray(visual, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.num_frames = num_frames
        self.audio_num_frames = audio_num_frames if audio_num_frames is not None else num_frames
        self.add_static_edges = add_static_edges
        self.add_temporal_edges = add_temporal_edges
        self.add_dynamic_edges = add_dynamic_edges
        self.dynamic_k = dynamic_k
        self.dynamic_metric = dynamic_metric
        self.dynamic_edge_weight = dynamic_edge_weight
        self.dynamic_feature = dynamic_feature
        self.dynamic_region_restricted = dynamic_region_restricted
        self.use_edge_type = use_edge_type
        self.add_region_onehot = add_region_onehot
        self.add_self_loops = add_self_loops
        self.use_velocity = use_velocity
        self.region_ablation = region_ablation
        self.augment = augment
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.translate_range = translate_range
        self.noise_std = noise_std
        self.temporal_mask_prob = temporal_mask_prob
        self.temporal_mask_max_ratio = temporal_mask_max_ratio
        self.audio_use_delta = audio_use_delta
        self.audio_self_loops = audio_self_loops
        self.audio_skip = audio_skip
        self.audio_norm_mean = audio_norm_mean
        self.audio_norm_std = audio_norm_std

        self.acoustic = None
        self.use_audio = False
        if acoustic is not None:
            self.acoustic = np.asarray(acoustic, dtype=np.float32)
            self.use_audio = True

        assert len(self.visual) == len(self.labels), (
            f"Mismatch: {len(self.visual)} visual samples vs {len(self.labels)} labels"
        )
        if self.acoustic is not None:
            assert len(self.acoustic) == len(self.labels), (
                f"Mismatch: {len(self.acoustic)} acoustic samples vs {len(self.labels)} labels"
            )

    def __len__(self):
        return len(self.visual)

    def __getitem__(self, idx):
        visual_seq = self.visual[idx]
        label = int(self.labels[idx])

        if self.augment:
            visual_seq = augment_landmark_sequence(
                visual_seq,
                rotation_range=self.rotation_range,
                scale_range=self.scale_range,
                translate_range=self.translate_range,
                noise_std=self.noise_std,
                temporal_mask_prob=self.temporal_mask_prob,
                temporal_mask_max_ratio=self.temporal_mask_max_ratio,
            )

        data = build_face_graph(
            visual_seq,
            label=label,
            num_frames=self.num_frames,
            add_static_edges=self.add_static_edges,
            add_temporal_edges=self.add_temporal_edges,
            add_dynamic_edges=self.add_dynamic_edges,
            dynamic_k=self.dynamic_k,
            dynamic_metric=self.dynamic_metric,
            dynamic_edge_weight=self.dynamic_edge_weight,
            dynamic_feature=self.dynamic_feature,
            dynamic_region_restricted=self.dynamic_region_restricted,
            use_edge_type=self.use_edge_type,
            add_region_onehot=self.add_region_onehot,
            add_self_loops=self.add_self_loops,
            use_velocity=self.use_velocity,
            region_ablation=self.region_ablation,
        )

        if self.use_audio and self.acoustic is not None:
            audio_data = build_audio_graph(
                self.acoustic[idx],
                label=label,
                num_frames=self.audio_num_frames,
                norm_mean=self.audio_norm_mean,
                norm_std=self.audio_norm_std,
                use_delta=self.audio_use_delta,
                self_loops=self.audio_self_loops,
                skip=self.audio_skip,
            )
            data.audio_x = audio_data.audio_x

        return data


def get_dvlog_face_loaders(
    data_dir: str,
    num_frames: int = 16,
    batch_size: int = 32,
    num_workers: int = 4,
    audio_num_frames: int = None,
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
    augment: bool = False,
    rotation_range: Tuple[float, float] = (-10.0, 10.0),
    scale_range: Tuple[float, float] = (0.95, 1.05),
    translate_range: Tuple[float, float] = (-0.05, 0.05),
    noise_std: float = 0.02,
    temporal_mask_prob: float = 0.2,
    temporal_mask_max_ratio: float = 0.15,
    audio_use_delta: bool = False,
    audio_self_loops: bool = False,
    audio_skip: int = 0,
    use_audio: bool = True,
    worker_init_fn=None,
):
    """Build train/valid/test DataLoaders for D-Vlog face (and optional audio) graphs."""
    # Compute audio normalization stats from training split once.
    audio_mean, audio_std = None, None
    train_acoustic_path = os.path.join(data_dir, "train_acoustic.npy")
    if use_audio and os.path.exists(train_acoustic_path):
        train_acoustic = np.load(train_acoustic_path).astype(np.float32)
        audio_mean, audio_std = compute_audio_norm_stats(train_acoustic)

    loaders = {}
    for split in ["train", "valid", "test"]:
        visual_path = os.path.join(data_dir, f"{split}_visual.npy")
        labels_path = os.path.join(data_dir, f"{split}_labels.npy")
        acoustic_path = os.path.join(data_dir, f"{split}_acoustic.npy")

        is_train = split == "train"

        dataset = DVlogFaceDataset(
            visual_path=visual_path,
            labels_path=labels_path,
            acoustic_path=acoustic_path if use_audio else None,
            num_frames=num_frames,
            audio_num_frames=audio_num_frames,
            add_static_edges=add_static_edges,
            add_temporal_edges=add_temporal_edges,
            add_dynamic_edges=add_dynamic_edges,
            dynamic_k=dynamic_k,
            dynamic_metric=dynamic_metric,
            dynamic_edge_weight=dynamic_edge_weight,
            dynamic_feature=dynamic_feature,
            dynamic_region_restricted=dynamic_region_restricted,
            use_edge_type=use_edge_type,
            add_region_onehot=add_region_onehot,
            add_self_loops=add_self_loops,
            use_velocity=use_velocity,
            region_ablation=region_ablation,
            random_static_edges=random_static_edges,
            random_edge_seed=random_edge_seed,
            augment=(augment and is_train),
            rotation_range=rotation_range,
            scale_range=scale_range,
            translate_range=translate_range,
            noise_std=noise_std,
            temporal_mask_prob=temporal_mask_prob,
            temporal_mask_max_ratio=temporal_mask_max_ratio,
            audio_use_delta=audio_use_delta,
            audio_self_loops=audio_self_loops,
            audio_skip=audio_skip,
            audio_norm_mean=audio_mean,
            audio_norm_std=audio_std,
        )

        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=is_train,
            worker_init_fn=worker_init_fn,
        )

    return loaders
