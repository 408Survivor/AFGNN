#!/usr/bin/env python3
"""
Late-fusion ensemble inference for AFGNN experts.

Usage:
    conda activate DVlog
    cd /home/ltq/DepressionCode/DepGNN/AFGNN
    python src/ensemble_test.py --split test

Combines five trained experts:
    1. Multimodal Baseline   (Face 16 + Audio 32 + Cross-Attn)
    2. Audio-only            (experiments/checkpoints/afgnn_audio_only.pt)
    3. Multimodal Enhanced   (Face 32 + region one-hot + self-loops + Cross-Attn)
    4. Multimodal Concat     (Face 32 + region one-hot + self-loops + Concat)
    5. Multimodal Focal      (Face 32 + region one-hot + self-loops + Focal Loss)

Ensemble weights are learned on the validation split by grid search to maximize
F1 (with threshold tuning). The learned weights and threshold are then applied
to the requested split.
"""

import argparse
import datetime
import itertools
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import torch

from data.dvlog_face_dataset import get_dvlog_face_loaders
from models.afgnn import AFGNN
from utils.builders import load_config
from utils.metrics import compute_metrics, find_best_threshold
from utils.trainer import get_predictions


# Shared architecture defaults (dropout, heads, etc.)
SHARED_KWARGS = {
    "face_hidden_channels": 128,
    "face_out_channels": 128,
    "face_num_layers": 3,
    "face_heads": 4,
    "audio_hidden_channels": 64,
    "audio_out_channels": 64,
    "audio_num_layers": 2,
    "audio_heads": 4,
    "mlp_hidden": 128,
    "dropout": 0.3,
    "fusion_type": "concat",
    "fusion_hidden_dim": 64,
}

EXPERTS = [
    {
        # This is the previous best multimodal model (Face 16 + Audio 32 + Cross-Attn).
        "name": "face_audio_baseline",
        "config": "experiments/configs/afgnn_face_only.yaml",
        "checkpoint": "experiments/checkpoints/afgnn_face_only_best.pt",
    },
    {
        "name": "audio_only",
        "checkpoint": "experiments/checkpoints/afgnn_audio_only.pt",
        "model_kwargs": {
            "use_face": False,
            "use_audio": True,
            "face_in_channels": 4,
            "audio_in_channels": 25,
        },
        "data_kwargs": {
            "num_frames": 16,
            "audio_num_frames": 32,
            "use_audio": True,
            "add_static_edges": True,
            "add_temporal_edges": True,
            "add_dynamic_edges": False,
            "add_region_onehot": False,
            "add_self_loops": False,
        },
    },
    {
        "name": "face_audio_enhanced",
        "config": "experiments/configs/afgnn_face_enhanced.yaml",
        "checkpoint": "experiments/checkpoints/afgnn_face_enhanced_best.pt",
    },
    {
        "name": "face_audio_enhanced_concat",
        "config": "experiments/configs/afgnn_face_enhanced_concat.yaml",
        "checkpoint": "experiments/checkpoints/afgnn_face_enhanced_concat_best.pt",
    },
    {
        "name": "face_audio_enhanced_focal",
        "config": "experiments/configs/afgnn_face_enhanced_focal.yaml",
        "checkpoint": "experiments/checkpoints/afgnn_face_enhanced_focal_best.pt",
    },
]


def build_expert(expert: Dict, device: torch.device) -> Tuple[AFGNN, dict]:
    """Instantiate an expert model and return its data-loader kwargs."""
    if "config" in expert:
        cfg = load_config(expert["config"])
        model_kwargs = {
            "use_face": cfg["model"].get("use_face", True),
            "use_audio": cfg["model"].get("use_audio", True),
            "face_in_channels": cfg["model"]["face_in_channels"],
            "face_hidden_channels": cfg["model"]["face_hidden_channels"],
            "face_out_channels": cfg["model"]["face_out_channels"],
            "face_num_layers": cfg["model"]["face_num_layers"],
            "face_heads": cfg["model"]["face_heads"],
            "audio_in_channels": cfg["model"].get("audio_in_channels", 25),
            "audio_hidden_channels": cfg["model"].get("audio_hidden_channels", 64),
            "audio_out_channels": cfg["model"].get("audio_out_channels", 64),
            "audio_num_layers": cfg["model"].get("audio_num_layers", 2),
            "audio_heads": cfg["model"].get("audio_heads", 4),
            "mlp_hidden": cfg["model"]["mlp_hidden"],
            "dropout": cfg["model"]["dropout"],
            "fusion_type": cfg["model"].get("fusion_type", "concat"),
            "fusion_hidden_dim": cfg["model"].get("fusion_hidden_dim", 64),
        }
        data_kwargs = {
            "num_frames": cfg["data"]["num_frames"],
            "audio_num_frames": cfg["data"].get("audio_num_frames", cfg["data"]["num_frames"]),
            "use_audio": cfg["model"].get("use_audio", True),
            "add_static_edges": cfg["model"].get("add_static_edges", True),
            "add_temporal_edges": cfg["model"].get("add_temporal_edges", True),
            "add_dynamic_edges": cfg["model"].get("add_dynamic_edges", False),
            "dynamic_k": cfg["model"].get("dynamic_k", 3),
            "dynamic_metric": cfg["model"].get("dynamic_metric", "cosine"),
            "dynamic_edge_weight": cfg["model"].get("dynamic_edge_weight", 1.0),
            "dynamic_feature": cfg["model"].get("dynamic_feature", "full"),
            "dynamic_region_restricted": cfg["model"].get("dynamic_region_restricted", False),
            "use_edge_type": cfg["model"].get("use_edge_type", False),
            "add_region_onehot": cfg["model"].get("add_region_onehot", False),
            "add_self_loops": cfg["model"].get("add_self_loops", False),
        }
    else:
        model_kwargs = {**SHARED_KWARGS, **expert["model_kwargs"]}
        data_kwargs = expert["data_kwargs"]

    model = AFGNN(**model_kwargs).to(device)
    checkpoint = torch.load(
        expert["checkpoint"], map_location=device, weights_only=True
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, data_kwargs


def get_expert_predictions(
    expert: Dict,
    split: str,
    data_dir: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return true labels and predicted probabilities for one expert on one split."""
    model, data_kwargs = build_expert(expert, device)
    loader = get_dvlog_face_loaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        **data_kwargs,
    )[split]
    labels, probs = get_predictions(model, loader, device)
    # Free GPU memory
    del model
    torch.cuda.empty_cache()
    return labels, probs


def grid_search_weights(
    val_probs_list: List[np.ndarray],
    val_labels: np.ndarray,
    step: float = 0.1,
) -> Tuple[Tuple[float, ...], float, float]:
    """Search non-negative weights summing to 1 that maximize validation F1."""
    n = len(val_probs_list)
    n_steps = int(round(1.0 / step))
    best_w = tuple(1.0 / n for _ in range(n))
    best_thr = 0.5
    best_f1 = -1.0

    # Integer partitions of n_steps into n parts, then convert to weights.
    for int_weights in itertools.product(range(n_steps + 1), repeat=n):
        if sum(int_weights) != n_steps:
            continue
        weights = tuple(w * step for w in int_weights)
        ens_probs = sum(w * p for w, p in zip(weights, val_probs_list))
        thr, f1 = find_best_threshold(val_labels, ens_probs, metric="f1")
        if f1 > best_f1:
            best_f1 = f1
            best_w = weights
            best_thr = thr

    return best_w, best_thr, best_f1


def main():
    parser = argparse.ArgumentParser(description="AFGNN late-fusion ensemble")
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "valid", "test"],
        help="Split to evaluate",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/data/ltq/DVlog/processed_official_features",
        help="Directory containing preprocessed D-Vlog features",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="DataLoader workers",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments/results",
        help="Directory to save ensemble results",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Gather predictions from each expert on validation (for weight learning)
    # and on the target split (for final evaluation).
    print("Collecting expert predictions...")
    val_probs_list = []
    target_probs_list = []
    target_labels = None

    for expert in EXPERTS:
        print(f"  -> {expert['name']}: {expert.get('checkpoint', expert.get('config'))}")
        val_labels, val_probs = get_expert_predictions(
            expert, "valid", args.data_dir, args.batch_size, args.num_workers, device
        )
        tgt_labels, tgt_probs = get_expert_predictions(
            expert, args.split, args.data_dir, args.batch_size, args.num_workers, device
        )
        val_probs_list.append(val_probs)
        target_probs_list.append(tgt_probs)
        if target_labels is None:
            target_labels = tgt_labels
        else:
            assert np.array_equal(target_labels, tgt_labels)

    # Simple average ensemble
    n_experts = len(EXPERTS)
    avg_probs = np.mean(target_probs_list, axis=0)
    avg_threshold, _ = find_best_threshold(val_labels, np.mean(val_probs_list, axis=0), metric="f1")
    avg_metrics = compute_metrics(target_labels, avg_probs, threshold=avg_threshold)
    avg_weights = [1.0 / n_experts] * n_experts

    print("\n--- Simple Average Ensemble ---")
    print(f"Weights: {tuple(round(w, 2) for w in avg_weights)}")
    print(f"Threshold (from val F1): {avg_threshold:.4f}")
    for k, v in avg_metrics.items():
        print(f"  {k.capitalize():10s}: {v:.4f}")

    # Weighted ensemble
    best_w, best_thr, best_val_f1 = grid_search_weights(val_probs_list, val_labels, step=0.1)
    weighted_probs = sum(w * p for w, p in zip(best_w, target_probs_list))
    weighted_metrics = compute_metrics(target_labels, weighted_probs, threshold=best_thr)

    print("\n--- Weighted Ensemble (val-F1 optimized) ---")
    weight_str = ", ".join(
        f"{expert['name']}={w:.2f}" for expert, w in zip(EXPERTS, best_w)
    )
    print(f"Weights: {weight_str}")
    print(f"Validation F1 (used for selection): {best_val_f1:.4f}")
    print(f"Threshold (from val F1): {best_thr:.4f}")
    for k, v in weighted_metrics.items():
        print(f"  {k.capitalize():10s}: {v:.4f}")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    result_path = os.path.join(args.output_dir, f"ensemble_{args.split}_results.json")
    result_record = {
        "split": args.split,
        "timestamp": datetime.datetime.now().isoformat(),
        "average_ensemble": {
            "weights": [round(w, 4) for w in avg_weights],
            "threshold": float(avg_threshold),
            "metrics": {k: float(v) for k, v in avg_metrics.items()},
        },
        "weighted_ensemble": {
            "weights": [round(w, 4) for w in best_w],
            "threshold": float(best_thr),
            "val_f1": float(best_val_f1),
            "metrics": {k: float(v) for k, v in weighted_metrics.items()},
        },
    }
    with open(result_path, "w") as f:
        json.dump(result_record, f, indent=2)
    print(f"\nResults saved to: {result_path}")


if __name__ == "__main__":
    main()
