#!/usr/bin/env python3
"""
Self-ensemble over multiple random seeds for the same AFGNN config.

Usage:
    conda activate DVlog
    cd /home/ltq/DepressionCode/DepGNN/AFGNN
    python src/seed_ensemble_test.py \
        --config experiments/configs/afgnn_face_enhanced_focal.yaml \
        --seeds 42 43 44 \
        --split test
"""

import argparse
import datetime
import json
import os

import numpy as np
import torch

from utils.builders import build_loaders, build_model, load_config
from utils.metrics import compute_metrics, find_best_threshold
from utils.trainer import get_predictions


def get_seed_checkpoint(base_path: str, seed: int) -> str:
    base, ext = os.path.splitext(base_path)
    return f"{base}_seed{seed}{ext}"


def main():
    parser = argparse.ArgumentParser(description="AFGNN self-ensemble over random seeds")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        required=True,
        help="Random seeds to ensemble",
    )
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

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    loaders = build_loaders(
        cfg,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=False,
    )

    base_checkpoint = cfg["training"]["checkpoint_path"]
    val_probs_list = []
    target_probs_list = []
    target_labels = None

    print("Collecting predictions from seed checkpoints...")
    for seed in args.seeds:
        checkpoint_path = get_seed_checkpoint(base_checkpoint, seed)
        print(f"  -> seed {seed}: {checkpoint_path}")

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        model = build_model(cfg, device)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        val_labels, val_probs = get_predictions(model, loaders["valid"], device)
        tgt_labels, tgt_probs = get_predictions(model, loaders[args.split], device)

        val_probs_list.append(val_probs)
        target_probs_list.append(tgt_probs)
        if target_labels is None:
            target_labels = tgt_labels
        else:
            assert np.array_equal(target_labels, tgt_labels)

        del model
        torch.cuda.empty_cache()

    # Simple average across seeds
    avg_probs = np.mean(target_probs_list, axis=0)
    avg_val_probs = np.mean(val_probs_list, axis=0)
    threshold, _ = find_best_threshold(val_labels, avg_val_probs, metric="f1")
    metrics = compute_metrics(target_labels, avg_probs, threshold=threshold)

    print(f"\n--- Seed Ensemble (seeds={args.seeds}) ---")
    print(f"Threshold (from val F1): {threshold:.4f}")
    for k, v in metrics.items():
        print(f"  {k.capitalize():10s}: {v:.4f}")

    os.makedirs(args.output_dir, exist_ok=True)
    result_path = os.path.join(args.output_dir, f"seed_ensemble_{args.split}_results.json")
    result_record = {
        "split": args.split,
        "config": args.config,
        "seeds": args.seeds,
        "timestamp": datetime.datetime.now().isoformat(),
        "threshold": float(threshold),
        "metrics": {k: float(v) for k, v in metrics.items()},
    }
    with open(result_path, "w") as f:
        json.dump(result_record, f, indent=2)
    print(f"\nResults saved to: {result_path}")


if __name__ == "__main__":
    main()
