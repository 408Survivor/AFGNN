#!/usr/bin/env python3
"""
Per-seed stability evaluation for AFGNN.

Unlike `seed_ensemble_test.py` (which averages probabilities across seeds into a
single ensemble), this script evaluates each single-seed checkpoint independently
and reports the mean +/- std of the test metrics across seeds. This quantifies the
variance caused by random initialization, addressing the "did you cherry-pick a
good seed?" question.

For each seed:
    1. load the single-model checkpoint,
    2. search the F1-optimal threshold on the validation split,
    3. apply that threshold on the target split and compute metrics.
Then aggregate mean and sample std (ddof=1) across seeds.

Usage:
    conda activate DVlog
    cd /home/ltq/DepressionCode/DepGNN/AFGNN
    python src/seed_stability_test.py \
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


METRIC_KEYS = ["accuracy", "precision", "recall", "f1", "auc"]


def get_seed_checkpoint(base_path: str, seed: int) -> str:
    base, ext = os.path.splitext(base_path)
    return f"{base}_seed{seed}{ext}"


def main():
    parser = argparse.ArgumentParser(description="AFGNN per-seed stability (mean +/- std)")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        required=True,
        help="Random seeds to evaluate independently",
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
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments/results",
        help="Directory to save stability results",
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
    per_seed = []

    print("Evaluating each seed independently...")
    for seed in args.seeds:
        checkpoint_path = get_seed_checkpoint(base_checkpoint, seed)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        model = build_model(cfg, device)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        val_labels, val_probs = get_predictions(model, loaders["valid"], device)
        tgt_labels, tgt_probs = get_predictions(model, loaders[args.split], device)

        threshold, _ = find_best_threshold(val_labels, val_probs, metric="f1")
        metrics = compute_metrics(tgt_labels, tgt_probs, threshold=threshold)

        per_seed.append(
            {
                "seed": seed,
                "checkpoint": checkpoint_path,
                "threshold": float(threshold),
                "metrics": {k: float(metrics[k]) for k in METRIC_KEYS},
            }
        )
        print(
            f"  seed {seed}: thr={threshold:.2f}  "
            + "  ".join(f"{k[:4].capitalize()}={metrics[k]:.4f}" for k in METRIC_KEYS)
        )

        del model
        torch.cuda.empty_cache()

    # Aggregate mean and sample std (ddof=1) across seeds.
    agg = {}
    for k in METRIC_KEYS:
        vals = np.array([s["metrics"][k] for s in per_seed], dtype=np.float64)
        std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        agg[k] = {"mean": float(np.mean(vals)), "std": std}

    print(f"\n--- Seed stability over seeds={args.seeds} (mean +/- sample std) ---")
    for k in METRIC_KEYS:
        print(f"  {k.capitalize():10s}: {agg[k]['mean']:.4f} +/- {agg[k]['std']:.4f}")

    os.makedirs(args.output_dir, exist_ok=True)
    seed_tag = "-".join(str(s) for s in args.seeds)
    result_path = os.path.join(
        args.output_dir, f"seed_stability_{seed_tag}_{args.split}_results.json"
    )
    result_record = {
        "split": args.split,
        "config": args.config,
        "seeds": args.seeds,
        "timestamp": datetime.datetime.now().isoformat(),
        "std_ddof": 1,
        "per_seed": per_seed,
        "aggregate": agg,
    }
    with open(result_path, "w") as f:
        json.dump(result_record, f, indent=2)
    print(f"\nResults saved to: {result_path}")


if __name__ == "__main__":
    main()
