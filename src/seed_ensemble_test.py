#!/usr/bin/env python3
"""
Self-ensemble over multiple random seeds for the same AFGNN config.

Usage (preferred, exp_N system):
    conda activate DVlog
    cd /home/ltq/Code/AFGNN
    python src/seed_ensemble_test.py \
        --exp_dirs experiments/exp_7 experiments/exp_11 experiments/exp_12 \
                   experiments/exp_13 experiments/exp_14 \
        --split test

Legacy mode (config + seeds, checkpoints from config's checkpoint_path):
    python src/seed_ensemble_test.py \
        --config experiments/configs/afgnn_face_enhanced_focal.yaml \
        --seeds 42 43 44 --split test
"""

import argparse
import datetime
import glob
import json
import os
import re

import numpy as np
import torch

from utils.builders import build_loaders, build_model, load_config
from utils.metrics import compute_metrics, find_best_threshold
from utils.trainer import get_predictions


def get_seed_checkpoint(base_path: str, seed: int) -> str:
    base, ext = os.path.splitext(base_path)
    return f"{base}_seed{seed}{ext}"


def collect_checkpoints(args):
    """Return (config_path, [(seed, checkpoint_path), ...]).

    Two modes:
    - --exp_dirs: exp_N folders from the current experiment system; the config
      snapshot comes from the first folder, each folder contributes its
      ``best_seed*.pt``.
    - default (legacy): --config + --seeds, checkpoints derived from the
      config's ``training.checkpoint_path``.
    """
    if args.exp_dirs:
        config_path = os.path.join(args.exp_dirs[0], "config.yaml")
        pairs = []
        for d in args.exp_dirs:
            ckpts = glob.glob(os.path.join(d, "best_seed*.pt"))
            if len(ckpts) != 1:
                raise FileNotFoundError(
                    f"{d}: expected exactly one best_seed*.pt, found {len(ckpts)}"
                )
            seed = int(re.search(r"best_seed(\d+)\.pt", ckpts[0]).group(1))
            pairs.append((seed, ckpts[0]))
        return config_path, sorted(pairs)
    if args.config is None or args.seeds is None:
        raise SystemExit("请提供 --exp_dirs，或同时提供 --config 和 --seeds")
    cfg = load_config(args.config)
    base = cfg["training"]["checkpoint_path"]
    return args.config, [(s, get_seed_checkpoint(base, s)) for s in args.seeds]


def main():
    parser = argparse.ArgumentParser(description="AFGNN self-ensemble over random seeds")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config file (legacy mode)")
    parser.add_argument(
        "--exp_dirs",
        type=str,
        nargs="+",
        default=None,
        help="exp_N folders to ensemble (preferred; config from the first folder)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Random seeds to ensemble (legacy mode)",
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
        default="/data/ltq/D-Vlog_Raw/processed_official_features",
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

    config_path, seed_ckpts = collect_checkpoints(args)
    cfg = load_config(config_path)
    seeds = [s for s, _ in seed_ckpts]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    loaders = build_loaders(
        cfg,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=False,
    )

    val_probs_list = []
    target_probs_list = []
    target_labels = None

    print("Collecting predictions from seed checkpoints...")
    for seed, checkpoint_path in seed_ckpts:
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

    print(f"\n--- Seed Ensemble (seeds={seeds}) ---")
    print(f"Threshold (from val F1): {threshold:.4f}")
    for k, v in metrics.items():
        print(f"  {k.capitalize():10s}: {v:.4f}")

    # Prefer the original config path recorded in the exp_N snapshot
    # (the snapshot itself is just named "config.yaml").
    name_source = cfg.get("_config_path", config_path)
    expid = os.path.splitext(os.path.basename(name_source))[0]
    expid = expid[len("afgnn_"):] if expid.startswith("afgnn_") else expid
    seed_tag = f"seed{min(seeds)}-{max(seeds)}" if len(seeds) > 1 else f"seed{seeds[0]}"
    os.makedirs(args.output_dir, exist_ok=True)
    result_path = os.path.join(
        args.output_dir, f"seed_ensemble_{expid}_{seed_tag}_{args.split}.json"
    )
    result_record = {
        "split": args.split,
        "config": config_path,
        "seeds": seeds,
        "checkpoints": [p for _, p in seed_ckpts],
        "timestamp": datetime.datetime.now().isoformat(),
        "threshold": float(threshold),
        "metrics": {k: float(v) for k, v in metrics.items()},
    }
    with open(result_path, "w") as f:
        json.dump(result_record, f, indent=2)
    print(f"\nResults saved to: {result_path}")


if __name__ == "__main__":
    main()
