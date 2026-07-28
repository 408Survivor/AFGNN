#!/usr/bin/env python3
"""
Evaluation script for AFGNN (Audio-Facial Graph Neural Network).

Works for all modalities; the model and data graphs are rebuilt from the config
embedded in the checkpoint when present, otherwise from --config.

Usage:
    conda activate DVlog
    cd /home/ltq/DepressionCode/DepGNN/AFGNN
    python src/test.py --config experiments/configs/afgnn_face_enhanced_focal.yaml --split test

Results are saved to `<output_dir>/test_{split}_results.json` (rename per GUIDE.md).
"""

import argparse
import json
import os
from datetime import datetime

import torch

from utils.builders import build_loaders, build_model, load_config
from utils.run_context import start_run
from utils.trainer import evaluate


def main():
    parser = argparse.ArgumentParser(description="Evaluate AFGNN")
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/configs/afgnn_face_only.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (defaults to config's training.checkpoint_path)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "valid", "test"],
        help="Which split to evaluate",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments/results",
        help="Directory to save test results",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Default checkpoint from config if not provided
    if args.checkpoint is None:
        args.checkpoint = cfg["training"]["checkpoint_path"]

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load checkpoint first, so we can rebuild the exact model/loaders from the
    # config embedded at training time (P0-2). Fall back to the --config YAML for
    # older checkpoints that predate embedded configs.
    checkpoint = torch.load(
        args.checkpoint, map_location=device, weights_only=True
    )
    embedded_cfg = checkpoint.get("config")
    if embedded_cfg is not None:
        run_cfg = embedded_cfg
        print(f"[Config] Using config embedded in checkpoint: {args.checkpoint}")
    else:
        run_cfg = cfg
        print("[Config] Checkpoint has no embedded config; falling back to --config")

    run = start_run(run_cfg, "eval")

    # Data loader + model, built from the resolved config
    loaders = build_loaders(run_cfg, augment=False)
    model = build_model(run_cfg, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint from: {args.checkpoint}")
    metric_key = checkpoint.get("early_stopping_metric", "f1")
    best_score = checkpoint.get("best_score", checkpoint.get("best_f1", "N/A"))
    best_threshold = checkpoint.get("best_threshold", 0.5)
    print(f"Best val {metric_key.upper()} (from training): {best_score}")
    print(f"Best validation threshold (F1-tuned): {best_threshold:.4f}")

    # Evaluate using the threshold tuned on the validation set
    metrics = evaluate(
        model, loaders[args.split], device, threshold=best_threshold
    )
    print(f"\n{args.split.upper()} Results:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1:        {metrics['f1']:.4f}")
    print(f"  AUC:       {metrics['auc']:.4f}")

    # Save results to JSON
    os.makedirs(args.output_dir, exist_ok=True)
    result_path = os.path.join(args.output_dir, f"test_{args.split}_results.json")

    result_record = {
        "split": args.split,
        "checkpoint": args.checkpoint,
        "config": args.config,
        "timestamp": datetime.now().isoformat(),
        "metrics": {k: float(v) for k, v in metrics.items()},
    }

    with open(result_path, "w") as f:
        json.dump(result_record, f, indent=2)

    print(f"\nResults saved to: {result_path}")

    run.finalize(
        checkpoint_path=args.checkpoint,
        metrics=metrics,
        extra={"split": args.split},
    )


if __name__ == "__main__":
    main()
