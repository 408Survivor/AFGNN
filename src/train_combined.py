#!/usr/bin/env python3
"""
Combined training script for AFGNN.

Merges the official train and valid splits, holds out 10% as an internal
validation set for early stopping, and evaluates on the official test split.

Usage:
    conda activate DVlog
    cd /home/ltq/DepressionCode/DepGNN/AFGNN
    python src/train_combined.py --config experiments/configs/afgnn_face_enhanced.yaml
"""

import argparse
import json
import os
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from data.dvlog_face_dataset import DVlogFaceArrayDataset
from models.graph_utils import compute_audio_norm_stats
from utils.builders import build_model, load_config
from utils.losses import FocalLoss, compute_class_weights
from utils.run_context import start_run
from utils.trainer import evaluate, train_model


def build_criterion(labels, loss_type="bce", focal_alpha=0.25, focal_gamma=2.0):
    """Build loss criterion based on labels and config."""
    if loss_type == "focal":
        print(f"[Loss] FocalLoss(alpha={focal_alpha}, gamma={focal_gamma})")
        return FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

    elif loss_type == "weighted_bce":
        weights = compute_class_weights(torch.from_numpy(labels))
        pos_weight = weights[1] / weights[0]
        print(f"[Loss] Weighted BCE (pos_weight={pos_weight:.4f})")
        return torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight], dtype=torch.float)
        )

    else:
        print("[Loss] Standard BCEWithLogitsLoss")
        return torch.nn.BCEWithLogitsLoss()


def build_scheduler(optimizer, cfg):
    """Build optional learning rate scheduler."""
    scheduler_cfg = cfg["training"].get("scheduler", None)
    if scheduler_cfg is None:
        return None

    sched_type = scheduler_cfg.get("type", "plateau")
    if sched_type == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=scheduler_cfg.get("factor", 0.5),
            patience=scheduler_cfg.get("patience", 5),
            verbose=True,
        )
        print(
            f"[Scheduler] ReduceLROnPlateau(factor={scheduler_cfg.get('factor', 0.5)}, "
            f"patience={scheduler_cfg.get('patience', 5)})"
        )
        return scheduler
    elif sched_type == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_cfg.get("step_size", 20),
            gamma=scheduler_cfg.get("gamma", 0.5),
        )
        print(
            f"[Scheduler] StepLR(step_size={scheduler_cfg.get('step_size', 20)}, "
            f"gamma={scheduler_cfg.get('gamma', 0.5)})"
        )
        return scheduler
    else:
        raise ValueError(f"Unsupported scheduler type: {sched_type}")


def make_array_dataset(
    visual, labels, acoustic, audio_mean, audio_std, cfg, augment=False
):
    """Create a DVlogFaceArrayDataset from in-memory arrays."""
    return DVlogFaceArrayDataset(
        visual=visual,
        labels=labels,
        acoustic=acoustic,
        num_frames=cfg["data"]["num_frames"],
        audio_num_frames=cfg["data"].get("audio_num_frames", cfg["data"]["num_frames"]),
        add_static_edges=cfg["model"]["add_static_edges"],
        add_temporal_edges=cfg["model"]["add_temporal_edges"],
        add_dynamic_edges=cfg["model"].get("add_dynamic_edges", False),
        dynamic_k=cfg["model"].get("dynamic_k", 3),
        dynamic_metric=cfg["model"].get("dynamic_metric", "cosine"),
        dynamic_edge_weight=cfg["model"].get("dynamic_edge_weight", 1.0),
        dynamic_feature=cfg["model"].get("dynamic_feature", "full"),
        dynamic_region_restricted=cfg["model"].get("dynamic_region_restricted", False),
        use_edge_type=cfg["model"].get("use_edge_type", False),
        add_region_onehot=cfg["model"].get("add_region_onehot", False),
        add_self_loops=cfg["model"].get("add_self_loops", False),
        augment=augment,
        rotation_range=cfg.get("augmentation", {}).get("rotation_range", (-10.0, 10.0)),
        scale_range=cfg.get("augmentation", {}).get("scale_range", (0.95, 1.05)),
        translate_range=cfg.get("augmentation", {}).get(
            "translate_range", (-0.05, 0.05)
        ),
        noise_std=cfg.get("augmentation", {}).get("noise_std", 0.02),
        temporal_mask_prob=cfg.get("augmentation", {}).get("temporal_mask_prob", 0.2),
        temporal_mask_max_ratio=cfg.get("augmentation", {}).get(
            "temporal_mask_max_ratio", 0.15
        ),
        audio_use_delta=cfg["model"].get("audio_use_delta", False),
        audio_self_loops=cfg["model"].get("audio_self_loops", False),
        audio_skip=cfg["model"].get("audio_skip", 0),
        audio_norm_mean=audio_mean,
        audio_norm_std=audio_std,
    )


def main():
    parser = argparse.ArgumentParser(description="Train AFGNN on combined train+valid")
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/configs/afgnn_face_enhanced.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Ratio of combined data to hold out for validation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/valid split",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments/results",
        help="Directory to save training history",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    run = start_run(cfg, "train_combined", seed=args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_dir = cfg["data"]["processed_dir"]
    use_audio = cfg["model"].get("use_audio", False)

    # Load official train/valid/test arrays
    train_visual = np.load(os.path.join(data_dir, "train_visual.npy"))
    train_labels = np.load(os.path.join(data_dir, "train_labels.npy"))
    valid_visual = np.load(os.path.join(data_dir, "valid_visual.npy"))
    valid_labels = np.load(os.path.join(data_dir, "valid_labels.npy"))
    test_visual = np.load(os.path.join(data_dir, "test_visual.npy"))
    test_labels = np.load(os.path.join(data_dir, "test_labels.npy"))

    train_acoustic, valid_acoustic, test_acoustic = None, None, None
    if use_audio:
        train_acoustic = np.load(os.path.join(data_dir, "train_acoustic.npy"))
        valid_acoustic = np.load(os.path.join(data_dir, "valid_acoustic.npy"))
        test_acoustic = np.load(os.path.join(data_dir, "test_acoustic.npy"))

    # Combine official train + valid
    combined_visual = np.concatenate([train_visual, valid_visual], axis=0)
    combined_labels = np.concatenate([train_labels, valid_labels], axis=0)
    combined_acoustic = None
    if use_audio:
        combined_acoustic = np.concatenate([train_acoustic, valid_acoustic], axis=0)

    print(
        f"[Combined] train+valid: {len(combined_visual)} samples "
        f"(train {len(train_visual)} + valid {len(valid_visual)})"
    )

    # Compute audio normalization stats on combined data
    audio_mean, audio_std = None, None
    if use_audio and combined_acoustic is not None:
        audio_mean, audio_std = compute_audio_norm_stats(combined_acoustic)

    # Build full combined dataset (no augmentation yet; set per-subset below)
    full_dataset = make_array_dataset(
        combined_visual,
        combined_labels,
        combined_acoustic,
        audio_mean,
        audio_std,
        cfg,
        augment=False,
    )

    # Random split into internal train / validation
    rng = np.random.RandomState(args.seed)
    indices = np.arange(len(full_dataset))
    rng.shuffle(indices)
    split_idx = int(len(full_dataset) * (1 - args.val_ratio))
    train_idx = indices[:split_idx].tolist()
    val_idx = indices[split_idx:].tolist()

    print(
        f"[Split] internal train: {len(train_idx)}, internal val: {len(val_idx)}, "
        f"val ratio: {args.val_ratio}"
    )

    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)

    # Enable augmentation only on internal training subset
    # Subset does not expose the underlying dataset flags, so we enable globally
    # and rely on DataLoader shuffle/drop_last for train vs val behavior.
    # Note: this means val will also have augment if enabled in config.
    # To avoid that, we create separate datasets with/without augment.
    if cfg.get("augmentation", {}).get("enabled", False):
        train_dataset = Subset(
            make_array_dataset(
                combined_visual,
                combined_labels,
                combined_acoustic,
                audio_mean,
                audio_std,
                cfg,
                augment=True,
            ),
            train_idx,
        )
        val_dataset = Subset(
            make_array_dataset(
                combined_visual,
                combined_labels,
                combined_acoustic,
                audio_mean,
                audio_std,
                cfg,
                augment=False,
            ),
            val_idx,
        )

    batch_size = cfg["training"]["batch_size"]
    num_workers = cfg["training"]["num_workers"]

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # Test dataset (no augmentation)
    test_dataset = make_array_dataset(
        test_visual,
        test_labels,
        test_acoustic,
        audio_mean,
        audio_std,
        cfg,
        augment=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # Model
    model = build_model(cfg, device)

    print(model)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_params:,}")

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    # Loss (computed on internal training labels)
    train_labels_subset = combined_labels[train_idx]
    criterion = build_criterion(
        train_labels_subset,
        loss_type=cfg["training"].get("loss", "bce"),
        focal_alpha=cfg["training"].get("focal_alpha", 0.25),
        focal_gamma=cfg["training"].get("focal_gamma", 2.0),
    ).to(device)

    # Scheduler
    scheduler = build_scheduler(optimizer, cfg)

    # Gradient clipping
    grad_clip = cfg["training"].get("grad_clip", None)
    if grad_clip is not None:
        print(f"[Training] Gradient clipping (max_norm={grad_clip})")

    # Train
    save_path = cfg["training"]["checkpoint_path"]
    # Avoid overwriting non-combined checkpoint if config reused
    if not save_path.endswith("_combined_best.pt"):
        base, ext = os.path.splitext(save_path)
        save_path = f"{base}_combined{ext}"
        print(f"[Checkpoint] Saving combined model to: {save_path}")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    history = train_model(
        model=model,
        train_loader=train_loader,
        valid_loader=val_loader,
        optimizer=optimizer,
        device=device,
        criterion=criterion,
        num_epochs=cfg["training"]["epochs"],
        patience=cfg["training"]["patience"],
        save_path=save_path,
        scheduler=scheduler,
        grad_clip=grad_clip,
        early_stopping_metric=cfg["training"].get("early_stopping_metric", "f1"),
        config=cfg,
    )

    # Evaluate on official test set
    test_metrics = evaluate(
        model, test_loader, device, threshold=0.5, criterion=None
    )
    print("\n[Official Test Set Results]")
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  F1:        {test_metrics['f1']:.4f}")
    print(f"  AUC:       {test_metrics['auc']:.4f}")

    # Save training history
    os.makedirs(args.output_dir, exist_ok=True)
    history_path = os.path.join(args.output_dir, "training_history_combined.json")
    history_record = {
        "config": args.config,
        "checkpoint": save_path,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "timestamp": datetime.now().isoformat(),
        "test_metrics": {k: float(v) for k, v in test_metrics.items()},
        "history": [
            {k: float(v) if isinstance(v, (np.floating, float)) else v for k, v in row.items()}
            for row in history
        ],
    }
    with open(history_path, "w") as f:
        json.dump(history_record, f, indent=2)
    print(f"Training history saved to: {history_path}")
    print(f"\nTraining completed!")
    print(f"Best checkpoint saved to: {save_path}")

    run.finalize(checkpoint_path=save_path, history=history, metrics=test_metrics)


if __name__ == "__main__":
    main()
