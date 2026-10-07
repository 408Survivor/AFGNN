#!/usr/bin/env python3
"""AUC-based facial region importance analysis (inference-time masking, multi-seed).

Distinct from the F1-based region ablation (`ablation/results/test_ablation_region_mask_*`),
which *retrains* one model per masked region and compares thresholded F1 scores.
This script instead:

    1. loads the already-trained full-model checkpoints (one per random seed),
    2. masks one facial region at *inference* time by zeroing its region one-hot
       column (the existing `region_ablation` mechanism in graph_utils.build_face_graph),
    3. computes the test-set AUC (a threshold-free ranking metric) for each
       full / masked condition,
    4. takes dAUC = AUC_full - AUC_masked per seed as the region's importance,
    5. runs paired significance tests across seeds (Wilcoxon signed-rank and
       paired t-test, both reported, plus Holm correction across the 9 regions).

No model is retrained; the existing F1-based results are left untouched.

Usage:
    conda activate DVlog
    cd /home/ltq/DepressionCode/DepGNN/AFGNN
    python src/scripts/region_importance_auc.py --seeds 42 43 44 45 46 --split test
"""

import argparse
import json
import math
import os
import sys
from datetime import datetime

import numpy as np
import torch
from scipy import stats as scipy_stats
from sklearn.metrics import roc_auc_score
from torch_geometric.loader import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from utils.builders import build_loaders, build_model, load_config  # noqa: E402
from utils.trainer import get_predictions  # noqa: E402
from models.graph_utils import LANDMARK_GROUPS_68  # noqa: E402

REGION_NAMES = list(LANDMARK_GROUPS_68.keys())


def get_seed_checkpoint(base_path: str, seed: int) -> str:
    base, ext = os.path.splitext(base_path)
    return f"{base}_seed{seed}{ext}"


def eval_auc(model, dataset, split_cfg, device) -> float:
    """Evaluate AUC on a dataset with its current `region_ablation` setting.

    A fresh DataLoader with num_workers=0 is built each time so the mutated
    `dataset.region_ablation` attribute is guaranteed to be visible (worker
    processes would hold a stale copy otherwise).
    """
    loader = DataLoader(
        dataset,
        batch_size=split_cfg["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    labels, probs = get_predictions(model, loader, device)
    return float(roc_auc_score(labels, probs))


def holm_correction(pvalues):
    """Holm-Bonferroni adjusted p-values (same order as input)."""
    p = np.asarray(pvalues, dtype=np.float64)
    n = len(p)
    order = np.argsort(p, kind="stable")
    adjusted = np.empty(n, dtype=np.float64)
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = (n - rank) * p[idx]
        running_max = max(running_max, val)
        adjusted[idx] = min(running_max, 1.0)
    return adjusted


def safe_float(x):
    """Convert to a JSON-safe float (NaN/inf -> None)."""
    x = float(x)
    return x if math.isfinite(x) else None


def region_stats(auc_full, auc_masked):
    """Paired statistics for one region across seeds.

    auc_full / auc_masked: np.ndarray of shape (n_seeds,).
    """
    delta = auc_full - auc_masked
    n = len(delta)
    mean = float(np.mean(delta))
    std = float(np.std(delta, ddof=1)) if n > 1 else 0.0

    if n > 1 and std > 0:
        ci_low, ci_high = scipy_stats.t.interval(0.95, df=n - 1, loc=mean, scale=std / math.sqrt(n))
    else:
        ci_low, ci_high = mean, mean

    # Wilcoxon signed-rank on the paired differences (H0: median dAUC = 0).
    wilcoxon_note = None
    try:
        with np.errstate(all="ignore"):
            p_wilcoxon = float(scipy_stats.wilcoxon(delta).pvalue)
        if not math.isfinite(p_wilcoxon):
            raise ValueError("non-finite p-value")
    except (ValueError, ZeroDivisionError) as exc:
        p_wilcoxon = None
        wilcoxon_note = f"Wilcoxon undefined ({exc}); likely all dAUC == 0"

    # Paired t-test (H0: mean dAUC = 0).
    if n > 1 and std > 0:
        p_ttest = float(scipy_stats.ttest_rel(auc_full, auc_masked).pvalue)
    else:
        p_ttest = None

    return {
        "delta_per_seed": [safe_float(d) for d in delta],
        "mean": safe_float(mean),
        "std": safe_float(std),
        "ci95": [safe_float(ci_low), safe_float(ci_high)],
        "p_wilcoxon": safe_float(p_wilcoxon) if p_wilcoxon is not None else None,
        "p_ttest": safe_float(p_ttest) if p_ttest is not None else None,
        "wilcoxon_note": wilcoxon_note,
    }


def main():
    parser = argparse.ArgumentParser(
        description="AUC-based region importance (inference-time masking, multi-seed)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/configs/afgnn_face_enhanced_focal.yaml",
        help="Path to config file (fallback when checkpoints carry no embedded config)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44, 45, 46],
        help="Random seeds of the pre-trained full-model checkpoints",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "valid", "test"],
        help="Split to evaluate",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="ablation/results",
        help="Directory to save the AUC region-importance results",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Regions ({len(REGION_NAMES)}): {REGION_NAMES}\n")

    base_checkpoint = cfg["training"]["checkpoint_path"]
    n_regions = len(REGION_NAMES)

    auc_full = np.zeros(len(args.seeds), dtype=np.float64)
    auc_masked = np.zeros((len(args.seeds), n_regions), dtype=np.float64)
    checkpoints_used = []

    for si, seed in enumerate(args.seeds):
        checkpoint_path = get_seed_checkpoint(base_checkpoint, seed)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        checkpoints_used.append(checkpoint_path)

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        run_cfg = checkpoint.get("config") or cfg
        if checkpoint.get("config") is not None:
            print(f"[Config] Using config embedded in checkpoint: {checkpoint_path}")
        else:
            print(f"[Config] No embedded config; falling back to --config ({args.config})")

        model = build_model(run_cfg, device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        loaders = build_loaders(run_cfg, augment=False)
        dataset = loaders[args.split].dataset
        split_cfg = {"batch_size": run_cfg["training"]["batch_size"]}

        # Full model (no masking).
        dataset.region_ablation = None
        auc_full[si] = eval_auc(model, dataset, split_cfg, device)
        print(f"seed {seed}: AUC_full = {auc_full[si]:.4f}")

        # Leave-one-region-out masking at inference time.
        for region_id, region_name in enumerate(REGION_NAMES):
            dataset.region_ablation = region_id
            auc_masked[si, region_id] = eval_auc(model, dataset, split_cfg, device)
            delta = auc_full[si] - auc_masked[si, region_id]
            print(
                f"  mask {region_id} {region_name:<14s}: "
                f"AUC = {auc_masked[si, region_id]:.4f}  dAUC = {delta:+.4f}"
            )

        del model
        torch.cuda.empty_cache()

    # Paired statistics per region across seeds.
    regions = {}
    for region_id, region_name in enumerate(REGION_NAMES):
        regions[region_name] = region_stats(auc_full, auc_masked[:, region_id])

    # Holm correction across regions, per test family (NaN/None treated as p=1).
    for key, adj_key in [("p_wilcoxon", "p_wilcoxon_holm"), ("p_ttest", "p_ttest_holm")]:
        raw = [regions[r][key] if regions[r][key] is not None else 1.0 for r in REGION_NAMES]
        adjusted = holm_correction(raw)
        for region_name, adj in zip(REGION_NAMES, adjusted):
            regions[region_name][adj_key] = safe_float(adj)

    # Ranked summary table.
    ranked = sorted(REGION_NAMES, key=lambda r: regions[r]["mean"], reverse=True)
    print(f"\n--- Region importance by dAUC over seeds={args.seeds} (split={args.split}) ---")
    header = f"{'rank':<5}{'region':<16}{'dAUC mean':>10}{'std':>8}{'95% CI':>18}{'p_wilc':>9}{'p_ttest':>9}{'p_holm':>9}"
    print(header)
    for rank, region_name in enumerate(ranked, start=1):
        s = regions[region_name]
        ci = f"[{s['ci95'][0]:+.4f},{s['ci95'][1]:+.4f}]"
        fmt = lambda p: f"{p:.4f}" if p is not None else "n/a"
        print(
            f"{rank:<5}{region_name:<16}{s['mean']:>+10.4f}{s['std']:>8.4f}{ci:>18}"
            f"{fmt(s['p_wilcoxon']):>9}{fmt(s['p_ttest']):>9}{fmt(s['p_ttest_holm']):>9}"
        )

    os.makedirs(args.output_dir, exist_ok=True)
    seed_tag = "-".join(str(s) for s in args.seeds)
    result_path = os.path.join(
        args.output_dir, f"region_importance_auc_seed{seed_tag}_{args.split}.json"
    )
    result_record = {
        "analysis": "region_importance_auc",
        "masking": "inference_time_onehot_zero",
        "note": (
            "AUC-based region importance, separate from the F1-based retraining "
            "ablation (test_ablation_region_mask_*). Full-model checkpoints are "
            "loaded per seed; each region's one-hot column is zeroed at inference "
            "time; dAUC = AUC_full - AUC_masked is compared across seeds with "
            "paired Wilcoxon signed-rank and paired t-test (Holm-corrected)."
        ),
        "config": args.config,
        "seeds": args.seeds,
        "split": args.split,
        "checkpoints": checkpoints_used,
        "timestamp": datetime.now().isoformat(),
        "per_seed": [
            {
                "seed": seed,
                "auc_full": safe_float(auc_full[si]),
                "auc_masked": {
                    region_name: safe_float(auc_masked[si, ri])
                    for ri, region_name in enumerate(REGION_NAMES)
                },
            }
            for si, seed in enumerate(args.seeds)
        ],
        "regions": regions,
        "ranking": ranked,
    }
    # Ensure plain-JSON output (no NaN literals).
    result_record = json.loads(json.dumps(result_record, allow_nan=False))
    with open(result_path, "w") as f:
        json.dump(result_record, f, indent=2)
    print(f"\nResults saved to: {result_path}")


if __name__ == "__main__":
    main()
