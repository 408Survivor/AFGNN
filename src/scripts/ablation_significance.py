#!/usr/bin/env python3
"""Paired significance tests for ablation configs vs the full AFGNN model.

Scans experiments/exp_*/run_info.txt, groups runs by config basename, collects
per-seed test metrics from each exp dir's test_test_results.json, and compares
every config group against the full model family
(`afgnn_face_enhanced_focal_adamw_cosine`, exp_7/11/12/13/14) on the seeds they
have in common, using the paired Wilcoxon signed-rank test (F1 and AUC
separately).

Note: with n=5 paired seeds the smallest achievable two-sided Wilcoxon p-value
is 0.0625, so p-values here are indicative rather than decisive; mean/std and
the signed delta are the primary evidence.

Usage:
    conda activate DVlog
    cd /home/ltq/Code/AFGNN
    python src/scripts/ablation_significance.py

Results are saved to experiments/analysis/ablation_significance.json.
"""

import argparse
import glob
import json
import math
import os
import re
from datetime import datetime

import numpy as np

try:
    from scipy import stats as scipy_stats
except ImportError:  # pragma: no cover
    scipy_stats = None
    print(
        "[WARNING] scipy is not installed in this environment; "
        "p-values will be reported as null. Install with: pip install scipy"
    )

BASELINE_CONFIG = "afgnn_face_enhanced_focal_adamw_cosine"


def parse_run_info(run_info_path):
    """Extract (config_basename, seed) from an exp dir's run_info.txt."""
    config_name = None
    seed = None
    with open(run_info_path, "r") as f:
        for line in f:
            if line.startswith("Config file:"):
                config_name = os.path.basename(line.split(":", 1)[1].strip())
                config_name = re.sub(r"\.ya?ml$", "", config_name)
            elif line.startswith("Command:"):
                m = re.search(r"--seed\s+(\d+)", line)
                if m:
                    seed = int(m.group(1))
    return config_name, seed


def collect_groups(exp_root):
    """Scan exp_* dirs -> {config_name: {seed: {metrics...}}}."""
    groups = {}
    for exp_dir in sorted(glob.glob(os.path.join(exp_root, "exp_*"))):
        run_info = os.path.join(exp_dir, "run_info.txt")
        results_path = os.path.join(exp_dir, "test_test_results.json")
        if not (os.path.exists(run_info) and os.path.exists(results_path)):
            continue
        config_name, seed = parse_run_info(run_info)
        if config_name is None or seed is None:
            continue
        with open(results_path, "r") as f:
            metrics = json.load(f)["metrics"]
        entry = {
            "exp": os.path.basename(exp_dir),
            "f1": float(metrics["f1"]),
            "auc": float(metrics["auc"]),
            "accuracy": float(metrics["accuracy"]),
            "precision": float(metrics["precision"]),
            "recall": float(metrics["recall"]),
        }
        groups.setdefault(config_name, {})[seed] = entry
    return groups


def safe_float(x):
    """Convert to a JSON-safe float (NaN/inf -> None)."""
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def paired_test(full_vals, other_vals):
    """Wilcoxon signed-rank on paired differences (other - full)."""
    diff = np.asarray(other_vals) - np.asarray(full_vals)
    if scipy_stats is None:
        return None, "scipy not installed"
    if len(diff) < 2:
        return None, "n<2 paired seeds; test skipped"
    if np.all(diff == 0):
        return None, "all paired differences are zero; Wilcoxon undefined"
    try:
        with np.errstate(all="ignore"):
            p = float(scipy_stats.wilcoxon(diff).pvalue)
        if not math.isfinite(p):
            raise ValueError("non-finite p-value")
        return p, None
    except ValueError as exc:
        return None, f"Wilcoxon undefined ({exc})"


def mean_std(vals):
    vals = np.asarray(vals, dtype=np.float64)
    std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return float(np.mean(vals)), std


def compare_group(full_group, other_group):
    """Compare one config group against the full model on common seeds."""
    common_seeds = sorted(set(full_group) & set(other_group))
    if not common_seeds:
        return None
    full_f1 = [full_group[s]["f1"] for s in common_seeds]
    other_f1 = [other_group[s]["f1"] for s in common_seeds]
    full_auc = [full_group[s]["auc"] for s in common_seeds]
    other_auc = [other_group[s]["auc"] for s in common_seeds]

    p_f1, note_f1 = paired_test(full_f1, other_f1)
    p_auc, note_auc = paired_test(full_auc, other_auc)

    f1_full_mean, f1_full_std = mean_std(full_f1)
    f1_other_mean, f1_other_std = mean_std(other_f1)
    auc_full_mean, auc_full_std = mean_std(full_auc)
    auc_other_mean, auc_other_std = mean_std(other_auc)

    return {
        "seeds": common_seeds,
        "n": len(common_seeds),
        "n_note": "n<5: limited statistical power" if len(common_seeds) < 5 else None,
        "exps": [other_group[s]["exp"] for s in common_seeds],
        "f1": {
            "full_mean": f1_full_mean,
            "full_std": f1_full_std,
            "other_mean": f1_other_mean,
            "other_std": f1_other_std,
            "delta": f1_other_mean - f1_full_mean,
            "p_wilcoxon": p_f1,
            "note": note_f1,
        },
        "auc": {
            "full_mean": auc_full_mean,
            "full_std": auc_full_std,
            "other_mean": auc_other_mean,
            "other_std": auc_other_std,
            "delta": auc_other_mean - auc_full_mean,
            "p_wilcoxon": p_auc,
            "note": note_auc,
        },
    }


def fmt_ms(mean, std):
    return f"{mean:.4f}±{std:.4f}"


def fmt_p(p):
    return f"{p:.4f}" if p is not None else "n/a"


def main():
    parser = argparse.ArgumentParser(
        description="Paired Wilcoxon significance of ablation configs vs full AFGNN"
    )
    parser.add_argument(
        "--exp_root",
        type=str,
        default="experiments",
        help="Root directory containing exp_* experiment folders",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default=BASELINE_CONFIG,
        help="Config basename (without .yaml) of the full-model reference family",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments/analysis",
        help="Directory to save the significance results",
    )
    args = parser.parse_args()

    groups = collect_groups(args.exp_root)
    if args.baseline not in groups:
        raise SystemExit(
            f"Baseline config group not found: {args.baseline} "
            f"(available: {sorted(groups)})"
        )

    full_group = groups[args.baseline]
    print(f"Baseline (full model): {args.baseline}")
    print(f"  seeds: {sorted(full_group)}")
    f1_mean, f1_std = mean_std([v["f1"] for v in full_group.values()])
    auc_mean, auc_std = mean_std([v["auc"] for v in full_group.values()])
    print(f"  test F1  = {f1_mean:.4f} ± {f1_std:.4f}")
    print(f"  test AUC = {auc_mean:.4f} ± {auc_std:.4f}\n")

    comparisons = {}
    for name in sorted(groups):
        if name == args.baseline:
            continue
        comp = compare_group(full_group, groups[name])
        if comp is not None:
            comparisons[name] = comp

    # Ranked table: largest AUC drop first.
    ranked = sorted(comparisons, key=lambda n: comparisons[n]["auc"]["delta"])

    header = (
        f"{'config':<44}{'n':>3}"
        f"{'F1 full':>15}{'F1 abl':>15}{'dF1':>9}{'p_f1':>8}"
        f"{'AUC full':>15}{'AUC abl':>15}{'dAUC':>9}{'p_auc':>8}"
    )
    print(header)
    print("-" * len(header))
    print(
        f"{args.baseline + ' [FULL]':<44}{len(full_group):>3}"
        f"{fmt_ms(f1_mean, f1_std):>15}{'-':>15}{'-':>9}{'-':>8}"
        f"{fmt_ms(auc_mean, auc_std):>15}{'-':>15}{'-':>9}{'-':>8}"
    )
    for name in ranked:
        c = comparisons[name]
        f1, auc = c["f1"], c["auc"]
        mark = "*" if c["n"] < 5 else " "
        print(
            f"{name:<44}{str(c['n']) + mark:>3}"
            f"{fmt_ms(f1['full_mean'], f1['full_std']):>15}"
            f"{fmt_ms(f1['other_mean'], f1['other_std']):>15}"
            f"{f1['delta']:>+9.4f}{fmt_p(f1['p_wilcoxon']):>8}"
            f"{fmt_ms(auc['full_mean'], auc['full_std']):>15}"
            f"{fmt_ms(auc['other_mean'], auc['other_std']):>15}"
            f"{auc['delta']:>+9.4f}{fmt_p(auc['p_wilcoxon']):>8}"
        )
    print("-" * len(header))
    print(
        "dF1/dAUC = ablation - full (negative = ablation is worse). "
        "* = n<5 paired seeds.\n"
        "With n=5 paired seeds the minimum two-sided Wilcoxon p is 0.0625."
    )

    os.makedirs(args.output_dir, exist_ok=True)
    result_path = os.path.join(args.output_dir, "ablation_significance.json")
    result_record = {
        "analysis": "ablation_significance",
        "method": (
            "Paired Wilcoxon signed-rank test (two-sided) on per-seed test F1 and "
            "AUC, ablation config vs the full model family, over the seeds the two "
            "groups have in common. Metrics are read from each exp dir's "
            "test_test_results.json; runs are grouped by the config basename in "
            "run_info.txt."
        ),
        "baseline_config": args.baseline,
        "baseline_seeds": sorted(full_group),
        "baseline_f1_mean": safe_float(f1_mean),
        "baseline_f1_std": safe_float(f1_std),
        "baseline_auc_mean": safe_float(auc_mean),
        "baseline_auc_std": safe_float(auc_std),
        "timestamp": datetime.now().isoformat(),
        "groups": {
            name: {
                seed: {k: safe_float(v) if isinstance(v, float) else v
                       for k, v in entry.items()}
                for seed, entry in group.items()
            }
            for name, group in sorted(groups.items())
        },
        "comparisons": json.loads(json.dumps(comparisons, default=safe_float)),
    }
    result_record = json.loads(json.dumps(result_record, allow_nan=False, default=safe_float))
    with open(result_path, "w") as f:
        json.dump(result_record, f, indent=2)
    print(f"Results saved to: {result_path}")


if __name__ == "__main__":
    main()
