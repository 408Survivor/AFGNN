#!/usr/bin/env python3
"""Subgroup (gender / subject-level) analysis of clean test predictions.

Reads the per-sample clean test probabilities produced by
src/scripts/robustness_eval.py (experiments/analysis/clean_test_probs_seed{N}.json),
joins them with D-Vlog metadata — gender from labels.csv and channelId (the
video uploader's channel id, i.e. the subject) from data.csv — and reports:

    1. per-gender (m / f) test AUC and F1 for each seed, plus mean±std;
    2. subject-level metrics: the predicted probabilities of all test videos
       from the same channelId are averaged into one subject score (subject
       label = video label; verified consistent within subject on this split),
       then subject-level AUC/F1 per seed and mean±std.

F1 uses each seed's own validation-tuned threshold (stored in the probs file).

Usage:
    conda activate DVlog
    cd /home/ltq/Code/AFGNN
    python src/scripts/subgroup_analysis.py
    python src/scripts/subgroup_analysis.py --seeds 42   # smoke run

Results are saved to experiments/analysis/subgroup_analysis.json.
"""

import argparse
import json
import math
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from utils.metrics import compute_metrics  # noqa: E402


def load_metadata(labels_csv, data_csv):
    """Join labels.csv with data.csv's channelId (positional row alignment).

    data.csv has a malformed header (two unnamed columns), so it is read
    positionally: column 0 = video_id, 5 = gender, 6 = duration, 7 = channelId.
    Row order of data.csv matches labels.csv one-to-one; this is asserted via
    the duration and gender columns.
    """
    labels_df = pd.read_csv(labels_csv)
    data_df = pd.read_csv(data_csv, header=None, skiprows=1)
    assert len(labels_df) == len(data_df), (
        f"row count mismatch: labels.csv {len(labels_df)} vs data.csv {len(data_df)}"
    )
    assert np.allclose(labels_df["duration"].values, data_df[6].values), (
        "duration columns disagree; data.csv row order does not match labels.csv"
    )
    assert (labels_df["gender"].values == data_df[5].values).all(), (
        "gender columns disagree; data.csv row order does not match labels.csv"
    )
    labels_df = labels_df.copy()
    labels_df["channelId"] = data_df[7].values
    return labels_df


def safe_float(x):
    """Convert to a JSON-safe float (NaN/inf -> None)."""
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def metric_block(labels, probs, threshold):
    m = compute_metrics(np.asarray(labels), np.asarray(probs), threshold=threshold)
    return {
        "n": int(len(labels)),
        "n_pos": int(np.sum(np.asarray(labels) == 1)),
        "n_neg": int(np.sum(np.asarray(labels) == 0)),
        "accuracy": safe_float(m["accuracy"]),
        "precision": safe_float(m["precision"]),
        "recall": safe_float(m["recall"]),
        "f1": safe_float(m["f1"]),
        "auc": safe_float(m["auc"]),
    }


def mean_std(vals):
    vals = np.asarray(vals, dtype=np.float64)
    std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return float(np.mean(vals)), std


def main():
    parser = argparse.ArgumentParser(
        description="Gender / subject-level subgroup analysis of clean test probs"
    )
    parser.add_argument(
        "--probs_dir",
        type=str,
        default="experiments/analysis",
        help="Directory containing clean_test_probs_seed{N}.json",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44, 45, 46],
        help="Seeds to include (a clean_test_probs_seed{N}.json must exist)",
    )
    parser.add_argument(
        "--labels_csv",
        type=str,
        default="/data/ltq/D-Vlog_Raw/labels.csv",
    )
    parser.add_argument(
        "--data_csv",
        type=str,
        default="/data/ltq/D-Vlog_Raw/data.csv",
        help="D-Vlog data.csv; column 8 (channelId) is the subject id",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments/analysis",
        help="Directory to save subgroup_analysis.json",
    )
    args = parser.parse_args()

    meta = load_metadata(args.labels_csv, args.data_csv)
    meta_by_index = meta.set_index("index")

    # per_seed[seed] = {"overall":..., "gender": {"m":..., "f":...}, "subject":...}
    per_seed = {}
    for seed in args.seeds:
        probs_path = os.path.join(args.probs_dir, f"clean_test_probs_seed{seed}.json")
        if not os.path.exists(probs_path):
            print(f"[WARNING] missing {probs_path}; seed {seed} skipped")
            continue
        with open(probs_path, "r") as f:
            record = json.load(f)
        threshold = float(record.get("threshold", 0.5))
        samples = record["samples"]

        idx = np.array([s["index"] for s in samples])
        probs = np.array([s["prob"] for s in samples], dtype=np.float64)
        labels = np.array([s["label"] for s in samples], dtype=np.int64)

        joined = meta_by_index.loc[idx]
        # Sanity: metadata label (depression/normal) must match the probs label.
        meta_label = (joined["label"].values == "depression").astype(int)
        assert (meta_label == labels).all(), (
            f"seed {seed}: label mismatch after joining metadata by index"
        )
        genders = joined["gender"].values
        channels = joined["channelId"].values

        entry = {
            "threshold": threshold,
            "overall": metric_block(labels, probs, threshold),
            "gender": {},
        }
        for g in ["f", "m"]:
            sel = genders == g
            entry["gender"][g] = metric_block(labels[sel], probs[sel], threshold)

        # Subject-level aggregation: mean prob over all test videos of a subject.
        subj_probs = {}
        subj_labels = {}
        mixed = 0
        for ch, p, y in zip(channels, probs, labels):
            subj_probs.setdefault(ch, []).append(p)
            if ch in subj_labels and subj_labels[ch] != y:
                mixed += 1
            subj_labels.setdefault(ch, y)
        s_labels = np.array([subj_labels[ch] for ch in subj_probs], dtype=np.int64)
        s_probs = np.array(
            [np.mean(subj_probs[ch]) for ch in subj_probs], dtype=np.float64
        )
        entry["subject"] = metric_block(s_labels, s_probs, threshold)
        entry["subject"]["n_mixed_label_subjects"] = mixed
        per_seed[seed] = entry
        print(
            f"seed {seed}: overall F1={entry['overall']['f1']:.4f} "
            f"AUC={entry['overall']['auc']:.4f} | "
            f"F F1={entry['gender']['f']['f1']:.4f} AUC={entry['gender']['f']['auc']:.4f} | "
            f"M F1={entry['gender']['m']['f1']:.4f} AUC={entry['gender']['m']['auc']:.4f} | "
            f"subject(n={entry['subject']['n']}) F1={entry['subject']['f1']:.4f} "
            f"AUC={entry['subject']['auc']:.4f}"
        )

    seeds_used = sorted(per_seed)
    if not seeds_used:
        raise SystemExit("No clean_test_probs_seed*.json found for the requested seeds.")

    # Aggregate across seeds.
    def aggregate(getter):
        out = {}
        for metric in ["accuracy", "precision", "recall", "f1", "auc"]:
            vals = [getter(s)[metric] for s in seeds_used]
            vals = [v for v in vals if v is not None]
            if vals:
                mean, std = mean_std(vals)
                out[metric] = {"mean": mean, "std": std, "per_seed": {
                    str(s): getter(s)[metric] for s in seeds_used}}
            else:
                out[metric] = None
        return out

    summary = {
        "overall": aggregate(lambda s: per_seed[s]["overall"]),
        "gender_f": aggregate(lambda s: per_seed[s]["gender"]["f"]),
        "gender_m": aggregate(lambda s: per_seed[s]["gender"]["m"]),
        "subject": aggregate(lambda s: per_seed[s]["subject"]),
    }

    # Terminal summary table.
    def ms(block, metric):
        agg = summary[block][metric]
        return f"{agg['mean']:.4f}±{agg['std']:.4f}" if agg else "n/a"

    print(f"\n=== Subgroup summary over seeds {seeds_used} (mean ± std) ===")
    header = f"{'subgroup':<22}{'n':>6}{'F1':>16}{'AUC':>16}{'Acc':>16}"
    print(header)
    print("-" * len(header))
    rows = [
        ("overall (video)", per_seed[seeds_used[0]]["overall"]["n"], "overall"),
        ("gender: female", per_seed[seeds_used[0]]["gender"]["f"]["n"], "gender_f"),
        ("gender: male", per_seed[seeds_used[0]]["gender"]["m"]["n"], "gender_m"),
        ("subject-level", per_seed[seeds_used[0]]["subject"]["n"], "subject"),
    ]
    for name, n, key in rows:
        print(
            f"{name:<22}{n:>6}{ms(key, 'f1'):>16}{ms(key, 'auc'):>16}{ms(key, 'accuracy'):>16}"
        )
    print("-" * len(header))
    print(
        "subject-level: probs of all test videos sharing a channelId are "
        "averaged into one subject score."
    )

    os.makedirs(args.output_dir, exist_ok=True)
    result_path = os.path.join(args.output_dir, "subgroup_analysis.json")
    result_record = {
        "analysis": "subgroup_analysis",
        "method": (
            "Clean test-set per-sample probabilities (from robustness_eval.py) "
            "joined with D-Vlog metadata: gender from labels.csv, channelId "
            "(subject id) from data.csv column 8 (row-aligned with labels.csv, "
            "verified via duration/gender). Per-gender metrics use each seed's "
            "validation-tuned threshold. Subject-level scores are the mean prob "
            "over all test videos of one channelId; subject labels are "
            "consistent within subject on this split."
        ),
        "seeds": seeds_used,
        "labels_csv": args.labels_csv,
        "data_csv": args.data_csv,
        "timestamp": datetime.now().isoformat(),
        "per_seed": {str(s): per_seed[s] for s in seeds_used},
        "summary": summary,
    }
    result_record = json.loads(json.dumps(result_record, allow_nan=False))
    with open(result_path, "w") as f:
        json.dump(result_record, f, indent=2)
    print(f"\nResults saved to: {result_path}")


if __name__ == "__main__":
    main()
