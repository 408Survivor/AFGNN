#!/usr/bin/env python3
"""Input-perturbation robustness evaluation for the full AFGNN model family.

Evaluates the pre-trained full-model checkpoints (one per random seed, default
exp_7/11/12/13/14 = afgnn_face_enhanced_focal_adamw_cosine, seeds 42~46) on the
test split under the following conditions:

    clean              : unmodified inputs.
    face_drop_<p>      : a fraction p in {0.1, 0.25, 0.5, 0.75, 1.0} of the
                         sampled frames of every sample is chosen uniformly at
                         random and the facial landmark features of those frames
                         are zeroed, simulating face-detection failure. Only the
                         coordinate/velocity feature block (the first 4 node
                         feature dims, or 2 if use_velocity=false) is zeroed;
                         the region one-hot suffix is kept, matching the natural
                         form of undetected frames in the D-Vlog data (zero
                         landmark rows). The frame mask is drawn from an RNG
                         seeded by (condition, sample position), so it is
                         reproducible and identical across model seeds and
                         batch sizes.
    audio_noise_<alpha>: zero-mean Gaussian noise with std = alpha * (global std
                         of the sample's audio features) is added to the audio
                         node features, alpha in {0.1, 0.25, 0.5, 1.0}. Same
                         per-(condition, sample) seeding scheme.

Perturbations are applied at inference time on the batched graph tensors
(Batch.x / Batch.audio_x); no model is retrained. Metrics use each checkpoint's
embedded validation-tuned best_threshold, exactly as src/test.py does.

Under `clean`, per-sample (index, prob, label) records are additionally saved
to experiments/analysis/clean_test_probs_seed{N}.json for the subgroup
analysis (src/scripts/subgroup_analysis.py). Sample indices are D-Vlog sample
ids: the processed test split is ordered by ascending `index` of labels.csv
(see src/scripts/build_processed_features.py) and the test loader is not
shuffled, so position i in the evaluation loop corresponds to the i-th test
row of labels.csv sorted by index.

Usage:
    conda activate DVlog
    cd /home/ltq/Code/AFGNN
    python src/scripts/robustness_eval.py                      # full run
    python src/scripts/robustness_eval.py --seeds 42 \
        --conditions clean,face_drop_0.5                       # smoke run

Results are saved to experiments/analysis/robustness.json.
"""

import argparse
import glob
import json
import math
import os
import re
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from utils.builders import build_loaders, build_model  # noqa: E402
from utils.metrics import compute_metrics  # noqa: E402

NUM_LANDMARKS = 68  # facial landmarks per frame (see models.graph_utils)

# Fixed RNG bases for perturbation masks. Per-sample masks/noise are drawn from
# np.random.default_rng(base + condition_code * 1_000_003 + sample_position),
# making them reproducible and independent of model seed and batch composition.
FACE_DROP_SEED_BASE = 20_260_000
AUDIO_NOISE_SEED_BASE = 20_270_000

DEFAULT_EXP_DIRS = [
    "experiments/exp_7",   # seed 42
    "experiments/exp_11",  # seed 43
    "experiments/exp_12",  # seed 44
    "experiments/exp_13",  # seed 45
    "experiments/exp_14",  # seed 46
]

FACE_DROP_PS = [0.1, 0.25, 0.5, 0.75, 1.0]
AUDIO_NOISE_ALPHAS = [0.1, 0.25, 0.5, 1.0]
ALL_CONDITIONS = (
    ["clean"]
    + [f"face_drop_{p}" for p in FACE_DROP_PS]
    + [f"audio_noise_{a}" for a in AUDIO_NOISE_ALPHAS]
)


def parse_condition(condition):
    """Parse a condition name -> ("clean"|"face_drop"|"audio_noise", value)."""
    if condition == "clean":
        return "clean", None
    m = re.fullmatch(r"face_drop_(\d+(?:\.\d+)?)", condition)
    if m:
        return "face_drop", float(m.group(1))
    m = re.fullmatch(r"audio_noise_(\d+(?:\.\d+)?)", condition)
    if m:
        return "audio_noise", float(m.group(1))
    raise ValueError(f"Unknown condition: {condition}")


def get_test_sample_ids(labels_csv, num_samples):
    """D-Vlog sample ids of the test split, in evaluation (dataset) order.

    The processed split arrays are ordered by ascending `index` within each
    fold (build_processed_features.py) and the test loader is not shuffled.
    """
    labels_df = pd.read_csv(labels_csv)
    test_ids = (
        labels_df[labels_df["fold"] == "test"].sort_values("index")["index"].tolist()
    )
    assert len(test_ids) == num_samples, (
        f"labels.csv test fold has {len(test_ids)} rows but the dataset has "
        f"{num_samples} samples"
    )
    return test_ids


def apply_face_drop(data, p, sample_positions, num_frames, coord_feat_dim):
    """Zero the coordinate/velocity features of a fraction p of frames.

    Operates in place on the batched face node features `data.x`
    (shape: (num_graphs * num_frames * 68, F)).
    """
    k = int(round(p * num_frames))
    if k <= 0:
        return
    batch_vec = data.batch
    for g, pos in enumerate(sample_positions):
        rng = np.random.default_rng(
            FACE_DROP_SEED_BASE + int(round(p * 1000)) * 1_000_003 + pos
        )
        dropped = rng.choice(num_frames, size=k, replace=False)
        node_rows = (batch_vec == g).nonzero(as_tuple=True)[0]
        # Nodes of one graph are contiguous and ordered frame-major:
        # local node index = frame * 68 + landmark.
        local = torch.arange(node_rows.numel(), device=node_rows.device)
        frame_of_node = local // NUM_LANDMARKS
        mask = torch.isin(
            frame_of_node, torch.as_tensor(dropped, device=node_rows.device)
        )
        data.x[node_rows[mask], :coord_feat_dim] = 0.0


def apply_audio_noise(data, alpha, sample_positions, audio_num_frames):
    """Add Gaussian noise with std = alpha * per-sample global audio std.

    Operates in place on `data.audio_x` (shape: (num_graphs * T_a, D); rows of
    one graph are contiguous because every graph has the same number of audio
    frames by dataset construction).
    """
    if alpha <= 0:
        return
    for g, pos in enumerate(sample_positions):
        rows = slice(g * audio_num_frames, (g + 1) * audio_num_frames)
        feats = data.audio_x[rows]
        std = float(feats.std())
        if std < 1e-12:
            continue
        rng = np.random.default_rng(
            AUDIO_NOISE_SEED_BASE + int(round(alpha * 1000)) * 1_000_003 + pos
        )
        noise = torch.from_numpy(
            rng.normal(0.0, alpha * std, size=tuple(feats.shape)).astype(np.float32)
        ).to(feats.device)
        data.audio_x[rows] = feats + noise


@torch.no_grad()
def evaluate_condition(
    model, loader, device, threshold, condition, num_frames, audio_num_frames,
    coord_feat_dim,
):
    """Run inference under one perturbation condition; return metrics + records."""
    kind, value = parse_condition(condition)
    model.eval()

    all_labels = []
    all_probs = []
    offset = 0  # position of the first sample of the current batch in the split
    for data in loader:
        data = data.to(device)
        num_graphs = data.num_graphs
        sample_positions = list(range(offset, offset + num_graphs))

        if kind == "face_drop":
            apply_face_drop(
                data, value, sample_positions, num_frames, coord_feat_dim
            )
        elif kind == "audio_noise":
            apply_audio_noise(
                data, value, sample_positions, audio_num_frames
            )

        logit = model(data)
        probs = torch.sigmoid(logit).squeeze(-1).cpu().numpy()
        labels = data.y.cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels)
        offset += num_graphs

    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)
    metrics = compute_metrics(all_labels, all_probs, threshold=threshold)
    return metrics, all_labels, all_probs


def safe_float(x):
    """Convert to a JSON-safe float (NaN/inf -> None)."""
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def main():
    parser = argparse.ArgumentParser(
        description="Input-perturbation robustness evaluation for full AFGNN"
    )
    parser.add_argument(
        "--exp_dirs",
        type=str,
        nargs="+",
        default=DEFAULT_EXP_DIRS,
        help="Experiment dirs of the full-model family (one best_seed*.pt each)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44, 45, 46],
        help="Only evaluate checkpoints whose seed is in this list",
    )
    parser.add_argument(
        "--conditions",
        type=str,
        default="all",
        help=(
            "Comma-separated condition list, or 'all'. Available: "
            + ",".join(ALL_CONDITIONS)
        ),
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "valid", "test"],
        help="Split to evaluate (perturbation protocol is defined for test)",
    )
    parser.add_argument(
        "--labels_csv",
        type=str,
        default="/data/ltq/D-Vlog_Raw/labels.csv",
        help="D-Vlog labels.csv (for mapping split positions to sample ids)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override eval batch size (defaults to the config's training.batch_size)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments/analysis",
        help="Directory to save robustness.json and clean per-sample probs",
    )
    args = parser.parse_args()

    if args.conditions.strip().lower() == "all":
        conditions = list(ALL_CONDITIONS)
    else:
        conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    for c in conditions:
        parse_condition(c)  # fail fast on typos

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Conditions ({len(conditions)}): {conditions}")

    os.makedirs(args.output_dir, exist_ok=True)

    # per_seed[seed][condition] = metrics dict
    per_seed = {}
    checkpoints_used = {}

    for exp_dir in args.exp_dirs:
        matches = sorted(glob.glob(os.path.join(exp_dir, "best_seed*.pt")))
        if not matches:
            raise FileNotFoundError(f"No best_seed*.pt checkpoint in {exp_dir}")
        checkpoint_path = matches[0]
        m = re.search(r"best_seed(\d+)\.pt$", checkpoint_path)
        seed = int(m.group(1))
        if seed not in args.seeds:
            continue
        checkpoints_used[seed] = checkpoint_path

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        run_cfg = checkpoint.get("config")
        if run_cfg is None:
            raise ValueError(
                f"Checkpoint has no embedded config: {checkpoint_path}"
            )
        best_threshold = float(checkpoint.get("best_threshold", 0.5))

        num_frames = int(run_cfg["data"]["num_frames"])
        audio_num_frames = int(
            run_cfg["data"].get("audio_num_frames", num_frames)
        )
        coord_feat_dim = 4 if run_cfg["model"].get("use_velocity", True) else 2

        model = build_model(run_cfg, device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        loaders = build_loaders(run_cfg, augment=False, batch_size=args.batch_size)
        loader = loaders[args.split]

        sample_ids = get_test_sample_ids(args.labels_csv, len(loader.dataset))

        print(
            f"\n=== seed {seed} ({exp_dir}, threshold={best_threshold:.4f}, "
            f"num_frames={num_frames}, audio_frames={audio_num_frames}) ==="
        )
        per_seed[seed] = {}
        for condition in conditions:
            metrics, labels, probs = evaluate_condition(
                model, loader, device, best_threshold, condition,
                num_frames, audio_num_frames, coord_feat_dim,
            )
            assert len(labels) == len(sample_ids)
            per_seed[seed][condition] = {k: float(v) for k, v in metrics.items()}
            print(
                f"  {condition:<18} acc={metrics['accuracy']:.4f} "
                f"prec={metrics['precision']:.4f} rec={metrics['recall']:.4f} "
                f"f1={metrics['f1']:.4f} auc={metrics['auc']:.4f}"
            )

            if condition == "clean":
                probs_path = os.path.join(
                    args.output_dir, f"clean_test_probs_seed{seed}.json"
                )
                record = {
                    "seed": seed,
                    "exp_dir": exp_dir,
                    "checkpoint": checkpoint_path,
                    "split": args.split,
                    "threshold": best_threshold,
                    "num_samples": len(sample_ids),
                    "samples": [
                        {
                            "index": int(sample_ids[i]),
                            "prob": float(probs[i]),
                            "label": int(labels[i]),
                        }
                        for i in range(len(sample_ids))
                    ],
                }
                with open(probs_path, "w") as f:
                    json.dump(record, f, indent=2)
                print(f"  -> clean per-sample probs saved to: {probs_path}")

        del model
        torch.cuda.empty_cache()

    seeds_used = sorted(per_seed)
    if not seeds_used:
        raise SystemExit("No checkpoints matched the requested seeds.")

    # Aggregate across seeds.
    summary = {}
    for condition in conditions:
        agg = {}
        for metric in ["accuracy", "precision", "recall", "f1", "auc"]:
            vals = np.array(
                [per_seed[s][condition][metric] for s in seeds_used],
                dtype=np.float64,
            )
            agg[metric] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "per_seed": {str(s): float(v) for s, v in zip(seeds_used, vals)},
            }
        summary[condition] = agg

    # Terminal summary table.
    print(f"\n=== Robustness summary over seeds {seeds_used} (mean ± std) ===")
    header = f"{'condition':<18}{'Acc':>16}{'Prec':>16}{'Rec':>16}{'F1':>16}{'AUC':>16}"
    print(header)
    print("-" * len(header))
    for condition in conditions:
        agg = summary[condition]
        cells = []
        for metric in ["accuracy", "precision", "recall", "f1", "auc"]:
            cells.append(f"{agg[metric]['mean']:.4f}±{agg[metric]['std']:.4f}")
        print(f"{condition:<18}" + "".join(f"{c:>16}" for c in cells))
    print("-" * len(header))
    print(
        "face_drop_p: fraction p of frames with facial coordinate/velocity "
        "features zeroed (face-detection failure).\n"
        "audio_noise_a: Gaussian noise with std = a x per-sample global audio std."
    )

    result_path = os.path.join(args.output_dir, "robustness.json")
    result_record = {
        "analysis": "robustness_eval",
        "method": (
            "Inference-time input perturbation of the full-model checkpoints. "
            "face_drop_<p>: a seeded-random fraction p of the sampled frames of "
            "each sample has its facial coordinate/velocity node features "
            "zeroed (region one-hot kept), simulating face-detection failure; "
            "audio_noise_<a>: Gaussian noise with std = a * per-sample global "
            "audio-feature std added to audio node features. Masks/noise are "
            "seeded by (condition, sample position) and are identical across "
            "model seeds. Metrics use each checkpoint's validation-tuned "
            "best_threshold, as in src/test.py."
        ),
        "exp_dirs": args.exp_dirs,
        "seeds": seeds_used,
        "checkpoints": {str(s): checkpoints_used[s] for s in seeds_used},
        "conditions": conditions,
        "split": args.split,
        "labels_csv": args.labels_csv,
        "timestamp": datetime.now().isoformat(),
        "per_seed": {
            str(s): {
                c: {k: safe_float(v) for k, v in per_seed[s][c].items()}
                for c in conditions
            }
            for s in seeds_used
        },
        "summary": json.loads(json.dumps(summary, allow_nan=False)),
    }
    result_record = json.loads(json.dumps(result_record, allow_nan=False))
    with open(result_path, "w") as f:
        json.dump(result_record, f, indent=2)
    print(f"\nResults saved to: {result_path}")


if __name__ == "__main__":
    main()
