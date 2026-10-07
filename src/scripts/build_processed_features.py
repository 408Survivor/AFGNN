#!/usr/bin/env python3
"""Build split-level processed feature arrays from the raw D-Vlog dataset.

Source: /data/ltq/D-Vlog_Raw/
  - dvlog-dataset/<i>/<i>_visual.npy    (T_i, 136) per-frame standardized,
    OpenFace block layout [x_0..x_67, y_0..y_67]; zero rows = face not detected
  - dvlog-dataset/<i>/<i>_acoustic.npy  (T_i, 25) eGeMAPS LLDs
  - labels.csv                          index,label,duration,gender,fold

Output (per D-Vlog paper: truncate or zero-pad every sequence to t=596):
  <out_dir>/{train,valid,test}_{visual,labels,acoustic}.npy

Samples are ordered by ascending `index` within each fold.
Labels: depression -> 1, normal -> 0.

Usage:
    conda activate DVlog
    python src/scripts/build_processed_features.py \
        --raw_dir /data/ltq/D-Vlog_Raw \
        --out_dir /data/ltq/D-Vlog_Raw/processed_official_features
"""

import argparse
import os

import numpy as np
import pandas as pd

T_TARGET = 596  # paper: mean vlog duration; truncate or zero-pad to this length


def pad_or_truncate(seq: np.ndarray, t: int = T_TARGET) -> np.ndarray:
    """Truncate to t frames or zero-pad at the end (D-Vlog paper convention)."""
    if seq.shape[0] >= t:
        return seq[:t]
    pad = np.zeros((t - seq.shape[0], seq.shape[1]), dtype=seq.dtype)
    return np.concatenate([seq, pad], axis=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw_dir", default="/data/ltq/D-Vlog_Raw")
    parser.add_argument("--out_dir", default="/data/ltq/D-Vlog_Raw/processed_official_features")
    args = parser.parse_args()

    labels_df = pd.read_csv(os.path.join(args.raw_dir, "labels.csv"))
    assert labels_df["index"].is_unique and labels_df["index"].min() == 0
    label_map = {"depression": 1, "normal": 0}
    unknown = set(labels_df["label"].unique()) - set(label_map)
    assert not unknown, f"unknown labels: {unknown}"

    os.makedirs(args.out_dir, exist_ok=True)

    for fold in ["train", "valid", "test"]:
        sub = labels_df[labels_df["fold"] == fold].sort_values("index")
        visuals, acoustics, labels = [], [], []
        n_mismatch = 0
        for _, row in sub.iterrows():
            i = int(row["index"])
            v = np.load(os.path.join(args.raw_dir, "dvlog-dataset", str(i), f"{i}_visual.npy"))
            a = np.load(os.path.join(args.raw_dir, "dvlog-dataset", str(i), f"{i}_acoustic.npy"))
            if v.shape[0] != a.shape[0]:
                # 14 known samples with visual/acoustic frame-count mismatch.
                # Paper scheme: each modality is truncated/zero-padded to t
                # independently, so the mismatch needs no special handling.
                n_mismatch += 1
            visuals.append(pad_or_truncate(v))
            acoustics.append(pad_or_truncate(a))
            labels.append(label_map[row["label"]])

        visual = np.stack(visuals).astype(np.float64)
        acoustic = np.stack(acoustics).astype(np.float64)
        labels = np.array(labels, dtype=np.int64)

        np.save(os.path.join(args.out_dir, f"{fold}_visual.npy"), visual)
        np.save(os.path.join(args.out_dir, f"{fold}_acoustic.npy"), acoustic)
        np.save(os.path.join(args.out_dir, f"{fold}_labels.npy"), labels)
        print(
            f"[{fold}] {len(sub)} samples -> visual {visual.shape}, "
            f"acoustic {acoustic.shape}, pos ratio {labels.mean():.3f}, "
            f"frame-mismatched samples: {n_mismatch}"
        )

    print(f"Done. Output written to: {args.out_dir}")


if __name__ == "__main__":
    main()
