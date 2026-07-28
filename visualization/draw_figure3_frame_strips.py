"""
Generate vertical 25-D feature strips for selected frames.

These strips match the column appearance of figure3a_features heatmap
(same color scale, same 25 rows) and are saved as individual files.

Usage:
    source /home/ltq/miniconda3/etc/profile.d/conda.sh && conda activate DVlog
    python visualization/draw_figure3_frame_strips.py
"""

import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

DPI = 300
C_TEXT = "#1D3557"
C_EDGE = "#3D5A80"

# Frames to export (0-indexed, as labelled on the figure3a x-axis).
FRAME_INDICES = [1, 2, 14, 15, 22]


def load_data():
    here = os.path.dirname(os.path.abspath(__file__))
    npz = os.path.join(here, "figure3_data.npz")
    if not os.path.exists(npz):
        raise FileNotFoundError(f"{npz} not found.")
    d = np.load(npz)
    return d["features"], d["weights"]


def save_strip(features, frame_idx, out_dirs):
    T, F = features.shape
    if frame_idx >= T:
        raise ValueError(f"frame {frame_idx} exceeds feature length {T}")

    vec = features[frame_idx]
    # Use the same normalization range as figure3a_features.
    vlim = float(np.percentile(np.abs(features), 98))

    fig, ax = plt.subplots(figsize=(1.0, 4.3))
    strip = vec.reshape(-1, 1)
    ax.imshow(strip, aspect="auto", cmap="RdBu_r", vmin=-vlim, vmax=vlim,
              extent=[0, 1, 0, F], origin="lower")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, F + 0.05)
    # Tight border matching the heatmap cell grid style.
    ax.add_patch(mpatches.Rectangle((0, 0), 1, F, fill=False,
                                    edgecolor=C_EDGE, lw=1.5, zorder=3))
    ax.axis("off")
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)

    for out_dir in out_dirs:
        os.makedirs(out_dir, exist_ok=True)
        for ext in ["pdf", "png", "svg"]:
            path = os.path.join(out_dir, f"figure3_framestrip_{frame_idx:02d}.{ext}")
            fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white", pad_inches=0.02)
            print(f"Saved: {path}")
    plt.close(fig)


def main():
    features, weights = load_data()   # original (32, 25)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    out_dirs = [script_dir, os.path.join(project_root, "paper", "overleaf_upload", "figures")]

    for frame_idx in FRAME_INDICES:
        save_strip(features, frame_idx, out_dirs)


if __name__ == "__main__":
    main()
