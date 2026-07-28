"""
Generate Figure 3(b): temporal chain graph with 5 representative frame strips.

Nodes: frame 1, 2, 14 (peak), 15, 22 (last real frame)
Edges: bidirectional arrows between adjacent frames

Output: figure3b_chain.{png,pdf,svg} in visualization/ and paper/overleaf_upload/figures/

Run:
    source /home/ltq/miniconda3/etc/profile.d/conda.sh && conda activate DVlog
    python visualization/draw_figure3b_chain.py
"""

import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

DPI = 300
C_TEXT = "#1D3557"
C_EDGE = "#3D5A80"
C_ACCENT = "#2A9D8F"
C_PEAK = "#E07A5F"

FRAME_INDICES = [1, 2, 14, 15, 22]
NODE_LABELS = [r"$a_1$", r"$a_2$", r"$a_t$", r"$a_{t+1}$", r"$a_T$"]


def load_data():
    here = os.path.dirname(os.path.abspath(__file__))
    npz = os.path.join(here, "figure3_data.npz")
    if not os.path.exists(npz):
        raise FileNotFoundError(f"{npz} not found.")
    d = np.load(npz)
    return d["features"], d["weights"]


def draw_frame_strip(ax, vec, vlim, is_peak=False):
    """Draw a single vertical 25-d feature strip on the given axes."""
    F = len(vec)
    strip = vec.reshape(-1, 1)
    ax.imshow(strip, aspect="auto", cmap="RdBu_r", vmin=-vlim, vmax=vlim,
              extent=[0, 1, 0, F], origin="lower")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, F + 0.05)
    edgecolor = C_PEAK if is_peak else C_EDGE
    lw = 2.5 if is_peak else 1.5
    ax.add_patch(mpatches.Rectangle((0, 0), 1, F, fill=False,
                                    edgecolor=edgecolor, lw=lw, zorder=3))
    ax.axis("off")


def main():
    features, weights = load_data()
    peak = int(weights.argmax())
    vlim = float(np.percentile(np.abs(features), 98))

    fig = plt.figure(figsize=(10, 4.2))
    fig.patch.set_facecolor("white")

    # 5 strip axes + small gaps for arrows.
    n = len(FRAME_INDICES)
    left = 0.08
    right = 0.92
    top = 0.82
    bottom = 0.12
    strip_w = 0.10
    gap = (right - left - n * strip_w) / (n - 1)

    axes = []
    for i, frame_idx in enumerate(FRAME_INDICES):
        x0 = left + i * (strip_w + gap)
        ax = fig.add_axes([x0, bottom, strip_w, top - bottom])
        is_peak = (frame_idx == peak)
        draw_frame_strip(ax, features[frame_idx], vlim, is_peak=is_peak)
        axes.append(ax)

    # Labels above strips.
    for i, (ax, label) in enumerate(zip(axes, NODE_LABELS)):
        pos = ax.get_position()
        color = C_PEAK if FRAME_INDICES[i] == peak else C_TEXT
        weight = "bold" if FRAME_INDICES[i] == peak else "normal"
        fig.text(pos.x0 + pos.width / 2, pos.y1 + 0.03, label,
                 ha="center", va="bottom", fontsize=12, color=color, fontweight=weight)

    # Bidirectional arrows between adjacent strips.
    fig_ax = fig.add_axes([0, 0, 1, 1])
    fig_ax.axis("off")
    for i in range(n - 1):
        pos1 = axes[i].get_position()
        pos2 = axes[i + 1].get_position()
        x1 = pos1.x1
        x2 = pos2.x0
        y = pos1.y0 + pos1.height / 2
        # Draw ellipsis between a2 and a_t to indicate skipped frames.
        if i == 1:
            mid = (x1 + x2) / 2
            fig_ax.annotate("", xy=(mid - 0.015, y), xytext=(x1 + 0.01, y),
                            arrowprops=dict(arrowstyle="<->", color=C_EDGE, lw=1.4, mutation_scale=9))
            fig_ax.annotate("", xy=(x2 - 0.01, y), xytext=(mid + 0.015, y),
                            arrowprops=dict(arrowstyle="<->", color=C_EDGE, lw=1.4, mutation_scale=9))
            fig_ax.text(mid, y, r"$\cdots$", ha="center", va="center",
                        fontsize=12, color=C_TEXT, zorder=3)
        else:
            fig_ax.annotate("", xy=(x2 - 0.01, y), xytext=(x1 + 0.01, y),
                            arrowprops=dict(arrowstyle="<->", color=C_EDGE, lw=1.6, mutation_scale=10))

    # Bottom caption.
    fig.text(0.5, 0.03,
             "Each node = one frame's 25-d feature vector; edges link adjacent frames bidirectionally",
             ha="center", va="bottom", fontsize=9, color=C_EDGE)

    # Title.
    fig.text(0.5, 0.96, "(b) Temporal chain graph", ha="center", va="top",
             fontsize=13, color=C_TEXT, fontweight="bold")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    out_dirs = [script_dir, os.path.join(project_root, "paper", "overleaf_upload", "figures")]
    for out_dir in out_dirs:
        os.makedirs(out_dir, exist_ok=True)
        for ext in ["pdf", "png", "svg"]:
            path = os.path.join(out_dir, f"figure3b_chain.{ext}")
            fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
            print(f"Saved: {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
