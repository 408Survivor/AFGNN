"""
Generate Figure 4: Main ablation results as 2x2 horizontal-lollipop small multiples.

Panels (matching AFGNN-V2.tex caption):
    (a) Facial graph structure            (Table III, F1)
    (b) Modality contribution and gated fusion (Table IV, F1)
    (c) Audio length and loss function    (Table V, F1)
    (d) Seed-ensemble size                (Table V, F1)

Colour semantics (shared legend):
    best = best configuration in the group / final model
    var  = other variants
    ctrl = degraded / control configurations

Output: figure4_ablation.{pdf,png,svg} in visualization and overleaf_upload/figures.

Run:
    source /home/ltq/miniconda3/etc/profile.d/conda.sh && conda activate DVlog
    python visualization/draw_figure4_ablation.py
"""

import os

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib.lines import Line2D

DPI = 300

# --- Use Times New Roman for all text ------------------------------------
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = ["Times New Roman"]
mpl.rcParams["axes.titlesize"] = 14
mpl.rcParams["axes.labelsize"] = 12
mpl.rcParams["xtick.labelsize"] = 11
mpl.rcParams["ytick.labelsize"] = 11
mpl.rcParams["legend.fontsize"] = 12

C_TEXT = "#1D3557"
C_ERR = "#2B2D42"

# Palette (Indigo / Periwinkle / Coral)
PALETTE = {"best": "#34488F", "var": "#9FB1DC", "ctrl": "#E0875E"}

XMIN, XMAX = 0.68, 0.84
XTICKS = np.arange(0.68, 0.84, 0.03)

# (panel letter, section title, [(label, value, std_or_None, role)])
SECTIONS = [
    ("a", "Facial graph structure", [
        ("Full", 0.794, None, "best"),
        ("$-$ Topology edges", 0.786, None, "var"),
        ("$-$ Temporal edges", 0.748, None, "ctrl"),
        ("$-$ Region one-hot", 0.789, None, "var"),
        ("$-$ Velocity $[dx,dy]$", 0.752, None, "ctrl"),
        ("$-$ Self-loop", 0.748, None, "ctrl"),
        ("Topology+Temporal+Dynamic ($k$=1)", 0.714, None, "ctrl"),
        ("Motion+Region ($k$=2)", 0.709, None, "ctrl")]),
    ("b", "Modality contribution and gated fusion", [
        ("Face only", 0.734, None, "var"),
        ("Audio only", 0.743, None, "var"),
        ("Concat", 0.764, None, "var"),
        ("Gated fusion", 0.787, None, "best")]),
    ("c", "Audio length and loss function", [
        ("16 frames", 0.767, None, "var"),
        ("32 frames", 0.787, None, "best"),
        ("64 frames", 0.762, None, "var"),
        ("Weighted BCE", 0.782, None, "var"),
        ("Focal loss", 0.787, None, "best")]),
    ("d", "Seed-ensemble size", [
        ("Single seed 42", 0.787, None, "var"),
        ("3-seed ensemble", 0.8077, None, "best"),
        ("5-seed ensemble", 0.793, None, "var")]),
]


def bar_panel(ax, letter, title, rows, palette, show_xlabel):
    ys = np.arange(len(rows))[::-1]
    for y, (label, val, err, role) in zip(ys, rows):
        ax.barh(y, val - XMIN, left=XMIN, height=0.62, color=palette[role],
                edgecolor="white", linewidth=0.8, zorder=2)
        if err:
            ax.errorbar(val, y, xerr=err, fmt="none", ecolor=C_ERR,
                        elinewidth=1.2, capsize=3, zorder=3)
        # Value label
        ax.text(val + 0.007, y, f"{val:.4f}" if val == 0.8077 else f"{val:.3f}",
                va="center", ha="left", fontsize=11, fontweight="bold", color=C_TEXT)

    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=11, color=C_TEXT)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlim(XMIN, XMAX)
    ax.set_xticks(XTICKS)
    ax.set_title(f"({letter}) {title}", fontsize=14, color=C_TEXT, fontweight="bold",
                 pad=8, loc="left")
    ax.grid(axis="x", linestyle="--", alpha=0.35, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=11, colors=C_TEXT)
    if show_xlabel:
        ax.set_xlabel("F1-score", fontsize=12, color=C_TEXT)


def build_figure(palette=PALETTE, out_stem="figure4_ablation", out_dirs=None):
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.0))
    fig.patch.set_facecolor("white")
    axes_flat = axes.ravel()
    for idx, (ax, (letter, name, rows)) in enumerate(zip(axes_flat, SECTIONS)):
        bar_panel(ax, letter, name, rows, palette, show_xlabel=(idx >= 2))

    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=palette["best"],
               markeredgecolor="white", markersize=13, label="Best in group / final model"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=palette["var"],
               markeredgecolor="white", markersize=13, label="Other variants"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=palette["ctrl"],
               markeredgecolor="white", markersize=13, label="Degraded / control"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               fontsize=12, bbox_to_anchor=(0.5, 1.01))

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if out_dirs is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        out_dirs = [script_dir, os.path.join(project_root, "paper", "overleaf_upload", "figures")]
    for out_dir in out_dirs:
        os.makedirs(out_dir, exist_ok=True)
        for ext in ["pdf", "png", "svg"]:
            path = os.path.join(out_dir, f"{out_stem}.{ext}")
            fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
            print(f"Saved: {path}")
    plt.close(fig)


def main():
    build_figure()


if __name__ == "__main__":
    main()
