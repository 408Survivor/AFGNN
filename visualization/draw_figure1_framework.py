"""
Generate Figure 1: AFGNN overall framework.

Output:
    - afgnn_framework.pdf
    - afgnn_framework.png
    - afgnn_framework.svg

Run:
    python visualization/draw_figure1_framework.py
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
FIG_W, FIG_H = 14, 10
DPI = 300

# Color palette
C_FACE = "#E07A5F"      # warm coral
C_FACE_LT = "#F2CC8F"   # light peach
C_AUDIO = "#3D5A80"     # steel blue
C_AUDIO_LT = "#98C1D9"  # light blue
C_FUSION = "#6D597A"    # muted purple
C_MLP = "#4A4E69"       # dark slate
C_OUTPUT = "#2A9D8F"    # teal
C_TEXT = "#1D3557"      # dark blue-gray
C_ARROW = "#555555"

# Box style
BOX_STYLE = "round,pad=0.03,rounding_size=0.15"
LW = 1.2
FONT_SIZE = 10
SMALL_FONT = 8


def add_box(ax, x, y, w, h, text, color, text_color="white", fontsize=FONT_SIZE,
            alpha=1.0, edgecolor="none", linewidth=LW):
    """Add a rounded rectangle with text."""
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle=BOX_STYLE,
        facecolor=color,
        edgecolor=edgecolor,
        linewidth=linewidth,
        alpha=alpha,
        zorder=2,
    )
    ax.add_patch(box)
    ax.text(
        x, y, text,
        ha="center", va="center",
        color=text_color,
        fontsize=fontsize,
        fontweight="bold",
        zorder=3,
        wrap=True,
    )
    return box


def add_text(ax, x, y, text, fontsize=SMALL_FONT, color=C_TEXT, ha="center", va="center",
             fontweight="normal"):
    ax.text(x, y, text, ha=ha, va=va, fontsize=fontsize, color=color,
            fontweight=fontweight, zorder=3)


def arrow(ax, x1, y1, x2, y2, color=C_ARROW, style="->", mutation_scale=15, lw=1.2):
    """Add an arrow between two points."""
    ax.annotate(
        "",
        xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle=style,
            color=color,
            lw=lw,
            connectionstyle="arc3,rad=0",
        ),
        zorder=1,
    )


def main():
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, FIG_H)
    ax.axis("off")

    # -----------------------------------------------------------------------
    # Vertical layout positions (top to bottom)
    # -----------------------------------------------------------------------
    y_input = 8.8
    y_landmarks = 7.6
    y_graph = 6.0
    y_gnn = 4.4
    y_emb = 3.0
    y_fusion = 1.8
    y_mlp = 0.9
    y_out = 0.15

    # Horizontal centers
    x_left = 3.5
    x_right = 10.5
    x_center = 7.0

    # -----------------------------------------------------------------------
    # Title
    # -----------------------------------------------------------------------
    ax.text(
        FIG_W / 2, 9.7,
        "Figure 1: AFGNN Framework",
        ha="center", va="center",
        fontsize=16, fontweight="bold", color=C_TEXT,
    )

    # -----------------------------------------------------------------------
    # Inputs
    # -----------------------------------------------------------------------
    add_box(ax, x_left, y_input, 2.4, 0.65, "Input Video\nClip", "#8D99AE",
            text_color="white", fontsize=FONT_SIZE)
    add_box(ax, x_right, y_input, 2.4, 0.65, "Acoustic\nFeatures", "#8D99AE",
            text_color="white", fontsize=FONT_SIZE)

    add_text(ax, x_left - 1.7, y_input, "vlog", fontsize=SMALL_FONT, color=C_TEXT)
    add_text(ax, x_right + 1.8, y_input, "25-D $\\times$ $T_a$", fontsize=SMALL_FONT, color=C_TEXT)

    # -----------------------------------------------------------------------
    # Face branch
    # -----------------------------------------------------------------------
    add_box(ax, x_left, y_landmarks, 2.6, 0.75,
            "68 Facial\nLandmarks", C_FACE_LT, text_color=C_TEXT, fontsize=FONT_SIZE)
    add_text(ax, x_left - 1.8, y_landmarks, "$T_v$ frames", fontsize=SMALL_FONT, color=C_TEXT)

    add_box(ax, x_left, y_graph, 3.2, 1.0,
            "Spatio-Temporal\nFacial Graph", C_FACE, fontsize=FONT_SIZE)
    add_text(ax, x_left - 2.0, y_graph + 0.25, "static", fontsize=SMALL_FONT, color=C_FACE)
    add_text(ax, x_left - 2.0, y_graph, "temporal", fontsize=SMALL_FONT, color=C_FACE)
    add_text(ax, x_left - 2.0, y_graph - 0.25, "dynamic*", fontsize=SMALL_FONT, color=C_FACE)

    add_box(ax, x_left, y_gnn, 2.8, 0.8,
            "Face GNN\n(GAT × $L_V$)", C_FACE, fontsize=FONT_SIZE)

    add_box(ax, x_left, y_emb, 2.2, 0.65,
            "$\\mathbf{H}_{\\mathrm{face}}$", C_FACE_LT, text_color=C_TEXT, fontsize=FONT_SIZE)
    add_text(ax, x_left - 1.6, y_emb, "$d_f$=128", fontsize=SMALL_FONT, color=C_TEXT)

    # -----------------------------------------------------------------------
    # Audio branch
    # -----------------------------------------------------------------------
    add_box(ax, x_right, y_landmarks, 2.6, 0.75,
            "Frame-Level\nAcoustic Seq.", C_AUDIO_LT, text_color=C_TEXT, fontsize=FONT_SIZE)
    add_text(ax, x_right + 1.8, y_landmarks, "$T_a$=32", fontsize=SMALL_FONT, color=C_TEXT)

    add_box(ax, x_right, y_graph, 3.0, 1.0,
            "Temporal Chain\nAudio Graph", C_AUDIO, fontsize=FONT_SIZE)
    add_text(ax, x_right + 1.9, y_graph, "bidirectional", fontsize=SMALL_FONT, color=C_AUDIO)

    add_box(ax, x_right, y_gnn, 2.8, 0.8,
            "Audio GNN\n(GAT × $L_A$)", C_AUDIO, fontsize=FONT_SIZE)

    add_box(ax, x_right, y_emb, 2.2, 0.65,
            "$\\mathbf{H}_{\\mathrm{audio}}$", C_AUDIO_LT, text_color=C_TEXT, fontsize=FONT_SIZE)
    add_text(ax, x_right + 1.6, y_emb, "$d_a$=64", fontsize=SMALL_FONT, color=C_TEXT)

    # -----------------------------------------------------------------------
    # Arrows: input -> landmarks/graph -> gnn -> embedding
    # -----------------------------------------------------------------------
    arrow(ax, x_left, y_input - 0.32, x_left, y_landmarks + 0.37)
    arrow(ax, x_left, y_landmarks - 0.37, x_left, y_graph + 0.5)
    arrow(ax, x_left, y_graph - 0.5, x_left, y_gnn + 0.4)
    arrow(ax, x_left, y_gnn - 0.4, x_left, y_emb + 0.32)

    arrow(ax, x_right, y_input - 0.32, x_right, y_landmarks + 0.37)
    arrow(ax, x_right, y_landmarks - 0.37, x_right, y_graph + 0.5)
    arrow(ax, x_right, y_graph - 0.5, x_right, y_gnn + 0.4)
    arrow(ax, x_right, y_gnn - 0.4, x_right, y_emb + 0.32)

    # -----------------------------------------------------------------------
    # Fusion
    # -----------------------------------------------------------------------
    add_box(ax, x_center, y_fusion, 3.0, 0.8,
            "Cross-Modal\nAttention Fusion", C_FUSION, fontsize=FONT_SIZE)
    arrow(ax, x_left, y_emb - 0.32, x_center - 0.7, y_fusion + 0.4)
    arrow(ax, x_right, y_emb - 0.32, x_center + 0.7, y_fusion + 0.4)

    # -----------------------------------------------------------------------
    # MLP classifier
    # -----------------------------------------------------------------------
    add_box(ax, x_center, y_mlp, 2.6, 0.65,
            "MLP Classifier", C_MLP, fontsize=FONT_SIZE)
    arrow(ax, x_center, y_fusion - 0.4, x_center, y_mlp + 0.32)

    # -----------------------------------------------------------------------
    # Output
    # -----------------------------------------------------------------------
    add_box(ax, x_center, y_out, 2.4, 0.5,
            "Depression Score  $\\hat{y}$", C_OUTPUT, fontsize=FONT_SIZE)
    arrow(ax, x_center, y_mlp - 0.32, x_center, y_out + 0.25)

    # -----------------------------------------------------------------------
    # Legend / notes
    # -----------------------------------------------------------------------
    legend_x, legend_y = 0.5, 2.3
    add_text(ax, legend_x, legend_y + 0.4, "Legend:", fontsize=SMALL_FONT, color=C_TEXT, ha="left")

    patches = [
        ("Face branch", C_FACE),
        ("Audio branch", C_AUDIO),
        ("Fusion", C_FUSION),
        ("Classifier / Output", C_OUTPUT),
    ]
    for i, (label, color) in enumerate(patches):
        rect = mpatches.Rectangle((legend_x, legend_y - i * 0.35 - 0.12), 0.35, 0.22,
                                   facecolor=color, edgecolor="none", zorder=3)
        ax.add_patch(rect)
        add_text(ax, legend_x + 0.5, legend_y - i * 0.35, label,
                 fontsize=SMALL_FONT, color=C_TEXT, ha="left", va="center")

    add_text(ax, 0.5, 0.5, "* Dynamic edges are ablated;\n  final model uses static + temporal.",
             fontsize=SMALL_FONT, color=C_TEXT, ha="left")

    add_text(ax, FIG_W - 0.5, 0.5, "< 5M parameters",
             fontsize=SMALL_FONT, color=C_TEXT, ha="right", fontweight="bold")

    plt.tight_layout()

    out_dir = os.path.dirname(os.path.abspath(__file__))
    for ext in ["pdf", "png", "svg"]:
        path = os.path.join(out_dir, f"afgnn_framework.{ext}")
        fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
        print(f"Saved: {path}")

    plt.close(fig)


if __name__ == "__main__":
    main()
