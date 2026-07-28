"""
Generate Figure 2 (v4): Single-panel 68-point facial landmark topology.

Characteristics:
    * Only subfigure (a): full face landmark layout.
    * No text, titles, labels, or legend.
    * No edges between landmarks.
    * Landmarks are coloured by facial region.
    * Face shape is intentionally slender / elongated.

Output:
    - visualization/figure2_face_graph_v4.{pdf,png,svg}
"""

import os

import matplotlib.pyplot as plt
import numpy as np

DPI = 300
C_NODE = "#1D3557"

# ---------------------------------------------------------------------------
# 68 facial landmark coordinates (schematic frontal face, symmetric)
# Based on the standard dlib/OpenFace 68-point convention.
# ---------------------------------------------------------------------------
LANDMARKS_68 = np.array([
    [1.60, 0.40], [1.55, 0.65], [1.45, 0.90], [1.30, 1.10],
    [1.10, 1.28], [0.85, 1.42], [0.55, 1.52], [0.00, 1.58],
    [-0.55, 1.52], [-0.85, 1.42], [-1.10, 1.28], [-1.30, 1.10],
    [-1.45, 0.90], [-1.55, 0.65], [-1.60, 0.40], [-1.55, 0.15], [-1.45, -0.05],
    [0.30, -0.55], [0.55, -0.72], [0.85, -0.78], [1.15, -0.72], [1.40, -0.55],
    [-0.30, -0.55], [-0.55, -0.72], [-0.85, -0.78], [-1.15, -0.72], [-1.40, -0.55],
    [0.00, -0.45], [0.00, -0.15], [0.00, 0.15], [0.00, 0.45],
    [-0.22, 0.72], [-0.10, 0.78], [0.00, 0.80], [0.10, 0.78], [0.22, 0.72],
    [0.38, 0.05], [0.58, -0.02], [0.88, -0.02], [1.08, 0.05], [0.88, 0.14], [0.58, 0.14],
    [-0.38, 0.05], [-0.58, -0.02], [-0.88, -0.02], [-1.08, 0.05], [-0.88, 0.14], [-0.58, 0.14],
    [-0.45, 1.02], [-0.25, 0.95], [-0.10, 0.92], [0.00, 0.91], [0.10, 0.92], [0.25, 0.95],
    [0.45, 1.02], [0.25, 1.12], [0.10, 1.16], [0.00, 1.17], [-0.10, 1.16], [-0.25, 1.12],
    [-0.22, 1.02], [-0.10, 0.98], [0.10, 0.98], [0.22, 1.02], [0.10, 1.10], [0.00, 1.11],
    [-0.10, 1.10], [-0.22, 1.02],
], dtype=float)
LANDMARKS_68[:, 1] *= -1

REGIONS = {
    "face_contour": list(range(0, 17)),
    "left_eyebrow": list(range(17, 22)),
    "right_eyebrow": list(range(22, 27)),
    "nose_bridge": list(range(27, 31)),
    "nose_bottom": list(range(31, 36)),
    "left_eye": list(range(36, 42)),
    "right_eye": list(range(42, 48)),
    "outer_mouth": list(range(48, 60)),
    "inner_mouth": list(range(60, 68)),
}

REGION_COLORS = {
    "face_contour": "#8D99AE",
    "left_eyebrow": "#E07A5F",
    "right_eyebrow": "#E07A5F",
    "nose_bridge": "#3D5A80",
    "nose_bottom": "#3D5A80",
    "left_eye": "#2A9D8F",
    "right_eye": "#2A9D8F",
    "outer_mouth": "#6D597A",
    "inner_mouth": "#6D597A",
}

# Scaling factors to make the face look slender / elongated.
# x_squeeze < 1 narrows the face; y_stretch > 1 elongates it.
X_SQUEEZE = 0.82
Y_STRETCH = 1.35


def main():
    coords = LANDMARKS_68.copy()
    # Center the face around the origin before applying non-uniform scaling.
    center = coords.mean(axis=0)
    coords -= center
    coords[:, 0] *= X_SQUEEZE
    coords[:, 1] *= Y_STRETCH
    coords += center

    # Build per-point colours.
    point_colors = np.empty(len(coords), dtype=object)
    for region_name, indices in REGIONS.items():
        point_colors[indices] = REGION_COLORS[region_name]

    # Use a tall, narrow figure to reinforce the slender look.
    fig, ax = plt.subplots(figsize=(3.6, 6.0), dpi=DPI)
    fig.patch.set_facecolor("white")
    fig.patch.set_alpha(1.0)

    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=point_colors,
        s=45,
        zorder=3,
        edgecolors="white",
        linewidths=0.6,
    )

    # Remove all text/axes/decorations.
    ax.axis("off")

    # Tight crop around the points with a small margin.
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    margin = 0.25
    ax.set_xlim(x_min - margin, x_max + margin)
    ax.set_ylim(y_min - margin, y_max + margin)

    # Ensure the aspect ratio matches the slender transformation.
    ax.set_aspect("equal")

    plt.tight_layout(pad=0)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = script_dir
    os.makedirs(out_dir, exist_ok=True)
    for ext in ["pdf", "png", "svg"]:
        path = os.path.join(out_dir, f"figure2_face_graph_v4.{ext}")
        fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white", pad_inches=0.02)
        print(f"Saved: {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
