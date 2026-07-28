"""
Generate Figure 2 (v5): 68-point facial landmark topology with AU-informed edges.

Based on v4 (slender / elongated face, no text) but adds intra-frame facial-topology
edges that correspond to the region definitions used in the AFGNN code:
    * chain connections for open components (contour, eyebrows, nose bridge)
    * ring/closed connections for eyes, nose bottom, and mouth

Output:
    - visualization/figure2_face_graph_v5.{pdf,png,svg}
"""

import os

import matplotlib.pyplot as plt
import numpy as np

DPI = 300
C_NODE = "#1D3557"

# ---------------------------------------------------------------------------
# 68 facial landmark coordinates (schematic frontal face, symmetric)
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

# Open components use chain connections; closed components use ring connections.
OPEN_REGIONS = {"face_contour", "left_eyebrow", "right_eyebrow", "nose_bridge"}

X_SQUEEZE = 0.82
Y_STRETCH = 1.35


def chain_edges(indices, closed=False):
    """Return consecutive edges; if closed, also connect last -> first."""
    e = [(indices[i], indices[i + 1]) for i in range(len(indices) - 1)]
    if closed and len(indices) > 1:
        e.append((indices[-1], indices[0]))
    return e


def region_edges(name):
    return chain_edges(REGIONS[name], closed=name not in OPEN_REGIONS)


def main():
    coords = LANDMARKS_68.copy()
    center = coords.mean(axis=0)
    coords -= center
    coords[:, 0] *= X_SQUEEZE
    coords[:, 1] *= Y_STRETCH
    coords += center

    # Per-point colours.
    point_colors = np.empty(len(coords), dtype=object)
    for region_name, indices in REGIONS.items():
        point_colors[indices] = REGION_COLORS[region_name]

    fig, ax = plt.subplots(figsize=(3.6, 6.0), dpi=DPI)
    fig.patch.set_facecolor("white")
    fig.patch.set_alpha(1.0)

    # Draw topology edges first so points sit on top.
    for region_name in REGIONS:
        color = REGION_COLORS[region_name]
        for i, j in region_edges(region_name):
            ax.plot(
                [coords[i, 0], coords[j, 0]],
                [coords[i, 1], coords[j, 1]],
                color=color,
                lw=1.0,
                alpha=0.85,
                zorder=1,
                solid_capstyle="round",
            )

    # Draw landmark nodes.
    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=point_colors,
        s=55,
        zorder=3,
        edgecolors="white",
        linewidths=0.7,
    )

    # No text, axes, or legend.
    ax.axis("off")

    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    margin = 0.25
    ax.set_xlim(x_min - margin, x_max + margin)
    ax.set_ylim(y_min - margin, y_max + margin)
    ax.set_aspect("equal")

    plt.tight_layout(pad=0)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = script_dir
    os.makedirs(out_dir, exist_ok=True)
    for ext in ["pdf", "png", "svg"]:
        path = os.path.join(out_dir, f"figure2_face_graph_v5.{ext}")
        fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white", pad_inches=0.02)
        print(f"Saved: {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
