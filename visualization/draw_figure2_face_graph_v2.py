"""
Generate Figure 2 (v2): Facial graph construction details.

Subfigures:
    (a) Single-frame 68-point facial landmark topology coloured by region.
    (b) Zoomed eye region showing intra-frame anatomical edges.
    (c) Spatio-temporal graph: same landmark across 3 consecutive frames.

Output:
    - overleaf_upload/figures/figure2_face_graph.pdf
    - overleaf_upload/figures/figure2_face_graph.png
    - overleaf_upload/figures/figure2_face_graph.svg
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

DPI = 300

# ---------------------------------------------------------------------------
# Improved 68 facial landmark coordinates (schematic frontal face)
# Based on the standard dlib/OpenFace 68-point convention.
# Coordinates are chosen to give a natural, symmetric face proportion.
# ---------------------------------------------------------------------------
# Start with a clean symmetric face, x=0 at center, y>0 at top.
LANDMARKS_68 = np.array([
    # Face contour (0-16): jawline from right ear to chin to left ear
    [1.60, 0.40], [1.55, 0.65], [1.45, 0.90], [1.30, 1.10],
    [1.10, 1.28], [0.85, 1.42], [0.55, 1.52], [0.00, 1.58],  # chin center
    [-0.55, 1.52], [-0.85, 1.42], [-1.10, 1.28], [-1.30, 1.10],
    [-1.45, 0.90], [-1.55, 0.65], [-1.60, 0.40], [-1.55, 0.15],
    [-1.45, -0.05],

    # Left eyebrow (17-21): person's left -> right side of image, x>0
    [0.30, -0.55], [0.55, -0.72], [0.85, -0.78], [1.15, -0.72], [1.40, -0.55],

    # Right eyebrow (22-26): person's right -> left side of image, x<0
    [-0.30, -0.55], [-0.55, -0.72], [-0.85, -0.78], [-1.15, -0.72], [-1.40, -0.55],

    # Nose bridge (27-30)
    [0.00, -0.45], [0.00, -0.15], [0.00, 0.15], [0.00, 0.45],

    # Nose bottom (31-35)
    [-0.22, 0.72], [-0.10, 0.78], [0.00, 0.80], [0.10, 0.78], [0.22, 0.72],

    # Left eye (36-41): person's left eye on right side of image
    [0.38, 0.05], [0.58, -0.02], [0.88, -0.02], [1.08, 0.05],
    [0.88, 0.14], [0.58, 0.14],

    # Right eye (42-47): person's right eye on left side of image
    [-0.38, 0.05], [-0.58, -0.02], [-0.88, -0.02], [-1.08, 0.05],
    [-0.88, 0.14], [-0.58, 0.14],

    # Outer mouth (48-59)
    [-0.45, 1.02], [-0.25, 0.95], [-0.10, 0.92], [0.00, 0.91],
    [0.10, 0.92], [0.25, 0.95], [0.45, 1.02], [0.25, 1.12],
    [0.10, 1.16], [0.00, 1.17], [-0.10, 1.16], [-0.25, 1.12],

    # Inner mouth (60-67)
    [-0.22, 1.02], [-0.10, 0.98], [0.10, 0.98], [0.22, 1.02],
    [0.10, 1.10], [0.00, 1.11], [-0.10, 1.10], [-0.22, 1.02],
], dtype=float)

# Mirror x so that the face faces the viewer in standard orientation
# (already symmetric; just ensure y increases downward for image-like view)
LANDMARKS_68[:, 1] *= -1

# ---------------------------------------------------------------------------
# Region definitions (dlib convention)
# ---------------------------------------------------------------------------
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


def chain_edges(indices, closed=False):
    """Return chain edges for a list of indices."""
    edges = []
    for i in range(len(indices) - 1):
        edges.append((indices[i], indices[i + 1]))
    if closed and len(indices) > 1:
        edges.append((indices[-1], indices[0]))
    return edges


def build_static_edges():
    """Build anatomical (static) edges."""
    edges = []
    edges.extend(chain_edges(REGIONS["face_contour"], closed=False))
    edges.extend(chain_edges(REGIONS["left_eyebrow"], closed=False))
    edges.extend(chain_edges(REGIONS["right_eyebrow"], closed=False))
    edges.extend(chain_edges(REGIONS["nose_bridge"], closed=False))
    edges.extend(chain_edges(REGIONS["nose_bottom"], closed=True))
    edges.extend(chain_edges(REGIONS["left_eye"], closed=True))
    edges.extend(chain_edges(REGIONS["right_eye"], closed=True))
    edges.extend(chain_edges(REGIONS["outer_mouth"], closed=True))
    edges.extend(chain_edges(REGIONS["inner_mouth"], closed=True))
    return edges


def plot_edges(ax, coords, edges, color="#555555", lw=1.0, alpha=1.0, zorder=1):
    """Plot edges as line segments."""
    for i, j in edges:
        ax.plot([coords[i, 0], coords[j, 0]], [coords[i, 1], coords[j, 1]],
                color=color, lw=lw, alpha=alpha, zorder=zorder, solid_capstyle="round")


def plot_nodes(ax, coords, color="#1D3557", size=25, alpha=1.0, zorder=2, edgecolors="white", lw=0.5):
    """Plot landmark nodes."""
    ax.scatter(coords[:, 0], coords[:, 1], c=color, s=size, alpha=alpha,
               zorder=zorder, edgecolors=edgecolors, linewidths=lw)


def add_region_legend(ax):
    """Add legend for facial regions."""
    handles = []
    shown = set()
    label_map = {
        "face_contour": "Face contour",
        "left_eyebrow": "Eyebrow",
        "right_eyebrow": "Eyebrow",
        "nose_bridge": "Nose",
        "nose_bottom": "Nose",
        "left_eye": "Eye",
        "right_eye": "Eye",
        "outer_mouth": "Mouth",
        "inner_mouth": "Mouth",
    }
    for name, color in REGION_COLORS.items():
        label = label_map[name]
        if label not in shown:
            handles.append(mpatches.Patch(color=color, label=label))
            shown.add(label)
    ax.legend(handles=handles, loc="lower right", fontsize=8, frameon=False,
              ncol=1, handlelength=1.2, handletextpad=0.4)


def main():
    coords = LANDMARKS_68
    static_edges = build_static_edges()

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    fig.patch.set_facecolor("white")

    # -----------------------------------------------------------------------
    # (a) Full 68-point topology with region coloring
    # -----------------------------------------------------------------------
    ax = axes[0]
    # Plot all static edges faintly first to avoid overlap artefacts
    plot_edges(ax, coords, static_edges, color="#DDDDDD", lw=0.8, alpha=0.7, zorder=1)

    # Plot region-specific edges and nodes on top
    for region_name, indices in REGIONS.items():
        closed = region_name not in ["face_contour", "left_eyebrow", "right_eyebrow", "nose_bridge"]
        region_edges = chain_edges(indices, closed=closed)
        color = REGION_COLORS[region_name]
        plot_edges(ax, coords, region_edges, color=color, lw=1.6, alpha=0.95, zorder=2)

    plot_nodes(ax, coords, color="#1D3557", size=35, zorder=3)
    ax.set_title("(a) 68-point facial landmark topology", fontsize=12, color="#1D3557", pad=10)
    add_region_legend(ax)
    ax.set_aspect("equal")
    ax.axis("off")

    # -----------------------------------------------------------------------
    # (b) Zoomed right eye region (indices 42-47 in dlib convention)
    # -----------------------------------------------------------------------
    ax = axes[1]
    eye_region = "right_eye"  # person's right eye, appears on left side of image
    eye_indices = REGIONS[eye_region]
    eye_edges = chain_edges(eye_indices, closed=True)

    # Plot contextual landmarks faintly
    context_indices = [i for r in ["face_contour", "nose_bridge", "right_eyebrow"] for i in REGIONS[r]]
    context_edges = []
    for r in ["face_contour", "nose_bridge", "right_eyebrow"]:
        context_edges.extend(chain_edges(REGIONS[r], closed=(r != "face_contour" and r != "nose_bridge" and r != "right_eyebrow")))

    plot_edges(ax, coords, context_edges, color="#CCCCCC", lw=1.0, alpha=0.45, zorder=1)
    plot_nodes(ax, coords[context_indices], color="#888888", size=20, alpha=0.45, zorder=1)

    # Plot eye edges and nodes prominently
    plot_edges(ax, coords, eye_edges, color="#2A9D8F", lw=2.4, alpha=0.95, zorder=2)
    plot_nodes(ax, coords[eye_indices], color="#2A9D8F", size=110, zorder=3)

    # Annotate landmark indices
    for idx in eye_indices:
        x, y = coords[idx]
        ax.text(x + 0.04, y + 0.04, str(idx), fontsize=7, color="#1D3557", zorder=4,
                fontweight="medium")

    margin = 0.35
    ax.set_xlim(coords[eye_indices, 0].min() - margin, coords[eye_indices, 0].max() + margin)
    ax.set_ylim(coords[eye_indices, 1].min() - margin, coords[eye_indices, 1].max() + margin)
    ax.set_title("(b) Intra-frame anatomical edges (right eye)", fontsize=12, color="#1D3557", pad=10)
    ax.set_aspect("equal")
    ax.axis("off")

    # -----------------------------------------------------------------------
    # (c) Temporal edges: same landmark across 3 frames (3-D perspective)
    # -----------------------------------------------------------------------
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    axes[2].remove()
    ax = fig.add_subplot(1, 3, 3, projection="3d")

    landmark_idx = 42  # right eye outer corner
    n_frames = 3
    z_depths = [0.0, 1.2, 2.4]
    frame_colors = ["#E07A5F", "#457B9D", "#2A9D8F"]
    frame_alphas = [0.55, 0.75, 0.95]
    frame_labels = ["$t$", "$t+1$", "$t+2$"]

    contour_edges = chain_edges(REGIONS["face_contour"], closed=False)
    traj_points = []

    for t, z in enumerate(z_depths):
        # Place each frame at a different depth; shift laterally so they do not overlap
        shift_x = -0.35 * z
        shift_y = -0.25 * z
        frame_coords = coords.copy()
        frame_coords[:, 0] += shift_x
        frame_coords[:, 1] += shift_y

        # Face contour at this time step
        for i, j in contour_edges:
            ax.plot3D(
                [frame_coords[i, 0], frame_coords[j, 0]],
                [frame_coords[i, 1], frame_coords[j, 1]],
                [z, z],
                color=frame_colors[t], lw=1.2, alpha=frame_alphas[t] * 0.6
            )
        # All landmarks as faint dots
        ax.scatter(
            frame_coords[:, 0], frame_coords[:, 1], [z] * len(frame_coords),
            c=frame_colors[t], s=8, alpha=frame_alphas[t] * 0.4, depthshade=False
        )

        # Highlight the tracked landmark
        hl = frame_coords[landmark_idx]
        ax.scatter(
            hl[0], hl[1], z, c=frame_colors[t], s=90,
            edgecolors="white", linewidths=1.0, depthshade=False
        )
        ax.text(
            hl[0] + 0.18, hl[1] + 0.12, z,
            frame_labels[t], fontsize=10, color=frame_colors[t],
            fontweight="bold"
        )
        traj_points.append((hl[0], hl[1], z))

    # Temporal edges connecting the tracked landmark across frames
    xs, ys, zs = zip(*traj_points)
    ax.plot3D(
        xs, ys, zs, color="#1D3557", lw=2.6, linestyle="--",
        marker="o", markersize=5,
        markerfacecolor="#1D3557", markeredgecolor="white", markeredgewidth=1.0
    )

    ax.set_title("(c) Temporal edges across consecutive frames", fontsize=12, color="#1D3557", pad=10)
    ax.set_box_aspect([1, 1.15, 1.2])
    ax.view_init(elev=22, azim=-55)
    ax.set_axis_off()

    plt.tight_layout()

    # figure lives in visualization; mirror to project-root overleaf_upload/figures
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    out_dirs = [
        script_dir,
        os.path.join(project_root, "paper", "overleaf_upload", "figures"),
    ]
    for out_dir in out_dirs:
        os.makedirs(out_dir, exist_ok=True)
        for ext in ["pdf", "png", "svg"]:
            path = os.path.join(out_dir, f"figure2_face_graph.{ext}")
            fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
            print(f"Saved: {path}")

    plt.close(fig)


if __name__ == "__main__":
    main()
