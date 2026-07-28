"""
Generate Figure 2: Facial graph construction details.

Subfigures:
    (a) Single-frame 68-point facial landmark topology.
    (b) Zoomed eye region showing anatomical edges.
    (c) Spatio-temporal graph: same landmark across 3 frames.

Output:
    - figure2_face_graph.pdf
    - figure2_face_graph.png
    - figure2_face_graph.svg
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

DPI = 300

# ---------------------------------------------------------------------------
# 68 facial landmark coordinates (schematic, front-facing face)
# Indices 0-67, based on dlib/OpenFace convention.
# ---------------------------------------------------------------------------
LANDMARKS_68 = np.array([
    # Face contour (0-16)
    [0.00, 1.80], [0.25, 1.55], [0.55, 1.35], [0.85, 1.15],
    [1.15, 0.95], [1.45, 0.75], [1.75, 0.55], [2.00, 0.30],
    [2.25, 0.05], [2.50, -0.25], [2.75, -0.55], [3.00, -0.80],
    [3.20, -1.05], [3.35, -1.30], [3.45, -1.55], [3.50, -1.80],
    [3.52, -2.05],
    # Left eyebrow (17-21)
    [0.60, 0.50], [1.05, 0.65], [1.55, 0.70], [2.05, 0.65], [2.50, 0.50],
    # Right eyebrow (22-26)
    [3.00, 0.50], [3.45, 0.65], [3.95, 0.70], [4.45, 0.65], [4.90, 0.50],
    # Nose bridge (27-30)
    [3.25, 0.35], [3.25, -0.05], [3.25, -0.40], [3.25, -0.75],
    # Nose bottom (31-35)
    [2.75, -0.95], [3.00, -1.05], [3.25, -1.10], [3.50, -1.05], [3.75, -0.95],
    # Left eye (36-41)
    [1.10, -0.05], [1.40, -0.15], [1.80, -0.15], [2.10, -0.05],
    [1.85, 0.10], [1.45, 0.10],
    # Right eye (42-47)
    [4.40, -0.05], [4.70, -0.15], [5.10, -0.15], [5.40, -0.05],
    [5.15, 0.10], [4.75, 0.10],
    # Outer mouth (48-59)
    [2.60, -1.75], [2.85, -1.90], [3.15, -1.95], [3.45, -1.95],
    [3.75, -1.90], [4.00, -1.75], [3.80, -1.55], [3.55, -1.50],
    [3.25, -1.50], [3.00, -1.55], [2.80, -1.65], [2.70, -1.70],
    # Inner mouth (60-67)
    [2.95, -1.70], [3.15, -1.78], [3.45, -1.78], [3.70, -1.70],
    [3.55, -1.62], [3.30, -1.60], [3.05, -1.62], [2.95, -1.68],
], dtype=float)

# Scale and center
LANDMARKS_68[:, 0] = (LANDMARKS_68[:, 0] - 2.5) * 1.2
LANDMARKS_68[:, 1] = (LANDMARKS_68[:, 1] + 0.6) * 1.2


# ---------------------------------------------------------------------------
# Edge topology
# ---------------------------------------------------------------------------
def chain_edges(indices, closed=False):
    edges = []
    for i in range(len(indices) - 1):
        edges.append((indices[i], indices[i + 1]))
    if closed and len(indices) > 1:
        edges.append((indices[-1], indices[0]))
    return edges


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


def build_static_edges():
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


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
def plot_landmarks_and_edges(ax, coords, edges, edge_color="#555555", node_color="#1D3557",
                              node_size=25, line_width=1.0, alpha=1.0, labels=None,
                              highlight_indices=None, highlight_color="#E07A5F"):
    """Plot landmarks and edges on the given axes."""
    # Draw edges
    for i, j in edges:
        ax.plot([coords[i, 0], coords[j, 0]], [coords[i, 1], coords[j, 1]],
                color=edge_color, lw=line_width, alpha=alpha, zorder=1)

    # Draw nodes
    ax.scatter(coords[:, 0], coords[:, 1], c=node_color, s=node_size, zorder=2, edgecolors="white", lw=0.5)

    if highlight_indices is not None:
        ax.scatter(coords[highlight_indices, 0], coords[highlight_indices, 1],
                   c=highlight_color, s=node_size * 2, zorder=3, edgecolors="white", lw=0.5)

    if labels is not None:
        for idx, (x, y) in enumerate(coords):
            if isinstance(labels, dict) and idx in labels:
                ax.text(x + 0.03, y + 0.03, str(labels[idx]), fontsize=5, color="#333333", zorder=4)
            elif isinstance(labels, (list, tuple)) and idx < len(labels):
                ax.text(x + 0.03, y + 0.03, str(labels[idx]), fontsize=5, color="#333333", zorder=4)

    ax.set_aspect("equal")
    ax.axis("off")


def add_region_legend(ax):
    """Add a small legend for facial regions."""
    handles = []
    shown = set()
    for name, color in REGION_COLORS.items():
        label = name.replace("_", " ").title().replace("Left ", "").replace("Right ", "")
        if label not in shown:
            handles.append(mpatches.Patch(color=color, label=label))
            shown.add(label)
    ax.legend(handles=handles, loc="lower right", fontsize=7, frameon=False)


def main():
    coords = LANDMARKS_68
    edges = build_static_edges()

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # -----------------------------------------------------------------------
    # (a) Full 68-point topology with region coloring
    # -----------------------------------------------------------------------
    ax = axes[0]
    for region_name, indices in REGIONS.items():
        region_edges = chain_edges(indices, closed=(region_name not in [
            "face_contour", "left_eyebrow", "right_eyebrow", "nose_bridge"
        ]))
        plot_landmarks_and_edges(
            ax, coords, region_edges,
            edge_color=REGION_COLORS[region_name],
            node_color="#1D3557",
            node_size=30,
            line_width=1.3,
        )
    ax.set_title("(a) 68-point facial landmark topology", fontsize=11, color="#1D3557", pad=10)
    add_region_legend(ax)

    # -----------------------------------------------------------------------
    # (b) Zoomed eye region
    # -----------------------------------------------------------------------
    ax = axes[1]
    eye_indices = REGIONS["left_eye"]
    eye_edges = chain_edges(eye_indices, closed=True)
    plot_landmarks_and_edges(
        ax, coords, eye_edges,
        edge_color="#2A9D8F",
        node_color="#2A9D8F",
        node_size=80,
        line_width=2.0,
        labels={i: str(i) for i in eye_indices},
    )

    # Add eyebrow and nose bridge connections for context
    context_edges = chain_edges(REGIONS["left_eyebrow"], closed=False) + \
                    chain_edges(REGIONS["nose_bridge"][:3], closed=False)
    plot_landmarks_and_edges(
        ax, coords, context_edges,
        edge_color="#BBBBBB",
        node_color="#888888",
        node_size=25,
        line_width=1.0,
        alpha=0.5,
    )
    ax.set_xlim(coords[eye_indices, 0].min() - 0.4, coords[eye_indices, 0].max() + 0.4)
    ax.set_ylim(coords[eye_indices, 1].min() - 0.3, coords[eye_indices, 1].max() + 0.5)
    ax.set_title("(b) Intra-frame anatomical edges (left eye)", fontsize=11, color="#1D3557", pad=10)

    # -----------------------------------------------------------------------
    # (c) Spatio-temporal: same landmark across 3 frames
    # -----------------------------------------------------------------------
    ax = axes[2]
    landmark_idx = 39  # left eye outer corner
    n_frames = 3
    frame_offsets = [-0.6, 0.0, 0.6]
    frame_colors = ["#A8DADC", "#457B9D", "#1D3557"]

    for t, offset in enumerate(frame_offsets):
        frame_coords = coords.copy()
        frame_coords[:, 1] += offset
        # Plot all landmarks faintly
        ax.scatter(frame_coords[:, 0], frame_coords[:, 1], c=frame_colors[t], s=15, alpha=0.4, zorder=1)
        # Highlight the tracked landmark
        ax.scatter(frame_coords[landmark_idx, 0], frame_coords[landmark_idx, 1],
                   c="#E07A5F", s=100, zorder=3, edgecolors="white", lw=0.5)
        ax.text(frame_coords[landmark_idx, 0] + 0.15, frame_coords[landmark_idx, 1],
                f"$t_{t+1}$", fontsize=9, color="#E07A5F", va="center")

    # Temporal edges connecting the tracked landmark across frames
    traj_x = [coords[landmark_idx, 0]] * n_frames
    traj_y = [coords[landmark_idx, 1] + off for off in frame_offsets]
    ax.plot(traj_x, traj_y, color="#E07A5F", lw=2.5, linestyle="--", marker="o", markersize=6, zorder=2)

    ax.set_title("(c) Temporal edges across consecutive frames", fontsize=11, color="#1D3557", pad=10)
    ax.set_aspect("equal")
    ax.axis("off")

    plt.tight_layout()

    out_dir = os.path.dirname(os.path.abspath(__file__))
    for ext in ["pdf", "png", "svg"]:
        path = os.path.join(out_dir, f"figure2_face_graph.{ext}")
        fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
        print(f"Saved: {path}")

    plt.close(fig)


if __name__ == "__main__":
    main()
