"""
Generate Figure 2 (v3): Facial graph construction - polished layout.

Improvements over v2:
    * Three panels share an identical coordinate frame -> aligned boxes & titles.
    * (a) carries a dashed zoom box around one eye, with leader lines into (b).
    * (b) fills the panel with the zoomed eye + faint eyebrow context.
    * (c) is a clean 2D three-frame strip (light wireframes) with temporal edges
      connecting the same landmarks across consecutive frames (no noisy 3D scatter).
    * Shared region legend at the bottom.

Output: figure2_face_graph.{pdf,png,svg} in visualization and overleaf_upload/figures.

Run:
    conda activate DVlog
    python visualization/draw_figure2_face_graph_v3.py
"""

import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import ConnectionPatch, Rectangle
import numpy as np

DPI = 300
C_NODE = "#1D3557"
C_TEXT = "#1D3557"
C_ACCENT = "#2A9D8F"
FRAME = 2.15

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
    "face_contour": list(range(0, 17)), "left_eyebrow": list(range(17, 22)),
    "right_eyebrow": list(range(22, 27)), "nose_bridge": list(range(27, 31)),
    "nose_bottom": list(range(31, 36)), "left_eye": list(range(36, 42)),
    "right_eye": list(range(42, 48)), "outer_mouth": list(range(48, 60)),
    "inner_mouth": list(range(60, 68)),
}
REGION_COLORS = {
    "face_contour": "#8D99AE", "left_eyebrow": "#E07A5F", "right_eyebrow": "#E07A5F",
    "nose_bridge": "#3D5A80", "nose_bottom": "#3D5A80", "left_eye": "#2A9D8F",
    "right_eye": "#2A9D8F", "outer_mouth": "#6D597A", "inner_mouth": "#6D597A",
}
OPEN_REGIONS = {"face_contour", "left_eyebrow", "right_eyebrow", "nose_bridge"}


def chain_edges(indices, closed=False):
    e = [(indices[i], indices[i + 1]) for i in range(len(indices) - 1)]
    if closed and len(indices) > 1:
        e.append((indices[-1], indices[0]))
    return e


def region_edges(name):
    return chain_edges(REGIONS[name], closed=name not in OPEN_REGIONS)


def transform(coords, center, scale, shift=(0.0, 0.0)):
    return (coords - center) * scale + np.asarray(shift)


def draw_edges(ax, pts, edges, color, lw, alpha=1.0, z=1):
    for i, j in edges:
        ax.plot([pts[i, 0], pts[j, 0]], [pts[i, 1], pts[j, 1]],
                color=color, lw=lw, alpha=alpha, zorder=z, solid_capstyle="round")


def frame_axis(ax):
    ax.set_xlim(-FRAME, FRAME)
    ax.set_ylim(-FRAME, FRAME)
    ax.set_aspect("equal")
    ax.axis("off")


def main():
    coords = LANDMARKS_68
    face_center = coords.mean(axis=0)
    A = transform(coords, face_center, 1.0)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.2))
    fig.patch.set_facecolor("white")
    ax_a, ax_b, ax_c = axes

    # ----- (a) full topology + zoom box on the (image-)right eye (left_eye) -----
    draw_edges(ax_a, A, [e for r in REGIONS for e in region_edges(r)], "#E2E5EC", 0.9, 1.0, 1)
    for r in REGIONS:
        draw_edges(ax_a, A, region_edges(r), REGION_COLORS[r], 1.7, 0.95, 2)
    ax_a.scatter(A[:, 0], A[:, 1], c=C_NODE, s=32, zorder=3, edgecolors="white", linewidths=0.5)

    zoom = REGIONS["left_eye"]
    zx0, zy0 = A[zoom, 0].min(), A[zoom, 1].min()
    zx1, zy1 = A[zoom, 0].max(), A[zoom, 1].max()
    mx, my = 0.22, 0.22
    ax_a.add_patch(Rectangle((zx0 - mx, zy0 - my), (zx1 - zx0) + 2 * mx, (zy1 - zy0) + 2 * my,
                             fill=False, edgecolor=C_ACCENT, lw=1.6, ls="--", zorder=4))
    ax_a.set_title("(a) 68-point landmark topology", fontsize=12, color=C_TEXT, pad=8)
    frame_axis(ax_a)

    # ----- (b) zoomed eye filling the panel (eyebrow as faint context) -----
    sel = REGIONS["left_eye"] + REGIONS["left_eyebrow"]
    sub = coords[sel]
    bc = sub.mean(axis=0)
    half = max(np.ptp(sub[:, 0]), np.ptp(sub[:, 1])) / 2
    s = 1.45 / half
    B = transform(coords, bc, s)
    draw_edges(ax_b, B, region_edges("left_eyebrow"), "#C9CFDA", 1.6, 0.6, 1)
    ax_b.scatter(B[REGIONS["left_eyebrow"], 0], B[REGIONS["left_eyebrow"], 1],
                 c="#C9CFDA", s=28, zorder=1)
    draw_edges(ax_b, B, region_edges("left_eye"), C_ACCENT, 3.0, 0.95, 2)
    ax_b.scatter(B[REGIONS["left_eye"], 0], B[REGIONS["left_eye"], 1], c=C_ACCENT,
                 s=210, zorder=3, edgecolors="white", linewidths=1.4)
    for idx in REGIONS["left_eye"]:
        ax_b.text(B[idx, 0], B[idx, 1] + 0.22, str(idx), fontsize=8.5, color=C_TEXT,
                  ha="center", va="bottom", fontweight="bold", zorder=4)
    ax_b.set_title("(b) Intra-frame anatomical edges (eye)", fontsize=12, color=C_TEXT, pad=8)
    frame_axis(ax_b)

    # leader lines: zoom-box right corners (a) -> enlarged-eye left corners (b)
    eyeB = B[REGIONS["left_eye"]]
    ex0, eby0, eby1 = eyeB[:, 0].min(), eyeB[:, 1].min(), eyeB[:, 1].max()
    for ya, yb in [(zy1 + my, eby1 + 0.15), (zy0 - my, eby0 - 0.15)]:
        fig.add_artist(ConnectionPatch(
            xyA=(zx1 + mx, ya), coordsA=ax_a.transData,
            xyB=(ex0 - 0.25, yb), coordsB=ax_b.transData,
            color="#9AA7B8", lw=1.0, ls="--", alpha=0.85, zorder=0))

    # ----- (c) 2D three-frame strip with temporal edges -----
    scale_c = 0.28
    centers = [-1.50, 0.0, 1.50]
    frame_lbl = ["$t$", "$t{+}1$", "$t{+}2$"]
    tracked = [8, 39, 54]  # chin, eye corner, mouth corner
    track_pts = {idx: [] for idx in tracked}

    for k, cx in enumerate(centers):
        Ck = transform(coords, face_center, scale_c, shift=(cx, 0.18))
        draw_edges(ax_c, Ck, [e for r in REGIONS for e in region_edges(r)], "#C2C9D4", 1.0, 0.9, 1)
        ax_c.scatter(Ck[:, 0], Ck[:, 1], c="#9AA7B8", s=5, zorder=2)
        for idx in tracked:
            ax_c.scatter(Ck[idx, 0], Ck[idx, 1], c=C_ACCENT, s=34, zorder=4,
                         edgecolors="white", linewidths=0.9)
            track_pts[idx].append(Ck[idx])
        ax_c.text(cx, -1.55, frame_lbl[k], fontsize=12, color=C_TEXT, ha="center",
                  va="top", fontweight="bold")

    for idx in tracked:
        pts = track_pts[idx]
        for k in range(len(pts) - 1):
            ax_c.annotate("", xy=pts[k + 1], xytext=pts[k],
                          arrowprops=dict(arrowstyle="-|>", color="#1D3557", lw=1.5,
                                          linestyle=(0, (4, 2)), shrinkA=5, shrinkB=5),
                          zorder=3)
    ax_c.set_title("(c) Temporal edges across frames", fontsize=12, color=C_TEXT, pad=8)
    frame_axis(ax_c)

    # ----- shared region legend at the bottom -----
    seen, handles = set(), []
    label_map = {"face_contour": "Face contour", "left_eyebrow": "Eyebrow",
                 "right_eyebrow": "Eyebrow", "nose_bridge": "Nose", "nose_bottom": "Nose",
                 "left_eye": "Eye", "right_eye": "Eye", "outer_mouth": "Mouth",
                 "inner_mouth": "Mouth"}
    for name, color in REGION_COLORS.items():
        lab = label_map[name]
        if lab not in seen:
            handles.append(mpatches.Patch(color=color, label=lab))
            seen.add(lab)
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, -0.01))

    fig.tight_layout(rect=[0, 0.06, 1, 1])

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    for out_dir in [script_dir, os.path.join(project_root, "paper", "overleaf_upload", "figures")]:
        os.makedirs(out_dir, exist_ok=True)
        for ext in ["pdf", "png", "svg"]:
            path = os.path.join(out_dir, f"figure2_face_graph.{ext}")
            fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
            print(f"Saved: {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
