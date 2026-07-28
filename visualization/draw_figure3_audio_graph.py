"""
Generate Figure 3: Audio temporal graph construction and interpretability.

Uses REAL data extracted from a trained model + one depressed D-Vlog sample
(run visualization/extract_figure3_data.py first to produce figure3_data.npz):

    (a) Frame-level acoustic feature heatmap        (real normalized features)
    (b) Bidirectional temporal chain graph          (schematic + real feature callout)
    (c) Audio readout attention weights             (real per-frame attention)

Output: figure3_audio_graph.{pdf,png,svg} in visualization and overleaf_upload/figures.

Run (after extraction):
    conda activate DVlog
    python visualization/draw_figure3_audio_graph.py
"""

import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

DPI = 300

# --- Palette (consistent with Figure 4) ------------------------------------
C_NODE = "#A8C5D6"
C_EDGE = "#3D5A80"
C_ACCENT = "#2A9D8F"   # salient / above-median
C_MUTED = "#9DB4C0"    # below-median
C_TEXT = "#1D3557"
C_CORAL = "#E07A5F"


def load_data():
    here = os.path.dirname(os.path.abspath(__file__))
    npz = os.path.join(here, "figure3_data.npz")
    if not os.path.exists(npz):
        raise FileNotFoundError(
            f"{npz} not found.\nRun the extraction first:\n"
            "  python visualization/extract_figure3_data.py "
            "--config experiments/configs/afgnn_face_enhanced_focal.yaml "
            "--checkpoint experiments/checkpoints/afgnn_face_enhanced_focal_best_seed42.pt"
        )
    d = np.load(npz)
    return d["features"], d["weights"], int(d["label"])


def detect_actual_length(features, tol=1e-6):
    """Return the first frame index that appears to be padded (constant to end)."""
    T = features.shape[0]
    for t in range(T):
        if np.all(np.abs(features[t:] - features[t]) < tol):
            return t
    return T


def main():
    features_raw, weights_raw, label = load_data()   # features (T,25), weights (T,)
    # Truncate trailing padded frames so heatmap/weights show only real audio.
    T_real = detect_actual_length(features_raw)
    features = features_raw[:T_real]
    weights = weights_raw[:T_real]
    T, F = features.shape
    frames = np.arange(T)
    peak = int(weights.argmax())

    fig = plt.figure(figsize=(14, 4.3))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 0.95, 1.15], wspace=0.32)

    # -----------------------------------------------------------------------
    # (a) Real acoustic feature heatmap
    # -----------------------------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    vlim = float(np.percentile(np.abs(features), 98))
    im = ax.imshow(features.T, aspect="auto", cmap="RdBu_r", vmin=-vlim, vmax=vlim,
                   extent=[0, T, 0, F], origin="lower")
    ax.set_xlabel("Time frame", fontsize=10, color=C_TEXT)
    ax.set_ylabel("Acoustic feature dim.", fontsize=10, color=C_TEXT)
    ax.set_title("(a) Frame-level acoustic features", fontsize=11.5, color=C_TEXT, pad=8,
                 fontweight="bold")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Normalized value", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    ax.set_xticks([0, 8, 16, 24, 32][: (T // 8) + 1])
    ax.set_yticks([0, 5, 10, 15, 20, 25])
    # Mark the most salient frame (links to panels b/c).
    ax.axvline(peak + 0.5, color=C_ACCENT, lw=1.6, ls="--", alpha=0.9)

    # -----------------------------------------------------------------------
    # (b) Clean temporal chain graph
    # -----------------------------------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.3, 1.1)
    ax.axis("off")
    ax.set_title("(b) Temporal chain graph", fontsize=11.5, color=C_TEXT,
                 pad=8, fontweight="bold")

    y_line = 0.55
    # Timeline backbone
    ax.plot([1.0, 9.0], [y_line, y_line], color="#B8C5D0", lw=2.5,
            solid_capstyle="round", zorder=1)

    # Nodes: a_t is the peak frame and is highlighted
    nodes = [("a_1", 1.5), ("a_2", 3.0), ("a_t", 5.0), ("a_{t+1}", 7.0), ("a_T", 8.5)]
    node_xs = [x for _, x in nodes]
    for label, x in nodes:
        is_peak = (label == "a_t")
        facecolor = C_ACCENT if is_peak else "white"
        edgecolor = C_ACCENT if is_peak else C_EDGE
        size = 110 if is_peak else 70
        lw = 2.0 if is_peak else 1.4
        ax.scatter(x, y_line, c=facecolor, s=size, zorder=3,
                   edgecolors=edgecolor, lw=lw)
        ax.text(x, y_line + 0.22, f"${label}$", ha="center", va="bottom",
                fontsize=10, color=C_ACCENT if is_peak else C_TEXT,
                fontweight="bold" if is_peak else "normal")

    # Ellipsis for gaps
    ax.text(4.0, y_line, r"$\cdots$", ha="center", va="center", fontsize=12, color=C_TEXT, zorder=3)
    ax.text(6.0, y_line, r"$\cdots$", ha="center", va="center", fontsize=12, color=C_TEXT, zorder=3)

    # Bidirectional edges between all adjacent nodes (representative double arrows)
    edge_pairs = [(1.5, 3.0), (5.0, 7.0), (7.0, 8.5)]
    for x1, x2 in edge_pairs:
        ax.annotate("", xy=(x2 - 0.13, y_line), xytext=(x1 + 0.13, y_line),
                    arrowprops=dict(arrowstyle="<->", color=C_EDGE, lw=1.6, mutation_scale=11))

    # Label
    ax.text(6.0, y_line - 0.28, "bidirectional edges", ha="center", va="top",
            fontsize=8, color=C_EDGE)

    # Bottom caption
    ax.text(5.0, 0.08, "Each node = one frame's 25-d feature vector",
            ha="center", va="top", fontsize=8.5, color=C_EDGE)

    # -----------------------------------------------------------------------
    # (c) Real audio readout attention weights
    # -----------------------------------------------------------------------
    ax = fig.add_subplot(gs[0, 2])
    med = float(np.median(weights))
    colors = [C_ACCENT if w >= med else C_MUTED for w in weights]
    ax.bar(frames, weights, color=colors, edgecolor="white", lw=0.4, zorder=2)
    ax.axhline(med, color="#888888", ls="--", lw=1.0, zorder=1)
    # Highlight the single most salient frame.
    ax.bar([peak], [weights[peak]], color=C_CORAL, edgecolor="white", lw=0.4, zorder=3)
    ax.annotate(f"peak\n(frame {peak})", xy=(peak, weights[peak]),
                xytext=(peak - 4, weights[peak] + 0.18 * (weights.max() - weights.min()) + weights.max() * 0.02),
                ha="center", va="bottom", fontsize=8, color=C_CORAL,
                arrowprops=dict(arrowstyle="->", color=C_CORAL, lw=1.0,
                                connectionstyle="arc3,rad=0.15"))
    ax.set_xlabel("Time frame", fontsize=10, color=C_TEXT)
    ax.set_ylabel("Readout weight", fontsize=10, color=C_TEXT)
    ax.set_title("(c) Audio readout attention", fontsize=11.5, color=C_TEXT, pad=8, fontweight="bold")
    ax.set_xlim(-0.5, T - 0.5)
    ax.set_xticks([0, 8, 16, 24, 32][: (T // 8) + 1])
    ax.set_ylim(0, weights.max() * 1.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8.5, colors=C_TEXT)
    # Explanatory note for readout gate.
    ax.text(0.5 * (T - 1), -0.16 * weights.max() * 1.45,
            "Higher weights indicate more informative acoustic frames",
            ha="center", va="top", fontsize=8, color=C_EDGE)
    handles = [
        mpatches.Patch(color=C_ACCENT, label="Above median"),
        mpatches.Patch(color=C_MUTED, label="Below median"),
        mpatches.Patch(color=C_CORAL, label="Peak frame"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=8, frameon=False)

    fig.subplots_adjust(left=0.06, right=0.98, top=0.86, bottom=0.20, wspace=0.32)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    out_dirs = [script_dir, os.path.join(project_root, "paper", "overleaf_upload", "figures")]
    for out_dir in out_dirs:
        os.makedirs(out_dir, exist_ok=True)
        for ext in ["pdf", "png", "svg"]:
            path = os.path.join(out_dir, f"figure3_audio_graph.{ext}")
            fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
            print(f"Saved: {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
