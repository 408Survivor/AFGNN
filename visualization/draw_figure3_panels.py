"""
Generate Figure 3 panels as separate files for flexible assembly.

Panels:
    figure3a_features.{pdf,png,svg}  : Frame-level acoustic feature heatmap
    figure3b_feature_bar.{pdf,png,svg}: Vertical 25-d feature strip of the peak frame
    figure3c_readout.{pdf,png,svg}   : Audio readout attention weights

Output: visualization/ and paper/overleaf_upload/figures/

Run:
    source /home/ltq/miniconda3/etc/profile.d/conda.sh && conda activate DVlog
    python visualization/draw_figure3_panels.py
"""

import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

DPI = 300

# --- Palette (consistent with Figure 3 combined plot) ----------------------
C_NODE = "#A8C5D6"
C_EDGE = "#3D5A80"
C_ACCENT = "#2A9D8F"
C_MUTED = "#9DB4C0"
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


def save_panel(fig, stem, out_dirs):
    for out_dir in out_dirs:
        os.makedirs(out_dir, exist_ok=True)
        for ext in ["pdf", "png", "svg"]:
            path = os.path.join(out_dir, f"{stem}.{ext}")
            fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
            print(f"Saved: {path}")
    plt.close(fig)


def build_panel_a(features, peak):
    T, F = features.shape
    fig, ax = plt.subplots(figsize=(5.2, 4.3))
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
    ax.axvline(peak + 0.5, color=C_ACCENT, lw=1.6, ls="--", alpha=0.9)
    ax.tick_params(labelsize=8.5, colors=C_TEXT)
    return fig


def build_panel_b(features, peak):
    """Vertical 25-d feature strip for the peak frame."""
    vec = features[peak]               # (25,)
    F = len(vec)
    vlim = float(np.percentile(np.abs(features), 98))

    fig, ax = plt.subplots(figsize=(1.6, 4.3))
    # Single-column heatmap: dims 0..F on y, one frame on x.
    strip = vec.reshape(-1, 1)
    im = ax.imshow(strip, aspect="auto", cmap="RdBu_r", vmin=-vlim, vmax=vlim,
                   extent=[0, 1, 0, F], origin="lower")
    ax.set_xlabel("← frame", fontsize=9, color=C_TEXT)
    ax.set_ylabel("Acoustic feature dim.", fontsize=10, color=C_TEXT)
    ax.set_title("(b) 25-d feature vector\n" + r"($a_t$, peak frame)",
                 fontsize=11.5, color=C_TEXT, pad=8, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([0, 5, 10, 15, 20, 25])
    ax.set_xlim(-0.3, 1.3)
    # Add a border around the strip.
    ax.add_patch(mpatches.Rectangle((0, 0), 1, F, fill=False,
                                    edgecolor=C_EDGE, lw=1.5, zorder=3))
    ax.tick_params(labelsize=8.5, colors=C_TEXT)
    return fig


def build_panel_c(weights, peak):
    T = len(weights)
    frames = np.arange(T)
    fig, ax = plt.subplots(figsize=(5.2, 4.3))
    med = float(np.median(weights))
    colors = [C_ACCENT if w >= med else C_MUTED for w in weights]
    ax.bar(frames, weights, color=colors, edgecolor="white", lw=0.4, zorder=2)
    ax.axhline(med, color="#888888", ls="--", lw=1.0, zorder=1)
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
    return fig


def main():
    features_raw, weights_raw, label = load_data()
    T_real = detect_actual_length(features_raw)
    features = features_raw[:T_real]
    weights = weights_raw[:T_real]
    peak = int(weights.argmax())

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    out_dirs = [script_dir, os.path.join(project_root, "paper", "overleaf_upload", "figures")]

    save_panel(build_panel_a(features, peak), "figure3a_features", out_dirs)
    save_panel(build_panel_b(features, peak), "figure3b_feature_bar", out_dirs)
    save_panel(build_panel_c(weights, peak), "figure3c_readout", out_dirs)


if __name__ == "__main__":
    main()
