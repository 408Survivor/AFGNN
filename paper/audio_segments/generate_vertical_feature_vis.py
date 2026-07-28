"""Generate vertical-strip heatmaps for a 25-D acoustic descriptor."""
import os

import matplotlib.pyplot as plt
import numpy as np

DPI = 300
np.random.seed(42)

# A plausible 25-D acoustic descriptor with both positive and negative values.
features = np.array([
    0.85, -0.32, 0.14, 0.66, -0.55,
    0.21, 0.91, -0.12, -0.78, 0.43,
    0.05, 0.62, -0.41, 0.29, -0.63,
    0.74, -0.19, 0.38, -0.84, 0.17,
    0.52, -0.47, 0.10, 0.68, -0.25,
])

out_dir = "paper/audio_segments"
os.makedirs(out_dir, exist_ok=True)


def save_vertical_strip(features, cmap, filename, add_labels=False, label_every=5):
    """Save a 25x1 vertical heatmap strip."""
    fig, ax = plt.subplots(figsize=(0.9, 3.5) if not add_labels else (1.4, 3.5), dpi=DPI)
    strip = features.reshape(-1, 1)  # 25 rows, 1 column
    vmax = np.max(np.abs(features))

    im = ax.imshow(strip, cmap=cmap, aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")

    if add_labels:
        for i in range(0, len(features), label_every):
            ax.text(
                0.5,
                i,
                f"d{i+1}",
                ha="left",
                va="center",
                fontsize=7,
                color="#333333",
                transform=ax.transData,
            )
        # Shift image left a bit to make room for labels.
        ax.set_xlim(-0.6, 0.5)

    fig.patch.set_alpha(0)
    plt.tight_layout(pad=0)
    plt.savefig(os.path.join(out_dir, f"{filename}.png"), bbox_inches="tight", pad_inches=0.02, transparent=True)
    plt.savefig(os.path.join(out_dir, f"{filename}.pdf"), bbox_inches="tight", pad_inches=0.02, transparent=True)
    plt.close()
    print(f"Saved: {filename}.png/pdf")


# More vibrant / varied colormaps.
save_vertical_strip(features, "turbo", "feature_25d_vertical_turbo")
save_vertical_strip(features, "plasma", "feature_25d_vertical_plasma")
save_vertical_strip(features, "coolwarm", "feature_25d_vertical_coolwarm")
save_vertical_strip(features, "RdYlBu_r", "feature_25d_vertical_rdyibu")
save_vertical_strip(features, "viridis", "feature_25d_vertical_viridis")

# One version with dimension labels (using turbo).
save_vertical_strip(features, "turbo", "feature_25d_vertical_turbo_labeled", add_labels=True)

print("All vertical strips saved in", out_dir)
