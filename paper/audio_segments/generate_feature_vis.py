"""Generate visual representations of 25-D acoustic descriptors."""
import os

import matplotlib.pyplot as plt
import numpy as np

DPI = 300
np.random.seed(42)

# A plausible-looking 25-D acoustic descriptor (e.g., openSMILE eGeMAPS-like).
# Values are standardized, so roughly N(0, 1) with some structure.
features = np.array([
    0.85, -0.32, 0.14, 0.66, -0.55,
    0.21, 0.91, -0.12, -0.78, 0.43,
    0.05, 0.62, -0.41, 0.29, -0.63,
    0.74, -0.19, 0.38, -0.84, 0.17,
    0.52, -0.47, 0.10, 0.68, -0.25,
])

# Normalize to [0, 1] for colormap.
fmin, fmax = features.min(), features.max()
features_norm = (features - fmin) / (fmax - fmin)

out_dir = "paper/audio_segments"
os.makedirs(out_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# Plot 1: 5x5 grid heatmap (fits nicely inside a square node)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(1.2, 1.2), dpi=DPI)
grid = features_norm.reshape(5, 5)
im = ax.imshow(grid, cmap="YlGnBu", aspect="equal", vmin=0, vmax=1)
ax.axis("off")
for i in range(5):
    for j in range(5):
        val = grid[i, j]
        # No text inside tiny cells; keep it clean.
        pass
fig.patch.set_alpha(0)
plt.tight_layout(pad=0)
plt.savefig(os.path.join(out_dir, "feature_25d_grid.png"), bbox_inches="tight", pad_inches=0.02, transparent=True)
plt.savefig(os.path.join(out_dir, "feature_25d_grid.pdf"), bbox_inches="tight", pad_inches=0.02, transparent=True)
plt.close()

# ---------------------------------------------------------------------------
# Plot 2: 1x25 horizontal strip (shows dimensionality clearly)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(3.5, 0.6), dpi=DPI)
strip = features_norm.reshape(1, -1)
ax.imshow(strip, cmap="YlGnBu", aspect="auto", vmin=0, vmax=1)
ax.axis("off")
fig.patch.set_alpha(0)
plt.tight_layout(pad=0)
plt.savefig(os.path.join(out_dir, "feature_25d_strip.png"), bbox_inches="tight", pad_inches=0.02, transparent=True)
plt.savefig(os.path.join(out_dir, "feature_25d_strip.pdf"), bbox_inches="tight", pad_inches=0.02, transparent=True)
plt.close()

# ---------------------------------------------------------------------------
# Plot 3: small bar chart (alternative representation)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(3.0, 1.0), dpi=DPI)
x = np.arange(1, 26)
colors = plt.cm.YlGnBu(features_norm)
ax.bar(x, features, color=colors, edgecolor="white", linewidth=0.3)
ax.set_xticks([])
ax.set_yticks([])
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.spines["left"].set_visible(False)
fig.patch.set_alpha(0)
plt.tight_layout(pad=0)
plt.savefig(os.path.join(out_dir, "feature_25d_bars.png"), bbox_inches="tight", pad_inches=0.02, transparent=True)
plt.savefig(os.path.join(out_dir, "feature_25d_bars.pdf"), bbox_inches="tight", pad_inches=0.02, transparent=True)
plt.close()

# ---------------------------------------------------------------------------
# Plot 4: T_a x 25 feature matrix (multiple frames)
# ---------------------------------------------------------------------------
T = 16
np.random.seed(7)
matrix = np.random.randn(T, 25)
# Add some temporal smoothness.
for i in range(1, T):
    matrix[i] = 0.6 * matrix[i] + 0.4 * matrix[i - 1]
matrix_norm = (matrix - matrix.min()) / (matrix.max() - matrix.min())

fig, ax = plt.subplots(figsize=(4.5, 2.2), dpi=DPI)
ax.imshow(matrix_norm, cmap="YlGnBu", aspect="auto", vmin=0, vmax=1)
ax.set_xlabel("Descriptor dimension", fontsize=9, color="#333333")
ax.set_ylabel("Frame", fontsize=9, color="#333333")
ax.set_xticks([])
ax.set_yticks([])
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.patch.set_facecolor("white")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "feature_matrix_Tx25.png"), dpi=DPI, bbox_inches="tight", facecolor="white", pad_inches=0.05)
plt.savefig(os.path.join(out_dir, "feature_matrix_Tx25.pdf"), bbox_inches="tight", facecolor="white", pad_inches=0.05)
plt.close()

print("Saved 25-D feature visualizations in", out_dir)
for name in ["feature_25d_grid", "feature_25d_strip", "feature_25d_bars", "feature_matrix_Tx25"]:
    print(f"  {name}.png/pdf")
