"""Generate a clean region-color palette reference figure."""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

DPI = 300

REGIONS = [
    ("Face contour", "#8D99AE"),
    ("Eyebrow", "#E07A5F"),
    ("Nose", "#3D5A80"),
    ("Eye", "#2A9D8F"),
    ("Mouth", "#6D597A"),
]

fig, ax = plt.subplots(figsize=(5.0, 1.1), dpi=DPI)
fig.patch.set_facecolor("white")
fig.patch.set_alpha(1.0)
ax.set_xlim(0, 5)
ax.set_ylim(0, 1)
ax.axis("off")

box_w = 0.85
box_h = 0.55
gap = 0.15
start_x = 0.15

for i, (name, color) in enumerate(REGIONS):
    x = start_x + i * (box_w + gap)
    # Color block
    rect = mpatches.FancyBboxPatch(
        (x, 0.32),
        box_w,
        box_h,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        facecolor=color,
        edgecolor="white",
        linewidth=1.5,
    )
    ax.add_patch(rect)
    # Label
    ax.text(
        x + box_w / 2,
        0.18,
        name,
        ha="center",
        va="top",
        fontsize=10,
        color="#1D3557",
        fontweight="medium",
    )

plt.tight_layout(pad=0)

script_dir = os.path.dirname(os.path.abspath(__file__))
out_dir = script_dir
os.makedirs(out_dir, exist_ok=True)
for ext in ["pdf", "png", "svg"]:
    path = os.path.join(out_dir, f"region_color_palette.{ext}")
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white", pad_inches=0.05)
    print(f"Saved: {path}")
plt.close(fig)
