"""Compose Figure 2: (a) raw face, (b) topology, (c) temporal edges."""
import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
base_dir = "paper/figure2_real_face"
img_a = os.path.join(base_dir, "figure2a_real_face_raw.png")
img_b = os.path.join(base_dir, "figure2b_real_face_topology.png")
img_c = os.path.join(base_dir, "figure2c_temporal_edges_real_v3.png")
out_png = os.path.join(base_dir, "figure2_composite.png")
out_pdf = os.path.join(base_dir, "figure2_composite.pdf")

# ---------------------------------------------------------------------------
# Load images
# ---------------------------------------------------------------------------
a = Image.open(img_a).convert("RGB")
b = Image.open(img_b).convert("RGB")
c = Image.open(img_c).convert("RGB")

a_np = np.array(a)
b_np = np.array(b)
c_np = np.array(c)

# ---------------------------------------------------------------------------
# Layout parameters
# ---------------------------------------------------------------------------
DPI = 300
fig_width = 14  # inches
bottom_height_ratio = 0.42  # fraction of total height for bottom row
margin = 0.04
label_size = 16
title_size = 12

# Determine sizes.
# Top row: a and b side by side, same height.
# Bottom row: c spans full width.
top_h = 4.0  # inches for top row
bottom_h = 4.0  # inches for bottom row, same as top row
gap = 0.25

fig_h = top_h + bottom_h + gap + 0.8  # extra for labels

fig = plt.figure(figsize=(fig_width, fig_h), dpi=DPI, facecolor="black")

# ---------------------------------------------------------------------------
# Top row: (a) and (b)
# ---------------------------------------------------------------------------
# Available width for each top image.
top_avail_w = (fig_width - 3 * margin) / 2

# Scale images to fit top_h while keeping aspect ratio.
def fit_to_height(img, target_h_inches, dpi):
    h, w = img.shape[:2]
    target_h_px = target_h_inches * dpi
    scale = target_h_px / h
    new_w = w * scale
    new_h = target_h_px
    return scale, new_w / dpi, new_h / dpi

scale_a, aw, ah = fit_to_height(a_np, top_h, DPI)
scale_b, bw, bh = fit_to_height(b_np, top_h, DPI)

# Center a and b in their halves.
left_a = margin + (top_avail_w - aw) / 2
left_b = 2 * margin + top_avail_w + (top_avail_w - bw) / 2
top_y = fig_h - margin - top_h

ax_a = fig.add_axes([left_a / fig_width, top_y / fig_h, aw / fig_width, ah / fig_h])
ax_a.imshow(a_np)
ax_a.axis("off")

ax_b = fig.add_axes([left_b / fig_width, top_y / fig_h, bw / fig_width, bh / fig_h])
ax_b.imshow(b_np)
ax_b.axis("off")

# ---------------------------------------------------------------------------
# Bottom row: (c)
# ---------------------------------------------------------------------------
# Scale c to fit fig_width - 2*margin.
c_h, c_w = c_np.shape[:2]
avail_w = fig_width - 2 * margin
scale_c = avail_w / (c_w / DPI)
cw_inch = avail_w
c_h_inch = (c_h * scale_c) / DPI

left_c = margin
bottom_y = top_y - gap - c_h_inch

ax_c = fig.add_axes([left_c / fig_width, bottom_y / fig_h, cw_inch / fig_width, c_h_inch / fig_h])
ax_c.imshow(c_np)
ax_c.axis("off")

# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------
fig.text(left_a / fig_width - 0.02, (top_y + ah) / fig_h - 0.02, "(a)",
         color="white", fontsize=label_size, fontweight="bold", va="top", ha="left")
fig.text(left_b / fig_width - 0.02, (top_y + bh) / fig_h - 0.02, "(b)",
         color="white", fontsize=label_size, fontweight="bold", va="top", ha="left")
fig.text(left_c / fig_width - 0.02, (bottom_y + c_h_inch) / fig_h - 0.02, "(c)",
         color="white", fontsize=label_size, fontweight="bold", va="top", ha="left")

# Optional sub-captions (can be removed if not needed).
cap_a = "Real face"
cap_b = "AU-informed facial topology"
cap_c = "Temporal edges across consecutive frames"

fig.text(left_a / fig_width + aw / (2 * fig_width), top_y / fig_h - 0.015, cap_a,
         color="white", fontsize=title_size, ha="center", va="top")
fig.text(left_b / fig_width + bw / (2 * fig_width), top_y / fig_h - 0.015, cap_b,
         color="white", fontsize=title_size, ha="center", va="top")
fig.text(left_c / fig_width + cw_inch / (2 * fig_width), bottom_y / fig_h - 0.015, cap_c,
         color="white", fontsize=title_size, ha="center", va="top")

plt.savefig(out_png, dpi=DPI, facecolor="black", bbox_inches="tight", pad_inches=0.1)
plt.savefig(out_pdf, dpi=DPI, facecolor="black", bbox_inches="tight", pad_inches=0.1)
plt.close()

print(f"Saved: {out_png}")
print(f"Saved: {out_pdf}")
