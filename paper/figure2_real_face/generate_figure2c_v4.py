"""Generate Figure 2(c) v4: white/transparent background, 3 landmarks, bidirectional arcs."""
import os

import cv2
import dlib
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Paths and config
# ---------------------------------------------------------------------------
video_path = "paper/-5jl7RzMMFU.mp4"
out_dir = "paper/figure2_real_face"
os.makedirs(out_dir, exist_ok=True)

predictor_path = "/home/ltq/miniconda3/envs/DVlog/lib/python3.10/site-packages/face_recognition_models/models/shape_predictor_68_face_landmarks.dat"

start_sec = 30
num_frames = 3
fps_target = 8
margin_ratio = 0.08
arc_height = 70

# Darker arc color for white/transparent background.
arc_color = (0, 120, 180)  # BGR: dark cyan/blue
arc_thickness = 2

# 3 landmarks: chin tip, left eye outer corner, right mouth corner.
tracked_indices = [8, 36, 54]

# Background: "white" or "transparent".
BACKGROUND = "white"  # change to "transparent" if needed

# ---------------------------------------------------------------------------
# dlib setup
# ---------------------------------------------------------------------------
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# Region colors (BGR), slightly adjusted for white background.
region_colors = {
    "face_contour": (100, 120, 150),
    "left_eyebrow": (60, 100, 200),
    "right_eyebrow": (60, 100, 200),
    "nose_bridge": (100, 80, 50),
    "nose_bottom": (100, 80, 50),
    "left_eye": (120, 160, 30),
    "right_eye": (120, 160, 30),
    "outer_mouth": (100, 70, 100),
    "inner_mouth": (100, 70, 100),
}

regions = {
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


def get_region_color(idx):
    for r_name, indices in regions.items():
        if idx in indices:
            return region_colors[r_name]
    return (150, 150, 150)


def detect_landmarks(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 1)
    if len(faces) == 0:
        return None
    face = max(faces, key=lambda f: (f.right() - f.left()) * (f.bottom() - f.top()))
    shape = predictor(gray, face)
    return np.array([[p.x, p.y] for p in shape.parts()])


def crop_to_landmarks(frame, landmarks, margin_ratio=0.08):
    h, w = frame.shape[:2]
    x_min, x_max = landmarks[:, 0].min(), landmarks[:, 0].max()
    y_min, y_max = landmarks[:, 1].min(), landmarks[:, 1].max()
    bw, bh = x_max - x_min, y_max - y_min
    margin = max(bw, bh) * margin_ratio

    x1 = max(0, int(x_min - margin))
    y1 = max(0, int(y_min - margin))
    x2 = min(w, int(x_max + margin))
    y2 = min(h, int(y_max + margin))

    return frame[y1:y2, x1:x2], (x1, y1)


def draw_landmarks_on_crop(crop, landmarks_crop, tracked_indices=None, radius=4, edge_thickness=1):
    img = crop.copy()

    # Draw topology edges.
    for r_name, indices in regions.items():
        color = region_colors[r_name]
        closed = r_name not in {"face_contour", "left_eyebrow", "right_eyebrow", "nose_bridge"}
        for i in range(len(indices) - 1):
            p1 = tuple(landmarks_crop[indices[i]].astype(int))
            p2 = tuple(landmarks_crop[indices[i + 1]].astype(int))
            cv2.line(img, p1, p2, color, edge_thickness, lineType=cv2.LINE_AA)
        if closed and len(indices) > 1:
            p1 = tuple(landmarks_crop[indices[-1]].astype(int))
            p2 = tuple(landmarks_crop[indices[0]].astype(int))
            cv2.line(img, p1, p2, color, edge_thickness, lineType=cv2.LINE_AA)

    # Draw landmarks.
    for idx, (x, y) in enumerate(landmarks_crop):
        color = get_region_color(idx)
        cv2.circle(img, (int(x), int(y)), radius, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(img, (int(x), int(y)), radius, (255, 255, 255), 1, lineType=cv2.LINE_AA)

    # Highlight tracked landmarks.
    if tracked_indices is not None:
        for idx in tracked_indices:
            x, y = landmarks_crop[idx]
            cv2.circle(img, (int(x), int(y)), radius + 3, arc_color, -1, lineType=cv2.LINE_AA)
            cv2.circle(img, (int(x), int(y)), radius + 3, (255, 255, 255), 1, lineType=cv2.LINE_AA)

    return img


def bezier_curve(p0, p1, p2, num=50):
    t = np.linspace(0, 1, num).reshape(-1, 1)
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2


def draw_arrowhead(img, p_tip, p_prev, color, thickness=2, arr_len=10, arr_ang=0.5):
    tangent = np.array(p_tip, dtype=float) - np.array(p_prev, dtype=float)
    tangent = tangent / (np.linalg.norm(tangent) + 1e-8)
    normal = np.array([-tangent[1], tangent[0]])

    a1 = p_tip - arr_len * (np.cos(arr_ang) * tangent - np.sin(arr_ang) * normal)
    a2 = p_tip - arr_len * (np.cos(arr_ang) * tangent + np.sin(arr_ang) * normal)
    cv2.line(img, tuple(p_tip.astype(int)), tuple(a1.astype(int)), color, thickness, lineType=cv2.LINE_AA)
    cv2.line(img, tuple(p_tip.astype(int)), tuple(a2.astype(int)), color, thickness, lineType=cv2.LINE_AA)


def draw_bidirectional_arc(img, pt1, pt2, color, thickness=2, arc_height=50, num=60):
    p0 = np.array(pt1, dtype=float)
    p2 = np.array(pt2, dtype=float)
    mid = (p0 + p2) / 2
    p1 = mid - np.array([0, arc_height], dtype=float)

    pts = bezier_curve(p0, p1, p2, num).astype(int)
    for i in range(len(pts) - 1):
        cv2.line(img, tuple(pts[i]), tuple(pts[i + 1]), color, thickness, lineType=cv2.LINE_AA)

    draw_arrowhead(img, p2, p1, color, thickness)
    draw_arrowhead(img, p0, p1, color, thickness)


# ---------------------------------------------------------------------------
# Extract frames and detect landmarks
# ---------------------------------------------------------------------------
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
start_frame = int(start_sec * fps)

frames = []
landmarks_list = []
for k in range(num_frames):
    frame_idx = start_frame + k * fps_target
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        print(f"Failed to read frame at index {frame_idx}")
        continue
    lms = detect_landmarks(frame)
    if lms is None:
        print(f"No face at frame {frame_idx}")
        continue
    frames.append(frame)
    landmarks_list.append(lms)

cap.release()

# ---------------------------------------------------------------------------
# Crop and draw landmarks
# ---------------------------------------------------------------------------
cropped_imgs = []
cropped_landmarks = []

for frame, lms in zip(frames, landmarks_list):
    crop, (ox, oy) = crop_to_landmarks(frame, lms, margin_ratio=margin_ratio)
    lms_crop = lms - np.array([ox, oy])
    crop_img = draw_landmarks_on_crop(crop, lms_crop, tracked_indices=tracked_indices, radius=4, edge_thickness=1)
    cropped_imgs.append(crop_img)
    cropped_landmarks.append(lms_crop)

# ---------------------------------------------------------------------------
# Composite
# ---------------------------------------------------------------------------
target_height = 599
resized = []
scales = []
for img in cropped_imgs:
    h, w = img.shape[:2]
    scale = target_height / h
    resized.append(cv2.resize(img, (int(w * scale), target_height)))
    scales.append(scale)

margin = 100
margin_y = 80
total_width = sum(img.shape[1] for img in resized) + margin * (len(resized) + 1)
total_height = target_height + 2 * margin_y + 60  # extra for arcs and labels

if BACKGROUND == "transparent":
    composite = np.zeros((total_height, total_width, 4), dtype=np.uint8)
    bg_color = (0, 0, 0, 0)
    text_color = (0, 0, 0, 255)
else:
    composite = np.full((total_height, total_width, 3), 255, dtype=np.uint8)
    bg_color = (255, 255, 255)
    text_color = (0, 0, 0)

x_offsets = []
x = margin
for img in resized:
    y = margin_y + 30  # leave some space above for arcs
    if BACKGROUND == "transparent":
        # Convert BGR to BGRA and paste.
        bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        composite[y:y + img.shape[0], x:x + img.shape[1]] = bgra
    else:
        composite[y:y + img.shape[0], x:x + img.shape[1]] = img
    x_offsets.append(x + img.shape[1] // 2)
    x += img.shape[1] + margin

# Draw arcs between tracked landmarks of consecutive frames.
for k in range(len(resized) - 1):
    lms1 = cropped_landmarks[k] * scales[k]
    lms2 = cropped_landmarks[k + 1] * scales[k + 1]

    x_base1 = x_offsets[k] - resized[k].shape[1] // 2
    x_base2 = x_offsets[k + 1] - resized[k + 1].shape[1] // 2
    y_base = margin_y + 30

    for idx in tracked_indices:
        x1 = int(lms1[idx, 0]) + x_base1
        y1 = int(lms1[idx, 1]) + y_base
        x2 = int(lms2[idx, 0]) + x_base2
        y2 = int(lms2[idx, 1]) + y_base

        draw_bidirectional_arc(composite, (x1, y1), (x2, y2), arc_color, arc_thickness,
                               arc_height=arc_height, num=60)

# Labels.
font = cv2.FONT_HERSHEY_SIMPLEX
for k, xo in enumerate(x_offsets):
    label = "t" if k == 0 else f"t+{k}"
    text_size = cv2.getTextSize(label, font, 1.2, 2)[0]
    if BACKGROUND == "transparent":
        # Draw black text with alpha.
        cv2.putText(composite, label, (xo - text_size[0] // 2, total_height - margin_y + 40),
                    font, 1.2, text_color, 2, cv2.LINE_AA)
    else:
        cv2.putText(composite, label, (xo - text_size[0] // 2, total_height - margin_y + 40),
                    font, 1.2, text_color, 2, cv2.LINE_AA)

# Save PNG.
out_path = os.path.join(out_dir, "figure2c_temporal_edges_real_v4.png")
if BACKGROUND == "transparent":
    cv2.imwrite(out_path, composite)
else:
    cv2.imwrite(out_path, composite)
print(f"Saved: {out_path}")

# Save PDF via matplotlib.
out_path_pdf = os.path.join(out_dir, "figure2c_temporal_edges_real_v4.pdf")
fig, ax = plt.subplots(figsize=(total_width / 150, total_height / 150), dpi=150)
if BACKGROUND == "transparent":
    ax.imshow(cv2.cvtColor(composite, cv2.COLOR_BGRA2RGBA))
else:
    ax.imshow(cv2.cvtColor(composite, cv2.COLOR_BGR2RGB))
ax.axis("off")
fig.patch.set_alpha(0 if BACKGROUND == "transparent" else 1)
fig.patch.set_facecolor("white" if BACKGROUND == "white" else "none")
plt.tight_layout(pad=0)
plt.savefig(out_path_pdf, bbox_inches="tight", pad_inches=0.02,
            facecolor="white" if BACKGROUND == "white" else "none",
            transparent=(BACKGROUND == "transparent"))
plt.close()
print(f"Saved: {out_path_pdf}")
