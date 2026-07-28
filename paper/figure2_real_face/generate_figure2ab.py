"""Generate Figure 2 (a) and (b) from a real face frame."""
import os

import cv2
import dlib
import numpy as np

# ---------------------------------------------------------------------------
# Paths and config
# ---------------------------------------------------------------------------
video_path = "paper/-5jl7RzMMFU.mp4"
out_dir = "paper/figure2_real_face"
os.makedirs(out_dir, exist_ok=True)

predictor_path = "/home/ltq/miniconda3/envs/DVlog/lib/python3.10/site-packages/face_recognition_models/models/shape_predictor_68_face_landmarks.dat"

start_sec = 30
margin_ratio = 0.08

# ---------------------------------------------------------------------------
# dlib setup
# ---------------------------------------------------------------------------
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# Region colors (BGR).
region_colors = {
    "face_contour": (141, 153, 174),
    "left_eyebrow": (95, 122, 224),
    "right_eyebrow": (95, 122, 224),
    "nose_bridge": (128, 90, 61),
    "nose_bottom": (128, 90, 61),
    "left_eye": (143, 157, 42),
    "right_eye": (143, 157, 42),
    "outer_mouth": (122, 89, 109),
    "inner_mouth": (122, 89, 109),
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
    return (200, 200, 200)


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


# ---------------------------------------------------------------------------
# Extract frame and detect landmarks
# ---------------------------------------------------------------------------
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_idx = int(start_sec * fps)
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
ret, frame = cap.read()
cap.release()

if not ret:
    raise RuntimeError("Failed to read frame")

landmarks = detect_landmarks(frame)
if landmarks is None:
    raise RuntimeError("No face detected")

# Crop to face region.
crop, _ = crop_to_landmarks(frame, landmarks, margin_ratio=margin_ratio)
lms_crop = landmarks - np.array([_.get('x', 0) for _ in [{'x': 0}]])  # placeholder

# Recompute crop offset properly.
h, w = frame.shape[:2]
x_min, x_max = landmarks[:, 0].min(), landmarks[:, 0].max()
y_min, y_max = landmarks[:, 1].min(), landmarks[:, 1].max()
bw, bh = x_max - x_min, y_max - y_min
margin = max(bw, bh) * margin_ratio
ox = max(0, int(x_min - margin))
oy = max(0, int(y_min - margin))
lms_crop = landmarks - np.array([ox, oy])

# ---------------------------------------------------------------------------
# (a) Raw cropped face, nothing drawn.
# ---------------------------------------------------------------------------
out_a = os.path.join(out_dir, "figure2a_real_face_raw.png")
cv2.imwrite(out_a, crop)
print(f"Saved: {out_a}")

# ---------------------------------------------------------------------------
# (b) Cropped face with static topology edges and landmarks.
# ---------------------------------------------------------------------------
crop_b = crop.copy()

# Draw topology edges.
for r_name, indices in regions.items():
    color = region_colors[r_name]
    closed = r_name not in {"face_contour", "left_eyebrow", "right_eyebrow", "nose_bridge"}
    for i in range(len(indices) - 1):
        p1 = tuple(lms_crop[indices[i]].astype(int))
        p2 = tuple(lms_crop[indices[i + 1]].astype(int))
        cv2.line(crop_b, p1, p2, color, 1, lineType=cv2.LINE_AA)
    if closed and len(indices) > 1:
        p1 = tuple(lms_crop[indices[-1]].astype(int))
        p2 = tuple(lms_crop[indices[0]].astype(int))
        cv2.line(crop_b, p1, p2, color, 1, lineType=cv2.LINE_AA)

# Draw landmarks.
for idx, (x, y) in enumerate(lms_crop):
    color = get_region_color(idx)
    cv2.circle(crop_b, (int(x), int(y)), 3, color, -1, lineType=cv2.LINE_AA)
    cv2.circle(crop_b, (int(x), int(y)), 3, (255, 255, 255), 1, lineType=cv2.LINE_AA)

out_b = os.path.join(out_dir, "figure2b_real_face_topology.png")
cv2.imwrite(out_b, crop_b)
print(f"Saved: {out_b}")

# Also save a larger / resized version for easy use.
scale = 1.5
crop_a_large = cv2.resize(crop, (int(crop.shape[1] * scale), int(crop.shape[0] * scale)))
crop_b_large = cv2.resize(crop_b, (int(crop_b.shape[1] * scale), int(crop_b.shape[0] * scale)))
cv2.imwrite(os.path.join(out_dir, "figure2a_real_face_raw_large.png"), crop_a_large)
cv2.imwrite(os.path.join(out_dir, "figure2b_real_face_topology_large.png"), crop_b_large)
print("Saved large versions too.")
