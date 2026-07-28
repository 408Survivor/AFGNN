"""Generate Figure 2(a) blur version: heavily blurred real face with high-contrast graph overlay."""
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
blur_kernel = (71, 71)
margin_ratio = 0.08

# Region colors (BGR) for white/blue-tint background.
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


def draw_landmarks_on_image(img, landmarks_crop, radius=8, edge_thickness=3):
    """Draw high-contrast landmarks and edges on the given image."""
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

    for idx, (x, y) in enumerate(landmarks_crop):
        color = get_region_color(idx)
        cv2.circle(img, (int(x), int(y)), radius, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(img, (int(x), int(y)), radius, (255, 255, 255), 2, lineType=cv2.LINE_AA)

    return img


# ---------------------------------------------------------------------------
# dlib setup
# ---------------------------------------------------------------------------
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# ---------------------------------------------------------------------------
# Extract one frame and detect landmarks
# ---------------------------------------------------------------------------
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
start_frame = int(start_sec * fps)

cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
ret, frame = cap.read()
cap.release()

if not ret:
    raise RuntimeError("Failed to read frame")

landmarks = detect_landmarks(frame)
if landmarks is None:
    raise RuntimeError("No face detected")

# ---------------------------------------------------------------------------
# Crop, blur, then overlay high-contrast graph
# ---------------------------------------------------------------------------
crop, (ox, oy) = crop_to_landmarks(frame, landmarks, margin_ratio=margin_ratio)
lms_crop = landmarks - np.array([ox, oy])

# Heavy blur to obscure identity.
blurred = cv2.GaussianBlur(crop, blur_kernel, 0)

# Optional: slightly desaturate / tint to make overlay pop.
blurred = cv2.addWeighted(blurred, 0.85, np.full_like(blurred, 240), 0.15, 0)

# Draw high-contrast landmarks.
result = draw_landmarks_on_image(blurred, lms_crop, radius=8, edge_thickness=3)

# Save.
out_path = os.path.join(out_dir, "figure2a_blur_highcontrast.png")
cv2.imwrite(out_path, result)
print(f"Saved: {out_path}")

# Save PDF.
import matplotlib.pyplot as plt
out_path_pdf = os.path.join(out_dir, "figure2a_blur_highcontrast.pdf")
fig, ax = plt.subplots(figsize=(result.shape[1] / 150, result.shape[0] / 150), dpi=150)
ax.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
ax.axis("off")
fig.patch.set_facecolor("white")
plt.tight_layout(pad=0)
plt.savefig(out_path_pdf, bbox_inches="tight", pad_inches=0.02, facecolor="white")
plt.close()
print(f"Saved: {out_path_pdf}")
