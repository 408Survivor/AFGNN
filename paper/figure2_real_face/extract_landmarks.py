"""Extract frames and detect 68 facial landmarks using dlib."""
import os
import sys

import cv2
import dlib
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
video_path = "paper/-5jl7RzMMFU.mp4"
out_dir = "paper/figure2_real_face"
os.makedirs(out_dir, exist_ok=True)

# dlib 68-point landmark model (bundled with face_recognition_models in DVlog env).
predictor_path = "/home/ltq/miniconda3/envs/DVlog/lib/python3.10/site-packages/face_recognition_models/models/shape_predictor_68_face_landmarks.dat"

# ---------------------------------------------------------------------------
# Extract frames at specified timestamps (seconds).
# ---------------------------------------------------------------------------
timestamps = [30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 540, 600, 660]

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Cannot open video: {video_path}")
    sys.exit(1)

fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Video FPS: {fps}")

# ---------------------------------------------------------------------------
# Detect landmarks
# ---------------------------------------------------------------------------
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

region_colors = {
    "face_contour": (141, 153, 174),   # BGR
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


def draw_landmarks(img, landmarks, draw_edges=True, radius=2, thickness=1):
    """Draw 68 landmarks on image. landmarks: np.array (68, 2)."""
    img = img.copy()
    h, w = img.shape[:2]

    # Draw topology edges first.
    if draw_edges:
        for r_name, indices in regions.items():
            color = region_colors[r_name]
            closed = r_name not in {"face_contour", "left_eyebrow", "right_eyebrow", "nose_bridge"}
            for i in range(len(indices) - 1):
                p1 = tuple(landmarks[indices[i]].astype(int))
                p2 = tuple(landmarks[indices[i + 1]].astype(int))
                cv2.line(img, p1, p2, color, thickness, lineType=cv2.LINE_AA)
            if closed and len(indices) > 1:
                p1 = tuple(landmarks[indices[-1]].astype(int))
                p2 = tuple(landmarks[indices[0]].astype(int))
                cv2.line(img, p1, p2, color, thickness, lineType=cv2.LINE_AA)

    # Draw points on top.
    for idx, (x, y) in enumerate(landmarks):
        color = get_region_color(idx)
        cv2.circle(img, (int(x), int(y)), radius, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(img, (int(x), int(y)), radius, (255, 255, 255), 1, lineType=cv2.LINE_AA)
    return img


# ---------------------------------------------------------------------------
# Process frames
# ---------------------------------------------------------------------------
for t_sec in timestamps:
    frame_idx = int(t_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        print(f"Failed to read frame at {t_sec}s")
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 1)
    print(f"Time {t_sec}s: found {len(faces)} face(s)")

    if len(faces) == 0:
        continue

    # Use the largest face.
    face = max(faces, key=lambda f: (f.right() - f.left()) * (f.bottom() - f.top()))
    shape = predictor(gray, face)
    landmarks = np.array([[p.x, p.y] for p in shape.parts()])

    # Save raw frame.
    raw_path = os.path.join(out_dir, f"frame_{t_sec}s_raw.png")
    cv2.imwrite(raw_path, frame)

    # Save landmarks overlay.
    overlay = draw_landmarks(frame, landmarks, draw_edges=True, radius=4, thickness=2)
    overlay_path = os.path.join(out_dir, f"frame_{t_sec}s_landmarks.png")
    cv2.imwrite(overlay_path, overlay)

    # Save landmark coordinates.
    coords_path = os.path.join(out_dir, f"frame_{t_sec}s_landmarks.npy")
    np.save(coords_path, landmarks)

    print(f"  Saved: {overlay_path}")
    break  # Stop after first successful detection for quick check.

cap.release()
print("Done.")
