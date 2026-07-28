"""Generate Figure 2(c): temporal edges across real face frames."""
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
num_frames = 3
fps_target = 8  # sample interval in frames (e.g., every 8 frames ≈ 1/3 s)

# ---------------------------------------------------------------------------
# dlib setup
# ---------------------------------------------------------------------------
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# ---------------------------------------------------------------------------
# Region colors (BGR for OpenCV) - matching Figure 1 / V5 palette.
# ---------------------------------------------------------------------------
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


def draw_face_with_landmarks(frame, landmarks, tracked_indices=None, radius=3, thickness=1):
    img = frame.copy()

    # Draw topology edges (subtle).
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

    # Draw all landmarks (small).
    for idx, (x, y) in enumerate(landmarks):
        color = get_region_color(idx)
        cv2.circle(img, (int(x), int(y)), radius, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(img, (int(x), int(y)), radius, (255, 255, 255), 1, lineType=cv2.LINE_AA)

    # Highlight tracked landmarks (larger).
    if tracked_indices is not None:
        for idx in tracked_indices:
            x, y = landmarks[idx]
            color = (0, 255, 255)  # cyan highlight
            cv2.circle(img, (int(x), int(y)), radius + 3, color, -1, lineType=cv2.LINE_AA)
            cv2.circle(img, (int(x), int(y)), radius + 3, (255, 255, 255), 1, lineType=cv2.LINE_AA)

    return img


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

if len(frames) < num_frames:
    print(f"Only got {len(frames)} valid frames, need {num_frames}")
    # Continue with what we have or exit.

# ---------------------------------------------------------------------------
# Choose tracked landmarks: pick a few representative ones across face.
# ---------------------------------------------------------------------------
tracked_indices = [8, 36, 45, 48, 54, 33]  # chin, left eye outer, right eye outer, mouth left, mouth right, nose tip

# Draw individual frames.
frame_imgs = []
for frame, lms in zip(frames, landmarks_list):
    img = draw_face_with_landmarks(frame, lms, tracked_indices=tracked_indices, radius=3, thickness=1)
    frame_imgs.append(img)

# Resize to a common height.
target_height = 720
resized = []
for img in frame_imgs:
    h, w = img.shape[:2]
    scale = target_height / h
    resized.append(cv2.resize(img, (int(w * scale), target_height)))

# ---------------------------------------------------------------------------
# Composite: frames side by side with temporal arrows between them.
# ---------------------------------------------------------------------------
margin = 60
total_width = sum(img.shape[1] for img in resized) + margin * (len(resized) + 1)
total_height = target_height + 120
composite = np.zeros((total_height, total_width, 3), dtype=np.uint8)

x_offsets = []
x = margin
for img in resized:
    y = 60
    composite[y:y + img.shape[0], x:x + img.shape[1]] = img
    x_offsets.append(x + img.shape[1] // 2)
    x += img.shape[1] + margin

# Draw temporal arrows between tracked landmarks of consecutive frames.
arrow_color = (0, 255, 255)  # cyan
for k in range(len(resized) - 1):
    lms1 = landmarks_list[k]
    lms2 = landmarks_list[k + 1]
    # Scaling factors because frames were resized.
    scale1 = target_height / frames[k].shape[0]
    scale2 = target_height / frames[k + 1].shape[0]

    for idx in tracked_indices:
        x1 = int(lms1[idx, 0] * scale1) + x_offsets[k] - resized[k].shape[1] // 2
        y1 = int(lms1[idx, 1] * scale1) + 60
        x2 = int(lms2[idx, 0] * scale2) + x_offsets[k + 1] - resized[k + 1].shape[1] // 2
        y2 = int(lms2[idx, 1] * scale2) + 60

        # Dashed line + arrowhead.
        cv2.line(composite, (x1, y1), (x2, y2), arrow_color, 2, lineType=cv2.LINE_AA)
        # Small arrowhead pointing right.
        angle = np.arctan2(y2 - y1, x2 - x1)
        arr_len = 12
        arr_ang = 0.5
        ax = int(x2 - arr_len * np.cos(angle - arr_ang))
        ay = int(y2 - arr_len * np.sin(angle - arr_ang))
        bx = int(x2 - arr_len * np.cos(angle + arr_ang))
        by = int(y2 - arr_len * np.sin(angle + arr_ang))
        cv2.line(composite, (x2, y2), (ax, ay), arrow_color, 2, lineType=cv2.LINE_AA)
        cv2.line(composite, (x2, y2), (bx, by), arrow_color, 2, lineType=cv2.LINE_AA)

# Add labels t, t+1, t+2.
font = cv2.FONT_HERSHEY_SIMPLEX
for k, xo in enumerate(x_offsets):
    label = f"t" if k == 0 else (f"t+{k}" if k > 0 else "t")
    text_size = cv2.getTextSize(label, font, 1.0, 2)[0]
    cv2.putText(composite, label, (xo - text_size[0] // 2, target_height + 100),
                font, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

# Save.
out_path = os.path.join(out_dir, "figure2c_temporal_edges_real.png")
cv2.imwrite(out_path, composite)
print(f"Saved: {out_path}")

# Also save a black-background friendly individual frames version.
