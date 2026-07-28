"""
Lightweight data augmentation for facial landmark sequences.

All transformations are applied to the raw (T, 68, 2) coordinate tensor before
velocity features are computed in `build_face_graph`.
"""

from typing import Tuple

import numpy as np


def _rotation_matrix(angle_deg: float) -> np.ndarray:
    """Return a 2x2 rotation matrix for the given angle in degrees."""
    angle_rad = np.deg2rad(angle_deg)
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.array([[c, -s], [s, c]], dtype=np.float32)


def random_affine_transform(
    coords: np.ndarray,
    rotation_range: Tuple[float, float] = (-10.0, 10.0),
    scale_range: Tuple[float, float] = (0.95, 1.05),
    translate_range: Tuple[float, float] = (-0.05, 0.05),
) -> np.ndarray:
    """Apply a random global affine transform to a landmark sequence.

    Args:
        coords: array of shape (T, 68, 2).
        rotation_range: random rotation angle range in degrees.
        scale_range: random isotropic scale range.
        translate_range: random x/y translation range.

    Returns:
        Augmented coords of shape (T, 68, 2).
    """
    angle = np.random.uniform(*rotation_range)
    scale = np.random.uniform(*scale_range)
    translation = np.random.uniform(*translate_range, size=2)
    R = _rotation_matrix(angle) * scale

    # Per-frame center to keep the face roughly in place
    centers = coords.mean(axis=1, keepdims=True)  # (T, 1, 2)
    centered = coords - centers
    transformed = centered @ R.T + centers + translation
    return transformed.astype(np.float32)


def add_gaussian_noise(
    coords: np.ndarray,
    noise_std: float = 0.02,
) -> np.ndarray:
    """Add Gaussian noise to landmark coordinates."""
    if noise_std <= 0:
        return coords
    noise = np.random.normal(0.0, noise_std, size=coords.shape).astype(np.float32)
    return coords + noise


def temporal_mask(
    coords: np.ndarray,
    mask_prob: float = 0.2,
    max_mask_ratio: float = 0.15,
) -> np.ndarray:
    """Randomly zero-out a contiguous span of frames to simulate missing detections.

    Args:
        coords: array of shape (T, 68, 2).
        mask_prob: probability of applying a temporal mask.
        max_mask_ratio: maximum ratio of frames to mask.

    Returns:
        Augmented coords of shape (T, 68, 2).
    """
    if mask_prob <= 0 or max_mask_ratio <= 0:
        return coords
    if np.random.rand() > mask_prob:
        return coords

    T = coords.shape[0]
    max_len = max(1, int(T * max_mask_ratio))
    length = np.random.randint(1, max_len + 1)
    start = np.random.randint(0, T - length + 1)

    aug = coords.copy()
    aug[start : start + length] = 0.0
    return aug


def augment_landmark_sequence(
    visual_seq: np.ndarray,
    rotation_range: Tuple[float, float] = (-10.0, 10.0),
    scale_range: Tuple[float, float] = (0.95, 1.05),
    translate_range: Tuple[float, float] = (-0.05, 0.05),
    noise_std: float = 0.02,
    temporal_mask_prob: float = 0.2,
    temporal_mask_max_ratio: float = 0.15,
) -> np.ndarray:
    """Apply a stochastic augmentation pipeline to a landmark sequence.

    Args:
        visual_seq: np.ndarray of shape (T, 136).

    Returns:
        Augmented visual_seq of shape (T, 136).
    """
    T = visual_seq.shape[0]
    coords = visual_seq.reshape(T, 68, 2).copy()

    # Order: affine -> noise -> temporal mask
    coords = random_affine_transform(
        coords,
        rotation_range=rotation_range,
        scale_range=scale_range,
        translate_range=translate_range,
    )
    coords = add_gaussian_noise(coords, noise_std=noise_std)
    coords = temporal_mask(
        coords,
        mask_prob=temporal_mask_prob,
        max_mask_ratio=temporal_mask_max_ratio,
    )

    return coords.reshape(T, 136).astype(np.float32)
