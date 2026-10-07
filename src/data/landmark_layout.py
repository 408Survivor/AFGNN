"""Single source of truth for the D-Vlog landmark coordinate layout.

D-Vlog official visual features (OpenFace convention) store each frame's 136
values as a BLOCK layout: ``[x_0..x_67, y_0..y_67]`` — first all 68 x's, then
all 68 y's. They are NOT interleaved ``(x, y)`` pairs.

Historical bug (found 2026-07-28, confirmed in the HiFAG project): the old
code parsed frames with ``reshape(T, 68, 2)``, i.e. as interleaved pairs, so
every landmark ended up with two x's or two y's of adjacent storage slots
instead of a true (x, y) coordinate. All parsing must go through
``flat_to_coords`` / ``coords_to_flat`` below — never reshape directly.

The data is normalized per frame (each frame's 68 x's and 68 y's are
independently standardized to mean 0 / std 1), but anatomical ordering still
holds on aggregate, which ``assert_landmark_layout`` exploits as a load-time
sanity check.
"""

from typing import Optional, Tuple

import numpy as np

NUM_LANDMARKS = 68
FLAT_DIM = 2 * NUM_LANDMARKS  # 136

# Landmark index ranges used by the anatomical sanity check
# (Dlib / OpenFace 68-point convention, same as graph_utils.LANDMARK_GROUPS_68).
_LEFT_EYE = range(36, 42)
_RIGHT_EYE = range(42, 48)
_OUTER_MOUTH = range(48, 60)
_LEFT_BROW = range(17, 22)
_RIGHT_BROW = range(22, 27)


def flat_to_coords(seq: np.ndarray) -> np.ndarray:
    """Convert a raw D-Vlog visual sequence to true (x, y) landmark coords.

    Args:
        seq: (T, 136) raw landmark sequence, OpenFace block layout
            ``[x_0..x_67, y_0..y_67]``.

    Returns:
        coords: (T, 68, 2) with ``[..., 0] = x``, ``[..., 1] = y``.
    """
    seq = np.asarray(seq)
    assert seq.ndim == 2 and seq.shape[1] == FLAT_DIM, (
        f"Expected (T, {FLAT_DIM}) OpenFace block-layout sequence, got {seq.shape}"
    )
    return np.stack([seq[:, :NUM_LANDMARKS], seq[:, NUM_LANDMARKS:]], axis=-1)


def coords_to_flat(coords: np.ndarray) -> np.ndarray:
    """Inverse of ``flat_to_coords``: (T, 68, 2) -> (T, 136) block layout."""
    coords = np.asarray(coords)
    assert coords.ndim == 3 and coords.shape[1:] == (NUM_LANDMARKS, 2), (
        f"Expected (T, {NUM_LANDMARKS}, 2) coords, got {coords.shape}"
    )
    return np.concatenate([coords[..., 0], coords[..., 1]], axis=-1)


def layout_margins(coords: np.ndarray) -> Tuple[float, float]:
    """Anatomical ordering margins of a landmark sequence, in data units.

    Args:
        coords: (..., 68, 2) landmark coordinates (any leading batch dims).

    Returns:
        (vertical_margin, horizontal_margin) where
        vertical_margin = mean(mouth_y) - mean(eye_y) — positive when the eyes
            sit above the mouth (image coordinates, y axis down);
        horizontal_margin = mean(right_brow_x) - mean(left_brow_x) — positive
            when landmark groups 17-21 lie to the left of 22-26.

    Both margins are clearly positive for the correct block layout on D-Vlog
    (measured on train: +0.91 / +1.12) and collapse/flip sign under the
    historical interleaved mis-parse (eye-above-mouth holds for only ~1% of
    frames), so they discriminate the two layouts decisively.
    """
    coords = np.asarray(coords, dtype=np.float64)
    pts = coords.reshape(-1, NUM_LANDMARKS, 2)
    eye_y = np.concatenate([pts[:, _LEFT_EYE, 1], pts[:, _RIGHT_EYE, 1]], axis=1).mean()
    mouth_y = pts[:, _OUTER_MOUTH, 1].mean()
    lbrow_x = pts[:, _LEFT_BROW, 0].mean()
    rbrow_x = pts[:, _RIGHT_BROW, 0].mean()
    return float(mouth_y - eye_y), float(rbrow_x - lbrow_x)


def assert_landmark_layout(coords: np.ndarray, name: str = "visual") -> None:
    """Raise AssertionError unless the anatomical layout margins are positive.

    Call this once when loading a visual feature array. It catches the
    historical interleaved-vs-block mis-parse (and gross transpositions)
    before any graph is built from garbage coordinates.
    """
    vertical, horizontal = layout_margins(coords)
    assert vertical > 0 and horizontal > 0, (
        f"{name}: landmark layout sanity check failed "
        f"(vertical margin {vertical:+.4f}, horizontal margin {horizontal:+.4f}; "
        f"both must be > 0). The data does not look like OpenFace block layout "
        f"[x_0..x_67, y_0..y_67] — check for an interleaved (x, y) mis-parse."
    )


def check_visual_array(
    visual: np.ndarray,
    name: str = "visual",
    max_samples: int = 64,
    frame_stride: int = 10,
) -> None:
    """Validate a loaded (N, T, 136) visual array: shape + anatomical layout.

    Checks a strided subset (up to ``max_samples`` sequences, every
    ``frame_stride``-th frame) so the cost is negligible at dataset load.
    """
    visual = np.asarray(visual)
    assert visual.ndim == 3 and visual.shape[2] == FLAT_DIM, (
        f"{name}: expected (N, T, {FLAT_DIM}) visual array, got {visual.shape}"
    )
    subset = visual[:max_samples, ::frame_stride, :]
    coords = flat_to_coords(subset.reshape(-1, FLAT_DIM))
    assert_landmark_layout(coords, name=name)
