"""Regression tests for the landmark coordinate layout (data.landmark_layout).

D-Vlog visual features use the OpenFace BLOCK layout [x_0..x_67, y_0..y_67].
The historical interleaved mis-parse (reshape(T, 68, 2)) silently corrupted
every coordinate; these tests pin the correct parsing end-to-end:
flat_to_coords / coords_to_flat, build_face_graph node features, and the
augmentation pipeline's write-back. CPU-only, no dataset required.
"""

import numpy as np
import pytest

from data.augmentation import augment_landmark_sequence
from data.landmark_layout import (
    assert_landmark_layout,
    check_visual_array,
    coords_to_flat,
    flat_to_coords,
)
from models.graph_utils import build_face_graph


def _known_seq(T=8):
    """(T, 136) sequence with landmark i at x=i, y=100+i in every frame."""
    frame = np.concatenate([np.arange(68), 100.0 + np.arange(68)])
    return np.tile(frame, (T, 1)).astype(np.float32)


def _upright_face_coords(T=4):
    """(T, 68, 2) synthetic upright face (image coords, y axis down)."""
    coords = np.zeros((T, 68, 2), dtype=np.float32)
    coords[:, 17:22, 0] = -1.0   # left brow to the left
    coords[:, 22:27, 0] = 1.0    # right brow to the right
    coords[:, 36:48, 1] = -1.0   # eyes above
    coords[:, 48:60, 1] = 1.0    # mouth below
    return coords


def test_flat_to_coords_restores_points_exactly():
    coords = flat_to_coords(_known_seq())
    assert coords.shape == (8, 68, 2)
    # point-by-point: landmark i must be (i, 100+i), NOT the interleaved
    # mis-parse which would give (2i, 2i+1).
    for i in [0, 1, 33, 67]:
        assert coords[0, i, 0] == pytest.approx(float(i))
        assert coords[0, i, 1] == pytest.approx(100.0 + i)
    # every frame identical
    assert np.allclose(coords, coords[0:1])


def test_coords_to_flat_is_exact_inverse():
    rng = np.random.default_rng(0)
    seq = rng.normal(size=(16, 136)).astype(np.float32)
    assert np.allclose(coords_to_flat(flat_to_coords(seq)), seq)


def test_flat_to_coords_rejects_bad_shape():
    with pytest.raises(AssertionError):
        flat_to_coords(np.zeros((4, 135), dtype=np.float32))
    with pytest.raises(AssertionError):
        flat_to_coords(np.zeros((4, 68, 2), dtype=np.float32))


def test_build_face_graph_uses_true_xy_pairs():
    """Node features of the face graph must be the true (x, y) pairs."""
    seq = _known_seq(T=8)
    data = build_face_graph(seq, label=1, num_frames=8, use_velocity=False)
    x = data.x.numpy().reshape(8, 68, 2)
    expected = flat_to_coords(seq)
    assert np.allclose(x, expected), (
        "build_face_graph node coords do not match the OpenFace block layout"
    )


def test_build_face_graph_velocity_from_true_coords():
    """Velocity [dx, dy] must be the temporal diff of the true coords."""
    T = 4
    xs = np.tile(np.arange(68), (T, 1)).astype(np.float32)
    # y block increases by a constant step per frame -> dy == step
    ys = 100.0 + np.arange(68)[None, :] + 3.0 * np.arange(T)[:, None]
    seq = np.concatenate([xs, ys], axis=1).astype(np.float32)

    data = build_face_graph(seq, label=0, num_frames=T, use_velocity=True)
    feats = data.x.numpy().reshape(T, 68, 4)
    assert np.allclose(feats[0, :, 2:], 0.0)          # zero-padded first frame
    assert np.allclose(feats[1:, :, 2], 0.0)           # dx = 0
    assert np.allclose(feats[1:, :, 3], 3.0)           # dy = 3


def test_augmentation_writes_back_block_layout():
    """A pure translation must land on the x-block / y-block respectively."""
    seq = _known_seq(T=4)
    out = augment_landmark_sequence(
        seq,
        rotation_range=(0.0, 0.0),
        scale_range=(1.0, 1.0),
        translate_range=(0.5, 0.5),   # fixed translation (tx=ty=0.5)
        noise_std=0.0,
        temporal_mask_prob=0.0,
    )
    assert out.shape == seq.shape
    coords = flat_to_coords(out)
    # x block shifted by +0.5, y block shifted by +0.5 — under the historical
    # interleaved write-back the offsets would land on the wrong storage slots.
    assert np.allclose(coords[..., 0], np.arange(68) + 0.5, atol=1e-5)
    assert np.allclose(coords[..., 1], 100.0 + np.arange(68) + 0.5, atol=1e-5)


def test_augmentation_zero_params_is_identity():
    seq = _known_seq(T=4)
    out = augment_landmark_sequence(
        seq,
        rotation_range=(0.0, 0.0),
        scale_range=(1.0, 1.0),
        translate_range=(0.0, 0.0),
        noise_std=0.0,
        temporal_mask_prob=0.0,
    )
    assert np.allclose(out, seq)


def test_sanity_accepts_upright_face():
    assert_landmark_layout(_upright_face_coords())


def test_sanity_rejects_flipped_face():
    flipped = _upright_face_coords().copy()
    flipped[..., 1] *= -1.0  # mouth above eyes
    with pytest.raises(AssertionError):
        assert_landmark_layout(flipped)


def test_sanity_rejects_interleaved_misparse():
    """An interleaved-style (x, y) stream must fail the layout check.

    Simulates the historical bug: a stream where even slots are x-like and
    odd slots are y-like, so reshape(T, 68, 2) looks tempting — the parsed
    "regions" no longer satisfy anatomical ordering.
    """
    rng = np.random.default_rng(1)
    bad = rng.normal(size=(8, 136)).astype(np.float32)
    with pytest.raises(AssertionError):
        check_visual_array(bad.reshape(1, 8, 136))


def test_check_visual_array_accepts_block_layout():
    coords = _upright_face_coords(T=32)
    visual = coords_to_flat(coords).reshape(2, 16, 136)
    check_visual_array(visual)


def test_check_visual_array_rejects_bad_shape():
    with pytest.raises(AssertionError):
        check_visual_array(np.zeros((2, 16, 135), dtype=np.float32))
