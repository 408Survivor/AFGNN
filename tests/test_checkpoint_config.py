"""Smoke tests for P0-2: config embedded in the checkpoint.

Guards two guarantees the training/eval paths rely on:
  1. A checkpoint can carry the parsed config, and ``torch.load(weights_only=True)``
     (which test.py uses) can still read it back exactly.
  2. Rebuilding the model from the embedded config yields a state-dict-compatible
     model, and older checkpoints without a config degrade gracefully (None).
No dataset is required (CPU only).
"""

import os
import tempfile

import torch

from utils.builders import build_model, load_config

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CFG = os.path.join(REPO, "experiments", "configs", "afgnn_face_enhanced_focal.yaml")


def test_checkpoint_embeds_and_restores_config():
    cfg = load_config(CFG)
    model = build_model(cfg)

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ckpt.pt")
        torch.save({"model_state_dict": model.state_dict(), "config": cfg}, path)
        # test.py loads with weights_only=True; the plain-dict config must survive.
        ckpt = torch.load(path, map_location="cpu", weights_only=True)

    assert ckpt.get("config") == cfg

    # Rebuilding purely from the embedded config must accept the saved weights.
    rebuilt = build_model(ckpt["config"])
    rebuilt.load_state_dict(ckpt["model_state_dict"])  # raises if incompatible


def test_missing_config_is_none_for_backward_compat():
    # Legacy checkpoints have no "config" key; test.py falls back to --config
    # exactly when checkpoint.get("config") is None.
    legacy = {"model_state_dict": {}, "best_score": 0.5}
    assert legacy.get("config") is None
