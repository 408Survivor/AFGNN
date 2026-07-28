"""Smoke tests for the config->model factory (utils.builders.build_model).

These are the regression net for the P0-1 builders refactor: they assert that
every experiment/ablation config can still be turned into a valid AFGNN model,
and that the model structure stays consistent with its modality flags.
No dataset is required (CPU only).
"""

import glob
import os

import pytest
import torch.nn as nn

from utils.builders import build_model, load_config

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIGS = sorted(
    glob.glob(os.path.join(REPO, "experiments", "configs", "*.yaml"))
    + glob.glob(os.path.join(REPO, "ablation", "configs", "*.yaml"))
)


def test_configs_discovered():
    assert CONFIGS, "no experiment/ablation configs found under */configs/*.yaml"


@pytest.mark.parametrize("config_path", CONFIGS, ids=lambda p: os.path.basename(p))
def test_build_model_from_every_config(config_path):
    """Every shipped config builds an nn.Module ending in a single-logit head."""
    cfg = load_config(config_path)
    model = build_model(cfg)
    assert isinstance(model, nn.Module)
    last_linear = [m for m in model.classifier.modules() if isinstance(m, nn.Linear)][-1]
    assert last_linear.out_features == 1


@pytest.mark.parametrize("config_path", CONFIGS, ids=lambda p: os.path.basename(p))
def test_model_structure_consistent_with_flags(config_path):
    """Submodules must match the use_face / use_audio flags; fusion only when both."""
    cfg = load_config(config_path)
    model = build_model(cfg)
    assert (model.face_gnn is not None) == model.use_face
    assert (model.audio_gnn is not None) == model.use_audio
    assert (model.fusion is not None) == (model.use_face and model.use_audio)
