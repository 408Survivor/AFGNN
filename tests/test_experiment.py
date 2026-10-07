"""Smoke tests for utils.experiment (exp_N auto-numbering + run metadata).

CPU-only, no dataset. Uses tmp_path so no real experiments/ dir is touched.
"""

import os

import torch.nn as nn
import yaml

from utils.experiment import (
    get_latest_exp_dir,
    get_next_exp_dir,
    infer_exp_dir_from_checkpoint,
    save_run_info,
)


def test_get_next_exp_dir_increments(tmp_path):
    exp1 = get_next_exp_dir(str(tmp_path))
    assert exp1.name == "exp_1" and exp1.is_dir()
    exp2 = get_next_exp_dir(str(tmp_path))
    assert exp2.name == "exp_2" and exp2.is_dir()


def test_get_next_exp_dir_ignores_non_exp_entries(tmp_path):
    (tmp_path / "configs").mkdir()
    (tmp_path / "exp_7").mkdir()
    (tmp_path / "exp_x").mkdir()
    (tmp_path / "exp_100.txt").write_text("not a dir")
    assert get_next_exp_dir(str(tmp_path)).name == "exp_8"


def test_get_next_exp_dir_fills_gap_with_max_plus_one(tmp_path):
    # A deleted exp_2 must not cause reuse of the number: always max + 1.
    (tmp_path / "exp_1").mkdir()
    (tmp_path / "exp_3").mkdir()
    assert get_next_exp_dir(str(tmp_path)).name == "exp_4"


def test_get_latest_exp_dir(tmp_path):
    assert get_latest_exp_dir(str(tmp_path / "nonexistent")) is None
    assert get_latest_exp_dir(str(tmp_path)) is None
    (tmp_path / "exp_1").mkdir()
    (tmp_path / "exp_5").mkdir()
    (tmp_path / "exp_3").mkdir()
    assert get_latest_exp_dir(str(tmp_path)).name == "exp_5"


def test_infer_exp_dir_from_checkpoint(tmp_path):
    exp = tmp_path / "exp_9"
    exp.mkdir()
    ckpt = exp / "best_seed42.pt"
    ckpt.write_bytes(b"w")
    assert infer_exp_dir_from_checkpoint(str(ckpt)) == exp.resolve()
    assert infer_exp_dir_from_checkpoint(str(tmp_path / "other" / "best.pt")) is None


def test_save_run_info_writes_all_artifacts(tmp_path):
    exp = tmp_path / "exp_1"
    exp.mkdir()
    cfg = {"_config_path": "experiments/configs/demo.yaml", "training": {"lr": 0.001}}
    model = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 1))

    save_run_info(exp, cfg, model, command="python src/train.py --config demo.yaml")

    with open(exp / "config.yaml") as f:
        saved = yaml.safe_load(f)
    assert saved["training"]["lr"] == 0.001

    summary = (exp / "model_summary.txt").read_text()
    assert "Trainable parameters: 49" in summary

    info = (exp / "run_info.txt").read_text()
    assert "python src/train.py --config demo.yaml" in info
    assert "experiments/configs/demo.yaml" in info
    assert os.path.isfile(exp / "model_summary.txt")
