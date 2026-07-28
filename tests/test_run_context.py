"""Smoke tests for P0-3: per-run output directory (utils.run_context).

CPU-only, no dataset. Uses log=False so pytest's stdout capture is untouched.
"""

import json
import os

from utils.run_context import resolve_expid, start_run


def test_resolve_expid_strips_prefix_suffix_and_adds_seed():
    cfg = {"training": {"checkpoint_path": "experiments/checkpoints/afgnn_face_enhanced_focal_best.pt"}}
    assert resolve_expid(cfg) == "face_enhanced_focal"
    assert resolve_expid(cfg, seed=42) == "face_enhanced_focal_seed42"


def test_start_run_snapshots_config_and_finalize_writes_manifest(tmp_path):
    cfg = {
        "training": {"checkpoint_path": "experiments/checkpoints/afgnn_demo_best.pt"},
        "model": {"use_face": True},
    }
    # a real file to be symlinked as best.pt
    ckpt = tmp_path / "afgnn_demo_best.pt"
    ckpt.write_bytes(b"weights")

    run = start_run(cfg, kind="train", base=str(tmp_path / "outputs"), log=False)

    # run dir + config snapshot exist
    assert os.path.isdir(run.dir)
    assert os.path.isfile(run.path("config.yaml"))

    history = [{"epoch": 1, "f1": 0.5}, {"epoch": 2, "f1": 0.6}]
    run.finalize(checkpoint_path=str(ckpt), history=history, metrics={"f1": 0.6})

    # history.json + run.json written
    with open(run.path("history.json")) as f:
        assert json.load(f) == history
    with open(run.path("run.json")) as f:
        manifest = json.load(f)
    assert manifest["kind"] == "train"
    assert manifest["metrics"]["f1"] == 0.6

    # best.pt is a symlink to the canonical checkpoint (recommended default)
    link = run.path("best.pt")
    assert os.path.islink(link)
    assert os.path.realpath(link) == os.path.realpath(str(ckpt))


def test_start_run_without_checkpoint_is_ok(tmp_path):
    # eval-style run: no checkpoint symlink, just manifest + metrics.
    cfg = {"training": {"checkpoint_path": "x/afgnn_demo_best.pt"}}
    run = start_run(cfg, kind="eval", base=str(tmp_path / "outputs"), log=False)
    run.finalize(metrics={"f1": 0.7}, extra={"split": "test"})
    with open(run.path("run.json")) as f:
        manifest = json.load(f)
    assert manifest["kind"] == "eval"
    assert manifest["split"] == "test"
    assert not os.path.exists(run.path("best.pt"))
