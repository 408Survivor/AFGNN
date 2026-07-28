"""
Per-run output directory + provenance (P0-3, additive variant).

Each train/eval run gets its own ``outputs/<expid>_<timestamp>/`` folder holding:
  - config.yaml : snapshot of the exact config used
  - run.log     : full stdout/stderr of the run
  - history.json: training curve (train runs only)
  - run.json    : manifest (kind, timestamps, argv, checkpoint, metrics)
  - best.pt     : symlink to the canonical checkpoint (train runs only)

This is purely additive: the canonical checkpoint still lives at
``cfg["training"]["checkpoint_path"]`` so every existing reader (test.py,
seed/ensemble scripts) keeps working unchanged. ``outputs/`` is a disposable
provenance layer, not a source of truth.
"""

import json
import os
import sys
from datetime import datetime

import numpy as np
import yaml


def resolve_expid(cfg, seed=None):
    """Derive an experiment id from the config's checkpoint_path.

    Mirrors the GUIDE convention: strip the ``afgnn_`` prefix and ``_best``
    suffix from the checkpoint filename, then append ``_seed{N}`` if given.
    """
    ckpt = cfg.get("training", {}).get("checkpoint_path", "model")
    stem = os.path.splitext(os.path.basename(ckpt))[0]
    if stem.startswith("afgnn_"):
        stem = stem[len("afgnn_"):]
    if stem.endswith("_best"):
        stem = stem[: -len("_best")]
    if seed is not None:
        stem = f"{stem}_seed{seed}"
    return stem or "run"


def _to_jsonable(obj):
    """Recursively convert numpy scalars so json.dump accepts the object."""
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


class _TeeStream:
    """A file-like object that writes to both the original stream and a file."""

    def __init__(self, stream, file):
        self._stream = stream
        self._file = file

    def write(self, data):
        self._stream.write(data)
        self._file.write(data)

    def flush(self):
        self._stream.flush()
        self._file.flush()


class _Tee:
    """Mirror stdout/stderr into a log file until closed."""

    def __init__(self, path):
        self._file = open(path, "w", buffering=1)
        self._orig_out = sys.stdout
        self._orig_err = sys.stderr
        sys.stdout = _TeeStream(self._orig_out, self._file)
        sys.stderr = _TeeStream(self._orig_err, self._file)

    def close(self):
        sys.stdout = self._orig_out
        sys.stderr = self._orig_err
        self._file.close()


class RunContext:
    """Handle to a single run's output directory."""

    def __init__(self, run_dir, kind, tee=None):
        self.dir = run_dir
        self.kind = kind
        self._tee = tee
        self._start = datetime.now()

    def path(self, name):
        return os.path.join(self.dir, name)

    def finalize(self, checkpoint_path=None, history=None, metrics=None, extra=None):
        """Write history/manifest, symlink the checkpoint, and stop logging."""
        manifest = {
            "kind": self.kind,
            "started": self._start.isoformat(),
            "finished": datetime.now().isoformat(),
            "argv": sys.argv,
            "checkpoint": checkpoint_path,
        }
        if metrics is not None:
            manifest["metrics"] = {k: float(v) for k, v in metrics.items()}
        if extra:
            manifest.update(extra)

        if history is not None:
            with open(self.path("history.json"), "w") as f:
                json.dump(_to_jsonable(history), f, indent=2)

        if checkpoint_path and os.path.exists(checkpoint_path):
            link = self.path("best.pt")
            try:
                if os.path.islink(link) or os.path.exists(link):
                    os.remove(link)
                os.symlink(os.path.abspath(checkpoint_path), link)
                manifest["checkpoint_link"] = "symlink"
            except OSError:
                manifest["checkpoint_link"] = "failed"

        with open(self.path("run.json"), "w") as f:
            json.dump(manifest, f, indent=2)

        if self._tee is not None:
            self._tee.close()
            self._tee = None
        return self.dir


def start_run(cfg, kind="train", expid=None, seed=None, base="outputs", log=True):
    """Create ``outputs/<expid>_<timestamp>/``, snapshot the config, tee logs.

    Args:
        cfg: parsed config dict (snapshotted to config.yaml).
        kind: "train" / "eval" / "train_combined" ... (recorded in run.json).
        expid: override the derived experiment id.
        seed: appended to the expid when given.
        base: parent directory for run folders.
        log: when True, mirror stdout/stderr into run.log until finalize().
    """
    if expid is None:
        expid = resolve_expid(cfg, seed=seed)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base, f"{expid}_{ts}")
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    tee = _Tee(os.path.join(run_dir, "run.log")) if log else None
    print(f"[Run] {kind} run dir: {run_dir}")
    return RunContext(run_dir, kind, tee=tee)
