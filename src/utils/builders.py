"""
Central factory helpers for building AFGNN models and D-Vlog data loaders
from a parsed YAML config.

Extracted so every train / test / eval script constructs models and loaders
through one place instead of duplicating the (large) keyword-argument plumbing.
Adding a new model or data hyper-parameter now means editing one function here,
not five call sites.
"""

import yaml

from data.dvlog_face_dataset import get_dvlog_face_loaders
from models.afgnn import AFGNN


def load_config(config_path):
    """Load a YAML config file into a plain dict."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_model(cfg, device=None):
    """Construct an AFGNN model from a config dict.

    Reproduces exactly the keyword arguments that were previously duplicated
    across train.py / test.py / train_combined.py / seed_ensemble_test.py /
    seed_stability_test.py. If ``device`` is given the model is moved onto it.
    """
    m = cfg["model"]
    model = AFGNN(
        use_face=m.get("use_face", True),
        use_audio=m.get("use_audio", False),
        face_in_channels=m["face_in_channels"],
        face_hidden_channels=m["face_hidden_channels"],
        face_out_channels=m["face_out_channels"],
        face_num_layers=m["face_num_layers"],
        face_heads=m["face_heads"],
        audio_in_channels=m.get("audio_in_channels", 25),
        audio_hidden_channels=m.get("audio_hidden_channels", 64),
        audio_out_channels=m.get("audio_out_channels", 64),
        audio_num_layers=m.get("audio_num_layers", 2),
        audio_heads=m.get("audio_heads", 4),
        mlp_hidden=m["mlp_hidden"],
        dropout=m["dropout"],
        num_edge_types=m.get("num_edge_types", None),
        edge_emb_dim=m.get("edge_emb_dim", 1),
        fusion_type=m.get("fusion_type", "concat"),
        fusion_hidden_dim=m.get("fusion_hidden_dim", 64),
    )
    if device is not None:
        model = model.to(device)
    return model


def build_loaders(
    cfg,
    *,
    data_dir=None,
    batch_size=None,
    num_workers=None,
    augment=False,
    worker_init_fn=None,
):
    """Build the train/valid/test DataLoaders from a config dict.

    Reproduces the full ``get_dvlog_face_loaders(...)`` argument set that was
    duplicated across train.py / test.py / seed_ensemble_test.py /
    seed_stability_test.py.

    Args:
        data_dir / batch_size / num_workers: fall back to the config when None,
            so eval scripts can override them from CLI flags.
        augment: explicit switch (default False) so evaluation never augments by
            accident; training passes the config's ``augmentation.enabled`` flag.
        worker_init_fn: optional DataLoader worker seeding hook (training only).
    """
    d = cfg["data"]
    m = cfg["model"]
    aug = cfg.get("augmentation", {})

    if data_dir is None:
        data_dir = d["processed_dir"]
    if batch_size is None:
        batch_size = cfg["training"]["batch_size"]
    if num_workers is None:
        num_workers = cfg["training"]["num_workers"]

    return get_dvlog_face_loaders(
        data_dir=data_dir,
        num_frames=d["num_frames"],
        audio_num_frames=d.get("audio_num_frames", d["num_frames"]),
        batch_size=batch_size,
        num_workers=num_workers,
        add_static_edges=m["add_static_edges"],
        add_temporal_edges=m["add_temporal_edges"],
        add_dynamic_edges=m.get("add_dynamic_edges", False),
        dynamic_k=m.get("dynamic_k", 3),
        dynamic_metric=m.get("dynamic_metric", "cosine"),
        dynamic_edge_weight=m.get("dynamic_edge_weight", 1.0),
        dynamic_feature=m.get("dynamic_feature", "full"),
        dynamic_region_restricted=m.get("dynamic_region_restricted", False),
        use_edge_type=m.get("use_edge_type", False),
        add_region_onehot=m.get("add_region_onehot", False),
        add_self_loops=m.get("add_self_loops", False),
        use_velocity=m.get("use_velocity", True),
        region_ablation=m.get("region_ablation", None),
        random_static_edges=m.get("random_static_edges", False),
        random_edge_seed=m.get("random_edge_seed", 0),
        augment=augment,
        rotation_range=aug.get("rotation_range", (-10.0, 10.0)),
        scale_range=aug.get("scale_range", (0.95, 1.05)),
        translate_range=aug.get("translate_range", (-0.05, 0.05)),
        noise_std=aug.get("noise_std", 0.02),
        temporal_mask_prob=aug.get("temporal_mask_prob", 0.2),
        temporal_mask_max_ratio=aug.get("temporal_mask_max_ratio", 0.15),
        audio_use_delta=m.get("audio_use_delta", False),
        audio_self_loops=m.get("audio_self_loops", False),
        audio_skip=m.get("audio_skip", 0),
        use_audio=m.get("use_audio", False),
        worker_init_fn=worker_init_fn,
    )
