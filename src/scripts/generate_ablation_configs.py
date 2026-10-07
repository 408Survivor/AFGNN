#!/usr/bin/env python3
"""Generate YAML configs for facial graph component ablations and region LOO experiments."""

import copy
import os

import yaml


BASE_CONFIG = {
    "data": {
        "processed_dir": "/data/ltq/D-Vlog_Raw/processed_official_features",
        "num_frames": 32,
        "audio_num_frames": 32,
    },
    "model": {
        "use_face": True,
        "use_audio": True,
        "fusion_type": "cross_attention",
        "fusion_hidden_dim": 64,
        "face_in_channels": 13,
        "face_hidden_channels": 128,
        "face_out_channels": 128,
        "face_num_layers": 3,
        "face_heads": 4,
        "audio_in_channels": 25,
        "audio_hidden_channels": 64,
        "audio_out_channels": 64,
        "audio_num_layers": 2,
        "audio_heads": 4,
        "mlp_hidden": 128,
        "dropout": 0.3,
        "add_static_edges": True,
        "add_temporal_edges": True,
        "add_dynamic_edges": False,
        "dynamic_k": 2,
        "dynamic_metric": "cosine",
        "dynamic_edge_weight": 1.0,
        "dynamic_feature": "motion",
        "dynamic_region_restricted": True,
        "add_region_onehot": True,
        "add_self_loops": True,
        "use_edge_type": False,
        "num_edge_types": None,
        "edge_emb_dim": 4,
        "use_velocity": True,
        "region_ablation": None,
    },
    "augmentation": {"enabled": False},
    "training": {
        "batch_size": 32,
        "num_workers": 4,
        "epochs": 100,
        "patience": 15,
        "lr": 0.001,
        "weight_decay": 0.0001,
        "checkpoint_path": "ablation/checkpoints/ablation_full_best.pt",
        "early_stopping_metric": "auc",
        "loss": "focal",
        "focal_alpha": 0.25,
        "focal_gamma": 2.0,
        "scheduler": {"type": "plateau", "factor": 0.5, "patience": 5},
        "grad_clip": 1.0,
    },
}

# Component ablations: name -> overrides dict
COMPONENT_ABLATIONS = {
    "full": {},
    "no_static": {"add_static_edges": False},
    "no_temporal": {"add_temporal_edges": False},
    "no_region": {"add_region_onehot": False, "face_in_channels": 4},
    "no_velocity": {"use_velocity": False, "face_in_channels": 11},
    "no_selfloop": {"add_self_loops": False},
}

# 9 facial regions defined in graph_utils.LANDMARK_GROUPS_68
REGION_NAMES = [
    "face_contour",
    "left_eyebrow",
    "right_eyebrow",
    "nose_bridge",
    "nose_bottom",
    "left_eye",
    "right_eye",
    "outer_mouth",
    "inner_mouth",
]


def make_config(overrides, checkpoint_name):
    cfg = copy.deepcopy(BASE_CONFIG)
    cfg["model"].update(overrides)
    cfg["training"]["checkpoint_path"] = f"ablation/checkpoints/{checkpoint_name}_best.pt"
    return cfg


def main():
    out_dir = "ablation/configs"
    os.makedirs(out_dir, exist_ok=True)

    # Component ablations
    for name, overrides in COMPONENT_ABLATIONS.items():
        cfg = make_config(overrides, f"ablation_{name}")
        path = os.path.join(out_dir, f"ablation_{name}.yaml")
        with open(path, "w") as f:
            yaml.dump(cfg, f, sort_keys=False)
        print(f"Wrote {path}")

    # Region leave-one-out ablations
    for region_id, region_name in enumerate(REGION_NAMES):
        cfg = make_config(
            {"region_ablation": region_id},
            f"ablation_region_mask_{region_id:02d}_{region_name}",
        )
        path = os.path.join(out_dir, f"ablation_region_mask_{region_id:02d}_{region_name}.yaml")
        with open(path, "w") as f:
            yaml.dump(cfg, f, sort_keys=False)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
