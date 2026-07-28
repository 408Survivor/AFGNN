#!/usr/bin/env python3
"""
Extract REAL data for Figure 3 (audio temporal graph interpretability).

Loads a trained AFGNN checkpoint and one real depressed D-Vlog sample, then dumps:
    - features: (T, 25) normalized acoustic feature matrix  -> panel (a) heatmap
    - weights : (T,)   audio readout attention weights       -> panel (c) bars
both taken from the SAME sample, so the figure is fully real and self-consistent.

The readout weights are the per-frame attention of the AudioGNN's
AttentionalAggregation gate (softmax of gate_nn(x) over the T frames).

Usage:
    conda activate DVlog
    cd /home/ltq/DepressionCode/DepGNN/AFGNN
    python visualization/extract_figure3_data.py \
        --config experiments/configs/afgnn_face_enhanced_focal.yaml \
        --checkpoint experiments/checkpoints/afgnn_face_enhanced_focal_best_seed42.pt
"""

import argparse
import os
import sys

import numpy as np
import torch
import yaml

# Allow running this script from anywhere: add the project root (two levels up
# from visualization/) to sys.path so `data` and `models` are importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dvlog_face_dataset import get_dvlog_face_loaders
from src.models.afgnn import AFGNN


def load_config(p):
    with open(p) as f:
        return yaml.safe_load(f)


def build_model(cfg, device):
    return AFGNN(
        use_face=cfg["model"].get("use_face", True),
        use_audio=cfg["model"].get("use_audio", False),
        face_in_channels=cfg["model"]["face_in_channels"],
        face_hidden_channels=cfg["model"]["face_hidden_channels"],
        face_out_channels=cfg["model"]["face_out_channels"],
        face_num_layers=cfg["model"]["face_num_layers"],
        face_heads=cfg["model"]["face_heads"],
        audio_in_channels=cfg["model"].get("audio_in_channels", 25),
        audio_hidden_channels=cfg["model"].get("audio_hidden_channels", 64),
        audio_out_channels=cfg["model"].get("audio_out_channels", 64),
        audio_num_layers=cfg["model"].get("audio_num_layers", 2),
        audio_heads=cfg["model"].get("audio_heads", 4),
        mlp_hidden=cfg["model"]["mlp_hidden"],
        dropout=cfg["model"]["dropout"],
        num_edge_types=cfg["model"].get("num_edge_types", None),
        edge_emb_dim=cfg["model"].get("edge_emb_dim", 1),
        fusion_type=cfg["model"].get("fusion_type", "concat"),
        fusion_hidden_dim=cfg["model"].get("fusion_hidden_dim", 64),
    ).to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="experiments/configs/afgnn_face_enhanced_focal.yaml")
    ap.add_argument("--checkpoint", default="experiments/checkpoints/afgnn_face_enhanced_focal_best_seed42.pt")
    ap.add_argument("--data_dir", default="/data/ltq/DVlog/processed_official_features")
    ap.add_argument("--sample_index", type=int, default=-1,
                    help="Test-set index to use; -1 = first depressed (label==1) sample.")
    ap.add_argument("--output", default=None, help="Output .npz path.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loaders = get_dvlog_face_loaders(
        data_dir=args.data_dir,
        num_frames=cfg["data"]["num_frames"],
        audio_num_frames=cfg["data"].get("audio_num_frames", cfg["data"]["num_frames"]),
        batch_size=1,
        num_workers=0,
        add_static_edges=cfg["model"]["add_static_edges"],
        add_temporal_edges=cfg["model"]["add_temporal_edges"],
        add_dynamic_edges=cfg["model"].get("add_dynamic_edges", False),
        dynamic_k=cfg["model"].get("dynamic_k", 3),
        dynamic_metric=cfg["model"].get("dynamic_metric", "cosine"),
        dynamic_edge_weight=cfg["model"].get("dynamic_edge_weight", 1.0),
        dynamic_feature=cfg["model"].get("dynamic_feature", "full"),
        dynamic_region_restricted=cfg["model"].get("dynamic_region_restricted", False),
        use_edge_type=cfg["model"].get("use_edge_type", False),
        add_region_onehot=cfg["model"].get("add_region_onehot", False),
        add_self_loops=cfg["model"].get("add_self_loops", False),
        use_velocity=cfg["model"].get("use_velocity", True),
        region_ablation=cfg["model"].get("region_ablation", None),
        random_static_edges=cfg["model"].get("random_static_edges", False),
        random_edge_seed=cfg["model"].get("random_edge_seed", 0),
        augment=False,
        audio_use_delta=cfg["model"].get("audio_use_delta", False),
        audio_self_loops=cfg["model"].get("audio_self_loops", False),
        audio_skip=cfg["model"].get("audio_skip", 0),
        use_audio=cfg["model"].get("use_audio", False),
    )

    model = build_model(cfg, device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Hook the readout gate to capture per-frame gate logits.
    captured = {}

    def hook(_module, _inp, out):
        captured["gate"] = out.detach().cpu()

    handle = model.audio_gnn.readout_gate.register_forward_hook(hook)

    # Pick the target sample.
    chosen = None
    with torch.no_grad():
        for i, data in enumerate(loaders["test"]):
            label = int(data.y.item())
            if args.sample_index >= 0:
                if i != args.sample_index:
                    continue
            elif label != 1:  # auto mode: first depressed sample
                continue
            data = data.to(device)
            _ = model(data)
            gate = captured["gate"].squeeze(-1)          # (T,)
            weights = torch.softmax(gate, dim=0).numpy()  # per-frame attention
            features = data.audio_x.detach().cpu().numpy()  # (T, 25) normalized
            chosen = dict(index=i, label=label, features=features, weights=weights)
            break

    handle.remove()
    if chosen is None:
        raise RuntimeError("No matching sample found.")

    out_path = args.output or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "figure3_data.npz"
    )
    np.savez(
        out_path,
        features=chosen["features"].astype(np.float32),
        weights=chosen["weights"].astype(np.float32),
        label=chosen["label"],
        index=chosen["index"],
        num_frames=chosen["features"].shape[0],
    )
    print(f"sample index = {chosen['index']}  label = {chosen['label']} (1=depressed)")
    print(f"features shape = {chosen['features'].shape}  weights shape = {chosen['weights'].shape}")
    print(f"weights: min={chosen['weights'].min():.4f} max={chosen['weights'].max():.4f} "
          f"argmax_frame={int(chosen['weights'].argmax())}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
