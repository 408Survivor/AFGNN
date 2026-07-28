# AFGNN: Audio-Facial Graph Neural Network for Depression Recognition

AFGNN is a fully graph-based multimodal framework for depression recognition.
Both the facial and audio modalities are represented as graphs and encoded by
Graph Attention Networks (GATs). A cross-modal attention fusion module then
combines the two graph-level embeddings for classification.

> **Key design principle**: everything is a graph. The face is a spatio-temporal
> landmark graph, and the audio is a temporal acoustic graph.

---

## Current Scope

- **Face GNN**: spatio-temporal graph built from 68 facial landmarks.
  - Nodes: facial landmarks with features `[x, y, Δx, Δy]`.
  - Edges: static anatomical edges + temporal edges.
  - Encoder: 3-layer GAT with attention-based readout.
- **Audio GNN**: temporal graph built from frame-level acoustic features.
  - Nodes: time frames with 25-dimensional acoustic features.
  - Edges: bidirectional chain edges connecting consecutive frames.
  - Encoder: 2-layer GAT with attention-based readout.
- **Fusion**: cross-modal attention between face and audio graph embeddings,
  followed by a 2-layer MLP classifier.
- **Dataset**: D-Vlog preprocessed visual and acoustic features
  (`{split}_visual.npy` and `{split}_acoustic.npy`).

---

## Environment

```bash
conda activate DVlog
```

Required packages: `torch`, `torch_geometric`, `numpy`, `scikit-learn`, `pyyaml`, `tqdm`.

---

## Quick Start

### 1. Train

```bash
cd /home/ltq/DepressionCode/DepGNN/AFGNN
conda activate DVlog
python src/train.py --config experiments/configs/afgnn_face_only.yaml
```

### 2. Test

```bash
cd /home/ltq/DepressionCode/DepGNN/AFGNN
conda activate DVlog
python src/test.py --config experiments/configs/afgnn_face_only.yaml --split test
```

Test results are automatically saved to `experiments/results/test_{split}_results.json`.
Training history is saved to `experiments/results/training_history.json`.

---

## Project Structure

```
AFGNN/
├── src/                       # Code
│   ├── data/                  # Dataset loaders and augmentation
│   ├── models/                # AFGNN model implementations
│   ├── utils/                 # Losses, metrics, trainer
│   ├── scripts/               # Experiment helper scripts
│   ├── train.py
│   ├── test.py
│   ├── train_combined.py
│   ├── ensemble_test.py
│   ├── seed_ensemble_test.py
│   └── seed_stability_test.py
├── visualization/             # Figure generation scripts
├── paper/                     # Paper source files and figures
├── ablation/                  # Ablation study artifacts
│   ├── configs/
│   ├── results/
│   └── checkpoints/
├── experiments/               # Other experiment artifacts
│   ├── configs/
│   ├── results/
│   └── checkpoints/
├── records/                   # Notes, logs, and reference documents
└── README.md
```

---

## Notes

- `processed_official_features` is expected to contain:
  - `{split}_visual.npy` and `{split}_labels.npy`
  - `{split}_acoustic.npy`
- Visual features have shape `(N, 596, 136)` where `136 = 68 landmarks × 2 (x, y)`.
- Acoustic features have shape `(N, 596, 25)`.
- Each face sample is converted into a graph with `T_v × 68` nodes.
- Each audio sample is converted into a chain graph with `T_a` nodes.
