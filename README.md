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

Required packages: `torch`, `torch_geometric`, `numpy`, `scikit-learn`, `pyyaml`,
`tqdm`, `matplotlib` (training-curve plots), `pytest` (smoke tests).

---

## Quick Start

```bash
cd /home/ltq/Code/AFGNN
conda activate DVlog
```

### 1. Train

Each training run automatically creates a self-contained experiment folder
`experiments/exp_N/` (auto-incremented) holding the config snapshot, run info,
model summary, best checkpoint, training history and training curves:

```bash
python src/train.py --config experiments/configs/afgnn_base.yaml --seed 42
```

### 2. Test

Evaluation results are written back into the same experiment folder:

```bash
python src/test.py --exp_dir experiments/exp_1 --split test
```

`experiments/INDEX.md` is the authoritative index mapping every `exp_N` to its
config, seed and metrics. To visualize a finished run's training curves:

```bash
python src/scripts/plot_training_history.py --exp_dir experiments/exp_1
```

### 3. Smoke tests

```bash
pytest tests/ -q
```

---

## Project Structure

```
AFGNN/
├── src/                       # Code
│   ├── data/                  # Dataset loaders, landmark layout parsing, augmentation
│   ├── models/                # AFGNN model (face/audio GAT branches, fusion)
│   ├── utils/                 # Builders, losses, metrics, trainer, experiment manager
│   ├── scripts/               # Helper scripts (processed-feature build, plotting, ...)
│   ├── train.py               # Training entry (auto-creates experiments/exp_N)
│   ├── test.py                # Evaluation entry (results written back to exp_N)
│   └── ...                    # Ensemble / seed-stability evaluation scripts
├── tests/                     # CPU smoke tests (pytest)
├── experiments/
│   ├── configs/               # One YAML per main experiment
│   ├── exp_N/                 # Self-contained run folders (created automatically)
│   └── INDEX.md               # Authoritative experiment index
├── ablation/
│   └── configs/               # Ablation configs (also numbered into experiments/exp_N)
├── paper/                     # Reference papers
├── DATA_DIMENSIONS.md         # Data shapes across processing stages
└── README.md
```

---

## Notes

- `processed_official_features` is expected to contain:
  - `{split}_visual.npy` and `{split}_labels.npy`
  - `{split}_acoustic.npy`
- Visual features have shape `(N, 596, 136)` in the OpenFace **block layout**
  `[x_0..x_67, y_0..y_67]` (NOT interleaved `(x, y)` pairs) — always parse them
  via `src/data/landmark_layout.py` (`flat_to_coords` / `coords_to_flat`).
- Acoustic features have shape `(N, 596, 25)`.
- Each face sample is converted into a graph with `T_v × 68` nodes.
- Each audio sample is converted into a chain graph with `T_a` nodes.
