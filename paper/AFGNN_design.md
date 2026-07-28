# AFGNN: Audio-Facial Graph Neural Network for Depression Recognition

> **Design Document for Paper Writing**  
> Model: AFGNN (Audio-Facial Graph Neural Network)  
> Task: Multimodal Depression Recognition from Vlogs  
> Modalities: Facial Landmarks (68 points) + Audio Temporal Graph  
> Status: Architecture Design Phase

---

## 1. Abstract (Paper-Ready Draft)

Depression recognition from social media vlogs has attracted increasing attention due to its potential for passive, large-scale mental health screening. Existing methods typically rely on heavy convolutional or vision-transformer backbones operating on raw video frames and audio waveforms, which often suffer from high computational cost and poor cross-dataset generalization. In this paper, we propose the **Audio-Facial Graph Neural Network (AFGNN)**, a lightweight and interpretable framework that represents both facial geometry and audio dynamics as graphs. Specifically, we construct a **spatio-temporal facial landmark graph** from 68 facial keypoints, where nodes encode geometric and dynamic features and edges capture anatomical connectivity and temporal dependencies. Concurrently, we construct a **temporal acoustic graph** from frame-level acoustic features to model prosodic dynamics. A cross-modal fusion module then integrates facial geometric and audio temporal representations for depression classification. Extensive experiments on D-Vlog and cross-dataset evaluations on AVEC2014 demonstrate that AFGNN achieves competitive accuracy with significantly fewer parameters compared to CNN/ViT-based counterparts, while offering improved interpretability through graph attention weights.

---

## 2. Introduction & Motivation

### 2.1 Background

Depression is a leading cause of disability worldwide. Traditional diagnosis relies on clinical interviews and self-reported scales, which are time-consuming and subject to recall bias. With the proliferation of video-sharing platforms, researchers have explored **automatic depression detection from vlogs**, leveraging behavioral signals such as facial expressions, gaze, and speech prosody.

### 2.2 Limitations of Existing Approaches

Most existing methods fall into two categories:

1. **CNN/RNN-based methods**: Extract frame-level visual features and audio features separately, then fuse them. These methods often ignore the topological structure of facial components.
2. **Vision-Transformer-based methods**: Such as ViDA-GCAM, use heavy backbones (ResNet + ViT, 1D/2D audio transformers) and complex cross-attention fusion. While effective on the source dataset, they tend to:
   - Require hundreds of millions of parameters (~467M).
   - Overfit on small datasets (e.g., AVEC2014 with ~191 clips).
   - Generalize poorly across datasets due to domain shift in pixel-level appearance.

### 2.3 Why GNN?

Graph Neural Networks (GNNs) are naturally suited for modeling:
- **Structured geometric data**: Facial landmarks have inherent anatomical topology.
- **Cross-modal relationships**: Different modalities can be represented as heterogeneous nodes.
- **Lightweight computation**: Small graphs (e.g., 68 nodes per frame) require far fewer parameters than CNN/ViT backbones.

### 2.4 Key Contributions

1. We propose AFGNN, the first depression recognition framework to jointly model **facial landmark graphs** and **audio temporal graphs**.
2. We design a **spatio-temporal facial landmark graph** with both anatomical edges and learned dynamic edges to capture depression-related facial dynamics.
3. We demonstrate that lightweight graph-based models can achieve competitive or superior cross-dataset generalization compared to heavy CNN/ViT baselines.
4. We provide interpretability through graph attention weights, revealing facial regions and time frames important for depression classification.

---

## 3. Problem Formulation

### 3.1 Input

Given a video clip $v$ of duration $T$ seconds, we extract two complementary representations:

- **Facial Landmark Sequence**:  
  $$\mathbf{L} = \{\mathbf{l}_t \in \mathbb{R}^{68 \times d}\}_{t=1}^{T_v}$$  
  where $d \in \{2, 3\}$ denotes 2D or 3D coordinates, and $T_v$ is the number of video frames sampled.

- **Frame-Level Acoustic Feature Sequence**:  
  $$\mathbf{A} = \{\mathbf{a}_t \in \mathbb{R}^{D_a}\}_{t=1}^{T_a}$$  
  where $D_a$ is the acoustic feature dimension and $T_a$ is the number of time frames sampled.

### 3.2 Output

The model predicts a depression probability:
$$\hat{y} = f(\mathbf{L}, \mathbf{A}) \in [0, 1]$$

The binary label $y \in \{0, 1\}$ indicates non-depressed (0) or depressed (1).

### 3.3 Objective

Minimize the binary cross-entropy loss:
$$\mathcal{L}_{\text{BCE}} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1-y_i) \log(1-\hat{y}_i) \right]$$

---

## 4. Model Architecture

AFGNN consists of three main components:
1. **Facial Landmark Graph Branch** (Section 4.2)
2. **Audio Temporal Graph Branch** (Section 4.3)
3. **Cross-Modal Fusion and Classification Head** (Section 4.4)

A high-level illustration is shown in Figure 1 (to be added).

### 4.1 Overall Pipeline

```
Input Video Clip
    ├── Facial Landmark Extractor ──► Spatio-Temporal Landmark Graph ──► Face GNN ──► H_face
    └── Acoustic Features ──► Temporal Graph ──► Audio GNN ──► H_audio
                                                          │
                                                          ▼
                                            Cross-Modal Fusion Module
                                                          │
                                                          ▼
                                                   Sigmoid Classifier
                                                          │
                                                          ▼
                                                   Depression Score
```

### 4.2 Facial Landmark Graph Branch

#### 4.2.1 Node Construction

For each video clip, we sample $T_v$ frames uniformly. For frame $t$ and landmark $i$, the node feature is defined as:

$$\mathbf{x}_{t,i}^{(V)} = \left[ \tilde{x}_{t,i}, \tilde{y}_{t,i}, \Delta x_{t,i}, \Delta y_{t,i}, c_{t,i} \right] \in \mathbb{R}^{5}$$

where:
- $(\tilde{x}_{t,i}, \tilde{y}_{t,i})$ are the normalized landmark coordinates.
- $(\Delta x_{t,i}, \Delta y_{t,i}) = (x_{t,i} - x_{t-1,i}, y_{t,i} - y_{t-1,i})$ are velocity features.
- $c_{t,i} \in [0,1]$ is the detection confidence (if available).

If 3D landmarks are used, add $\tilde{z}_{t,i}$ and $\Delta z_{t,i}$.

**Total nodes in the facial graph**: $N_V = T_v \times 68$.

#### 4.2.2 Edge Construction

We construct a spatio-temporal graph $\mathcal{G}_V = (\mathcal{V}_V, \mathcal{E}_V)$ with two types of edges:

**Static Anatomical Edges ($\mathcal{E}_V^{\text{static}}$)**:

Intra-frame edges follow the standard 68-point facial landmark topology:

| Region | Landmark Indices | Edge Pattern |
| :--- | :--- | :--- |
| Face contour | 1–17 | Chain: $1\!\!-\!\!2\!\!-\!\!\dots\!\!-\!\!17$ |
| Left eyebrow | 18–22 | Chain: $18\!\!-\!\!19\!\!-\!\!20\!\!-\!\!21\!\!-\!\!22$ |
| Right eyebrow | 23–27 | Chain: $23\!\!-\!\!24\!\!-\!\!25\!\!-\!\!26\!\!-\!\!27$ |
| Nose bridge | 28–31 | Chain: $28\!\!-\!\!29\!\!-\!\!30\!\!-\!\!31$ |
| Nose bottom | 32–36 | Ring: $32\!\!-\!\!33\!\!-\!\!34\!\!-\!\!35\!\!-\!\!36\!\!-\!\!32$ |
| Left eye | 37–42 | Ring: $37\!\!-\!\!38\!\!-\!\!39\!\!-\!\!40\!\!-\!\!41\!\!-\!\!42\!\!-\!\!37$ |
| Right eye | 43–48 | Ring: $43\!\!-\!\!44\!\!-\!\!45\!\!-\!\!46\!\!-\!\!47\!\!-\!\!48\!\!-\!\!43$ |
| Outer mouth | 49–60 | Ring: $49\!\!-\!\!50\!\!-\!\!\dots\!\!-\!\!60\!\!-\!\!49$ |
| Inner mouth | 61–68 | Ring: $61\!\!-\!\!62\!\!-\!\!\dots\!\!-\!\!68\!\!-\!\!61$ |

Temporal edges connect the same landmark across consecutive frames:
$$\mathcal{E}_V^{\text{temp}} = \{(v_{t,i}, v_{t+1,i}) \mid 1 \leq t < T_v, 1 \leq i \leq 68\}$$

**Dynamic Edges ($\mathcal{E}_V^{\text{dynamic}}$)**:

Within each frame, we add $K$ learned edges per landmark based on motion similarity or feature attention. Specifically, for node $v_{t,i}$, we compute its dynamic neighbors by:

$$\mathcal{N}^{\text{dyn}}(v_{t,i}) = \text{TopK}_{j \neq i} \left( \text{MLP}_{\text{edge}}([\mathbf{x}_{t,i}^{(V)} \|\| \mathbf{x}_{t,j}^{(V)}]) \right)$$

where $\|\|$ denotes concatenation and $\text{MLP}_{\text{edge}}$ predicts an edge affinity score. The final edge set is:

$$\mathcal{E}_V = \mathcal{E}_V^{\text{static}} \cup \mathcal{E}_V^{\text{temp}} \cup \mathcal{E}_V^{\text{dynamic}}$$

#### 4.2.3 Graph Encoder

We use a **Graph Attention Network (GAT)** with $L_V$ layers. The message passing rule at layer $\ell$ is:

$$\mathbf{h}_{t,i}^{(\ell+1)} = \sigma\left( \sum_{j \in \mathcal{N}(t,i)} \alpha_{t,i,j}^{(\ell)} \mathbf{W}^{(\ell)} \mathbf{h}_{t,j}^{(\ell)} \right)$$

where the attention coefficient $\alpha_{t,i,j}$ is computed as:

$$\alpha_{t,i,j} = \frac{\exp(\text{LeakyReLU}(\mathbf{a}^\top [\mathbf{W}\mathbf{h}_{t,i} \|\| \mathbf{W}\mathbf{h}_{t,j}]))}{\sum_{k \in \mathcal{N}(t,i)} \exp(\text{LeakyReLU}(\mathbf{a}^\top [\mathbf{W}\mathbf{h}_{t,i} \|\| \mathbf{W}\mathbf{h}_{t,k}]))}$$

Multi-head attention is used to stabilize learning:

$$\mathbf{h}_{t,i}^{(\ell+1)} = \Big\|_{m=1}^{M} \sigma\left( \sum_{j \in \mathcal{N}(t,i)} \alpha_{t,i,j}^{(m,\ell)} \mathbf{W}_m^{(\ell)} \mathbf{h}_{t,j}^{(\ell)} \right)$$

where $M$ is the number of attention heads.

#### 4.2.4 Readout

To obtain a clip-level facial representation, we apply a global attention pooling:

$$\mathbf{H}_V = \sum_{t,i} \beta_{t,i} \mathbf{h}_{t,i}^{(L_V)}$$

where the attention weights are:

$$\beta_{t,i} = \frac{\exp(\mathbf{q}^\top \mathbf{h}_{t,i}^{(L_V)})}{\sum_{t',i'} \exp(\mathbf{q}^\top \mathbf{h}_{t',i'}^{(L_V)})}$$

and $\mathbf{q}$ is a learned query vector.

---

### 4.3 Audio Temporal Graph Branch

#### 4.3.1 Acoustic Feature Extraction

For each video clip, we extract a sequence of frame-level acoustic features:

$$\mathbf{A} = \{\mathbf{a}_t \in \mathbb{R}^{D_a}\}_{t=1}^{T}$$

where $D_a$ is the acoustic feature dimension and $T$ is the number of original time frames. We uniformly sample $T_a$ frames and apply z-score normalization using statistics computed on the training set:

$$\tilde{\mathbf{a}}_t = \frac{\mathbf{a}_t - \boldsymbol{\mu}}{\boldsymbol{\sigma}}$$

In our implementation, $D_a = 25$ and $T_a = 32$.

#### 4.3.2 Graph Construction

We construct a temporal graph $\mathcal{G}_A = (\mathcal{V}_A, \mathcal{E}_A)$ where:
- Each node corresponds to a sampled time frame $\tau$.
- Node feature: $\mathbf{x}_{\tau}^{(A)} = \tilde{\mathbf{a}}_\tau \in \mathbb{R}^{D_a}$.
- Edges connect consecutive time frames (bidirectional chain):
  $$\mathcal{E}_A = \{(\tau, \tau+1), (\tau+1, \tau) \mid 1 \leq \tau < T_a\}.$$

This graph structure captures local prosodic dynamics without requiring handcrafted spectrograms.

#### 4.3.3 Graph Encoder

We use a GAT with $L_A$ layers:

$$\mathbf{h}_{\tau}^{(\ell+1)} = \text{GAT}^{(\ell)}\left( \mathbf{h}_{\tau}^{(\ell)}, \{\mathbf{h}_{\tau'}^{(\ell)}\}_{\tau' \in \mathcal{N}(\tau)} \right)$$

#### 4.3.4 Readout

Global attention-based pooling:

$$\mathbf{H}_A = \sum_{\tau} \gamma_{\tau} \mathbf{h}_{\tau}^{(L_A)}$$

---

### 4.4 Cross-Modal Fusion and Classification

#### 4.4.1 Fusion Strategy (Baseline)

The simplest and effective fusion strategy is concatenation followed by an MLP:

$$\mathbf{H}_{\text{fuse}} = [\mathbf{H}_V \|\| \mathbf{H}_A]$$

$$\hat{y} = \sigma(\text{MLP}(\mathbf{H}_{\text{fuse}}))$$

#### 4.4.2 Cross-Modal Attention Fusion (Optional)

For stronger interaction, we use a bilinear attention matrix between face and audio features:

$$\mathbf{A} = \text{softmax}\left( \frac{\mathbf{H}_V \mathbf{W}_Q (\mathbf{H}_A \mathbf{W}_K)^\top}{\sqrt{D}} \right)$$

$$\mathbf{H}_V' = \mathbf{A} \mathbf{H}_A, \quad \mathbf{H}_A' = \mathbf{A}^\top \mathbf{H}_V$$

$$\mathbf{H}_{\text{fuse}} = [\mathbf{H}_V + \mathbf{H}_V' \|\| \mathbf{H}_A + \mathbf{H}_A']$$

#### 4.4.3 Classification Head

A 2-layer MLP with dropout:

$$\hat{y} = \sigma(\mathbf{W}_2 \text{ReLU}(\text{Dropout}(\mathbf{W}_1 \mathbf{H}_{\text{fuse}} + \mathbf{b}_1)) + \mathbf{b}_2)$$

where $\sigma$ is the sigmoid function.

---

## 5. Mathematical Summary

**Facial graph**:
$$\mathcal{G}_V = (\mathcal{V}_V, \mathcal{E}_V, \mathbf{X}_V), \quad |\mathcal{V}_V| = T_v \times 68$$

**Audio graph**:
$$\mathcal{G}_A = (\mathcal{V}_A, \mathcal{E}_A, \mathbf{X}_A), \quad |\mathcal{V}_A| = T_a$$

**Graph encoders**:
$$\mathbf{H}_V = \text{Readout}_V(\text{GNN}_V(\mathcal{G}_V))$$
$$\mathbf{H}_A = \text{Readout}_A(\text{GNN}_A(\mathcal{G}_A))$$

**Fusion and prediction**:
$$\mathbf{H}_{\text{fuse}} = \text{Fusion}(\mathbf{H}_V, \mathbf{H}_A)$$
$$\hat{y} = \sigma(\text{MLP}(\mathbf{H}_{\text{fuse}}))$$

**Loss**:
$$\mathcal{L} = \mathcal{L}_{\text{BCE}} + \lambda_{\text{reg}} \|\Theta\|_2^2$$

where $\Theta$ denotes all learnable parameters and $\lambda_{\text{reg}}$ is the weight decay coefficient.

---

## 6. Implementation Details

### 6.1 Facial Landmark Extraction

**Tool**: OpenFace 2.0 (recommended) or MediaPipe Face Mesh.

**Output from OpenFace**:
- `x_0` to `x_67`, `y_0` to `y_67` (2D landmarks)
- `X_0` to `X_67`, `Y_0` to `Y_67`, `Z_0` to `Z_67` (3D landmarks)
- Confidence values for each landmark

**Preprocessing steps**:
1. Detect faces in each frame.
2. Extract 68 landmarks.
3. Handle missing detections by forward-fill or interpolation.
4. Normalize coordinates:
   - Translate: center at the nose tip (landmark 30).
   - Scale: divide by the inter-ocular distance (landmarks 37 and 46).
5. Compute velocity features frame-by-frame.
6. Uniformly sample $T_v$ frames per video (e.g., $T_v = 16$).

### 6.2 Audio Processing

**Steps**:
1. Load precomputed frame-level acoustic features (e.g., 25 dims per frame).
2. Uniformly sample $T_a$ frames (e.g., $T_a = 32$).
3. Apply z-score normalization using training-set statistics.
4. Construct a temporal chain graph from the sampled frames.

### 6.3 Graph Construction Code Sketch

```python
# Facial graph
nodes = []
for t in range(T_v):
    for i in range(68):
        feat = [norm_x[t,i], norm_y[t,i], vel_x[t,i], vel_y[t,i], conf[t,i]]
        nodes.append(feat)

edges = static_edges + temporal_edges + dynamic_edges
face_graph = Data(x=nodes, edge_index=edges)

# Audio graph
nodes = []
for tau in range(T_a):
    feat = acoustic_features[tau]  # D_a-dimensional normalized feature
    nodes.append(feat)

edges = chain_neighbors(T_a)  # bidirectional consecutive edges
audio_graph = Data(x=nodes, edge_index=edges)
```

### 6.4 Model Hyperparameters (Initial Guess)

| Hyperparameter | Value | Notes |
| :--- | :--- | :--- |
| Video frames $T_v$ | 16 | Uniform sampling |
| Audio frames $T_a$ | 32 | Uniform sampling |
| Acoustic feature dim $D_a$ | 25 | Frame-level acoustic descriptors |
| Face GNN layers $L_V$ | 3–4 | Small graph |
| Audio GNN layers $L_A$ | 2–3 | Larger graph |
| Hidden dim $D$ | 128–256 | Shared across branches |
| GAT heads $M$ | 4 | Multi-head attention |
| Dropout | 0.3–0.5 | Prevent overfitting |
| Learning rate | 1e-4 to 1e-3 | Adam / AdamW |
| Weight decay | 1e-4 to 1e-3 | L2 regularization |
| Batch size | 16–32 | Depends on GPU memory |
| Epochs | 50–100 | Early stopping with patience 10–15 |

---

## 7. Experimental Setup

### 7.1 Datasets

#### D-Vlog
- 831 YouTube videos: 451 depressed, 380 non-depressed.
- Official train/val/test split.
- OpenFace features provided.
- Primary evaluation dataset.

#### AVEC2014 (Cross-Dataset)
- Depression sub-challenge: 150 training, 50 development, 50 test clips.
- Labels converted to binary (depressed vs. non-depressed) using PHQ-8 threshold (e.g., $\geq 10$).
- Used for zero-shot and fine-tuning experiments.

### 7.2 Evaluation Metrics

For binary depression classification:
- Accuracy (Acc)
- Precision (P)
- Recall (R)
- F1-score (F1)
- AUC-ROC

For video-level evaluation (if multiple clips per video):
- Average clip scores per video, then threshold at 0.5.

Metric averaging: `average="binary"` for Precision, Recall, F1 (focus on depressed class).

### 7.3 Baselines

1. **MDAVIF**: Multimodal depression recognition with autoencoder reconstruction.
2. **ViDA-GCAM**: Vision-transformer and cross-attention based method (~467M params).
3. **ViDA**: Simpler version of ViDA-GCAM.
4. **CNN-LSTM (Visual only)**: ResNet18 + LSTM on video frames.
5. **1D-CNN (Audio only)**: CNN on raw audio waveform.
6. **AFGNN-V (Ablated)**: AFGNN with only facial landmark graph.
7. **AFGNN-A (Ablated)**: AFGNN with only audio temporal graph.

### 7.4 Training Protocol

**Stage 1: D-Vlog Main Experiment**
- Train on D-Vlog train split.
- Validate on D-Vlog val split.
- Report test split performance.

**Stage 2: Cross-Dataset Generalization**
- **Zero-shot**: D-Vlog trained model directly tested on AVEC2014.
- **Fine-tune**: D-Vlog pretrained weights fine-tuned on AVEC2014.
- **Scratch**: Train from scratch on AVEC2014 (expected to be weak due to small data).

**Stage 3: Cross-Dataset Reverse**
- Train on AVEC2014, fine-tune on D-Vlog (optional).

---

## 8. Ablation Study Design

The following ablations should be performed to validate each design choice:

| ID | Ablation | Expected Impact | What It Shows |
| :--- | :--- | :--- | :--- |
| A1 | Remove audio branch (AFGNN-V) | F1 ↓ 3–5% | Audio contributes complementary information. |
| A2 | Remove face branch (AFGNN-A) | F1 ↓ 5–8% | Facial geometry is crucial. |
| A3 | Replace GNN with MLP on flattened landmarks | F1 ↓ | Graph structure matters for facial data. |
| A4 | Remove dynamic edges | No clear gain / possible drop | Feature-similarity dynamic edges do not consistently help on D-Vlog. |
| A5 | Remove temporal edges | F1 ↓ 2–4% | Temporal dynamics are important. |
| A6 | Replace audio graph with MLP on flattened acoustic features | F1 ↓ | Graph structure matters for audio dynamics. |
| A7 | Replace GAT with GCN | F1 ↓ 1–3% | Attention mechanisms help. |
| A8 | Replace cross-modal fusion with simple concat | Minor | Advanced fusion is optional. |

---

## 9. Expected Results & Discussion

### 9.1 Hypotheses

1. **Efficiency**: AFGNN will achieve comparable F1 to ViDA-GCAM with **< 5M parameters** (vs. 467M).
2. **Cross-dataset generalization**: AFGNN will outperform ViDA-GCAM on AVEC2014 zero-shot due to using geometric and prosodic features rather than pixel-level appearance.
3. **Interpretability**: Attention weights will highlight mouth and eye regions, consistent with depression literature.
4. **Audio contribution**: Temporal acoustic graph will capture slow/flat speech patterns associated with depression.

### 9.2 Potential Limitations

1. **Landmark quality**: Performance depends on accurate and stable landmark detection.
2. **Occlusion and pose**: Extreme head pose or occlusion may degrade landmark-based features.
3. **Small datasets**: On AVEC2014, even lightweight models may overfit; data augmentation or self-supervised pretraining may be needed.
4. **Audio segment length**: Short clips may not capture enough prosodic information.

### 9.3 Future Extensions

1. **Self-supervised pretraining**: Mask landmark nodes or audio time frames and reconstruct them.
2. **Heterogeneous graph**: Treat face nodes and audio nodes as different types in a single graph (HGT).
3. **Subject-level graph**: Build population graphs across samples for semi-supervised learning.
4. **3D landmarks + head pose**: Include head pose and gaze features.

---

## 10. Figures for Paper

### Figure 1: AFGNN Framework
Overall architecture showing:
- Input video → landmark sequence → spatio-temporal graph → Face GNN → $\mathbf{H}_V$
- Input acoustic features → temporal graph → Audio GNN → $\mathbf{H}_A$
- Fusion module → MLP → depression score

### Figure 2: Facial Landmark Graph Construction
- Subfigure (a): 68 landmark topology on a face
- Subfigure (b): Spatio-temporal graph with intra-frame and inter-frame edges
- Subfigure (c): Dynamic edge examples for smiling vs. neutral expressions

### Figure 3: Audio Temporal Graph Construction
- Subfigure (a): Frame-level acoustic feature sequence
- Subfigure (b): Chain graph with time-frame nodes
- Subfigure (c): Attention weights highlighting salient time frames

### Figure 4: Cross-Modal Fusion
- Subfigure (a): Concatenation fusion
- Subfigure (b): Cross-modal attention fusion

### Figure 5: Interpretability Visualization
- Heatmap of landmark attention weights on depressed vs. non-depressed samples.

---

## 11. Tables for Paper

### Table 1: Dataset Statistics

| Dataset | # Videos | # Clips | # Depressed | # Non-Depressed | Avg. Duration |
| :--- | :---: | :---: | :---: | :---: | :---: |
| D-Vlog Train | — | — | — | — | — |
| D-Vlog Val | — | — | — | — | — |
| D-Vlog Test | — | — | — | — | — |
| AVEC2014 Train | 150 | — | — | — | — |
| AVEC2014 Dev | 50 | — | — | — | — |
| AVEC2014 Test | 50 | — | — | — | — |

### Table 2: Comparison with State-of-the-Art on D-Vlog

| Method | Modality | # Params | Acc | Precision | Recall | F1 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| MDAVIF | V+A+AU | — | — | — | — | — |
| ViDA | V+A+AU | — | — | — | — | — |
| ViDA-GCAM | V+A+AU | 467M | — | 77.94% | — | 78.25% |
| AFGNN (Ours) | L+A (graphs) | <5M | — | — | — | — |

### Table 3: Cross-Dataset Performance (AVEC2014)

| Method | Zero-shot F1 | Fine-tune F1 | Scratch F1 |
| :--- | :---: | :---: | :---: |
| ViDA-GCAM | 65.65% | 71.30% | 58.25% |
| AFGNN (Ours) | — | — | — |

### Table 4: Ablation Study on D-Vlog

| Model | Acc | Precision | Recall | F1 |
| :--- | :---: | :---: | :---: | :---: |
| AFGNN (full) | — | — | — | — |
| - Audio branch | — | — | — | — |
| - Face branch | — | — | — | — |
| - Dynamic edges | — | — | — | — |
| - Temporal edges | — | — | — | — |
| - Cross-modal attention → concat | — | — | — | — |
| - GAT → GCN | — | — | — | — |

---

## 12. Paper Section Outline (Suggested)

1. **Abstract**
2. **Introduction**
   - Depression background and motivation
   - Limitations of CNN/ViT methods
   - Contribution
3. **Related Work**
   - Depression recognition from videos
   - Graph neural networks in affective computing
4. **Method**
   - Problem formulation
   - Facial landmark graph branch
   - Audio temporal graph branch
   - Cross-modal fusion
   - Loss function
5. **Experiments**
   - Datasets and metrics
   - Implementation details
   - Main results
   - Ablation study
   - Cross-dataset evaluation
6. **Visualization and Interpretability**
   - Landmark attention maps
   - Audio temporal attention maps
7. **Conclusion**
8. **References**

---

## 13. Code Structure Plan

```
AFGNN/
├── src/                          # Source code
│   ├── data/                     # Dataset loaders and augmentation
│   ├── models/                   # Model implementations
│   ├── utils/                    # Losses, metrics, trainer
│   ├── scripts/                  # Helper scripts
│   ├── train.py
│   ├── test.py
│   ├── train_combined.py
│   ├── ensemble_test.py
│   ├── seed_ensemble_test.py
│   └── seed_stability_test.py
├── visualization/                # Figure generation scripts
├── paper/                        # Paper source files and figures
├── ablation/                     # Ablation study artifacts
│   ├── experiments/configs/
│   ├── experiments/results/
│   └── experiments/checkpoints/
├── experiments/                  # Other experiment artifacts
│   ├── experiments/configs/
│   ├── experiments/results/
│   └── experiments/checkpoints/
├── records/                      # Notes, logs, and references
└── README.md
```

---

## 14. Notes for Writing

- Emphasize that AFGNN is **not** an incremental improvement of ViDA-GCAM; it is a fundamentally different representation learning paradigm based on graphs.
- Highlight **parameter efficiency** and **cross-dataset generalization** as key selling points.
- Use LaTeX math from this document directly in the paper.
- Keep figures clean: avoid clutter; use color consistently for face/audio branches.
- In the introduction, avoid positioning AFGNN as “fixing ViDA-GCAM”; instead, position it as “exploring a lightweight and interpretable alternative.”

---

*End of Design Document*
