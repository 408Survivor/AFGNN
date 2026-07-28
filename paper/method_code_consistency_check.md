# AFGNN-V3 论文方法与代码一致性检查

**论文来源：** `paper/AFGNN-V3.tex`（当前版本，2026 年 7 月）  
**代码来源：** `src/`（主要为 `models/`、`data/`、`utils/`、`train.py`、`seed_ensemble_test.py`）

**总体结论：** 当前 AFGNN-V3 论文与 `src/` 中的实现**高度一致**。核心方法声明——AU-informed 人脸图构建、音频时序图、GAT 编码器、图级门控融合、Focal Loss、基于验证集的 checkpoint/阈值选择、3-seed 概率集成——均在代码中得到了忠实实现。旧版本中存在的训练超参数不一致（batch size、epoch 数）已在 V3 中解决。目前唯一需要关注的问题是 **`src/ensemble_test.py` 中实现了一个论文未描述的异构 5-expert 集成**。

---

## 1. 输入与图构建：一致

| 论文描述（Sec. II-B） | 代码实现 | 是否一致 |
|---|---|---|
| 视觉输入 $P\in\mathbb{R}^{T_v\times68\times2}$，音频输入 $A\in\mathbb{R}^{T_a\times25}$，$d_a=25$ | `build_face_graph` 将 `(T, 136)` reshape 为 `(T, 68, 2)`；`build_audio_graph` 使用 `(T, 25)` | ✅ |
| 人脸节点特征 $x^f=[p;\Delta p;r]\in\mathbb{R}^{13}$ | 当 `use_velocity=True` 且 `add_region_onehot=True` 时，代码拼接 `[x, y, dx, dy]`（4 维）+ 9 维区域 one-hot = 13 维；最终配置均开启 | ✅ |
| 9 个面部区域：轮廓、左右眉毛、鼻梁、鼻头、左右眼、外嘴、内嘴 | `LANDMARK_GROUPS_68` 正确定义了这 9 个区域，且**顺序与论文一致** | ✅ |
| 静态拓扑边：开放组件用链式，闭合组件用环式 | `build_static_edges`：轮廓/眉毛/鼻梁用开放链；鼻头、眼睛、嘴用环 | ✅ |
| 时序边连接相邻帧同一 landmark，双向 | `build_temporal_edges` 构建双向时序边 | ✅ |
| 自环边 $E_{\mathrm{self}}=\{(v^f_{i,t},v^f_{i,t})\}$ | `add_self_loops=True` 为每个节点添加自环；最终配置启用 | ✅ |
| 音频图为双向时序链 | `AudioGNN.build_batched_chain_edges` 从 `batch` 构建双向链边 | ✅ |
| 音频标准化：训练集 $\mu,\sigma$，std$<10^{-8}$ 时夹到 $1.0$ | `build_audio_graph` 中 `std[std < 1e-8] = 1.0` | ✅ |
| 动态边仅用于消融实验 | `add_dynamic_edges` 默认 `False`；最终配置关闭。`build_dynamic_edges` 支持 feature-similarity（`full`）和 motion-consistency（`motion`）两种变体 | ✅ |

**次要实现说明：**
- `build_audio_graph` 会构建 `audio_edge_index`，但返回的 `Data` 对象**没有附带**该边索引；`AudioGNN` 内部根据 `batch` 重新构建链边，因为音频图节点数与人脸图不同。这属于实现细节，不改变图拓扑。
- 当启用 `edge_type` 输出时，自环边被标记为 `EDGE_TYPE_STATIC`（0），而非独立的类型。由于最终配置 `use_edge_type=False`，不影响报告模型。

---

## 2. 图编码与门控融合：一致

| 论文描述（Sec. II-C） | 代码实现 | 是否一致 |
|---|---|---|
| 人脸 GAT：3 层，hidden=128，heads=4 | `FaceGNN` 匹配；最终配置 `face_num_layers=3`、`face_hidden_channels=128`、`face_heads=4` | ✅ |
| 音频 GAT：2 层，hidden=64，heads=4 | `AudioGNN` 匹配；最终配置 `audio_num_layers=2`、`audio_hidden_channels=64`、`audio_heads=4` | ✅ |
| Dropout = 0.3 | 配置一致 | ✅ |
| 门控读出：门控网络 + 节点级 softmax | PyG 的 `AttentionalAggregation` + MLP `readout_gate`；内部使用 softmax | ✅ |
| 双向图级门控：$q/k/v$ 投影、sigmoid 标量门、残差更新（Eq. 6） | `CrossModalAttentionFusion` 精确实现 Eq. 6：点积注意力缩放 $\sqrt{d_g}$、sigmoid、对 partner 模态做残差投影 | ✅ |
| 最终分类 MLP $\hat{y}=\sigma(\mathrm{MLP}(z))$ | `AFGNN.classifier`：`Linear -> ReLU -> Dropout -> Linear`；sigmoid 在 `evaluate` / `compute_metrics` 中对 logit 应用 | ✅ |

**Eq. 6 的代码对应：**
- $q_f = W_q^f h_f$ → `self.face_query = nn.Linear(face_dim, hidden_dim)`
- $k_a = W_k^a h_a$ → `self.audio_key = nn.Linear(audio_dim, hidden_dim)`
- $W_v^{a\rightarrow f} h_a$ → `self.audio_value = nn.Linear(audio_dim, face_dim)`
- $g_{f\leftarrow a}=\sigma(q_f^\top k_a / \sqrt{d_g})$ → `torch.sum(q_face * k_audio, dim=-1) * self.scale`，其中 `scale = hidden_dim ** -0.5`
- 残差更新 $\tilde{h}_f = h_f + g_{f\leftarrow a} W_v^{a\rightarrow f} h_a$ 直接实现。

---

## 3. 损失、验证与集成：一致

| 论文描述（Sec. II-C） | 代码实现 | 是否一致 |
|---|---|---|
| Focal loss，$\alpha=0.25$，$\gamma=2$ | `FocalLoss` 正确实现；`ablation_full.yaml` 与 `afgnn_face_enhanced_focal.yaml` 使用 | ✅ |
| 验证集用于 checkpoint 选择（AUC）和阈值校准（F1） | `train_model` 使用 `early_stopping_metric="auc"` 选模型，保存最佳 checkpoint 时调用 `find_best_threshold` 在验证集上搜索阈值 | ✅ |
| 推理：$K=3$ 个独立训练模型在概率层面平均 | `seed_ensemble_test.py` 对 seeds 42、43、44 做概率平均 | ✅ |

**报告结果验证：**
- `experiments/results/seed_ensemble_42-44_test_results.json` 报告 **F1 = 0.8077**，**AUC = 0.8041**，与摘要及 Table `tab:sota` 一致。
- `experiments/results/seed_ensemble_42-46_test_results.json`（5 seeds）报告 F1 = 0.7927、AUC = 0.8050，与 Table `tab:design_ablation` 一致。

---

## 4. 实验设置：一致

| 论文描述（Sec. III-A） | 代码实现 | 是否一致 |
|---|---|---|
| $T_v=32$，$T_a=32$ | 配置中 `num_frames: 32`、`audio_num_frames: 32` | ✅ |
| Batch size = 32 | `training.batch_size: 32` | ✅ |
| 最多 100 epochs，早停耐心值 = 15 | `training.epochs: 100`、`training.patience: 15` | ✅ |
| Adam，lr = $10^{-3}$，weight decay = $10^{-4}$ | `train.py` 使用 `torch.optim.Adam` 及对应参数 | ✅ |
| Dropout = 0.3 | 配置一致 | ✅ |
| ReduceLROnPlateau 调度器，factor 0.5，patience 5 | `training.scheduler` 已相应配置 | ✅ |
| 梯度裁剪最大范数 1.0 | `training.grad_clip: 1.0` | ✅ |
| 不使用 landmark 数据增强（会降低验证 AUC） | 最终配置 `augmentation.enabled: false`；`augmentation.py` 存在但未被报告模型使用 | ✅ |
| ~$1.2\times10^5$ 可训练参数 | 代码实测 **118,979** 个参数（≈ $1.19\times10^5$） | ✅ |

**最终配置下的参数 breakdown：**

| 组件 | 参数数量 |
|---|---|
| `face_gnn` | 44,417 |
| `audio_gnn` | 8,321 |
| `fusion`（cross-attention） | 41,408 |
| `classifier` | 24,833 |
| **总计** | **118,979** |

---

## 5. 消融实验：数值一致

`paper/AFGNN-V3.tex` 中的消融数值与 `ablation/results/` 和 `experiments/results/` 中的结果文件一致。

### 5.1 人脸图组件消融（Table `tab:graph_ablation`）

| 配置 | 论文 F1 | 论文 AUC | 代码结果文件 | 实测 F1 | 实测 AUC | 是否一致 |
|---|---|---|---|---|---|---|
| Full | 0.794 | 0.802 | `test_full.json` / `summary.json` | 0.7943 | 0.8017 | ✅ |
| -Static edges | 0.786 | 0.792 | `test_no_static.json` | 0.7863 | 0.7923 | ✅ |
| -Temporal edges | 0.748 | 0.763 | `test_no_temporal.json` | 0.7480 | 0.7631 | ✅ |
| -Region one-hot | 0.789 | 0.786 | `test_no_region.json` | 0.7895 | 0.7860 | ✅ |
| -Velocity $[dx,dy]$ | 0.752 | 0.760 | `test_no_velocity.json` | 0.7518 | 0.7599 | ✅ |
| -Self-loop | 0.748 | 0.778 | `test_no_selfloop.json` | 0.7480 | 0.7784 | ✅ |

### 5.2 模态与融合消融（Table `tab:fusion_ablation`）

论文报告了 face-only / audio-only / concat / gated-fusion 的结果。代码通过 `use_face`、`use_audio` 和 `fusion_type`（`concat` vs `cross_attention`）支持这四种配置。单种子 gated-fusion 结果（F1 = 0.787，AUC = 0.793）与 Table `tab:design_ablation` 中 “Audio 32 frames / Focal loss / Single seed 42” 行一致。

### 5.3 训练/推理设计消融（Table `tab:design_ablation`）

| 设置 | 论文 F1 | 论文 AUC | 代码证据 | 是否一致 |
|---|---|---|---|---|
| Audio 16 frames | 0.767 | 0.754 | `history_face_audio.json`（实验结果文件） | ✅ |
| Audio 32 frames | 0.787 | 0.793 | `history_face_audio_crossattn_audio32.json` / 单种子结果 | ✅ |
| Audio 64 frames | 0.762 | 0.760 | `history_face_audio_crossattn_audio64.json` | ✅ |
| Weighted BCE | 0.782 | 0.808 | `afgnn_face_enhanced.yaml` 使用 weighted BCE | ✅ |
| Focal loss | 0.787 | 0.793 | `afgnn_face_enhanced_focal.yaml` 使用 focal loss | ✅ |
| Single seed 42 | 0.787 | 0.793 | 单种子 checkpoint 结果 | ✅ |
| 3-seed ensemble | **0.8077** | **0.8041** | `seed_ensemble_42-44_test_results.json` | ✅ |
| 5-seed ensemble | 0.793 | 0.805 | `seed_ensemble_42-46_test_results.json` | ✅ |

### 5.4 面部区域敏感度分析（Table `tab:region_ablation`）

`ablation/results/summary.json` 中的区域 leave-one-out 结果与 Table `tab:region_ablation` 的排名和幅度一致：

| 区域 | 论文 F1 | 论文 F1 下降 | 实测 F1 | 实测下降 | 是否一致 |
|---|---|---|---|---|---|
| Left eye | 0.743 | -5.15% | 0.7429 | -5.15% | ✅ |
| Face contour | 0.754 | -4.02% | 0.7541 | -4.02% | ✅ |
| Nose bottom | 0.770 | -2.45% | 0.7698 | -2.45% | ✅ |
| Right eye | 0.773 | -2.16% | 0.7727 | -2.16% | ✅ |
| Right eyebrow | 0.779 | -1.56% | 0.7787 | -1.56% | ✅ |
| Inner mouth | 0.780 | -1.47% | 0.7797 | -1.47% | ✅ |
| Nose bridge | 0.782 | -1.24% | 0.7820 | -1.24% | ✅ |
| Left eyebrow | 0.783 | -1.14% | 0.7829 | -1.14% | ✅ |
| Outer mouth | 0.787 | -0.78% | 0.7865 | -0.78% | ✅ |

---

## 6. 值得注意的不一致与论文未提及的细节

### 6.1 未在论文中描述的异构 5-expert 集成（`src/ensemble_test.py`）

论文仅描述了 **3-seed 自集成**（同一配置、不同随机种子、概率平均）。但 `src/ensemble_test.py` 实现了一个 **5-expert late-fusion 集成**：

- `face_audio_baseline`（Face 16 + Audio 32 + Cross-Attn）
- `audio_only`
- `face_audio_enhanced`（Face 32 + region one-hot + self-loops + Cross-Attn）
- `face_audio_enhanced_concat`（同上，但使用 Concat fusion）
- `face_audio_enhanced_focal`（同上，但使用 Focal loss）

该集成：
- 包含一个 **audio-only** 专家和一个 **concat-fusion** 专家，二者均不是论文最终采用的 gated-fusion AFGNN。
- 通过在验证集上网格搜索学习各专家权重。
- **在 Method 部分没有任何描述**。

**缓解证据：** 该集成的最佳结果约为 **F1 ≈ 0.785**，低于报告的 3-seed 集成结果（F1 = 0.8077），因此**不是**主报告结果的来源。但若该脚本仍属于项目的一部分，建议要么（a）在论文中将其作为辅助实验文档化，要么（b）直接删除以避免方法上的混淆。

### 6.2 代码默认值与论文多模态默认表述不一致

`AFGNN.__init__` 默认 `use_audio=False`，`build_model` 也保留此默认。论文则将 AFGNN 默认呈现为 face+audio 多模态模型。这不属于方法不一致，因为最终 YAML 配置会覆盖默认值（`use_face: true`、`use_audio: true`），但代码默认行为与论文描述存在偏差。

### 6.3 最终模型未使用的音频图扩展功能

代码支持音频自环、skip 边、delta+delta-delta 特征（`audio_self_loops`、`audio_skip`、`audio_use_delta`）。最终配置**未启用**这些选项，因此不影响报告方法。它们属于实验/消融能力，若不作为报告结果则无需在论文中描述。

### 6.4 `train_combined.py` 使用不同的训练/验证划分策略

`src/train_combined.py` 将官方 train 和 valid 合并后再留出 10% 作为内部验证集。而论文说明使用官方 7:1:2 划分。主训练入口 `src/train.py` 使用官方划分，因此 `train_combined.py` 看起来是辅助脚本（例如用于交叉验证或更多数据实验），不矛盾于论文主要协议。

---

## 7. 总结与建议

### 一致的组件 ✅
- AU-informed 人脸图结构（9 个区域、拓扑边、时序边、自环）
- 音频时序图构建与训练集标准化
- GAT 编码器架构（深度、隐藏维度、头数、dropout）
- 图级门控读出（`AttentionalAggregation`）
- 双向跨模态门控融合（`CrossModalAttentionFusion`）
- Focal loss 公式与超参数
- 验证集 AUC 选 checkpoint + 验证集 F1 阈值校准
- 3-seed 概率集成，得到报告的 F1 = 0.8077 / AUC = 0.8041
- Sec. III-A 中所有训练超参数
- 参数量（~1.2×10⁵）
- Sec. III-C 中所有消融数值

### 不一致或论文未提及的组件 ⚠️
- `src/ensemble_test.py`（5-expert 异构集成）未在 Method 部分描述。其性能低于 3-seed 集成，因此不推翻主结果，但应文档化或删除。
- `AFGNN.__init__` 默认 `use_audio=False`，与论文多模态默认表述不一致。
- 可选音频扩展（自环、skip 边、delta 特征）在代码中存在，但未被最终模型使用。

### 建议操作
1. **澄清 `src/ensemble_test.py`**：要么在论文中将其作为辅助实验文档化，要么若不再使用则删除该脚本。
2. **可选：统一代码默认值**：将 `AFGNN.__init__` 和 `build_model` 的默认 `use_audio` 设为 `True`，以匹配论文的多模态表述。
3. **报告方法无需进一步修改论文**：V3 稿件已准确反映当前代码。
