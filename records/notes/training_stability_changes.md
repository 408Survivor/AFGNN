# AFGNN 训练稳定性优化变更记录

> 记录人：Kimi Code CLI  
> 日期：2026-06-22  
> 背景：第一次 face-only 训练在 16 epoch 后因早停结束，但验证集 F1 卡在 0.7170，accuracy 0.5588，recall 1.0，precision 0.56，AUC ~0.4-0.5。模型几乎把所有样本预测为抑郁类，说明存在类别不平衡或训练不稳定问题。

---

## 1. 问题诊断

| 现象 | 分析 |
|---|---|
| Recall ≈ 1.0，Precision ≈ 0.56 | 模型倾向于把所有样本判为抑郁类（正类），可能是类别不平衡导致 |
| Accuracy ≈ 0.56 | 与验证集抑郁样本比例接近，模型没有真正学到判别特征 |
| AUC ≈ 0.4-0.5 | 模型输出几乎无区分度 |
| 训练 loss 下降极慢 | 可能存在优化不稳定、BN 与小 batch 不兼容、或 readout 注意力不稳定 |

根因假设（按优先级）：
1. **类别不平衡**：D-Vlog 抑郁样本偏多，标准 BCE 会让模型偏向多数类。
2. **BatchNorm 不适合图数据 + 小 batch**：每个样本是一个图，batch 内节点数波动大，BatchNorm 统计量不稳定。
3. **Readout 注意力实现简单**：之前用 `torch_geometric.utils.softmax` 配合手动 `readout_scale`，可能不够稳定。
4. **Dropout 0.5 对 647 训练样本可能过强**。
5. **缺少梯度裁剪和学习率调度**。

---

## 2. 修改策略

### 2.1 损失函数：处理类别不平衡

**修改文件**：新增 `src/utils/losses.py`，修改 `src/train.py`

- 新增 `compute_class_weights()`：根据训练集正负样本数计算逆频率权重。
- 新增 `FocalLoss`：降低简单样本权重，聚焦难分样本。
- `src/train.py` 支持三种 loss：
  - `"bce"`：标准 BCEWithLogitsLoss
  - `"weighted_bce"`：按类别权重加权（推荐作为 D-Vlog 默认）
  - `"focal"`：Focal Loss

**默认配置**：`loss: "weighted_bce"`

### 2.2 归一化：BatchNorm → LayerNorm

**修改文件**：`src/models/face_gnn.py`

- 将 `nn.BatchNorm1d(hidden_channels)` 替换为 `nn.LayerNorm(hidden_channels)`。
- **理由**：LayerNorm 对每个样本的节点特征独立归一化，不依赖 batch 统计量，更适合图级别任务和小 batch。

### 2.3 Readout：自定义 Attention → PyG AttentionalAggregation

**修改文件**：`src/models/face_gnn.py`

- 移除手动 `readout_scale + softmax + global_add_pool` 实现。
- 使用 `torch_geometric.nn.GlobalAttention(gate_nn=MLP)`，由 PyG 内部保证按 graph 做 softmax。
- **理由**：更稳定、经过充分测试，避免 batch 划分错误导致 attention 跨图计算。

### 2.4 Dropout：0.5 → 0.3

**修改文件**：`experiments/configs/afgnn_face_only.yaml`，默认值从 0.5 改为 0.3。

- **理由**：D-Vlog 训练集仅 647 个样本，过大的 dropout 会严重削弱模型拟合能力。

### 2.5 权重初始化

**修改文件**：`src/models/face_gnn.py`，`src/models/afgnn.py`

- GAT 层权重由 PyG 内部初始化。
- 新增 `_reset_parameters()` 方法，对 `nn.Linear` 使用 Xavier Uniform，对 `nn.LayerNorm` 初始化 weight=1, bias=0。
- **理由**：避免初始输出过大/过小，稳定早期训练。

### 2.6 梯度裁剪

**修改文件**：`src/utils/trainer.py`，`src/train.py`，`experiments/configs/afgnn_face_only.yaml`

- `train_epoch()` 增加 `grad_clip` 参数，训练时调用 `torch.nn.utils.clip_grad_norm_`。
- 配置文件中设置 `grad_clip: 1.0`。
- **理由**：抑制图注意力中可能出现的梯度尖峰。

### 2.7 学习率调度

**修改文件**：`src/utils/trainer.py`，`src/train.py`，`experiments/configs/afgnn_face_only.yaml`

- 新增 `build_scheduler()`，支持：
  - `ReduceLROnPlateau`：验证 F1 不提升时降低学习率
  - `StepLR`：固定步长衰减
- 配置文件中默认启用 `plateau` scheduler（factor=0.5, patience=5）。
- **理由**：当模型收敛平台期时自动降低 LR，避免震荡。

### 2.8 Trainer 与 train.py 接口调整

**修改文件**：`src/utils/trainer.py`，`src/train.py`

- `train_model()` 新增参数 `criterion`、`scheduler`、`grad_clip`。
- `evaluate()` 支持传入 `criterion` 计算验证 loss，也支持不传（测试时）。
- 日志增加当前 learning rate 输出。

---

## 3. 文件变更清单

| 文件 | 变更类型 | 说明 |
|---|---|---|
| `src/utils/losses.py` | 新增 | 加权 BCE、Focal Loss、类别权重计算 |
| `src/models/face_gnn.py` | 修改 | LayerNorm、GlobalAttention、权重初始化 |
| `src/models/afgnn.py` | 修改 | 分类器权重初始化 |
| `src/utils/trainer.py` | 修改 | 支持 criterion/scheduler/grad_clip |
| `src/train.py` | 修改 | 构建 criterion、scheduler，传入新参数 |
| `src/test.py` | 修改 | 适配新的 `evaluate()` 签名 |
| `experiments/configs/afgnn_face_only.yaml` | 修改 | dropout=0.3，loss=weighted_bce，scheduler，grad_clip |
| `notes/training_stability_changes.md` | 新增 | 本文档 |

---

## 4. 下一步建议（不运行实验）

1. 再次运行 `src/train.py` 观察验证 F1 是否能突破 0.72，AUC 是否能明显提升。
2. 如果仍然偏向正类，可尝试：
   - 进一步加大负类权重（调整 `pos_weight`）
   - 使用 Focal Loss 替代 weighted BCE
   - 尝试更小的 `T_v`（如 8 或 12）减少图规模
   - 增加数据增强（如对 landmark 坐标加噪声）
3. 如果验证 AUC 提升但 F1 仍低，可调整分类阈值（当前 0.5）或改用 AUC 作为早停指标。

---

## 5. 关键配置项速查

```yaml
# 当前默认配置
training:
  loss: "weighted_bce"      # 可选 bce / weighted_bce / focal
  focal_alpha: 0.25         # focal 参数
  focal_gamma: 2.0          # focal 参数
  scheduler:
    type: "plateau"         # 可选 null / plateau / step
    factor: 0.5
    patience: 5
  grad_clip: 1.0            # 设为 null 关闭

model:
  dropout: 0.3
```


---

## 6. 动态边扩展（方案 A：特征相似度 top-K）

> 新增日期：2026-06-22  
> 实现：方案 A（特征相似度 top-K，非可学习 MLP）

### 6.1 实现思路

对每一帧内的 68 个 landmark 节点，基于节点特征 `[x, y, dx, dy]` 计算两两相似度：
- **cosine**：余弦相似度
- **euclidean**：负欧氏距离

对每个节点，取相似度最高的 `dynamic_k` 个节点（排除自身和已有的静态/时序邻居）建立双向边。

### 6.2 涉及文件

| 文件 | 变更 |
|---|---|
| `src/models/graph_utils.py` | 新增 `build_dynamic_edges()`；`build_face_graph()` 增加 `add_dynamic_edges`、`dynamic_k`、`dynamic_metric` 参数 |
| `src/data/dvlog_face_dataset.py` | Dataset 和 DataLoader 透传动态边参数 |
| `src/models/afgnn.py`（间接） | 无需修改，模型结构不变 |
| `src/train.py` / `src/test.py` | 从 config 读取动态边参数并传入 DataLoader |
| `experiments/configs/afgnn_face_only.yaml` | 新增 `add_dynamic_edges: true`、`dynamic_k: 3`、`dynamic_metric: "cosine"` |

### 6.3 默认配置

```yaml
model:
  add_static_edges: true
  add_temporal_edges: true
  add_dynamic_edges: true
  dynamic_k: 3
  dynamic_metric: "cosine"
```

### 6.4 消融说明

若要验证动态边是否有用，可直接改配置：

```yaml
model:
  add_dynamic_edges: false
```

然后重新训练对比 F1/AUC。


---

## 7. 动态边调参实验计划（进行中）

> 记录日期：2026-06-22  
> 目标：不放弃动态边的前提下，找到有效的动态边配置

### 已完成的消融

| 配置 | 验证 F1 | 测试 F1 | 测试 Recall | 结论 |
|---|---|---|---|---|
| 无动态边 | 0.6986 | **0.7333** | 0.8943 | 当前最佳基线 |
| k=3, cosine | 0.6531 | 0.6058 | 0.5122 | 动态边太密，引入噪声 |
| k=1, cosine | 0.7170 | 0.7343 | **1.0000** | F1 与基线持平，但模型退化为全正预测 |
| k=1, euclidean | 0.7002 (AUC) | 0.5492 | 0.4309 | AUC 早停避免全正，但 F1 下降明显 |
| k=2, euclidean | 0.7170 (AUC) | 0.5291 | 0.4065 | 验证 AUC 最高，但测试 F1 仍低 |

### 结论

简单 top-K 动态边（k=1/2/3，cosine/euclidean）均未能稳定超过无动态边基线。**跳过 k=2 cosine，直接尝试 edge weight**。

### 当前实验

- **配置**：`add_dynamic_edges: true`, `dynamic_k: 2`, `dynamic_metric: "euclidean"`, `dynamic_edge_weight: 0.3`
- **早停指标**：`auc`
- **目的**：降低动态边在 GAT attention 中的影响力，保留静态/时序边的主导地位


---

## 8. 早停指标可配置化

> 新增日期：2026-06-22  
> 背景：k=1 cosine 实验保存的“最佳模型”其实是全正预测（val_rec=1.0, val_prec=0.56），说明 F1 作为早停指标容易被“作弊”。

### 8.1 修改内容

| 文件 | 变更 |
|---|---|
| `src/utils/trainer.py` | `train_model()` 新增 `early_stopping_metric` 参数，支持 `"f1"`、`"auc"`、`"accuracy"`、`"loss"` |
| `src/train.py` | 从 config 读取 `early_stopping_metric` 并传入 |
| `src/test.py` | 加载 checkpoint 时显示对应的早停指标名称和分数 |
| `experiments/configs/afgnn_face_only.yaml` | 新增 `early_stopping_metric: "auc"` |

### 8.2 推荐默认

```yaml
training:
  early_stopping_metric: "auc"
```

**理由**：
- AUC 衡量排序能力，不受阈值影响
- 不会被“全预测为正类”这种 trivial 策略作弊
- 在类别不平衡的二分类任务中更稳定

### 8.3 为什么不只用 F1？

F1 仍然重要，是最终报告指标。但它作为**早停指标**不稳定，因为模型可以通过极端提高 recall 来获得较高 F1，即使 precision 很低。

---

## 9. 常见项目早停指标选择

在抑郁识别、情感计算等多模态不平衡分类任务中，文献和比赛常用的早停/模型选择指标：

| 指标 | 优点 | 缺点 | 适用场景 |
|---|---|---|---|
| **AUC-ROC** | 阈值无关；对类别不平衡鲁棒；不会被全正/全负预测欺骗 | 不直接反映精确率/召回率 | **最常用**，推荐作为默认早停指标 |
| **F1 / Macro-F1** | 直接反映 Precision-Recall 平衡 | 容易被极端 recall 拉高 | 可作为辅助指标，不建议单独早停 |
| **Balanced Accuracy** | 考虑类别不平衡 | 对类别比例敏感 | 类别不平衡严重时可用 |
| **Validation Loss** | 简单直接 | 可能过拟合到 loss _surface 而不是真实性能 | 数据量大、类别平衡时可用 |

**主流做法**：
- 早停用 **AUC**
- 最终报告 **Accuracy / Precision / Recall / F1 / AUC**
- 有些工作会用 **Macro-F1** 或 **Weighted-F1** 做最终比较



---

## 10. 动态边 Edge Weight（当前实验）

> 新增日期：2026-06-22  
> 背景：简单 top-K 动态边始终未能超过基线，尝试通过降低动态边 attention 权重来保留其信息同时减少噪声。

### 10.1 实现思路

- 静态边和时序边：weight = 1.0
- 动态边：weight = `dynamic_edge_weight`（可配置，如 0.3）
- GAT attention 分数在 softmax 前乘以 edge_weight
- 这样动态边仍然参与消息传递，但对目标节点的影响被削弱

### 10.2 涉及文件

| 文件 | 变更 |
|---|---|
| `src/models/weighted_gat_conv.py` | 新增自定义 `WeightedGATConv`，支持 `edge_weight` |
| `src/models/face_gnn.py` | 用 `WeightedGATConv` 替换 `GATConv` |
| `src/models/graph_utils.py` | `build_face_graph()` 生成 `edge_weight`；动态边 duplicates 取最大 weight |
| `src/data/dvlog_face_dataset.py` | 透传 `dynamic_edge_weight` |
| `src/train.py` / `src/test.py` | 从 config 读取 `dynamic_edge_weight` |
| `experiments/configs/afgnn_face_only.yaml` | `dynamic_edge_weight: 0.3` |

### 10.3 当前配置

```yaml
model:
  add_dynamic_edges: true
  dynamic_k: 2
  dynamic_metric: "euclidean"
  dynamic_edge_weight: 0.3
```

### 10.4 消融说明

若 0.3 有效，可继续尝试 0.1、0.5；若无效，则动态边可能需要更复杂的建模（如边类型 embedding）。


---

## 11. 边类型 Embedding 实验（待跑）

> 记录日期：2026-06-22  
> 目标：让模型自动学习如何区分 static / temporal / dynamic 边，而不是靠手工权重抑制动态边。

### 11.1 实现思路

- 定义三种边类型：
  - `0`: static（解剖边）
  - `1`: temporal（时序边）
  - `2`: dynamic（动态边）
- 每种边类型学习一个 embedding
- 在 GAT attention 计算中，把 edge embedding 投影到 attention 空间并加到 attention logit 上
- 让模型自己学习：static/temporal 边应该被重视，dynamic 边应该被抑制

### 11.2 涉及文件

| 文件 | 变更 |
|---|---|
| `src/models/weighted_gat_conv.py` | 新增 `edge_emb` 和 `edge_emb_proj`，支持 `edge_type` |
| `src/models/face_gnn.py` | 透传 `num_edge_types`、`edge_emb_dim`、`edge_type` |
| `src/models/afgnn.py` | 透传 `num_edge_types`、`edge_emb_dim` |
| `src/models/graph_utils.py` | `build_face_graph()` 生成 `edge_type`；duplicate edges 按优先级保留类型 |
| `src/data/dvlog_face_dataset.py` | 透传 `use_edge_type` |
| `src/train.py` / `src/test.py` | 从 config 读取并传入 |
| `experiments/configs/afgnn_face_only.yaml` | `use_edge_type: true`, `num_edge_types: 3`, `edge_emb_dim: 4` |

### 11.3 当前配置

```yaml
model:
  add_dynamic_edges: true
  dynamic_k: 2
  dynamic_metric: "euclidean"
  dynamic_edge_weight: 1.0    # 使用 edge type embedding 时设为 1.0
  use_edge_type: true
  num_edge_types: 3
  edge_emb_dim: 4
training:
  early_stopping_metric: "auc"
```

### 11.4 明天实验命令

```bash
cd /home/ltq/DepressionCode/DepGNN/AFGNN
conda activate DVlog
python src/train.py --config experiments/configs/afgnn_face_only.yaml
```

训练结束后测试：

```bash
cd /home/ltq/DepressionCode/DepGNN/AFGNN
conda activate DVlog
python src/test.py --config experiments/configs/afgnn_face_only.yaml --split test
```

### 11.5 若仍无效

如果边类型 embedding 仍无法让动态边超过基线，建议：
1. 暂时放弃 feature-similarity 动态边
2. 尝试**运动一致性动态边**：只连接运动方向/幅度相似的 landmark（更合理的面部动作假设）
3. 或尝试**区域限制动态边**：只在同面部区域内连动态边
4. 把重心转回**音频 STFT 图分支**的构建
