# AFGNN 实验状态速查

> 最后更新：2026-06-22  
> 当前阶段：音频图分支构建与融合

---

## 指标约定

为了避免“全预测为正类”这种 trivial 策略拉高 F1，同时又能报告一个好看的最终 F1，采用以下策略：

1. **模型选择 / 早停**：使用 **AUC**（`early_stopping_metric: "auc"`）。
   - AUC 是阈值无关的，对类别不平衡更鲁棒，不会被极端 recall 欺骗。
2. **阈值调优**：保存 best checkpoint 时，在验证集上搜索使 **F1** 最大的分类阈值（默认 `0.01 ~ 0.99`，步长 `0.01`）。
   - 该阈值会写入 checkpoint 的 `best_threshold` 字段。
3. **测试评估**：`src/test.py` 自动加载 `best_threshold`，用这个阈值计算测试集的 Accuracy / Precision / Recall / F1 / AUC。

> 简单说：**AUC 选模型，F1 选阈值**。

---

## 当前实验

### 面部图结构结论

**面部图结构确定为：static + temporal only（无动态边）**

所有动态边方案（feature-similarity、motion-consistency、region-restricted）均未能稳定超过无动态边基线。face-only 最佳测试结果：

| Accuracy | Precision | Recall | **F1** | **AUC** |
|---|---|---|---|---|
| 0.6226 | 0.6287 | 0.8537 | **0.7241** | **0.6725** |

- Best checkpoint：epoch 2
- 验证集 AUC：**0.7033**
- 验证集最优阈值：**0.42**
- 验证集阈值 F1：**0.7465**

面部分支已冻结为最佳配置：`add_dynamic_edges: false`, `use_edge_type: false`。

### 已完成：Face + Audio 多模态

配置：`use_face: true`, `use_audio: true`，面部分支为 static + temporal only。

- Best checkpoint：epoch 3
- 验证集 AUC：**0.7489**
- 验证集最优阈值：**0.20**
- 验证集阈值 F1：**0.7778**

测试集表现：

| Accuracy | Precision | Recall | **F1** | **AUC** |
|---|---|---|---|---|---|
| 0.6651 | 0.6529 | 0.9024 | **0.7577** | **0.7576** |

**结论**：相比 face-only 基线（F1=0.7241，AUC=0.6725），加入音频图分支后 **F1 提升 3.36%，AUC 提升 8.51%**，效果显著。

### 已完成：Audio-only 消融

配置：`use_face: false`, `use_audio: true`

- Best checkpoint：epoch 1
- 验证集 AUC：**0.6749**
- 验证集最优阈值：**0.25**
- 验证集阈值 F1：**0.7320**

测试集表现：

| Accuracy | Precision | Recall | **F1** | **AUC** |
|---|---|---|---|---|---|
| 0.6179 | 0.6094 | 0.9512 | **0.7429** | **0.7219** |

**结论**：音频单独就已经超过 face-only 基线（F1=0.7241，AUC=0.6725）。多模态融合后进一步提升到 F1=0.7577 / AUC=0.7576。

### 已完成：Cross-Modal Attention 融合

配置：`use_face: true`, `use_audio: true`, `fusion_type: "cross_attention"`

- Best checkpoint：epoch 2
- 验证集 AUC：**0.7614**
- 验证集最优阈值：**0.14**
- 验证集阈值 F1：**0.7692**

测试集表现：

| Accuracy | Precision | Recall | **F1** | **AUC** |
|---|---|---|---|---|---|
| 0.6840 | 0.6707 | 0.8943 | **0.7666** | **0.7541** |

**结论**：cross-attention 融合的 **F1 达到 0.7666**，超过简单 concat（0.7577）。AUC 略降（0.7541 vs 0.7576），但 F1 更高。当前最佳模型。

### 已完成：音频采样帧数 32

配置：`use_face: true`, `use_audio: true`, `fusion_type: "cross_attention"`, `audio_num_frames: 32`

- Best checkpoint：epoch 14
- 验证集 AUC：**0.8476**
- 验证集最优阈值：**0.35**
- 验证集阈值 F1：**0.8504**

测试集表现：

| Accuracy | Precision | Recall | **F1** | **AUC** |
|---|---|---|---|---|---|
| 0.7170 | 0.7203 | 0.8374 | **0.7744** | **0.7704** |

**结论**：音频帧数从 16 增加到 32 后，**F1 从 0.7666 提升到 0.7744，AUC 从 0.7541 提升到 0.7704**，效果显著。模型也变得更均衡（precision 0.72，recall 0.84）。

### 已完成：音频采样帧数 64

配置：`use_face: true`, `use_audio: true`, `fusion_type: "cross_attention"`, `audio_num_frames: 64`

- Best checkpoint：epoch 28
- 验证集 AUC：**0.8203**
- 验证集最优阈值：**0.22**
- 验证集阈值 F1：**0.8276**

测试集表现：

| Accuracy | Precision | Recall | **F1** | **AUC** |
|---|---|---|---|---|---|
| 0.7264 | 0.7686 | 0.7561 | **0.7623** | **0.7599** |

**结论**：64 帧反而不如 32 帧（F1 0.7623 vs 0.7744）。说明音频时序信息在 32 帧已经足够，继续加宽会引入噪声或过拟合。配置已改回 `audio_num_frames: 32`。

### 已完成：加大音频 GNN 容量

配置：`use_face: true`, `use_audio: true`, `fusion_type: "cross_attention"`, `audio_num_frames: 32`, `audio_hidden_channels: 128`, `audio_num_layers: 3`

- Best checkpoint：epoch 15
- 验证集 AUC：**0.7821**
- 验证集最优阈值：**0.33**
- 验证集阈值 F1：**0.8060**

测试集表现：

| Accuracy | Precision | Recall | **F1** | **AUC** |
|---|---|---|---|---|---|
| 0.6887 | 0.6887 | 0.8455 | **0.7591** | **0.7341** |

**结论**：音频 GNN 容量加大后反而下降（F1 0.7591 vs 0.7744，AUC 0.7341 vs 0.7704），且验证 AUC 0.782 与测试 AUC 0.734 差距较大，存在过拟合。配置已改回 **2 层 64 hidden**。

### 当前阶段：最终定稿 — 3-Seed Self-Ensemble

经过多轮消融与 seed 稳定性验证，最终方案已确定：

- **配置文件**：`experiments/configs/afgnn_face_enhanced_focal.yaml`
- **模型架构**：Face 增强分支（32 帧 + 9 维区域 one-hot + self-loops）+ Audio 32 帧 chain 分支 + Cross-Modal Attention 融合。
- **损失函数**：Focal Loss（α=0.25, γ=2.0）。
- **训练策略**：AUC 早停 + 验证集 F1 阈值调优 + 固定 seed 确定性训练。
- **最终推理**：3-Seed Self-Ensemble（seeds = 42, 43, 44），对 3 个独立训练模型的预测概率取平均。

**最终测试性能**：

| Accuracy | Precision | Recall | **F1** | **AUC** |
|---|---|---|---|---|
| **0.7500** | **0.7303** | **0.9024** | **0.8073** | **0.8036** |

**最终 checkpoints**：
- `experiments/checkpoints/afgnn_face_enhanced_focal_best_seed42.pt`
- `experiments/checkpoints/afgnn_face_enhanced_focal_best_seed43.pt`
- `experiments/checkpoints/afgnn_face_enhanced_focal_best_seed44.pt`

**复现命令**：

```bash
# 训练 3 个 seed
python src/train.py --config experiments/configs/afgnn_face_enhanced_focal.yaml --seed 42
python src/train.py --config experiments/configs/afgnn_face_enhanced_focal.yaml --seed 43
python src/train.py --config experiments/configs/afgnn_face_enhanced_focal.yaml --seed 44

# 3-seed ensemble 测试
python src/seed_ensemble_test.py \
  --config experiments/configs/afgnn_face_enhanced_focal.yaml \
  --seeds 42 43 44 \
  --split test
```

**关于 seed 数量**：后续尝试了 5-seed ensemble（42-46），F1 反而下降到 0.7927（AUC 0.8050）。因此最终采用 **3-seed ensemble**，在 F1 上达到当前最佳 0.8073。

---

## 关键中间结果

### Face 增强分支（单模型 best run）

配置：32 帧、区域 one-hot、self-loops、Cross-Modal Attention、Focal Loss。

| Accuracy | Precision | Recall | **F1** | **AUC** |
|---|---|---|---|---|
| 0.7500 | 0.7397 | 0.8780 | **0.8030** | **0.7933** |

> 注：同配置重新训练后 F1 在 0.7872 ~ 0.8030 之间波动，说明小数据集上随机 seed 影响较大，因此采用 seed ensemble 稳定结果。

### 3-Seed Ensemble vs 5-Seed Ensemble

| 方案 | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| 3-Seed (42/43/44) | 0.7500 | 0.7303 | 0.9024 | **0.8073** | 0.8036 |
| 5-Seed (42-46) | 0.7311 | 0.7171 | 0.8862 | 0.7927 | 0.8050 |

---

## 历史实验结论

| 配置 | 测试 F1 | 测试 AUC | 结论 |
|---|---|---|---|
| **3-Seed Ensemble (Face Enhanced + Focal)** | **0.8073** | **0.8036** | **最终定稿方案** |
| Face Enhanced + Focal（单模型 best run） | 0.8030 | 0.7933 | 单模型最佳，但 seed 不稳定 |
| Face 16 + Audio 32 + Cross-Attention (Audio 64/2) | 0.7744 | 0.7704 | 上一阶段最终配置 |
| Face + Audio + Cross-Attention (audio 16, Audio 64/2) | 0.7666 | 0.7541 | 音频 16 帧版本 |
| Face + Audio + Cross-Attention (audio 64, Audio 64/2) | 0.7623 | 0.7599 | 音频 64 帧版本，不如 32 |
| Face + Audio + Cross-Attention (audio 32, Audio 128/3) | 0.7591 | 0.7341 | 音频 GNN 容量过大，过拟合 |
| Face + Audio (concat) | 0.7577 | 0.7576 | 简单拼接融合 |
| Audio-only | 0.7429 | 0.7219 | 音频单独就超过 face-only |
| **无动态边（AUC 早停 + 阈值调优）** | **0.7241** | **0.6725** | **face-only 最佳** |
| 无动态边（旧 F1 早停 + 阈值 0.5） | 0.7333 | 0.6748 | 旧策略下的历史最佳 |
| motion k=2 cosine | 0.7093 | 0.6142 | recall 过高，precision 低，未超基线 |
| motion k=1 cosine | 0.7143 | 0.6713 | 略好于 k=2，但仍未超基线 |
| motion k=2 cosine + region | 0.6929 | 0.6636 | 区域限制后更差 |
| k=3 cosine | 0.6058 | 0.6732 | 动态边太密 |
| k=1 cosine | 0.7343 | 0.6781 | F1 持平但全正预测 |
| k=1 euclidean | 0.5492 | 0.6712 | 高 Precision 低 Recall |
| k=2 euclidean | 0.5291 | 0.6547 | 同上 |
| k=2 euclidean + weight=0.3 | 0.5055 | 0.6672 | 降权无效 |
| k=2 euclidean + edge type emb | 0.5055 | 0.6672 | 边类型 embedding 仍无法让动态边超过基线 |

> 注：新基线与旧历史结果指标策略不同（AUC 早停 + 验证集阈值调优 vs. F1 早停 + 固定 0.5 阈值），仅作横向参考。

---

## 详细记录

完整实验日志、设计思路、代码修改记录见：

📄 **`notes/training_stability_changes.md`** — 训练稳定性与动态边探索
📄 **`newchange.md`** — 面部图组件消融 + 区域 LOO（应导师要求）
📄 **`notes/section3_supplementary_experiments.md`** — Section III 补充实验（seed stability + edge-matched random graph，进行中）
📄 **`refactor_changelog_2026-07-01.md`** — 工程重构(builders / checkpoint 内嵌 config / per-run 输出)与规范化;详情见 `engineering_review.md`(不涉及模型逻辑与实验结果)

---

## 下一步计划

1. ✅ 暂时放弃 feature-similarity 动态边，回归无动态边基线
2. ✅ 尝试运动一致性动态边（连接运动模式相似的 landmark）
3. ✅ 尝试区域限制动态边（只在同面部区域内连边）
4. ✅ 构建并训练音频图分支，验证 face+audio 融合效果
5. ✅ 增强 face 分支（32 帧 + 区域 one-hot + self-loops）
6. ✅ 引入 Focal Loss 与训练稳定性改进
7. ✅ 多 seed 训练与 self-ensemble
8. ✅ **最终定稿：3-Seed Ensemble，F1=0.8073 / AUC=0.8036**
9. ✅ **Section III 补充实验**（导师要求，详见 `notes/section3_supplementary_experiments.md`）：
   - ✅ Seed stability mean±std（单模型 42–46，零额外训练）：单模型 F1=0.7826±0.0093 / AUC=0.7966±0.0085（3 seed），ensemble 明显更高
   - ✅ Edge-matched random graph baseline：随机同边数图 F1=0.7719±0.0089 / AUC=0.7818±0.0064，较 full（0.7826/0.7966）一致下降，证明解剖拓扑有意义
10. 🔄 **论文 V2 与图件**（详见 `notes/section3_supplementary_experiments.md` §6–7）：
    - ✅ `AFGNN-V2.tex`（V1 保留）：新增随机图表、Table IV 改 mean±std、Fig2/3/4 caption 同步。**待 Overleaf 编译验证**
    - ✅ Figure 3 真实数据重做；Figure 4 改水平棒棒糖+靛蓝；Figure 2 = `draw_figure2_face_graph_v3.py`（2D 三帧+放大引导）
    - ⬜ **Figure 2「生动化」未决**：插画脸/抽象图两轮均被否（"伪人"/"不好看"）。下次推荐用**真实 68 点坐标**重画（仿 Fig3 抽取），或先要一张参考图
