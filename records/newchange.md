# AFGNN 面部图组件消融实验 — 修改与实验记录

> 记录日期：2026-06-25  
> 记录人：Kimi Code CLI  
> 背景：导师反馈 Table III 未能充分拆清面部图各组件贡献，要求补充组件消融与区域重要性分析。

---

## 1. 导师要求

论文 Table III 已说明 `static + temporal + region + self-loop` 优于 dynamic graph，但尚未拆清楚以下组件各自的贡献：

1. **anatomical edges**（静态解剖边）
2. **temporal edges**（时序边）
3. **region features**（面部区域 one-hot 特征）
4. **velocity/motion features**（速度/运动特征 `[dx, dy]`）
5. **self-loop**（自环边）

另外要求给出一份 **脸部区域 top-k 重要性** 分析（即 9 个面部区域按贡献排序）。

---

## 2. 代码修改

### 2.1 `src/models/graph_utils.py`

在 `build_face_graph()` 中新增两个参数：

- `use_velocity: bool = True`
  - `True`：节点特征为 `[x, y, dx, dy]`（含速度）。
  - `False`：节点特征仅为 `[x, y]`，用于消融 velocity 贡献。
- `region_ablation: Optional[int] = None`
  - 若指定区域 id（0–8），则将该面部区域的 one-hot 向量置零，用于 leave-one-out 区域重要性分析。

同时更新了函数 docstring。

### 2.2 `src/data/dvlog_face_dataset.py`

`DVlogFaceDataset`、`DVlogFaceArrayDataset`、`get_dvlog_face_loaders` 均新增并透传：

- `use_velocity: bool = True`
- `region_ablation: Optional[int] = None`

文件顶部 import 增加 `Optional`。

### 2.3 `src/train.py`

从配置读取并透传：

```python
use_velocity=cfg["model"].get("use_velocity", True),
region_ablation=cfg["model"].get("region_ablation", None),
```

### 2.4 `src/test.py`

1. 从配置读取并透传 `use_velocity` 与 `region_ablation`。
2. **修复 checkpoint 默认路径**：原默认硬编码为 `experiments/checkpoints/afgnn_face_only_best.pt`，与 `--config` 无关。现改为若未指定 `--checkpoint`，自动从配置文件的 `training.checkpoint_path` 读取，避免加载错误模型。

---

## 3. 新增文件

### 3.1 `src/scripts/generate_ablation_configs.py`

自动生成消融实验 YAML：

- 6 个组件消融配置
- 9 个面部区域 leave-one-out 配置

每个配置使用独立的 `checkpoint_path`，避免互相覆盖。

### 3.2 `ablation/configs/*.yaml`

共生成 15 个配置文件：

| 配置 | 说明 |
|---|---|
| `ablation_full.yaml` | 完整模型：static + temporal + region + velocity + self-loop |
| `ablation_no_static.yaml` | 去掉 anatomical edges |
| `ablation_no_temporal.yaml` | 去掉 temporal edges |
| `ablation_no_region.yaml` | 去掉 region one-hot（`face_in_channels: 4`） |
| `ablation_no_velocity.yaml` | 去掉 velocity（`face_in_channels: 11`） |
| `ablation_no_selfloop.yaml` | 去掉 self-loop |
| `ablation_region_mask_00_face_contour.yaml` | mask face_contour 区域 one-hot |
| `ablation_region_mask_01_left_eyebrow.yaml` | mask left_eyebrow 区域 one-hot |
| `ablation_region_mask_02_right_eyebrow.yaml` | mask right_eyebrow 区域 one-hot |
| `ablation_region_mask_03_nose_bridge.yaml` | mask nose_bridge 区域 one-hot |
| `ablation_region_mask_04_nose_bottom.yaml` | mask nose_bottom 区域 one-hot |
| `ablation_region_mask_05_left_eye.yaml` | mask left_eye 区域 one-hot |
| `ablation_region_mask_06_right_eye.yaml` | mask right_eye 区域 one-hot |
| `ablation_region_mask_07_outer_mouth.yaml` | mask outer_mouth 区域 one-hot |
| `ablation_region_mask_08_inner_mouth.yaml` | mask inner_mouth 区域 one-hot |

### 3.3 `src/scripts/run_ablations.sh`

批量顺序运行所有消融实验：

- 对每个实验执行 `src/train.py` + `src/test.py`
- 将 `experiments/results/test_test_results.json` 复制为 `ablation/results/test_{name}.json`
- 将 `experiments/results/training_history.json` 复制为 `ablation/results/history_{name}.json`
- 若某实验结果已存在则自动跳过

---

## 4. 实验设计

### 4.1 组件消融（基于最终模型 face+audio + cross-attention + Focal Loss）

| 实验名 | 改动 | 目的 |
|---|---|---|
| `full` | 完整配置 | 基线 |
| `no_static` | `add_static_edges: false` | 评估 anatomical edges 贡献 |
| `no_temporal` | `add_temporal_edges: false` | 评估 temporal edges 贡献 |
| `no_region` | `add_region_onehot: false` | 评估 region features 贡献 |
| `no_velocity` | `use_velocity: false` | 评估 velocity/motion features 贡献 |
| `no_selfloop` | `add_self_loops: false` | 评估 self-loop 贡献 |

### 4.2 面部区域 leave-one-out

每次将一个面部区域的 one-hot 特征置零，其余设置保持不变：

| 区域 id | 区域名称 |
|---|---|
| 0 | face_contour |
| 1 | left_eyebrow |
| 2 | right_eyebrow |
| 3 | nose_bridge |
| 4 | nose_bottom |
| 5 | left_eye |
| 6 | right_eye |
| 7 | outer_mouth |
| 8 | inner_mouth |

通过比较 `full` 与每个 mask 配置的 F1/AUC 下降幅度，得到区域重要性排名 Top-k。

---

## 5. 运行状态与结果

### 已验证基线

`ablation_full` 已完成训练与测试：

| Metric | Value |
|---|---|
| Accuracy | 0.7264 |
| Precision | 0.7044 |
| Recall | 0.9106 |
| F1 | **0.7943** |
| AUC | **0.8017** |

### 已完成

全部 15 个消融实验已完成。汇总结果保存在 `ablation/results/summary.json`。

---

## 6. 实验结果汇总

### 6.1 组件消融结果

基线（`full`）：F1 = 0.7943，AUC = 0.8017

| 配置 | 说明 | Test F1 | Test AUC | F1 下降 | AUC 下降 |
|---|---|---:|---:|---:|---:|
| `full` | 完整配置 | 0.7943 | 0.8017 | — | — |
| `no_temporal` | 去掉 temporal edges | 0.7480 | 0.7631 | **-4.64%** | -3.85% |
| `no_selfloop` | 去掉 self-loop | 0.7480 | 0.7784 | **-4.63%** | -2.33% |
| `no_velocity` | 去掉 velocity `[dx, dy]` | 0.7518 | 0.7599 | -4.25% | -4.18% |
| `no_static` | 去掉 anatomical edges | 0.7863 | 0.7923 | -0.81% | -0.94% |
| `no_region` | 去掉 region one-hot | 0.7895 | 0.7860 | -0.49% | -1.57% |

**结论**：
- **temporal edges** 和 **self-loop** 贡献最大，去掉后 F1 均下降约 4.6%。
- **velocity/motion features** 次之，去掉后 F1 下降 4.25%。
- **anatomical edges** 和 **region one-hot** 贡献相对较小。

### 6.2 面部区域重要性 Top-k（leave-one-out）

每次将一个区域的 one-hot 特征置零，按 F1 下降幅度排序：

| 排名 | 区域 | Test F1 | Test AUC | F1 下降 | AUC 下降 |
|---:|---|---:|---:|---:|---:|
| 1 | left_eye | 0.7429 | 0.7886 | **-5.15%** | -1.31% |
| 2 | face_contour | 0.7541 | 0.7714 | -4.02% | -3.03% |
| 3 | nose_bottom | 0.7698 | 0.7965 | -2.45% | -0.52% |
| 4 | right_eye | 0.7727 | 0.7904 | -2.16% | -1.12% |
| 5 | right_eyebrow | 0.7787 | 0.8121 | -1.56% | **+1.04%** |
| 6 | inner_mouth | 0.7797 | 0.7948 | -1.47% | -0.69% |
| 7 | nose_bridge | 0.7820 | 0.7876 | -1.24% | -1.41% |
| 8 | left_eyebrow | 0.7829 | 0.7913 | -1.14% | -1.04% |
| 9 | outer_mouth | 0.7865 | 0.7701 | -0.78% | -3.16% |

**结论**：
- **left_eye** 是最重要的区域，mask 后 F1 下降 5.15%。
- **face_contour** 次之，F1 下降 4.02%。
- **outer_mouth** 在 F1 指标上最不敏感，但 AUC 下降 3.16%，说明该区域对排序能力仍有帮助。
- **right_eyebrow** mask 后 AUC 反而上升 1.04%，可能是随机波动或轻微正则化效应。

### 6.3 对论文 Table III 的建议

可在 Table III 中新增两行/一列：

1. **组件消融子表**：列出 `-static`、`-temporal`、`-region`、`-velocity`、`-self-loop` 的 F1/AUC。
2. **区域重要性子表**：列出 Top-3 或 Top-5 关键区域（left_eye、face_contour、nose_bottom）。

这样既回应了导师"拆清楚每个组件贡献"的要求，也提供了"脸部区域 top-k"。

### 6.4 结果文件

- 单个实验结果：`ablation/results/test_*.json`
- 训练历史：`ablation/results/history_*.json`
- 汇总文件：`ablation/results/summary.json`

---

## 7. LaTeX 论文修改建议

根据导师反馈和新的消融结果，为论文 `AFGNN-V1.tex` 生成了详细修改建议，保存在：

- **`V1fill.md`**

主要修改点包括：

1. **第 329–332 行**：更新 Graph structure 小节引导文字，说明是“组件级消融”。
2. **第 333–350 行**：重写 Table III，拆清楚 static / temporal / region / velocity / self-loop 各自贡献。
3. **第 350 行之后**：新增 `\subsubsection{Facial region importance}` 子小节和表格，给出 9 个面部区域 Top-k 排名。
4. **（可选）第 421–423 行**：调整 Visualization 中关于 mouth 的描述，与区域 LOO 结果保持一致。
5. **（提醒）Figure 4**：Table III 更新后，建议同步重绘图 4(a)。

`V1fill.md` 中每条建议都包含：行号、当前内容、建议改成的 LaTeX 代码、修改原因。

---

## 8. 本次会话最终状态

- 全部 15 个消融实验已完成。
- 组件消融和区域 LOO 结果已汇总。
- `newchange.md`（本文件）和 `V1fill.md` 已更新。
- 后台任务 `bash-av1qh5z1` 已正常结束，无遗留运行中任务。
- 如需进一步修改论文或补跑其他实验，可基于 `ablation/configs/*.yaml` 和 `src/scripts/run_ablations.sh` 继续。
