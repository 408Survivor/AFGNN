# Checkpoint ↔ Config 溯源表 (experiments/)

> 目的:`experiments/checkpoints/` 里 25 个 `.pt`,但 `experiments/configs/` 只有 10 个
> 配置文件。本表理清每个 checkpoint 的来源。
>
> **重要背景**:`.pt` 内部只存了 `model_state_dict / optimizer_state_dict / epoch /
> best_score / early_stopping_metric / best_threshold*`,**不含超参**;而
> `history_*.json` / `test_*.json` 里的 `config`、`checkpoint` 两个字段是训练脚本
> `src/train.py` 的默认值(`configs/afgnn_face_only.yaml`、`checkpoints/afgnn_face_only_best.pt`),
> **对所有实验都一样、并未随实验更新,不能作为溯源依据**(见 `src/train.py` L107-110、L267-269)。
> 因此孤儿 checkpoint 的真实 config 已丢失,下表的架构/超参列为「从 checkpoint 名 +
> state_dict 模块 + 指标反推」的结论,仅供参考。
>
> 生成时间:2026-07-01。best_score 为验证集早停指标(见各 `.pt` 的 `best_score`);
> test 指标取自对应 `experiments/results/test_*.json` 的 `metrics`。

---

## A. 有对应 config 的 checkpoint(15 个,来源明确)

| Checkpoint (`.pt`) | 对应 Config (`.yaml`) | 说明 |
|---|---|---|
| `afgnn_audio_enhanced_best` | `afgnn_audio_enhanced` | 音频增强(delta/delta-delta + skip-1 边) |
| `afgnn_face_enhanced_best` | `afgnn_face_enhanced` | 人脸增强(32帧 + region one-hot + 自环) |
| `afgnn_face_enhanced_aug_best` | `afgnn_face_enhanced_aug` | 人脸增强 + 训练期数据增广 |
| `afgnn_face_enhanced_concat_best` | `afgnn_face_enhanced_concat` | concat 融合变体 |
| `afgnn_face_enhanced_focal_best` | `afgnn_face_enhanced_focal` | Focal Loss(默认 α=0.25/γ=2.0) |
| `afgnn_face_enhanced_focal_best_seed42..46` | `afgnn_face_enhanced_focal` | **同一 config**,5 个随机种子(train.py `--seed` 自动加 `_seed{N}` 后缀,见 L241-243) |
| `afgnn_face_enhanced_focal_a01_g3_best` | `afgnn_face_enhanced_focal_a01_g3` | Focal α=0.1/γ=3.0 |
| `afgnn_face_enhanced_focal_a05_g15_best` | `afgnn_face_enhanced_focal_a05_g15` | Focal α=0.5/γ=1.5 |
| `afgnn_face_enhanced_focal_adamw_cosine_best` | `afgnn_face_enhanced_focal_adamw_cosine` | Focal + AdamW + Cosine |
| `afgnn_face_only_best` | `afgnn_face_only` | 人脸单模态基线(一阶段验证) |
| `afgnn_face_only_enhanced_best` | `afgnn_face_only_enhanced` | 人脸单模态增强(禁用音频分支) |

> `afgnn_face_enhanced_focal` 一份 config 覆盖 6 个 checkpoint(best + seed42~46),故 10 配置 → 15 checkpoint。

---

## B. 无对应 config 的孤儿 checkpoint(10 个,config 已丢失)

架构列的 modules 为 `.pt` 内 `model_state_dict` 顶层模块;有 `fusion` 表示跨模态注意力融合,
仅 `face_gnn`+`classifier` 表示人脸单模态,`audio_gnn`+`classifier` 表示音频单模态。

| Checkpoint (`.pt`) | 结果文件 (`history_*` / `test_*`) | 时间戳 | 架构(state_dict 模块) | val best_score | test acc/f1/auc | 名称推断的变体含义 |
|---|---|---|---|---|---|---|
| `afgnn_audio_only` | `*_audio_only` | 06-22 12:59 | audio_gnn + classifier | 0.675 | .618 / .743 / .722 | 音频单模态 |
| `afgnn_face_audio` | `*_face_audio` (另有 `test_face_audio_baseline`) | 06-22 12:57 | face+audio+classifier(无 fusion) | 0.749 | .665 / .758 / .758 | 人脸+音频,简单拼接 |
| `afgnn_face_audio_crossattn` | `*_face_audio_crossattn` | 06-22 13:05 | face+audio+fusion+classifier | 0.761 | .684 / .767 / .754 | 跨模态注意力融合 |
| `afgnn_face_audio_crossattn_audio32` | `*_face_audio_crossattn_audio32` | 06-22 13:09 | face+audio+fusion+classifier | **0.848** | .717 / .774 / .770 | crossattn,音频 32 帧 |
| `afgnn_face_audio_crossattn_audio64` | `*_face_audio_crossattn_audio64` | 06-22 13:13 | face+audio+fusion+classifier | 0.820 | .726 / .762 / .760 | crossattn,音频 64 帧 |
| `afgnn_face_audio_crossattn_audio32_audio128` | `*_face_audio_crossattn_audio32_audio128` | 06-22 13:17 | face+audio+fusion+classifier | 0.782 | .689 / .759 / .734 | crossattn,音频 32 帧 + hidden 128(?) |
| `afgnn_face_only_edge_type_emb` | `*_edge_type_emb` | 06-22 03:51 | face_gnn + classifier | 0.718 | .575 / .505 / .667 | 人脸单模态 + 边类型 embedding |
| `afgnn_face_only_motion_k1_cosine` | `*_motion_k1_cosine` | 06-22 12:26 | face_gnn + classifier | 0.702 | .623 / .714 / .671 | 运动特征 k=1,cosine 调度 |
| `afgnn_face_only_motion_k2_cosine` | `*_motion_k2_cosine` | 06-22 12:16 | face_gnn + classifier | 0.700 | .571 / .709 / .614 | 运动特征 k=2,cosine 调度 |
| `afgnn_face_only_motion_k2_region` | `*_motion_k2_region` | 06-22 12:42 | face_gnn + classifier | 0.701 | .632 / .693 / .664 | 运动特征 k=2 + region |

### 备注
- `test_face_audio_baseline.json` 未匹配到独立 checkpoint,推测与 `afgnn_face_audio` 为同一实验的两次测试输出(命名不一致,详见 results 命名方案)。
- 这批孤儿实验时间戳集中在 06-22,且都属于早期探索(crossattn/motion/edge 变体),推测是在 `afgnn_face_only.yaml` 上原地反复修改 + 手动重命名 checkpoint 得到,未保存 config 快照。
- **建议**:若这些 checkpoint 仍需复现或写进论文,应尽快对照其 state_dict 手动补写 config(至少记录 audio 帧数、hidden dim、motion k、是否 fusion 等关键项);否则可视为已被 `enhanced`/`focal` 系列取代的过时产物,考虑归档。
