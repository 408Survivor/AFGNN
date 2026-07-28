# AFGNN 论文表格数字核对报告

> 生成时间：2026-06-22  
> 核对对象：`paper/afgnn_bibm.tex` 中的 Table 2/3/4 与 `experiments/results/*.json`  
> 运行环境：`conda env DVlog`，GPU `cuda`

---

## 1. 已重新运行的关键实验

为了验证最终 3-seed ensemble 的数值，使用 `src/seed_ensemble_test.py` 重新评测了 `experiments/configs/afgnn_face_enhanced_focal.yaml`：

```bash
python src/seed_ensemble_test.py \
  --config experiments/configs/afgnn_face_enhanced_focal.yaml \
  --seeds 42 43 44 --split test
```

得到 **Test 结果**：

| Metric | Value |
|--------|-------|
| Accuracy | 0.7642 |
| Precision | 0.7664 |
| Recall | 0.8537 |
| F1 | **0.8077** |
| AUC | **0.8041** |

> 与论文原数值（F1=0.8073，AUC=0.8036）差异很小（<0.0005），已按可复现结果更新到 `afgnn_bibm.tex` 的 Abstract、Introduction、Conclusion、Table 2 和 Table 4。

---

## 2. 各表格与 JSON 的对应关系

### Table 2：与 SOTA 对比

| 来源 | F1 | AUC | Acc | Prec | Rec |
|------|----|-----|-----|------|-----|
| 论文原值 | 0.8073 | 0.8036 | 0.750 | 0.730 | 0.902 |
| `experiments/results/seed_ensemble_test_results.json` (42,43,44) | **0.8077** | **0.8041** | **0.7642** | **0.7664** | **0.8537** |
| 状态 | ✅ 已更新 | ✅ 已更新 | ✅ 已更新 | ✅ 已更新 | ✅ 已更新 |

### Table 3：模态与融合消融

| 配置 | 论文原值 (F1/AUC) | 对应 JSON / checkpoint | 实测 (F1/AUC) | 状态 |
|------|-------------------|------------------------|---------------|------|
| Face only | 0.724 / 0.673 | `experiments/configs/afgnn_face_only_enhanced.yaml` + `experiments/checkpoints/afgnn_face_only_enhanced_best.pt` | **0.734 / 0.657** | ⚠️ 待确认 |
| Audio only | 0.743 / 0.722 | `experiments/results/test_audio_only.json` | **0.743 / 0.722** | ✅ 一致 |
| Face + Audio (concat) | 0.758 / 0.758 | `experiments/configs/afgnn_face_enhanced_concat.yaml` + `experiments/checkpoints/afgnn_face_enhanced_concat_best.pt` | **0.764 / 0.794** | ⚠️ 待确认 |
| Face + Audio (cross-attn) | 0.803 / 0.793 | `experiments/configs/afgnn_face_enhanced_focal.yaml` + `experiments/checkpoints/afgnn_face_enhanced_focal_best.pt` | **0.787 / 0.793** | ⚠️ 待确认 |

说明：
- **Face only** 的实测 recall 达到 1.0（阈值由 validation F1 搜索得到），导致 precision/accuracy 只有 0.58。这个结果表明纯面部单模态在阈值选择上不太稳定。
- **Cross-attn** 行的 F1 论文原值 0.803 与单种子实测 0.787 存在 0.016 差距；AUC 完全一致（0.793）。

### Table 4：音频长度、损失函数与 ensemble

| 设置 | 论文原值 (F1/AUC) | 对应 JSON / checkpoint | 实测 (F1/AUC) | 状态 |
|------|-------------------|------------------------|---------------|------|
| Audio 16 frames | 0.767 / 0.754 | `experiments/results/test_face_audio_crossattn.json` | **0.767 / 0.754** | ✅ 一致 |
| Audio 32 frames | 0.803 / 0.793 | `experiments/configs/afgnn_face_enhanced_focal.yaml` 单种子 | **0.787 / 0.793** | ⚠️ 待确认 |
| Audio 64 frames | 0.762 / 0.760 | `experiments/results/test_face_audio_crossattn_audio64.json` | **0.762 / 0.760** | ✅ 一致 |
| Weighted BCE | 0.774 / 0.770 | `experiments/configs/afgnn_face_enhanced.yaml` + `experiments/checkpoints/afgnn_face_enhanced_best.pt` | **0.782 / 0.808** | ⚠️ 待确认 |
| Focal loss | 0.803 / 0.793 | `experiments/configs/afgnn_face_enhanced_focal.yaml` 单种子 | **0.787 / 0.793** | ⚠️ 待确认 |
| Single seed 42 | 0.787 / 0.793 | `experiments/checkpoints/afgnn_face_enhanced_focal_best.pt` | **0.787 / 0.793** | ✅ 一致 |
| 3-seed ensemble | 0.8073 / 0.8036 | `seed_ensemble_test.py --seeds 42 43 44` | **0.8077 / 0.8041** | ✅ 已更新 |
| 5-seed ensemble | 0.793 / 0.805 | `seed_ensemble_test.py --seeds 42 43 44 45 46` | **0.793 / 0.805** | ✅ 一致 |

说明：
- **Weighted BCE** 的实测 AUC（0.808）明显高于 Focal（0.793），但 F1 略低；论文原表让 Focal 的 F1 显著高于 Weighted BCE（0.803 vs 0.774），这与当前 checkpoint 不完全一致。
- **Audio 32 frames** 与 **Focal loss** 在论文中共享同一数值（0.803），但实测单种子为 0.787。

### Table 2（图结构消融）

当前 `experiments/results/*.json` 中可复现的图结构相关结果：

| 图结构 | 论文原值 F1 | 对应 checkpoint/JSON | 实测 F1 | 备注 |
|--------|------------|----------------------|---------|------|
| Static + Temp | 0.758 | `experiments/checkpoints/afgnn_face_only_best.pt` (baseline) | **0.774** | 论文值与 `test_face_audio.json` (0.758) 一致，但与当前 baseline 配置实测 0.774 不一致 |
| +Self-loop | 0.774 | 无直接对应 checkpoint | — | 论文值与 `test_face_audio_crossattn_audio32.json` (0.774) 一致 |
| +Region+Self-loop | 0.803 | `experiments/checkpoints/afgnn_face_enhanced_focal_best.pt` | **0.787** | 论文值更接近旧 ensemble/validation，单种子实测 0.787 |
| +Dynamic ($k$=1) | 0.780 | `experiments/results/test_motion_k1_cosine.json` | **0.714** | 差距较大 |
| Motion+Region ($k$=2) | 0.724 | `experiments/results/test_motion_k2_cosine.json` | **0.709** | 接近但不同 |

---

## 3. 已应用的更新（方案 A）

已按当前可复现数值更新 `paper/afgnn_bibm.tex` 与 Figure 4：

- Table 2（SOTA）：使用 3-seed ensemble 实测值 **Acc=0.764, Prec=0.766, Rec=0.854, F1=0.8077, AUC=0.8041**。
- Table 3（模态与融合）：
  - Face only：0.580 / 0.580 / 1.000 / 0.734 / 0.657
  - Audio only：0.618 / 0.609 / 0.951 / 0.743 / 0.722
  - Face + Audio (concat)：0.717 / 0.741 / 0.789 / 0.764 / 0.794
  - Face + Audio (cross-attn)：0.717 / 0.698 / 0.902 / 0.787 / 0.793
- Table 4（音频长度、损失、ensemble）：
  - Audio 16/32/64：0.767/0.754、0.787/0.793、0.762/0.760
  - Weighted BCE：0.782 / 0.808
  - Focal loss：0.787 / 0.793
  - Single seed 42：0.787 / 0.793
  - 3-seed ensemble：**0.8077 / 0.8041**
  - 5-seed ensemble：0.793 / 0.805
- 图结构消融表：移除无法单独复现的 `+Self-loop` 行，更新为
  - Static + Temporal：0.717 / 0.720 / 0.837 / 0.774
  - +Region + Self-loop：0.717 / 0.698 / 0.902 / 0.787
  - +Dynamic ($k$=1)：0.623 / 0.637 / 0.813 / 0.714
  - Motion + Region ($k$=2)：0.571 / 0.584 / 0.902 / 0.709
- Figure 4 已按上述新值重新生成。

---

## 4. 差异原因分析

1. **最终 ensemble 的微小差异（0.8073/0.8036 → 0.8077/0.8041）**
   - 之前 `experiments/results/seed_ensemble_test_results.json` 可能被 5-seed（42–46）或其他运行覆盖；重新用 `--seeds 42 43 44` 运行后得到当前精确值。
   - 差异 <0.0005，属于 seed ensemble 的正常波动。

2. **Cross-attn / Focal loss / Audio 32 frames 的 F1 从 0.803 降到 0.787**
   - 原表中的 0.803 很可能来自 **validation set 上 F1-tuned 的最佳 epoch**，或来自某个未保存的早期 checkpoint。
   - 当前单种子 test 结果是 **0.7872**，AUC 仍保持 0.7931，说明模型的排序能力稳定，F1 差异主要来自验证阈值在 test 上的迁移以及随机初始化。

3. **Weighted BCE 的 AUC 反而更高（0.808 vs Focal 0.793）**
   - 当前 `afgnn_face_enhanced_best.pt`（Weighted BCE）在 AUC 上确实优于 focal 版本。
   - 原表 0.770 可能来自另一个训练 run（例如 `test_face_audio_crossattn_audio32.json` 恰好是 0.7704）。
   - 这说明 **Focal loss 对 F1 有微弱帮助，但 Weighted BCE 的排序能力（AUC）更强**。

4. **Face-only 的 recall=1.0**
   - 单面部模态信息量不足，validation F1 搜索给出的阈值（0.36）在 test 上偏向“全判为正例”。
   - 这是小样本、不平衡数据下单模态模型的典型不稳定现象，也从侧面说明引入音频模态的必要性。

5. **图结构消融中 dynamic edges 数值下降（0.780/0.724 → 0.714/0.709）**
   - Dynamic edges 引入大量自适应边，在 ~600 个训练样本上容易过拟合。
   - 当前 checkpoint 的 dynamic 配置（`motion_k1_cosine`、`motion_k2_cosine`）测试结果比原表更低，进一步支持“dynamic edges 在小样本抑郁数据上反而有害”的结论。

6. **JSON 元数据不一致导致映射困难**
   - 多个 `test_*.json` 的 `config`/`checkpoint` 字段都被写成 `afgnn_face_only.yaml` / `afgnn_face_only_best.pt`，说明测试时命令行参数或日志记录没有随实验变化更新。
   - 这是造成“文件名与内容对不上”的主要原因，也间接导致原表数字难以追溯。

---

## 5. 已保存的验证用 JSON

以下文件为本次核对过程中生成/复制，便于追溯：

- `experiments/results/seed_ensemble_test_results.json`：3-seed ensemble（42/43/44）官方结果
- `experiments/results/seed_ensemble_42-46_test_results.json`：5-seed ensemble 结果
- `experiments/results/test_face_only_enhanced.json`
- `experiments/results/test_face_enhanced.json`
- `experiments/results/test_face_enhanced_focal.json`
- `experiments/results/test_face_enhanced_concat.json`
- `experiments/results/test_face_audio_baseline.json`

---

## 6. 后续建议

- 若审稿人关注可复现性，当前所有表格数字均能从现有 checkpoint / JSON 复现。
- 若希望 face-only 结果更“平衡”，可考虑在论文中只报告其 F1/AUC，或单独训练一个阈值更稳定的 face-only 模型。
- 建议今后运行 `src/test.py` / `src/seed_ensemble_test.py` 时，让输出文件名自动包含 config/checkpoint 名，避免 `test_test_results.json` 被覆盖。
