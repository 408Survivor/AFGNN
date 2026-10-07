# AFGNN 论文初稿完善工作总结

> 记录本次会话中对 `paper/afgnn_bibm.tex` 及相关材料的全部修改与完善工作。
> 生成时间：2026-06-22

---

## 一、整体目标

在已有 LaTeX 论文骨架基础上，逐步完善 AFGNN（Adaptive Facial-Audio Graph Neural Network）论文初稿，包括：
- 补充文献与 Related Work
- 生成并插入论文插图
- 统一公式、符号与术语
- 核对并修正实验表格数字
- 打包可上传 Overleaf 的论文文件

---

## 二、已完成工作清单

### 1. Related Work 与参考文献
- 扩展 `paper/afgnn_bibm.bib`，新增 8 篇核心引用：
  - AVEC 2014 / 2017
  - DAIC-WOZ
  - He et al. 2021（CNN + attention）
  - Kipf & Welling 2017（GCN）
  - Zhong et al. 2019（Graph-BRNN）
  - Liao et al. 2022（FERGCN）
  - Tsai et al. 2019（MulT cross-modal attention）
- 重写 `Related Work` 章节，分为三个小节：
  - Depression Detection from Multimodal Video
  - Graph Neural Networks for Facial Behaviour
  - Cross-Modal Fusion

### 2. 论文插图生成与插入

| 图号 | 内容 | 生成脚本 | 插入位置 |
|------|------|----------|----------|
| Figure 1 | AFGNN 整体架构 | `paper/figures/draw_figure1_framework.py` | Method 章节开头 |
| Figure 2 | 面部图构造（68 点拓扑、眼部解剖边、时序边） | `paper/figures/draw_figure2_face_graph.py` | Face Graph Construction 小节 |
| Figure 3 | 音频时序图（特征热图、链图、readout 权重） | `paper/figures/draw_figure3_audio_graph.py` | Audio Graph Construction 小节 |
| Figure 4 | 消融实验柱状图（2×2 子图） | `paper/figures/draw_figure4_ablation.py` | Visualization 小节之前 |

所有图均生成 `pdf/png/svg` 三种格式，并统一配色风格。

### 3. 公式与代码一致性
- 核对 `Cross-Modal Attention Fusion` 公式与 `src/models/fusion.py` 中 `CrossModalAttentionFusion` 的实现。
- 将 tex 中的融合公式重写为：投影 → 缩放量积 → sigmoid gate → residual，与代码完全一致。

### 4. 新增 Visualization/Interpretability 小节
- 在 Experiments 章节新增 `Visualization and Interpretability` 小节。
- 结合 Figure 2 和 Figure 3 解释面部图拓扑与音频 readout 权重的可解释性。
- 讨论 GAT attention 对眼部、嘴部区域的聚焦。

### 5. Introduction 重写
- 强化问题动机：从全球抑郁负担到自动视频筛查需求。
- 批判现有 CNN/RNN/Transformer 方法：忽视面部结构、参数量大、易过拟合。
- 突出图建模优势：面部关键点天然图结构、轻量、可解释。
- 明确核心发现：动态边在小样本 D-Vlog 上反而掉点。
- 统一 AFGNN 展开为 **Adaptive Facial-Audio Graph Neural Network**，与标题一致。
- 结果表述对齐最终实验：F1 = 0.8077，AUC = 0.8041。

### 6. 符号与术语统一
- 统一时间/帧数符号：
  - 原始视频帧数：`$T$`
  - 面部采样长度：`$T_v$`（原 `$N_f$`）
  - 音频采样长度：`$T_a$`（原 `$N_a$`）
- 统一术语：
  - 图对象统一为 **face graph** / **audio graph**
  - 章节标题 `Facial Graph Construction` → `Face Graph Construction`
  - 表格与正文中的 `facial graph` 统一为 `face graph`

### 7. 表格数字核对与更新
- 重新运行 `seed_ensemble_test.py --seeds 42 43 44`，得到最终 Test 结果：
  - **F1 = 0.8077，AUC = 0.8041**
  - Acc = 0.7642，Prec = 0.7664，Rec = 0.8537
- 更新所有表格为当前可复现数值：
  - Table 2（SOTA 对比）
  - Table 3（模态与融合消融）
  - Table 4（音频长度、损失函数、seed ensemble）
  - 图结构消融表（移除无法复现的 `+Self-loop` 行）
- 同步更新 Abstract、Introduction、Conclusion 中的最终数字。
- 按新表格数字重新生成 Figure 4。
- 生成详细核对报告：`paper/table_number_verification.md`

### 8. Overleaf 上传包
- 创建 `overleaf_upload/` 目录与 `overleaf_upload.zip`。
- 包含：
  - `main.tex`（论文主文件副本）
  - `afgnn_bibm.bib`
  - `figures/*.pdf`
  - `README.md`（上传与编译说明）

---

## 三、关键文件变更

### 新增文件
- `paper/figures/draw_figure1_framework.py`
- `paper/figures/draw_figure2_face_graph.py`
- `paper/figures/draw_figure3_audio_graph.py`
- `paper/figures/draw_figure4_ablation.py`
- `paper/figures/afgnn_framework.{pdf,png,svg}`
- `paper/figures/figure2_face_graph.{pdf,png,svg}`
- `paper/figures/figure3_audio_graph.{pdf,png,svg}`
- `paper/figures/figure4_ablation.{pdf,png,svg}`
- `paper/paper_writing_roadmap.md`
- `paper/table_number_verification.md`
- `paper/work_summary.md`
- `overleaf_upload/` 目录及 `overleaf_upload.zip`
- `experiments/results/test_face_only_enhanced.json`
- `experiments/results/test_face_enhanced.json`
- `experiments/results/test_face_enhanced_focal.json`
- `experiments/results/test_face_enhanced_concat.json`
- `experiments/results/test_face_audio_baseline.json`
- `experiments/results/seed_ensemble_42-46_test_results.json`

### 修改文件
- `paper/afgnn_bibm.tex`
  - Abstract、Introduction、Related Work、Method、Experiments、Conclusion
  - 插入 Figure 1/2/3/4
  - 统一 fusion 公式
  - 更新所有表格数字
  - 统一符号术语
- `paper/afgnn_bibm.bib`
  - 新增 8 篇引用
- `paper/paper_writing_roadmap.md`
  - 多次更新实际进度

---

## 四、关键实验结论（已固定）

- **最终模型**：3-seed ensemble（seeds 42, 43, 44），配置 `experiments/configs/afgnn_face_enhanced_focal.yaml`
- **最终 Test 结果**：F1 = 0.8077，AUC = 0.8041
- **最佳图结构**：Static + Temporal + Region one-hot + Self-loop
- **关键发现**：动态边在小样本 D-Vlog 上导致过拟合，性能低于静态-时序图
- **模型规模**：< 5M 参数

---

## 五、剩余待办

- [ ] **编译检查**：当前环境 `pdflatex` format 文件损坏、`bibtex` 缺失，需在本地/Overleaf 完成。
- [ ] **作者信息替换**：`main.tex` 中匿名作者块需替换为真实作者和单位。
- [ ] **可选**：改造 `src/test.py` / `src/seed_ensemble_test.py`，使输出 JSON 文件名自动包含配置名，避免后续覆盖。

---

## 六、如何继续

1. 上传 `overleaf_upload.zip` 到 Overleaf，选择 **pdfLaTeX** 编译。
2. 检查 Figure 1/2/3/4、表格、公式、引用是否显示正常。
3. 替换匿名作者信息。
4. 根据编译报错逐项修复。
