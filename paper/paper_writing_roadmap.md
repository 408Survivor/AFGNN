# AFGNN 论文初稿完成路线图

> 本文档汇总当前项目状态、论文结构、图表方案和分步执行计划。
> 基于 `paper/AFGNN_design.md` 和 `paper/paper_writing_guide.md` 整理。
> 更新时间：2026-06-22

---

## 一、当前项目状态总览

### 已完成代码与实验
- 模型实现：`src/models/afgnn.py`, `src/models/face_gnn.py`, `src/models/audio_gnn.py`, `src/models/fusion.py`, `src/models/weighted_gat_conv.py`, `src/models/graph_utils.py`
- 数据加载与增强：`src/data/dvlog_face_dataset.py`, `src/data/augmentation.py`
- 训练/测试脚本：`src/train.py`, `src/test.py`, `src/train_combined.py`, `src/ensemble_test.py`, `src/seed_ensemble_test.py`
- 工具模块：`src/utils/losses.py`, `src/utils/metrics.py`, `src/utils/trainer.py`
- 实验配置：`experiments/configs/` 下多个 YAML
- 模型权重：`experiments/checkpoints/` 下多个 `.pt`
- 训练历史：`experiments/results/` 下多个 JSON

### 已完成的论文材料
- `paper/AFGNN_design.md`：完整模型设计文档、数学公式、实验设计、图表规划
- `paper/paper_writing_guide.md`：写作侧重点、核心发现、推荐结构
- `paper/afgnn_bibm.tex`：论文 LaTeX 源文件（**骨架已基本完整**，包含 Abstract、Introduction、Related Work、Method、Experiments、Discussion、Conclusion、References）
- `paper/afgnn_bibm.bib`：参考文献（基础已具备，含 D-Vlog、TAMFN、DepMSTAT、ASYM、GAT、Focal Loss 等）

### 关键实验结论（已确定）
- **最佳模型**：3-Seed Ensemble（seeds=42,43,44），Test F1 = 0.8073，AUC = 0.8036
- **最佳图结构**：面部图仅使用 `static + temporal` 边，动态边在小数据集上反而更差
- **最佳融合**：Cross-modal attention fusion 优于简单 concat
- **单模态性能**：audio-only 0.7429，face-only 0.7241

---

## 二、推荐论文结构

参考 `paper_writing_guide.md` 与 `AFGNN_design.md`，建议初稿结构如下：

1. **Abstract**
2. **Introduction**
   - 抑郁症自动筛查背景与动机
   - 现有 CNN/ViT 方法的局限（参数量大、跨数据集泛化差）
   - GNN 用于面部几何与音频时序建模的优势
   - 本文贡献
3. **Related Work**
   - 视频抑郁识别（CNN/RNN/Transformer 方法）
   - 图神经网络在情感计算中的应用
   - 多模态融合与跨模态注意力
4. **Method**
   - 问题定义
   - 面部时空 landmark 图（节点、静态边、时序边、动态边）
   - 音频时序图
   - GAT 编码器与 Readout
   - 跨模态融合模块
   - 损失函数与训练策略
5. **Experiments**
   - 数据集与评价指标
   - 实现细节
   - 主实验结果
   - 消融实验（图结构、融合方式、损失函数）
   - Seed ensemble 分析
6. **Visualization and Interpretability**
   - Landmark attention 热力图
   - 时间帧 attention 可视化
7. **Conclusion**
8. **References**

---

## 三、图表绘制方案

### Figure 1：AFGNN 整体架构图
- **布局**：左右双分支 + 中间融合，横向流式结构
- **左支（Face，暖色）**：
  - Input Video → 68 Landmarks × T_v frames
  - Spatio-Temporal Facial Graph（static / temporal / dynamic edges）
  - Face GNN（GAT × L_V）→ Global Attention Pooling → H_face
- **右支（Audio，冷色）**：
  - Acoustic Features（25-D × T_a）
  - Temporal Chain Graph
  - Audio GNN（GAT × L_A）→ Global Attention Pooling → H_audio
- **中间/下方**：
  - Cross-Modal Fusion（concat / cross-attention）
  - MLP Classifier → Sigmoid → Depression Score
- **要点**：标注维度（H_face, H_audio, T_v, T_a），角落标注 `< 5M params`

### Figure 2：面部图构造细节
- **(a)** 单帧 68 点面部拓扑（按部位分组）
- **(b)** 时空展开：intra-frame anatomical edges + inter-frame temporal edges
- **(c)** 动态边示例：局部放大展示基于运动相似度的 top-K 连接

### Figure 3：音频时序图与可解释性
- **(a)** 帧级声学特征序列
- **(b)** 时序链图
- **(c)** Attention/readout 权重热力条

### Figure 4：消融实验结果图（可选）
- 各种动态边方案 vs static+temporal 的 F1/AUC 柱状图
- 强调“复杂动态边反而更差”这一反直觉发现

### 工具推荐
| 场景 | 推荐工具 |
|------|----------|
| 最终论文投稿 | **TikZ（LaTeX）** |
| 快速出图/迭代 | **draw.io / diagrams.net** |
| 程序化生成/复现 | **Python + matplotlib**（已安装） |
| 草稿/组会 | Excalidraw |

---

## 四、分步完成论文初稿计划

### Step 0：准备与整理（已完成）
- [x] 确认最终实验结果表格所需数据已整理到 `experiments/results/`
- [x] 确认 checkpoints 中最终模型权重可用
- [x] 阅读 `paper/AFGNN_design.md` 与 `paper/paper_writing_guide.md`

### Step 1：补齐 LaTeX 论文骨架（**已基本完成**）
- [x] 在 `paper/afgnn_bibm.tex` 中搭建完整章节结构
- [x] 填充 Abstract 与 Introduction
- [x] 填充 Related Work（基础版本已具备）
- [x] 补充更多 Related Work 文献（CNN/RNN/Transformer 抑郁识别、GNN 面部行为分析、跨模态注意力，已新增 8 篇引用）

### Step 2：完善 Method 章节（**已基本完成**）
- [x] 复制/改写 `AFGNN_design.md` §4 的数学公式到 `afgnn_bibm.tex`
- [x] 统一符号：landmark 节点、边类型、GAT 公式、Readout、Fusion
- [x] 加入训练策略（Focal Loss、AUC 早停、阈值搜索、seed ensemble）
- [x] 核对 fusion 公式与代码实现 `CrossModalAttentionFusion` 的一致性（已重写为显式的投影 + 缩放量积 + sigmoid gate + residual，与代码一致）

### Step 3：生成论文图表（**Figure 1/2/3 已完成**）
- [x] 绘制 Figure 1 整体架构图（matplotlib，PDF/SVG/PNG 已生成）
- [x] 绘制 Figure 2 面部图构造细节（matplotlib，PDF/SVG/PNG 已生成）
- [x] 绘制 Figure 3 音频时序图（matplotlib，PDF/SVG/PNG 已生成）
- [x] 将 Figure 1/2/3 插入 `afgnn_bibm.tex`
- [x] 生成 Figure 4 消融柱状图（2×2 子图：图结构、模态融合、音频长度/损失、seed ensemble，PDF/SVG/PNG 已生成并插入 tex）
- [x] 创建 `paper/figures/` 目录存放图表

### Step 4：整理实验结果表格（**已基本完成**）
- [x] Table 1：数据集统计
- [x] Table 2：与 SOTA 对比（D-Vlog）
- [x] Table 3：消融实验（图结构、融合、损失）
- [x] Table 4：Seed ensemble 结果
- [ ] **待验证**：表格中数字与 `experiments/results/` 中最新 JSON 是否一致

### Step 5：撰写 Experiments 与 Analysis（**已完成**）
- [x] 数据集与评价指标
- [x] 实现细节（超参数、优化器、数据增强）
- [x] 主实验结果分析
- [x] 消融实验分析（重点解释 dynamic edges 的负面效果）
- [x] 独立的 **Visualization and Interpretability** 小节（已加入 Experiments 章节，引用 Figure 2/3）

### Step 6：完善 Introduction、Conclusion 与 Related Work
- [x] 根据实验结果回头精炼 Introduction 的贡献表述（已重写，强调 lightweight、dynamic edges 反直觉发现、引用 Figure 1）
- [x] 撰写 Conclusion 并讨论局限性
- [x] 补充 Related Work 缺失文献（已新增 8 篇核心引用）

### Step 7：编译与校对
- [ ] 编译 `afgnn_bibm.tex` 生成 PDF
- [ ] 检查公式、图表、引用、表格
- [ ] 统一术语与符号

---

## 五、当前状态与下一步建议

`paper/afgnn_bibm.tex` 的论文初稿已较完整：
- **Figure 1/2/3/4 已生成并插入**
- **fusion 公式已与代码统一**
- **Visualization/Interpretability 小节已补充**
- **Introduction 已根据最终实验结果重写**
- **符号术语已统一（$T_v$ / $T_a$）**
- **表格数字已与 `experiments/results/` JSON 交叉验证并更新**，最终 3-seed ensemble 更新为 F1=0.8077 / AUC=0.8041

当前剩余主要工作：
- 在本地完整 LaTeX 环境中编译校对（当前环境 `pdflatex` 格式文件损坏、`bibtex` 缺失）。
- 替换 `main.tex` 中的匿名作者信息。

因此当前优先级调整为：

1. **Step 7 本地编译校对**
   - 在你的机器或 Overleaf 上编译，确认 Figure 1/2/3/4、公式、引用、表格无误。
2. **Step 9 作者信息替换**
   - 将 `overleaf_upload/main.tex` 中的匿名作者块替换为真实作者和单位。

工作过程已全部记录到 `paper/work_summary.md`。

我可以继续帮你：
- **A.** 编译检查并记录报错（环境受限，可能只能给出本地/Overleaf 建议）；
- **B.** 打包 Overleaf 上传文件（tex + bib + figures PDF/SVG + README）；
- **C.** 对 `src/test.py` / `src/seed_ensemble_test.py` 做小幅改造，让输出文件名自动包含配置名，避免以后覆盖。

我可以继续帮你：
- **A.** 完善 Introduction；
- **B.** 生成 Figure 4 消融柱状图；
- **C.** 统一全文字符号/术语；
- **D.** 检查全文字数/篇幅，看是否需要压缩 Figure 2/3。

请选择下一步。
