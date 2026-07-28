# AFGNN 论文写作建议

> 面向抑郁症检测的多模态自适应面部-音频图神经网络

---

## 一、推荐论文结构

### 1. Introduction / Motivation
- 抑郁症自动筛查的重要性，视频多模态信号（面部外观 + 语音）的潜力。
- 现有方法的问题：
  - 面部 landmark 图大多只考虑静态解剖连接，忽略了面部区域语义/时序信息。
  - 多模态融合常是简单拼接，没有建模模态间关系。
  - 小样本医疗数据集上训练不稳定、seed 敏感。
- **本文贡献**：提出 AFGNN，通过面部图、音频图和跨模态注意力融合实现抑郁症检测，并系统分析图结构、损失函数和集成策略的影响。

### 2. Related Work
- 图神经网络在抑郁症检测中的应用（AFGNN/GRU/Transformer 类工作）。
- 多模态抑郁识别（视觉 + 音频）。
- 动态图 / 自适应图结构相关研究。

### 3. Method（重点章节）

#### 3.1 面部图（Face Graph）
- 输入：面部关键点序列，下采样到 32 帧。
- 节点：面部 landmark + 9 维面部区域 one-hot（眉毛、眼、鼻、嘴等）。
- 边：
  - **静态解剖边**：预定义的面部连接。
  - **时序边**：相邻帧同一 landmark。
  - 自环边（self-loops）。
- **关键发现**：动态边（feature-similarity / motion-consistency）在小数据集上反而更差，最终采用 static + temporal only。

#### 3.2 音频图（Audio Graph）
- 输入：声学特征序列，下采样到 32 帧。
- 节点：每帧的声学特征。
- 边：双向时序 chain，捕捉声学动态变化。

#### 3.3 模态融合
- 分别用 GAT 编码面部图和音频图。
- **Cross-Modal Attention Fusion**：让面部特征去 attend 音频特征，反之亦然，得到互补表示。
- 最后接分类器输出抑郁概率。

#### 3.4 训练策略
- Focal Loss 处理类别不平衡。
- AUC 早停 + 验证集 F1 阈值搜索。
- 梯度裁剪、ReduceLROnPlateau、LayerNorm。

#### 3.5 Seed Ensemble
- 小数据集上随机性大，用固定多 seed 模型做概率平均，提高稳定性和 F1。

---

## 二、侧重点建议

### 重点 1：图结构设计的“反直觉”发现
- **不要花大篇幅鼓吹动态图**。实验反复证明：在 D-Vlog 这种小数据集上，复杂的动态边（motion、feature similarity、region restricted）都输给了简单的 static + temporal 边。
- 这是一个很好的卖点：**“不是越复杂的图越好”**，说明在小样本医学任务上，预定义结构 + 时序连接更稳健，避免过拟合。
- 可以把这部分作为 ablation study 的核心表格。

### 重点 2：多模态融合的有效性
- 单模态对比：audio-only 0.7429，face-only 0.7241，说明音频单独就很强。
- 多模态融合后提升到 0.8073（3-seed ensemble），证明融合有价值。
- Cross-attention 优于 concat，可以强调模态交互机制。

### 重点 3：训练稳定性和可复现性
- AUC 早停避免被高 recall 的 trivial 策略欺骗。
- 验证集 F1 阈值调优，而不是固定 0.5。
- 多 seed 集成提升稳定性，报告最终指标时说明 ensemble 策略。

### 重点 4：诚实地讨论局限性
- 数据集只有 D-Vlog，样本量小。
- Seed 敏感，需要 ensemble 才能稳定。
- 只在单一数据集验证，泛化性待进一步验证。

---

## 三、建议插图

1. **AFGNN 整体架构图**：Face Graph → GAT，Audio Graph → GAT，Cross-Modal Attention Fusion → Classifier。
2. **面部图构造示意图**：landmark + 区域 one-hot + 静态边 + 时序边。
3. **动态边消融对比柱状图**：各种动态边方案的 F1/AUC 都不如 static+temporal。
4. **多模态消融与 ensemble 结果表**。

---

## 四、论文标题方向

- `AFGNN: Adaptive Facial and Audio Graph Neural Network for Depression Detection`
- `Multimodal Graph Neural Networks with Cross-Modal Attention for Depression Recognition`
- `Static-Temporal Facial Graphs with Audio Fusion for Depression Detection: A Revisiting of Dynamic Edges`

---

## 五、一句话总结

> 系统探索了面部图结构、音频图和多模态融合在抑郁症检测中的有效性，并发现简单的 static-temporal 边配合 cross-modal attention 和 seed ensemble，在小数据集上取得了稳定的最佳性能。

---

*最终定稿模型：3-Seed Ensemble（seeds=42,43,44），Test F1 = 0.8073，AUC = 0.8036。*
