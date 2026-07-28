# AFGNN：完整流程的公式化表达

本文根据论文 `AFGNN: Adaptive Facial-Audio Graph Neural Network with Cross-Modal Attention for Depression Detection`，将整个抑郁识别流程从输入到输出、从图构建到训练协议，全部用数学符号系统化地写出来。

---

## 1. 问题定义

给定一段 vlog 视频，输入包含两类预处理后的特征：

- **面部关键点序列**：
  $$
  oldsymbol{V} \in \mathbb{R}^{T \times 136}
  $$
  其中 $T$ 为原始帧数，$136 = 68 \text{ 个关键点} \times 2 \text{ 维坐标}$。

- **声学特征序列**：
  $$
  oldsymbol{A} \in \mathbb{R}^{T \times 25}
  $$
  每帧为 25 维 openSMILE 低级声学描述子。

对两种模态分别均匀下采样到固定长度：
- 面部图：$T_v = 32$ 帧
- 音频图：$T_a = 32$ 帧

目标是预测二分类抑郁标签：
$$
y \in \{0, 1\}
$$
其中 $y=1$ 表示抑郁，$y=0$ 表示对照/非抑郁。

最终输出为抑郁概率：
$$
\hat{y} = f(\boldsymbol{V}, \boldsymbol{A}) \in [0, 1]
$$

---

## 2. 面部图构建（Face Graph Construction）

### 2.1 节点定义

对采样得到的 $T_v$ 帧、每帧 68 个关键点，构建面部图：
$$
G_f = (\mathcal{V}_f, \mathcal{E}_f), \quad |\mathcal{V}_f| = T_v \times 68
$$

每个节点对应第 $t$ 帧第 $i$ 个关键点，其特征为：
$$
\mathbf{x}_{i,t} = \big[ \mathbf{p}_{i,t};\; \dot{\mathbf{p}}_{i,t};\; \mathbf{r}_i \big] \in \mathbb{R}^{13}
$$

其中：
- $\mathbf{p}_{i,t} \in \mathbb{R}^2$：第 $t$ 帧第 $i$ 个关键点的坐标（已归一化）；
- $\dot{\mathbf{p}}_{i,t} \in \mathbb{R}^2$：该关键点在相邻帧之间的位移速度，即
  $$
  \dot{\mathbf{p}}_{i,t} = \mathbf{p}_{i,t} - \mathbf{p}_{i,t-1}
  $$
  首帧可用零填充或后向差分；
- $\mathbf{r}_i \in \{0, 1\}^9$：关键点所属面部区域（轮廓、左眉、右眉、鼻梁、鼻底、左眼、右眼、外嘴、内嘴）的 one-hot 向量。

因此节点维度为：
$$
2 + 2 + 9 = 13
$$

所有节点特征可拼接为矩阵：
$$
\mathbf{X}_f \in \mathbb{R}^{(T_v \times 68) \times 13}
$$

### 2.2 边定义

边集合 $\mathcal{E}_f$ 由以下部分组成：

#### (1) 静态解剖边（Static Anatomical Edges）

每帧内按 68 点面部拓扑连接：

| 面部区域 | 关键点索引 | 连接方式 |
|---|---|---|
| 面部轮廓 | 1–17 | 链：$1\!-\!2\!-\!\cdots\!-\!17$ |
| 左眉 | 18–22 | 链：$18\!-\!19\!-\!20\!-\!21\!-\!22$ |
| 右眉 | 23–27 | 链：$23\!-\!24\!-\!25\!-\!26\!-\!27$ |
| 鼻梁 | 28–31 | 链：$28\!-\!29\!-\!30\!-\!31$ |
| 鼻底 | 32–36 | 环：$32\!-\!33\!-\!34\!-\!35\!-\!36\!-\!32$ |
| 左眼 | 37–42 | 环：$37\!-\!38\!-\!39\!-\!40\!-\!41\!-\!42\!-\!37$ |
| 右眼 | 43–48 | 环：$43\!-\!44\!-\!45\!-\!46\!-\!47\!-\!48\!-\!43$ |
| 外嘴 | 49–60 | 环：$49\!-\!50\!-\!\cdots\!-\!60\!-\!49$ |
| 内嘴 | 61–68 | 环：$61\!-\!62\!-\!\cdots\!-\!68\!-\!61$ |

形式化地，对每一帧 $t$：
$$
\mathcal{E}_f^{\text{static}}(t) = \{(v_{t,i}, v_{t,j}) \mid i,j \text{ 按上述解剖链/环相邻}\}
$$

#### (2) 时序边（Temporal Edges）

同一关键点在相邻帧之间连接：
$$
\mathcal{E}_f^{\text{temp}} = \{(v_{t,i}, v_{t+1,i}) \mid 1 \le t < T_v,\; 1 \le i \le 68\}
$$

#### (3) 自环边（Self-loops）

为数值稳定性加入：
$$
\mathcal{E}_f^{\text{self}} = \{(v_{t,i}, v_{t,i}) \mid \forall t,i\}
$$

#### (4) 动态边（Dynamic Edges，消融实验用）

在每帧内，对每个节点基于特征相似度或运动一致性取 $k$ 近邻。例如基于运动一致性：
$$
\mathcal{N}_k^{\text{dyn}}(v_{t,i}) = \text{TopK}_{j \ne i}\; \text{sim}\big(\dot{\mathbf{p}}_{i,t}, \dot{\mathbf{p}}_{j,t}\big)
$$

最终实验中，**静态 + 时序 + 区域 one-hot + 自环** 表现最好。

#### 边类型与权重

每条边 $e$ 被赋予一个类型标签（static / temporal / dynamic）以及一个标量权重 $w_e$（实验中通常 $w_e=1$，或按边类型学习）。

---

## 3. 音频图构建（Audio Graph Construction）

### 3.1 节点特征

对采样得到的 $T_a$ 帧声学特征，先进行 z-score 标准化：
$$
\tilde{\mathbf{a}}_t = \frac{\mathbf{a}_t - \boldsymbol{\mu}}{\boldsymbol{\sigma}}
$$
其中 $\boldsymbol{\mu}, \boldsymbol{\sigma} \in \mathbb{R}^{25}$ 由训练集统计得到。

音频图节点：
$$
G_a = (\mathcal{V}_a, \mathcal{E}_a), \quad |\mathcal{V}_a| = T_a
$$

每个节点特征为：
$$
\mathbf{x}_t^{(a)} = \tilde{\mathbf{a}}_t \in \mathbb{R}^{25}
$$

所有节点特征矩阵：
$$
\mathbf{X}_a \in \mathbb{R}^{T_a \times 25}
$$

### 3.2 边定义

音频图采用简单的**双向链式结构**：
$$
\mathcal{E}_a = \{(t, t+1), (t+1, t) \mid 1 \le t < T_a\}
$$

也可选加入自环或跳边，但实验中普通双向链已足够。

---

## 4. GNN 编码器

两个分支均采用 **Graph Attention Network (GAT)**。

### 4.1 GAT 单头消息传递

对任意节点 $u$，设其邻居集合为 $\mathcal{N}(u)$。在第 $\ell$ 层：

首先对邻居特征做线性变换：
$$
\mathbf{h}_j^{(\ell)} = \mathbf{W}^{(\ell)} \mathbf{h}_j^{(\ell-1)}
$$

注意力系数：
$$
e_{u,j}^{(\ell)} = \text{LeakyReLU}\Big( \mathbf{a}^{(\ell)\top} \big[ \mathbf{W}^{(\ell)} \mathbf{h}_u^{(\ell-1)} \;\|\; \mathbf{W}^{(\ell)} \mathbf{h}_j^{(\ell-1)} \big] \Big)
$$

归一化：
$$
\alpha_{u,j}^{(\ell)} = \frac{\exp(e_{u,j}^{(\ell)})}{\sum_{k \in \mathcal{N}(u)} \exp(e_{u,k}^{(\ell)})}
$$

更新：
$$
\mathbf{h}_u^{(\ell)} = \sigma\left( \sum_{j \in \mathcal{N}(u)} \alpha_{u,j}^{(\ell)} \mathbf{W}^{(\ell)} \mathbf{h}_j^{(\ell-1)} \right)
$$
其中 $\sigma(\cdot)$ 为 ELU（中间层）或恒等映射（最后一层）。

### 4.2 多头注意力

设注意力头数为 $M$：
$$
\mathbf{h}_u^{(\ell)} = \Big\|_{m=1}^{M} \sigma\left( \sum_{j \in \mathcal{N}(u)} \alpha_{u,j}^{(m,\ell)} \mathbf{W}_m^{(\ell)} \mathbf{h}_j^{(\ell-1)} \right)
$$

中间层使用 $M=4$ 头，最后一层使用单头。

### 4.3 面部分支

- 层数：$L_f = 3$
- 隐藏维度：$d_f^{\text{hidden}} = 128$
- 注意力头数：4（中间层），1（最后一层）
- 每层后接 LayerNorm、ELU、Dropout($p=0.3$)

输出最后一层节点表示：
$$
\{\mathbf{h}_{i,t}^{(L_f)} \in \mathbb{R}^{d_f}\}_{i=1,t=1}^{68,T_v}
$$

### 4.4 音频分支

- 层数：$L_a = 2$
- 隐藏维度：$d_a^{\text{hidden}} = 64$
- 注意力头数：4（中间层），1（最后一层）
- 每层后接 LayerNorm、ELU、Dropout($p=0.3$)

输出最后一层节点表示：
$$
\{\mathbf{h}_t^{(L_a)} \in \mathbb{R}^{d_a}\}_{t=1}^{T_a}
$$

---

## 5. 图级读出（Graph-Level Readout）

采用 **Attentional Aggregation** 将节点表示聚合为图级嵌入。

对面部图：
$$
\mathbf{h}_f = \sum_{u \in \mathcal{V}_f} \sigma\big(g_f(\mathbf{h}_u^{(L_f)})\big) \, \mathbf{h}_u^{(L_f)} \in \mathbb{R}^{d_f}
$$

对音频图：
$$
\mathbf{h}_a = \sum_{u \in \mathcal{V}_a} \sigma\big(g_a(\mathbf{h}_u^{(L_a)})\big) \, \mathbf{h}_u^{(L_a)} \in \mathbb{R}^{d_a}
$$

其中：
- $g_f(\cdot), g_a(\cdot)$ 为各自的小门控 MLP；
- $\sigma(\cdot)$ 为 sigmoid 函数；
- 权重 $\beta_u = \sigma(g(\mathbf{h}_u))$ 表示该节点对图级分类的重要性。

---

## 6. 跨模态注意力融合（Cross-Modal Attention Fusion）

### 6.1 投影

将面部和音频图级嵌入分别投影到共享隐藏空间 $\mathbb{R}^{d_h}$ 和各自的值空间：

$$
\begin{aligned}
\mathbf{q}_f &= \mathbf{W}_q^{(f)} \mathbf{h}_f, &\quad \mathbf{k}_a &= \mathbf{W}_k^{(a)} \mathbf{h}_a, &\quad \mathbf{v}_a &= \mathbf{W}_v^{(a)} \mathbf{h}_a, \\
\mathbf{q}_a &= \mathbf{W}_q^{(a)} \mathbf{h}_a, &\quad \mathbf{k}_f &= \mathbf{W}_k^{(f)} \mathbf{h}_f, &\quad \mathbf{v}_f &= \mathbf{W}_v^{(f)} \mathbf{h}_f.
\end{aligned}
$$

其中：
- $\mathbf{W}_q^{(f)}, \mathbf{W}_q^{(a)} \in \mathbb{R}^{d_h \times d_f \text{ 或 } d_a}$
- $\mathbf{W}_k^{(f)}, \mathbf{W}_k^{(a)} \in \mathbb{R}^{d_h \times d_f \text{ 或 } d_a}$
- $\mathbf{W}_v^{(f)}, \mathbf{W}_v^{(a)} \in \mathbb{R}^{d_v \times d_f \text{ 或 } d_a}$

### 6.2 注意力门控

因为每个模态被压缩为单一图级向量，每个方向只有一个 key/value，所以用 **sigmoid 标量门控** 而非 softmax 分布。

面部对音频的注意力：
$$
\alpha_{f \to a} = \sigma\!\left( \frac{1}{\sqrt{d_h}} \sum_{i=1}^{d_h} q_{f,i} \, k_{a,i} \right)
$$

音频对面的注意力：
$$
\alpha_{a \to f} = \sigma\!\left( \frac{1}{\sqrt{d_h}} \sum_{i=1}^{d_h} q_{a,i} \, k_{f,i} \right)
$$

### 6.3 残差更新

$$
\begin{aligned}
\tilde{\mathbf{h}}_f &= \mathbf{h}_f + \alpha_{f \to a} \, \mathbf{v}_a, \\
\tilde{\mathbf{h}}_a &= \mathbf{h}_a + \alpha_{a \to f} \, \mathbf{v}_f.
\end{aligned}
$$

### 6.4 融合向量

$$
\mathbf{z} = \big[ \tilde{\mathbf{h}}_f \;\|\; \tilde{\mathbf{h}}_a \big] \in \mathbb{R}^{d_f + d_a}
$$

---

## 7. 分类头

融合向量 $\mathbf{z}$ 输入一个 2 层 MLP：

$$
\hat{y} = \sigma\big( \mathbf{W}_2 \, \text{ReLU}\big( \text{Dropout}(\mathbf{W}_1 \mathbf{z} + \mathbf{b}_1) \big) + \mathbf{b}_2 \big)
$$

其中 $\sigma(\cdot)$ 为 sigmoid：
$$
\sigma(s) = \frac{1}{1 + e^{-s}}
$$

---

## 8. 损失函数

### 8.1 Focal Loss

训练采用 Focal Loss 处理类别不平衡：

$$
\mathcal{L}_{\text{focal}} = -\alpha_t (1 - p_t)^\gamma \log(p_t)
$$

其中：
$$
p_t = \begin{cases}
\hat{y}, & y = 1 \\
1 - \hat{y}, & y = 0
\end{cases}
$$

参数设置为：
$$
\alpha = 0.25, \quad \gamma = 2
$$

### 8.2 带 L2 正则的总损失

$$
\mathcal{L} = \mathcal{L}_{\text{focal}} + \lambda_{\text{reg}} \|\Theta\|_2^2
$$

其中 $\Theta$ 为所有可学习参数，$\lambda_{\text{reg}} = 10^{-4}$。

---

## 9. 训练协议

### 9.1 优化器

- **Adam**
- 学习率：$\eta = 10^{-3}$
- 权重衰减：$\lambda_{\text{reg}} = 10^{-4}$
- 梯度裁剪：最大范数 1.0
- 学习率调度：ReduceLROnPlateau

### 9.2 AUC 早停

每个 epoch 后在验证集上计算 AUC，保存验证 AUC 最高的模型：
$$
\theta^* = \arg\max_\theta \; \text{AUC}_{\text{val}}(\theta)
$$

早停耐心值（patience）为 15 个 epoch。

### 9.3 验证集 F1 阈值搜索

训练时每个 epoch 在验证集上搜索最优阈值：
$$
\tau^* = \arg\max_{\tau \in [0.01, 0.99]} \text{F1}_{\text{val}}(\tau)
$$

测试时使用 $\tau^*$ 将概率转换为二分类标签：
$$
\hat{y}_{\text{bin}} = \mathbb{1}[\hat{y} \ge \tau^*]
$$

---

## 10. 模型集成（Model Ensemble）

训练 $K$ 个独立初始化（不同随机种子）的模型，对测试样本平均预测概率：

$$
\hat{y}^{\text{ens}} = \frac{1}{K} \sum_{k=1}^{K} \hat{y}^{(k)}
$$

最终论文使用 $K=3$ 的集成：
$$
\hat{y}_{\text{final}} = \frac{1}{3} \sum_{k=1}^{3} \hat{y}^{(k)}
$$

再用验证集搜索到的阈值 $\tau^*$ 进行二值化：
$$
\hat{y}_{\text{bin}}^{\text{ens}} = \mathbb{1}[\hat{y}^{\text{ens}} \ge \tau^*]
$$

---

## 11. 完整前向流程总结

1. **输入**：
   $$
   \boldsymbol{V} \in \mathbb{R}^{T \times 136}, \quad \boldsymbol{A} \in \mathbb{R}^{T \times 25}
   $$

2. **下采样**：
   $$
   \boldsymbol{V} \to \mathbb{R}^{T_v \times 136}, \quad \boldsymbol{A} \to \mathbb{R}^{T_a \times 25}
   $$

3. **面部图构建**：
   $$
   G_f = (\mathcal{V}_f, \mathcal{E}_f, \mathbf{X}_f), \quad |\mathcal{V}_f| = T_v \times 68
   $$

4. **音频图构建**：
   $$
   G_a = (\mathcal{V}_a, \mathcal{E}_a, \mathbf{X}_a), \quad |\mathcal{V}_a| = T_a
   $$

5. **GNN 编码**：
   $$
   \mathbf{h}_f = \text{Readout}_f\big(\text{GAT}_f(G_f)\big)
   $$
   $$
   \mathbf{h}_a = \text{Readout}_a\big(\text{GAT}_a(G_a)\big)
   $$

6. **跨模态注意力融合**：
   $$
   \mathbf{z} = \big[ \mathbf{h}_f + \alpha_{f \to a} \mathbf{v}_a \;\|\; \mathbf{h}_a + \alpha_{a \to f} \mathbf{v}_f \big]
   $$

7. **分类**：
   $$
   \hat{y} = \sigma\big(\text{MLP}(\mathbf{z})\big)
   $$

8. **训练**：
   $$
   \mathcal{L} = -\alpha_t (1 - p_t)^\gamma \log(p_t) + \lambda_{\text{reg}} \|\Theta\|_2^2
   $$

9. **测试集成**：
   $$
   \hat{y}^{\text{ens}} = \frac{1}{K} \sum_{k=1}^{K} \hat{y}^{(k)}, \quad \hat{y}_{\text{bin}}^{\text{ens}} = \mathbb{1}[\hat{y}^{\text{ens}} \ge \tau^*]
   $$

---

## 12. 关键超参数

| 超参数 | 取值 |
|---|---|
| 面部采样帧数 $T_v$ | 32 |
| 音频采样帧数 $T_a$ | 32 |
| 面部节点特征维度 | 13 |
| 音频节点特征维度 | 25 |
| 面部 GAT 层数 | 3 |
| 音频 GAT 层数 | 2 |
| 面部隐藏维度 | 128 |
| 音频隐藏维度 | 64 |
| GAT 头数（中间层） | 4 |
| Dropout | 0.3 |
| 学习率 | $10^{-3}$ |
| 权重衰减 | $10^{-4}$ |
| Batch size | 16 |
| Focal loss $\alpha$ | 0.25 |
| Focal loss $\gamma$ | 2 |
| 早停耐心 | 15 epochs |
| 集成模型数 $K$ | 3 |
| 总参数量 | 约 $1.2 \times 10^5$ |

---

## 13. 符号速查表

| 符号 | 含义 |
|---|---|
| $\boldsymbol{V}$ | 原始面部关键点序列 |
| $\boldsymbol{A}$ | 原始声学特征序列 |
| $T_v, T_a$ | 面部/音频采样长度 |
| $G_f, G_a$ | 面部图、音频图 |
| $\mathcal{V}_f, \mathcal{V}_a$ | 面部/音频节点集合 |
| $\mathcal{E}_f, \mathcal{E}_a$ | 面部/音频边集合 |
| $\mathbf{x}_{i,t}$ | 面部节点 $(t,i)$ 的特征 |
| $\mathbf{x}_t^{(a)}$ | 音频节点 $t$ 的特征 |
| $\mathbf{h}_f, \mathbf{h}_a$ | 图级面部/音频嵌入 |
| $\tilde{\mathbf{h}}_f, \tilde{\mathbf{h}}_a$ | 跨模态注意力更新后的嵌入 |
| $\mathbf{z}$ | 融合向量 |
| $\hat{y}$ | 预测的抑郁概率 |
| $\tau^*$ | 验证集最优 F1 阈值 |
| $\Theta$ | 所有可学习参数 |

---

*本文档基于 `overleaf_upload/AFGNN.tex` 与 `paper/AFGNN_design.md` 整理而成。*
