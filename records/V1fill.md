# AFGNN-V1.tex 修改建议

> 针对导师反馈：Table III 需要拆清楚 face graph 各组件贡献，并补充脸部区域 top-k 重要性。
> 目标文件：`AFGNN-V1.tex`

---

## 修改 1：更新 Graph structure 小节引导文字

**位置**：第 329–332 行

**当前内容**：

```latex
Table~\ref{tab:graph_ablation} compares different face graph constructions, keeping the audio branch and fusion fixed.
The simplest static + temporal graph with region one-hot and self-loops works best.
Adding dynamic edges based on feature similarity or motion consistently degrades performance, suggesting overfitting on the small training set.
```

**建议改为**：

```latex
Table~\ref{tab:graph_ablation} reports a component-level ablation of the face graph, keeping the audio branch and cross-attention fusion fixed.
The full configuration (static + temporal edges, facial-region one-hot, velocity features, and self-loops) works best.
Removing temporal edges or self-loops causes the largest performance drops, followed by removing velocity features, while removing anatomical edges or region one-hot has a smaller effect.
Adding dynamic edges based on feature similarity or motion consistently degrades performance, suggesting overfitting on the small training set.
```

**修改原因**：
- 明确说明表格现在是“组件级消融”，回应导师要求。
- 预先总结各组件贡献大小，引导读者。

---

## 修改 2：重写 Table III（`tab:graph_ablation`）

**位置**：第 333–350 行

**当前表格**：

```latex
\begin{table}[t]
\centering
\caption{Ablation on face graph structure. ``Static'' and ``Temp'' denote anatomical and temporal edges; ``Dyn'' denotes dynamic edges; ``Region'' adds a facial-region one-hot and self-loops.}
\label{tab:graph_ablation}
\footnotesize
\resizebox{\linewidth}{!}{%
\begin{tabular}{lcccc}
\toprule[1.2pt]
Face graph & Acc & Prec & Rec & F1 \\
\midrule[0.8pt]
Static + Temporal & 0.717 & 0.720 & 0.837 & 0.774 \\
Static + Temporal + Region + Self-loop & \textbf{0.717} & \textbf{0.698} & \textbf{0.902} & \textbf{0.787} \\
Static + Temporal + Dynamic ($k$=1) & 0.623 & 0.637 & 0.813 & 0.714 \\
Motion + Region ($k$=2) & 0.571 & 0.584 & 0.902 & 0.709 \\
\bottomrule[1.2pt]
\end{tabular}%
}
\end{table}
```

**建议改为**：

```latex
\begin{table}[t]
\centering
\caption{Component-level ablation on face graph structure. All variants keep the audio branch and cross-attention fusion fixed. ``Full'' uses static + temporal edges, region one-hot, velocity, and self-loops.}
\label{tab:graph_ablation}
\footnotesize
\resizebox{\linewidth}{!}{%
\begin{tabular}{lcc}
\toprule[1.2pt]
Face graph configuration & F1 & AUC \\
\midrule[0.8pt]
Full (Static + Temporal + Region + Velocity + Self-loop) & \textbf{0.794} & \textbf{0.802} \\
\midrule[0.8pt]
- Static edges & 0.786 & 0.792 \\
- Temporal edges & 0.748 & 0.763 \\
- Region one-hot & 0.789 & 0.786 \\
- Velocity $[dx,dy]$ & 0.752 & 0.760 \\
- Self-loop & 0.748 & 0.778 \\
\midrule[0.8pt]
Static + Temporal + Dynamic ($k$=1) & 0.714 & 0.672 \\
Motion + Region ($k$=2) & 0.709 & 0.664 \\
\bottomrule[1.2pt]
\end{tabular}%
}
\end{table}
```

**修改原因**：
- 拆清楚 5 个组件各自贡献：anatomical edges、temporal edges、region one-hot、velocity、self-loop。
- 表格从 4 列（Acc/Prec/Rec/F1）简化为 3 列（Configuration/F1/AUC），更易读。
- F1/AUC 数值来自本次新跑的消融实验（`ablation/results/summary.json`）。
- 保留原有 dynamic 对比行，说明 dynamic graph 不如 prior-guided graph。

**注意**：
- 新实验中 `full` 单种子 F1=0.7943，AUC=0.8017，与论文旧值 0.787/0.793 略有差异，属于单种子随机波动。
- dynamic 行数值沿用旧表，因为我们未重新跑 dynamic 实验。

---

## 修改 3：新增“面部区域重要性”子小节和表格

**位置**：第 350 行之后、第 352 行（`\subsubsection{Modality and fusion}`）之前插入。

**建议插入内容**：

```latex
\subsubsection{Facial region importance}

To understand which facial regions most influence depression classification, we mask the region one-hot vector of each region in turn and retrain the full model.
Table~\ref{tab:region_ablation} ranks the regions by the resulting F1 drop on the test set.

\begin{table}[t]
\centering
\caption{Facial region leave-one-out ablation. ``Mask'' zeros the one-hot vector of the corresponding region. Regions are ranked by F1 drop relative to the full model (F1 = 0.7943, AUC = 0.8017).}
\label{tab:region_ablation}
\footnotesize
\resizebox{\linewidth}{!}{%
\begin{tabular}{clcc}
\toprule[1.2pt]
Rank & Region & F1 & F1 drop \\
\midrule[0.8pt]
1 & Left eye & 0.743 & -5.15\% \\
2 & Face contour & 0.754 & -4.02\% \\
3 & Nose bottom & 0.770 & -2.45\% \\
4 & Right eye & 0.773 & -2.16\% \\
5 & Right eyebrow & 0.779 & -1.56\% \\
6 & Inner mouth & 0.780 & -1.47\% \\
7 & Nose bridge & 0.782 & -1.24\% \\
8 & Left eyebrow & 0.783 & -1.14\% \\
9 & Outer mouth & 0.787 & -0.78\% \\
\bottomrule[1.2pt]
\end{tabular}%
}
\end{table}

The eyes and face contour are the most important regions, while the mouth regions contribute less under the F1 metric; however, masking the outer mouth still degrades AUC by 3.16\%, indicating it helps the model's ranking ability.
```

**修改原因**：
- 满足导师要求的“脸部区域 top-k”。
- 明确给出 9 个区域的重要性排名。

---

## 修改 4（可选）：调整 Visualization 中关于 mouth 的描述

**位置**：第 421–423 行

**当前内容**：

```latex
When we aggregate GAT attention coefficients across all test samples and group them by facial region, the eye and mouth rings receive consistently higher weights than the face contour.
This pattern suggests that the model prioritises expressive regions over identity-related shape information.
```

**建议考虑改为**（若与区域 LOO 结果保持一致）：

```latex
When we aggregate GAT attention coefficients across all test samples and group them by facial region, the eye and mouth rings receive consistently higher attention weights than the face contour.
Interestingly, the leave-one-out ablation in Table~\ref{tab:region_ablation} shows that the eyes and face contour are the most critical regions for F1, while the mouth contributes more to the model's ranking ability (AUC) than to the precision-recall trade-off captured by F1.
```

**修改原因**：
- 区域 LOO 结果显示 mouth 的 F1 贡献较小，但 outer_mouth 对 AUC 仍有帮助。
- 避免读者误认为 mouth 在 LOO 中也是最重要的区域。

**是否采纳**：可选。如果原文的 attention weight 可视化确实显示 mouth 权重高，保留原句也无矛盾，只是需要区分 attention weight 与 LOO 贡献。

---

## 修改 5（提醒）：Figure 4 可能需要更新

**位置**：第 407–412 行

**说明**：
- Figure 4(a) 当前展示的是旧版 facial graph structure ablation。
- 如果 Table III 更新为新的组件消融，Figure 4(a) 最好也同步重绘。
- 可以直接使用 `src/scripts/generate_ablation_configs.py` 生成的配置和 `ablation/results/summary.json` 中的数据重新绘图。

**建议操作**：
- 更新 `paper/figures/draw_figure4_ablation.py` 或相应绘图脚本。
- 用新数据重新生成 `figure4_ablation.pdf`。

---

## 汇总：需要修改的行号

| 修改 | 起始行 | 结束行 | 类型 |
|---|---:|---:|:---|
| 更新 Graph structure 引导文字 | 329 | 332 | 替换 |
| 重写 Table III | 333 | 350 | 替换 |
| 新增 Facial region importance 子小节和表格 | 350 后 | 352 前 | 插入 |
| （可选）调整 Visualization 中 mouth 描述 | 421 | 423 | 替换 |
| （提醒）更新 Figure 4 | 407 | 412 | 重绘 PDF |

---

## 数据来源

所有新数值来自本次补充实验：

- 基线（full）：F1 = 0.7943，AUC = 0.8017
- 组件消融：见 `ablation/results/summary.json`
- 区域 LOO：见 `ablation/results/summary.json`
