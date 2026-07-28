# AFGNN Section III 补充实验 — 计划与记录

> 记录日期：2026-06-30
> 记录人：Kimi Code CLI
> 背景：导师反馈 Section III（实验章节）内容基本足够，建议**只补两个轻量实验**：
> 1. **Seed stability 的 mean±std**（量化随机种子带来的方差）
> 2. **Edge-matched random graph baseline**（证明解剖图结构本身有用，而非只靠边数/消息传递）

本文件记录这两个补充实验的设计、改动、运行命令与结果。

---

## 0. 当前状态

| 实验 | 状态 |
|---|---|
| 实验一：Seed stability mean±std | ✅ 已完成（2026-06-30） |
| 实验二：Edge-matched random graph baseline | ✅ 已完成（2026-06-30）：随机图 F1=0.7719±0.0089 / AUC=0.7818±0.0064，较 full 一致下降 |

> 状态标记：⬜ 待开始 / 🔄 进行中 / ✅ 已完成

---

## 1. 实验目的（回应导师）

两个都是审稿对照实验，不为提性能，而为堵质疑：

1. **Seed stability**：堵「0.807 是否挑了好种子」。论文 Table IV 目前只有孤立的
   `Single seed 42 = 0.787` 与 `3-seed ensemble = 0.8077`，无法判断方差。补单模型
   mean±std 后，证明单模型本身稳定，ensemble 不是靠运气。
2. **Edge-matched random graph**：堵「解剖图结构到底有没有用，还是有边能传消息就行」。
   把解剖边换成**边数完全相同**的随机连边，其余不变重训；若掉点，则证明是解剖先验在
   起作用。这是图网络论文证明「结构有意义」的标准 control。

---

## 2. 实验一：Seed stability（mean ± std）

### 2.1 关键事实：零额外训练

5 个单模型 checkpoint 与对应 training history 都已存在：

- `experiments/checkpoints/afgnn_face_enhanced_focal_best_seed{42,43,44,45,46}.pt`
- `experiments/results/training_history_seed{42..46}.json`

因此**不需要重训**，只需新增一个评估脚本，逐个 checkpoint 单独评估并汇总 mean±std。

### 2.2 做法

新增脚本 `src/seed_stability_test.py`（或给 `src/seed_ensemble_test.py` 加 `--per-seed` 模式）：

1. 逐个加载 seed 42–46 的单模型 checkpoint；
2. 每个模型**单独**在验证集上搜 F1 最优阈值；
3. 用该阈值在测试集上算 Acc / Prec / Rec / F1 / AUC；
4. 对种子集合求 **mean ± std**（主报告 F1 与 AUC）。

> 协议与实验二保持一致：默认报告 **seeds = 42, 43, 44** 三个种子的 mean±std
> （与最终 3-seed ensemble 同一组种子），并可附 5-seed 版本作参考。

### 2.3 运行命令（占位，脚本完成后填实）

```bash
conda activate DVlog
cd /home/ltq/DepressionCode/DepGNN/AFGNN
python src/seed_stability_test.py \
  --config experiments/configs/afgnn_face_enhanced_focal.yaml \
  --seeds 42 43 44 --split test
```

### 2.4 结果（已完成 2026-06-30）

脚本：`src/seed_stability_test.py`。每个种子单独在验证集搜 F1 阈值，再在测试集评估。
结果文件：`experiments/results/seed_stability_42-43-44_test_results.json`、
`experiments/results/seed_stability_42-43-44-45-46_test_results.json`。

**Per-seed 明细（单模型）**：

| Seed | thr | Acc | Prec | Rec | F1 | AUC |
|---|---|---|---|---|---|---|
| 42 | 0.30 | 0.7358 | 0.7343 | 0.8537 | 0.7895 | 0.7962 |
| 43 | 0.29 | 0.7358 | 0.7410 | 0.8374 | 0.7863 | 0.8052 |
| 44 | 0.27 | 0.7075 | 0.7047 | 0.8537 | 0.7721 | 0.7883 |
| 45 | 0.36 | 0.7217 | 0.7192 | 0.8537 | 0.7807 | 0.8096 |
| 46 | 0.22 | 0.7170 | 0.7086 | 0.8699 | 0.7810 | 0.7844 |

**汇总（mean ± 样本标准差 ddof=1）**：

| 种子集合 | Acc | Prec | Rec | F1 (mean±std) | AUC (mean±std) |
|---|---|---|---|---|---|
| 单模型 42/43/44（主报告） | 0.7264±0.0163 | 0.7267±0.0193 | 0.8482±0.0094 | **0.7826±0.0093** | **0.7966±0.0085** |
| 单模型 42–46（参考） | 0.7236±0.0123 | 0.7216±0.0158 | 0.8537±0.0115 | 0.7819±0.0066 | 0.7967±0.0108 |
| 3-seed ensemble | 0.764 | 0.766 | 0.854 | **0.8077** | **0.8041** |

**结论**：
- 单模型 F1/AUC 方差很小（F1 std≈0.009，AUC std≈0.009；5 种子下 std 更小），说明结果
  不依赖某个幸运种子，配置本身稳定。
- 3-seed ensemble（0.8077/0.8041）明显高于单模型均值（0.7826/0.7966）：ensemble 相对
  单模型平均 **F1 +2.5%，AUC +0.8%**，验证了 ensemble 降方差 + 提性能。
- 注：本次 seed 42 单模型 F1=0.7895/AUC=0.7962，与论文旧 Table IV 的「Single seed 42
  =0.787/0.793」基本一致（微小差异来自阈值搜索口径），可直接用本轮 mean±std 替换该行。

---

## 3. 实验二：Edge-matched random graph baseline

### 3.1 设计选择（已定）

- **随机化范围（变体 A）**：只随机化 **解剖（static）边**，保留 temporal edges、
  region one-hot、self-loop、velocity 与 audio/cross-attention 分支。结论最干净——
  「解剖拓扑有用，不只是帧内边数」。temporal 边的贡献已在组件消融里单独证明。
- **边数匹配**：随机图的帧内边数与 `build_static_edges()` **完全相同**
  （当前解剖边 = 每帧 64 条无向 / 128 条有向）。
- **固定随机拓扑**：用固定 seed 生成**一张**随机 intra-frame 拓扑，在所有帧、所有样本
  上复用——与解剖边「每个样本拓扑相同」严格对齐，唯一变量是「连哪些点」。
- **训练种子**：跑 seeds 42/43/44，报 mean±std，与实验一同协议。

### 3.2 代码改动计划

1. `src/models/graph_utils.py`：新增
   `build_random_static_edges(num_landmarks=68, num_edges=None, seed=0)`，
   生成与解剖边同边数的随机无向（双向）intra-frame 连边（无自环、无重复）。
2. `src/models/graph_utils.py::build_face_graph()`：新增参数
   `random_static_edges: bool=False`、`random_edge_seed: int=0`；为 True 时用随机边
   替换 `build_static_edges()` 的输出（其余逻辑不变）。
3. `src/data/dvlog_face_dataset.py`：`DVlogFaceDataset` 与 `get_dvlog_face_loaders`
   透传上述两个参数（沿用 `add_static_edges` 同一链路）。
4. `src/train.py` / `src/test.py`：从 config 读取并透传
   `random_static_edges` / `random_edge_seed`。
5. 新增配置 `ablation/configs/ablation_random_graph.yaml`：克隆 full 配置，
   `add_static_edges: true` + `random_static_edges: true`，其余全保持。

### 3.3 运行命令（占位）

```bash
conda activate DVlog
cd /home/ltq/DepressionCode/DepGNN/AFGNN
for s in 42 43 44; do
  python src/train.py --config ablation/configs/ablation_random_graph.yaml --seed $s
done
python src/seed_stability_test.py \
  --config ablation/configs/ablation_random_graph.yaml \
  --seeds 42 43 44 --split test
```

### 3.4 结果（已完成 2026-06-30）

训练 3 个种子（42/43/44），checkpoint =
`ablation/checkpoints/ablation_random_graph_best_seed{42,43,44}.pt`。
评估脚本 `src/seed_stability_test.py`，结果文件
`experiments/results/seed_stability_42-43-44_test_results.json`（注意：与实验一 3-seed 文件**同名已覆盖**，
实验一数据已存于本文件 §2.4，且 5-seed JSON 仍完整保留 42–44 的单模型值，可复原）。

**Per-seed 明细（随机图）**：

| Seed | thr | Acc | Prec | Rec | F1 | AUC |
|---|---|---|---|---|---|---|
| 42 | 0.33 | 0.7311 | 0.7538 | 0.7967 | 0.7747 | 0.7890 |
| 43 | 0.34 | 0.7217 | 0.7222 | 0.8455 | 0.7790 | 0.7768 |
| 44 | 0.32 | 0.7170 | 0.7442 | 0.7805 | 0.7619 | 0.7795 |

**对比（两行均为 seeds 42/43/44 的 mean ± 样本标准差 ddof=1，同一 `src/seed_stability_test.py` 协议）**：

| Face graph | F1 (mean±std) | AUC (mean±std) | vs full |
|---|---|---|---|
| Full（anatomical static + temporal + region + self-loop） | **0.7826±0.0093** | **0.7966±0.0085** | — |
| Edge-matched random graph | 0.7719±0.0089 | 0.7818±0.0064 | F1 −1.07%，AUC −1.48% |

**结论**：
- 把解剖边换成**边数相同**的随机图后，F1 与 AUC **在均值上一致下降**（F1 −1.1%，
  AUC −1.5%）。AUC 的分离更干净（差约 2 个 std），F1 约 1 个 std。说明解剖拓扑的贡献
  **超过单纯的「边数 / 消息传递容量」**，但幅度温和。
- 这与组件消融结果**自洽**：`no_static`（完全去掉解剖边）F1 仅降 0.81%，本就说明解剖边
  是相对次要的组件；因此边数匹配的随机 control 也只呈现温和下降，逻辑一致。
- 论文表述应强调「**一致下降 + 与组件消融自洽**」，不夸大为巨幅掉点，更可信。
- 注意：随机图保留了 temporal/region/self-loop/velocity（已证明是主要贡献者），所以本实验
  隔离的是**纯解剖空间连边**的增量价值，结论应限定在此范围。

---

## 4. 论文落点（Section III）—— 已落到 `AFGNN-V2.tex`（保留 V1 原样）

已在 **`AFGNN-V2.tex`**（从 `AFGNN-V1.tex` 复制后修改）完成以下改动：

1. **新增 Table `tab:random_graph`**（位于 Graph structure 小节、`tab:graph_ablation` 之后）：
   full vs edge-matched random，两行均为 seeds 42/43/44 的 3-seed mean±std（F1 + AUC），
   外加一段说明文字（强调「一致下降 + 与组件消融自洽」，不夸大）。
   - 单独建表而非塞进 Table III：因 Table III 现有行是单次运行 Acc/Prec/Rec/F1，
     与 3-seed mean±std 协议/列都不一致，分开报告口径更干净。
2. **Table IV（`tab:other_ablation`）**：把孤立的 `Single seed 42 & 0.787 & 0.793`
   替换为 `Single model (3 seeds) & 0.783±0.009 & 0.797±0.009`，caption 注明 mean±std。
3. **正文（Audio length/loss/ensemble 段）**：新增一句「单模型跨种子稳定（0.783±0.009 /
   0.797±0.009），结果不靠幸运 seed；ensemble 进一步抹平残余方差」。

校验：V2 共 6 张表，`table`/`tabular` 各 6/6 配平；V1 未改动。
（编译仍需在 Overleaf/本地完成——本机 pdflatex 环境此前已知损坏。）

---

## 5. 备注

- GPU：8× RTX 3090 全空闲；数据在 `/data/ltq/DVlog/processed_official_features`。
- 实验二单 seed 约 5–15 分钟，三种子 < 1 小时，属轻量。
- 数据一致性提醒：论文表格的 full 数值口径（单种子 0.794/0.802 vs ensemble 0.807/0.804）
  需与本轮新行口径对齐，避免横向不可比。

---

## 6. Figure 3 / Figure 4 重做（导师反馈"不好看"，2026-06-30）

### Figure 4（消融图）—— 已完成（最终：水平棒棒糖小图）
- 旧版问题：实心柱偏"重"、配色无语义、数据与 V2 脱节、无误差棒。导师反馈"不好看"后，
  又评估了"换构图方式"（不只是换色）。
- 比对了 3 种构图（水平棒棒糖小图 / 单张合并竖排 / 垂直棒棒糖）与 6 套配色，
  **最终选定：水平棒棒糖小图（Design A）+ Indigo/Periwinkle/Coral 配色**。
- 新版 `paper/figures/draw_figure4_ablation.py`：**2×3 六子图**，点+细线代替柱：
  (a) 面部图结构 (b) edge-matched random graph（带 std 误差棒）(c) 模态与融合
  (d) 音频长度 (e) 损失函数 (f) seed 稳定性与 ensemble（单模型带误差棒）。
- 语义配色（顶部点状图例）：靛蓝=最佳/最终、浅蓝=普通变体、珊瑚=退化/对照。
  配色可在脚本顶部 `PALETTE` 一行切换。数值逐一对齐 V2 各表。纯绘图，无需模型。

### Figure 3（音频图）—— 已完成（改用真实数据）
- 旧版问题：(b) 大片空白、三面板密度失衡、数据全为随机合成。
- 新流程分两步：
  1. `paper/figures/extract_figure3_data.py`：加载 checkpoint（seed42）+ test 集
     真实抑郁样本，导出 `figure3_data.npz`（真实归一化特征 + 真实 readout 注意力权重）。
     readout 权重 = AudioGNN 的 `AttentionalAggregation` gate 在 32 帧上的 softmax。
  2. `draw_figure3_audio_graph.py`：从 npz 绘图。(a) 真实热图 + 标注最显著帧；
     (b) 链图 + "节点=25维特征向量"真实 callout（修掉空白）；(c) 真实注意力、峰值帧高亮。
- 本次抽取样本：test index 0，label=1（抑郁），峰值帧=14。三个面板互相印证
  （尾部平坦段被注意力自动降权）。
- 注：抽取需要加载模型，由用户运行；若要换更有代表性的样本可加 `--sample_index N` 重抽。

### tex 同步
- `AFGNN-V2.tex` 中 Figure 3 与 Figure 4 的 caption 已更新（Fig4 改为 (a)–(f) 六子图 +
  配色语义说明；Fig3 改为真实样本描述）。正文无依赖旧子图数的引用。
- 输出已同时写入 `paper/figures/` 与 `overleaf_upload/figures/`（pdf/png/svg）。

### Figure 2（面部图构造）—— 已完成（2D 三帧 + 放大引导）
- 旧版问题：三个子图标题高低不齐、(b) 大片空白、(a)/(b) 无放大关联、(c) 3D 点云杂乱、整体散。
- 新版脚本 `paper/figures/draw_figure2_face_graph_v3.py`（保留 v2/v1 备用）：
  - 三面板共用同一坐标框 → 盒子等大、标题对齐。
  - (a) 在右眼处加虚线放大框，两条引线连到 (b) 放大后的眼睛。
  - (b) 放大的眼睛填满面板（眉毛作淡上下文），节点 36–41 标号。
  - (c) 改为 **2D 三帧并排**线框人脸，跟踪 chin/eye/mouth 三个关键点、用虚线箭头连成
    时序边（t→t+1→t+2），去掉旧版杂乱散点。
  - 底部统一区域图例。
- caption 已在 `AFGNN-V2.tex` 同步更新（提到放大框、多关键点、t/t+1/t+2）。
- 注：NumPy 2.0 下 `arr.ptp()` 已移除，脚本用 `np.ptp()`。


---

## 7. 会话暂停状态（2026-06-30，session pause）

### 7.1 已完成且定稿
- **实验一 Seed stability**：单模型 F1=0.7826±0.0093 / AUC=0.7966±0.0085（seeds 42/43/44）。
  脚本 `src/seed_stability_test.py`；结果 `experiments/results/seed_stability_*.json`。
- **实验二 Edge-matched random graph**：F1=0.7719±0.0089 / AUC=0.7818±0.0064，较 full 一致下降。
  代码改动见 §3；配置 `ablation/configs/ablation_random_graph.yaml`；checkpoint
  `ablation/checkpoints/ablation_random_graph_best_seed{42,43,44}.pt`。
- **论文 `AFGNN-V2.tex`**（从 V1 复制，V1 原样保留）：新增 `tab:random_graph`、Table IV 改 mean±std、
  Figure 2/3/4 caption 全部同步。表格 6/6 配平。**尚未在 Overleaf 编译验证排版**。
- **Figure 3**：真实数据重做（`extract_figure3_data.py` 抽取 → `draw_figure3_audio_graph.py` 绘图）。定稿。
- **Figure 4**：水平棒棒糖小图（Design A）+ 靛蓝配色（`draw_figure4_ablation.py`，顶部 `PALETTE` 可改）。定稿。
- **Figure 2**：当前定稿 = `draw_figure2_face_graph_v3.py`（2D 三帧 + 放大引导 + 对齐 + 底部图例）。

### 7.2 唯一未决：Figure 2 的"生动化"美化（OPEN）
导师/用户觉得 Figure 2 仍偏"死板"（纯点+线）。本会话尝试过两轮、**均被否**：
1. **画成真人脸**（柔和插画脸 / 线描脸）→ 用户评价"伪人"，否。
2. **抽象优雅图**（平滑曲线 + 节点柔光 / 柔色卡片发光）→ 用户仍觉"不好看"，否。
   - 临时 mockup 脚本/图仍在：`paper/figures/_fig2_graph_styles.py` 与 `_fig2_graph_styles.png`
     （**throwaway，可删**）。

**根因判断**：Figure 2 用的是**手写合成坐标** `LANDMARKS_68`，比例偏假（气球脸），所以怎么描都别扭。

**下次推荐的方向（按优先级）**：
1. **用真实关键点画**（最推荐）：D-Vlog `{split}_visual.npy` 是真实 68 点坐标。仿照 Fig3 写一个
   抽取脚本，从某真实帧取 68 点（reshape 136→68×2），用真实比例画图 → 自然、不伪人、与数据一致。
2. 叠在 CC0/合成头像底图上（关键点论文常见画法，需引入图片资源）。
3. TikZ/PGF 矢量版（出版级，但耗时）。
4. **已向用户征求一张"参考图"** 以对齐审美目标，用户尚未提供 → 下次可先要参考图再动手。

### 7.3 环境备注
- conda env：`DVlog`；数据：`/data/ltq/DVlog/processed_official_features`；8× RTX 3090。
- 约定：**所有需要加载模型/训练的命令交给用户运行**；纯绘图（matplotlib）由助手直接跑并自检。
- 所有实验图输出同时写入 `paper/figures/` 与 `overleaf_upload/figures/`。

### 7.4 其他待办（之前提到、未做）
- Overleaf 编译 V2，检查 Fig2/3/4 在双栏下的宽高与表格排版。
- `paper/figures/` 有 3 个 fig2 脚本（v1/v2/v3），v3 为现役；是否清理 v1/v2 待定。
- 是否清理 `_fig2_graph_styles.*` 临时文件待定。
