# D-Vlog 数据维度记录

> 记录数据在各处理阶段的维度变化，每次生成新版本特征时追加一节。
> 最近更新：2026-08-21（阶段 0 建立）

## 阶段 0：官方原始特征（`dvlog-dataset/`，未做任何处理）

- 组织方式：`dvlog-dataset/<i>/<i>_visual.npy` + `<i>_acoustic.npy`，i = 0~960，共 961 个样本
- 元信息：`labels.csv`（index, label, duration, gender, fold），
  fold = train 647 / valid 102 / test 212；label = depression 555 / normal 406（阳性率 57.8%）

### 逐样本维度

| 项目 | visual | acoustic |
|------|--------|----------|
| 单样本形状 | (T_i, 136) 变长 | (T_i, 25) 变长 |
| dtype | float64 | float64 |
| 特征布局 | OpenFace 分块 `[x_0..x_67, y_0..y_67]`，逐帧标准化（非零帧均值 0 方差 1）；零行 = 未检测到人脸 | eGeMAPS 25 维 LLD 原始值 |
| 值域（抽样）| ≈ [-3.2, 3.2] | ≈ [-201, 3200] |

### 维度的实际意义（依据：工作目录 `paper/D-Vlog.pdf` p.3-4 + `src/models/graph_utils.py`）

**时间维 T_i** = 视频秒数：visual 按 1 FPS 每秒取 1 帧、acoustic 按 1 秒切 1 段，
两者逐秒对齐（样本 0：labels.csv duration 823.31s → T_i=823）。

**visual 的 136 维** = 68 个 dlib 面部 landmark 的归一化 (x, y) 坐标，**分块布局
`[x_0..x_67, y_0..y_67]`**（前 68 维全是 x，后 68 维全是 y；**不是**交错 `[x0,y0,x1,y1,...]`）：
dim 0~67 是 x 块（dim i = 第 i 个 landmark 的 x），dim 68~135 是 y 块（dim 68+i = 第 i 个 landmark 的 y）。

布局已经真实数据实证（2026-08-21，样本 0 第 10 帧）：分块解析下 眼y=−0.876 < 嘴y=+0.837
（眼在嘴上 ✅）、左眉x=−1.084 < 右眉x=+1.077（左眉在左 ✅）；交错解析下眼嘴关系颠倒 ❌。
统计层面：分块解析 train 集眼嘴垂直 margin ≈ +0.91，交错解析下"眼在嘴上"仅 ~1.3% 帧成立。
按 9 个解剖区域拆分（Dlib/OpenFace 68 点约定，定义见 `graph_utils.LANDMARK_GROUPS_68`）：

| landmark idx | x 块 dim | y 块 dim | 区域 | 点数 |
|---|---|---|---|---|
| 0~16 | 0~16 | 68~84 | 面部轮廓 face_contour | 17 |
| 17~21 | 17~21 | 85~89 | 左眉 left_eyebrow | 5 |
| 22~26 | 22~26 | 90~94 | 右眉 right_eyebrow | 5 |
| 27~30 | 27~30 | 95~98 | 鼻梁 nose_bridge | 4 |
| 31~35 | 31~35 | 99~103 | 鼻底 nose_bottom | 5 |
| 36~41 | 36~41 | 104~109 | 左眼 left_eye | 6 |
| 42~47 | 42~47 | 110~115 | 右眼 right_eye | 6 |
| 48~59 | 48~59 | 116~127 | 外唇 outer_mouth | 12 |
| 60~67 | 60~67 | 128~135 | 内唇 inner_mouth | 8 |

注意：逐帧标准化（每帧 68 个 x / 68 个 y 各自均值 0 方差 1）意味着每维的值是
"该点相对全脸中心的位置，以全脸尺度为单位"——**绝对像素位置和人脸大小信息已被抹掉**。

**acoustic 的 25 维** = OpenSmile eGeMAPS 低阶声学描述子（LLD），
提取参数：帧长 0.06s / 帧移 0.01s，每秒一段并对段内全部 25 维取平均（论文 p.3-4）。
25 维的构成依据工作目录 `paper/eGeMAPS.pdf`（Eyben et al. 2015，Sec. 3.1/3.2）：
eGeMAPS = 极简集 18 LLD + 扩展 7 LLD = 25 LLD，与 D-Vlog 的"25 LLD"精确吻合：

| 组 | 参数 | 含义 |
|---|---|---|
| 频率（8） | Pitch | 对数 F0，半音阶（27.5 Hz 起为 0） |
| | Jitter | 相邻 F0 周期长度的偏差 |
| | Formant 1/2/3 frequency | 第 1/2/3 共振峰中心频率 |
| | Formant 1/2/3 bandwidth | 共振峰带宽（F2/F3 带宽为扩展集补充） |
| 能量/幅度（3） | Shimmer | 相邻 F0 周期峰值幅度差 |
| | Loudness | 基于听觉频谱的感知响度估计 |
| | HNR | 谐波-噪声比（谐波能量 / 噪声能量） |
| 频谱（14） | Alpha Ratio | 50–1000 Hz 与 1–5 kHz 的能量比 |
| | Hammarberg Index | 0–2 kHz 最强峰 / 2–5 kHz 最强峰 |
| | Spectral Slope 0–500 / 500–1500 Hz | 两个频带内对数功率谱的线性回归斜率 |
| | Formant 1/2/3 relative energy | 共振峰处谐波峰能量相对 F0 峰能量之比 |
| | H1–H2 | 第 1 / 第 2 谐波能量比 |
| | H1–A3 | 第 1 谐波 / 第三共振峰区最高谐波能量比 |
| | MFCC 1–4 | 梅尔频率倒谱系数 1–4（低阶 ≈ 谱倾斜/谱形） |
| | Spectral Flux | 相邻两帧频谱之差（谱动态） |

**25 列的排列顺序**（2026-08-21 确认）：依据 openSMILE `eGeMAPSv02.conf` 的
`lldconcat`（`lld = lldsetE_smo ++ lldsetF_smo`）及两个 core.lld.conf.inc 的
`cDataSelector` 字段顺序，并经数据目录真实数据逐列统计实证
（F1<F2<F3 对 99.9% 帧成立且共振峰范围教科书级；spectralFlux/jitter/shimmer 全非负；
幅度列下限恰为 −201 哨兵值）：

| col | 特征名（openSMILE 全名） | 含义 |
|---|---|---|
| 0 | Loudness_sma3 | 感知响度 |
| 1 | alphaRatio_sma3 | 50–1000 Hz / 1–5 kHz 能量比 (dB) |
| 2 | hammarbergIndex_sma3 | 0–2k / 2–5k Hz 峰值比 |
| 3 | slope0-500_sma3 | 0–500 Hz 谱斜率 |
| 4 | slope500-1500_sma3 | 500–1500 Hz 谱斜率 |
| 5 | spectralFlux_sma3 | 谱通量（相邻帧谱差） |
| 6 | mfcc1_sma3 | MFCC 1（≈谱倾斜） |
| 7 | mfcc2_sma3 | MFCC 2 |
| 8 | mfcc3_sma3 | MFCC 3 |
| 9 | mfcc4_sma3 | MFCC 4 |
| 10 | F0semitoneFrom27.5Hz_sma3nz | 对数 F0 半音阶（0 = 清音/无声段） |
| 11 | jitterLocal_sma3nz | 局部 jitter |
| 12 | shimmerLocaldB_sma3nz | 局部 shimmer (dB) |
| 13 | HNRdBACF_sma3nz | 谐波噪声比 (dB) |
| 14 | logRelF0-H1-H2_sma3nz | H1/H2 谐波能量对数比 |
| 15 | logRelF0-H1-A3_sma3nz | H1/A3 能量对数比 |
| 16 | F1frequency_sma3nz | 第 1 共振峰频率 (Hz) |
| 17 | F1bandwidth_sma3nz | 第 1 共振峰带宽 (Hz) |
| 18 | F1amplitudeLogRelF0_sma3nz | F1 处幅度（相对 F0，dB） |
| 19 | F2frequency_sma3nz | 第 2 共振峰频率 (Hz) |
| 20 | F2bandwidth_sma3nz | 第 2 共振峰带宽 (Hz) |
| 21 | F2amplitudeLogRelF0_sma3nz | F2 处幅度（相对 F0，dB） |
| 22 | F3frequency_sma3nz | 第 3 共振峰频率 (Hz) |
| 23 | F3bandwidth_sma3nz | 第 3 共振峰带宽 (Hz) |
| 24 | F3amplitudeLogRelF0_sma3nz | F3 处幅度（相对 F0，dB） |

注：`sma3`/`sma3nz` 后缀 = 3 帧滑动平均平滑（nz 版只在非零段内平滑）；
幅度列的 −201.00 为官方提取的下限哨兵值；共振峰频率列为 0 的帧对应无声段。
⚠️ 残留风险：D-Vlog 作者未声明 openSMILE 版本，以上顺序与 v02 config 及实测数据
高度自洽，但若作者当年用的是 pre-v2.0.0 版本（官方承认当时部分 LLD 被误输出为 delta），
个别列的语义可能有偏差；如需 100% 确认，需向作者索要提取代码。

### 帧长 T_i 分布（以 visual 为准）

- 全数据集：min 23 / p5 111 / 中位 472 / 均值 595.8 / p95 1374 / max 3968
- T < 596：601 个；T = 596：0 个；T > 596：360 个
- train（647）：T ∈ [23, 3968]，404 个短于 596
- valid（102）：T ∈ [57, 1843]，64 个短于 596
- test（212）：T ∈ [64, 2209]，133 个短于 596

### 已知数据质量问题（原样保留，未修）

- **视觉/声学帧数不对齐**：14 个样本（多数差 1 帧；454 差 26、465 差 23）：
  87, 151, 163, 183, 236, 363, 444, 454, 465, 564, 609, 772, 785, 925
- **全零视觉样本**（整段视频未检测到人脸）：92 个：
  1, 11, 16, 29, 34, 46, 53, 103, 106, 110, 113, 138, 144, 151, 165, 179, 188, 196,
  206, 207, 212, 218, 224, 230, 231, 232, 233, 234, 236, 238, 259, 276, 278, 292,
  293, 305, 308, 318, 330, 331, 334, 336, 353, 360, 361, 367, 373, 387, 393, 418,
  420, 422, 434, 435, 438, 441, 453, 468, 475, 477, 487, 491, 494, 496, 503, 505,
  506, 517, 521, 530, 531, 532, 535, 537, 542, 551, 589, 630, 637, 679, 683, 719,
  758, 799, 803, 811, 827, 927
- 全数据集零行总数（视觉）：134,052 行

## 阶段 1：split 级 processed 特征 ✅（2026-08-21 生成并验证）

- 生成脚本：工作目录 `src/scripts/build_processed_features.py`
- 输出位置：`/data/ltq/D-Vlog_Raw/processed_official_features/`
- 方案（按 D-Vlog 论文 p.4 的字面描述）：**每个模态各自独立**截断/零填充到 T=596，
  fold 内按 index 升序，depression→1 / normal→0
- 实际维度：

| 文件 | 形状 | 说明 |
|---|---|---|
| train_visual.npy | (647, 596, 136) float64 | 逐帧标准化，分块布局，尾部零填充 |
| valid_visual.npy | (102, 596, 136) float64 | 同上 |
| test_visual.npy | (212, 596, 136) float64 | 同上 |
| {fold}_acoustic.npy | (N, 596, 25) float64 | 25 列顺序见上表 |
| {fold}_labels.npy | (N,) int64 | 阳性率 train .580 / valid .559 / test .580 |

- 14 个帧数不对齐样本（train 10 / valid 2 / test 2）：按论文方案无需特殊处理，
  7 个双模态均 ≥596 的样本截断后差异消失，7 个短样本末尾存在 ≤26 帧的单模态真实数据。
- 校验结果：
  1. 与旧流水线产物（迁移前的旧 processed 特征）**9 个文件逐字节完全一致**——
     重建正确，且证实旧流水线用的正是"各模态独立截断/填充"方案（此次比对为
     一次性验证，此后不再参考旧备份）。
  2. 解剖学 sanity 通过：眼嘴垂直 margin train +1.125 / valid +1.084 / test +1.069，
     眉毛左右 margin +1.386 / +1.313 / +1.306。
  3. `DVlogFaceDataset` 加载验证通过（图构建 + 音频挂载正常）。
