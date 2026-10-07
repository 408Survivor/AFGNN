# AFGNN 重建进度日志

> 倒序记录。实验登记的权威索引在 `experiments/INDEX.md`。

## 2026-10-08：GitHub 仓库同步修复后项目 + git 本体迁入工作目录

- 背景：论文（BIBM 2026）被接收。检查发现 GitHub 公开仓库（git 本体在旧路径）
  仍是坐标修复前的 bug 版本代码，与论文结果不一致；工作目录则完全不在版本控制内。
- **归档先行**：旧仓库修复前完整状态（含未提交的作废 region_importance_auc 分析）
  提交到 `archive/pre-coordinate-fix` 分支并推送。
- **main 同步（commit 32a2c15）**：工作目录的修复后代码、exp_1~103 全部实验记录、
  文档体系整体同步进 GitHub 仓库；删除作废的旧 checkpoints/results/records/GUIDE.md；
  `paper/`（camera_ready 等）与 `visualization/`（图表脚本）原样保留。
- **git 本体迁移**：在 `/home/ltq/Code/AFGNN` 执行 init + fetch + 对齐 main，
  paper/、visualization/ 迁回工作目录；>100MB 的 mp4（gitignored）从旧路径拷贝过来。
  **用户规定：今后一切操作只在工作目录进行，旧路径弃用。**
- 本地排除（`.git/info/exclude`，不进公开仓库）：ReadmeForKimi.md、
  paper/D-Vlog.pdf、paper/eGeMAPS.pdf。
- 另：同日完成 Phase D（exp_59~103）补登记，详见下一条。

## 2026-10-08：Phase D 登记归档（exp_59~exp_103）——全部重跑实验收官

- 背景：Phase D（45 单元）实际已于 **2026-08-22 当晚跑完**（exp_59 起始时间戳
  19:32），但当时未做登记；本次补登记，INDEX.md 下一个可用编号更新为 **exp_104**。
- **Part 1 no_velocity × seeds 43~46（exp_59~62）**：5 seeds test AUC
  **0.7816 ± 0.0184**，vs full（0.7861 ± 0.0136）Δ = **-0.0045，落在种子噪声内**。
  ⚠️ Phase C 单 seed（42）观测到的 -0.023 跌幅经多 seed 验证**不成立**——
  至此面部图所有组件（含速度特征）均无超过种子噪声的贡献。
- **Part 2 afgnn_audio_only × seeds 42~46（exp_63~67）**：test AUC
  **0.7419 ± 0.0138**。纯音频远好于纯面部（exp_9 单 seed 0.6401），
  多模态融合（0.7881）较纯音频 +0.046；音频单模态普遍早峰（3/5 在 ep≤4 见顶）。
- **Part 3 区域 mask 00~08 × seeds 43~46（exp_68~103）**：9 区域 5 seeds 均值差
  全部在 ±0.015 以内（最大为外唇 +0.0146），**无任一区域贡献超过种子噪声**；
  单 seed 下 exp_33 外唇 +0.022 的现象多 seed 不成立。
- 同步更新：INDEX.md（45 行登记 + 消融汇总改写 + 新增单模态/区域 mask 汇总表）、
  ReadmeForKimi.md（当前进度）、AFGNN介绍.md（第 3 节结果表补音频单模态 +
  消融结论改写为 Phase D 多 seed 终版）。

## 2026-08-22：收工（Phase D 在 tmux 中运行）

- **文档整顿**：README.md 重写 Quick Start/目录结构（对齐 exp_N 与新数据路径）；
  GUIDE.md 删除，有效内容（config 命名语义、一实验一 yaml、环境纪律）并入
  ReadmeForKimi.md"实验规范"节；ReadmeForKimi.md 新增"文档分工"节；
  `afgnn_face_only.yaml` 改名 `afgnn_base.yaml`（实为多模态基准）并更新全部引用；
  新增 `experiments/configs/afgnn_audio_only.yaml`（纯音频配置）。
- **工具**：`plot_training_history.py`（train.py 自动出 training_curves.png）；
  `seed_ensemble_test.py` 支持 `--exp_dirs`；批量脚本 run_phase_b/c/c2/d.sh
  （断点跳过）。测试基线 78 项全绿。
- **Phase D 已启动**：`bash src/scripts/run_phase_d.sh` 在 **tmux** 中运行，
  45 单元（no_velocity 多 seed + audio_only 5 seeds + 区域 mask 多 seed），
  预计 exp_59~103、2~2.5 小时。**跑完后待办：登记 exp_59+、出最终消融结论、
  把音频单模态补进 AFGNN介绍.md 结果表**。

## 2026-08-22：Phase C-2 完成（exp_35~exp_58，消融多 seed）——任务 3 重跑收官

- `bash src/scripts/run_phase_c2.sh`：6 配置 × seeds 43~46 = 24 个单元全部完成。
- **消融多 seed 结论（test AUC，5 seeds）**：full 0.7861±0.0136；
  no_temporal +0.0047、random_graph +0.0035、no_static +0.0008、no_region -0.0017、
  no_selfloop -0.0071——**全部落在种子噪声内**。
- 唯一单 seed 超噪声的是 no_velocity（-0.023，仅 seed 42，未经多 seed 确认）。
- **旧项目"去组件即显著掉点"的消融结论在修复后特征上不成立**；961 样本规模下
  图结构组件贡献 < 种子方差。AFGNN介绍.md 结果表已同步更新。
- 至此任务 3 全部完成：Phase A（10 配置 seed42）+ Phase B（top-2 × 5 seeds）+
  种子集成（test AUC 0.7990）+ Phase C/C-2（消融 5 seeds）。

## 2026-08-22：种子集成评估完成（最优配置 5 seeds）

- `python src/seed_ensemble_test.py --exp_dirs experiments/exp_7 exp_11..exp_14 --split test`
- **5 种子集成（seed 42~46 概率平均，阈值取 valid F1 最优 0.31）**：
  **test AUC 0.7990**，acc 0.726 / precision 0.715 / recall 0.878 / F1 0.788。
  结果存于 `experiments/results/seed_ensemble_face_enhanced_focal_adamw_cosine_seed42-46_test.json`。
- `seed_ensemble_test.py` 已适配 exp_N 体系：新增 `--exp_dirs`（config 取首个目录快照、
  checkpoint 取各目录 best_seed*.pt），输出文件名含配置名+种子范围；
  旧 `--config + --seeds` 模式保留。76 项测试全绿。

## 2026-08-22：Phase C 完成（exp_19~exp_34，消融 seed 42）

- `bash src/scripts/run_phase_c.sh`：16 个消融配置全部完成，编号与预测一致（exp_19~34）。
- 基准 `ablation_full`（exp_19）：test AUC 0.7824。
- **组件消融**：唯一明显掉点的是 no_velocity（0.7593，-0.023）；no_static（-0.008）、
  no_selfloop（-0.005）、no_region（-0.001）、no_temporal（+0.002）均在噪声内；
  random_graph 反而 +0.006。
- **区域 mask**：9 个区域全部不低于 full，外唇 mask 甚至 +0.022（0.8048）。
- ⚠️ **关键结论**：单 seed 下消融差异（多为 ±0.01 内）小于种子间方差（Phase B 实测
  std ≈ 0.009~0.019），**除速度特征外无法得出"模块必要"的结论**；旧项目"去时间边
  AUC -3.9%"的结论在修复后特征上不成立。消融若要下结论，需对关键项补多 seed。

## 2026-08-21：Phase B 完成（exp_11~exp_18，多 seed 确认）

- `bash src/scripts/run_phase_b.sh`：top-2 配置 × seeds 43~46，8 个 训练+测试 单元全部完成。
- **afgnn_face_enhanced_focal_adamw_cosine.yaml**（含 exp_7 共 5 seeds）：
  test AUC **0.7881 ± 0.0089**，valid 范围 [0.8136, 0.8320]；五个 seed 全部 ≥ 0.780，
  稳定性与均值双优，确定为当前最优配置。
- **afgnn_face_enhanced_focal_a05_g15.yaml**（含 exp_6 共 5 seeds）：
  test AUC 0.7824 ± 0.0185，波动大（exp_16 seed44 早峰 ep3 仅 0.7594，exp_17 达 0.7992）。
- 新增 `src/scripts/run_phase_b.sh`：批量训练+测试脚本，带 (config, seed) 断点跳过。

## 2026-08-21：exp_10 完成，Phase A（主配置 seed 42 摸底）收官

- `python src/train.py --config experiments/configs/afgnn_base.yaml --seed 42`
- 57 epochs，best valid AUC **0.8023** @ ep42；**test AUC 0.7302**
  （acc 0.670 / precision 0.665 / recall 0.870 / F1 0.754）——valid 尚可但 test 落差最大（0.072）。
- **Phase A 横向对比（test AUC）**：exp_7 adamw_cosine 0.8031 > exp_6 focal_a05_g15 0.7979
  > exp_1 focal 0.7857 > exp_3 concat 0.7827 > exp_5 focal_a01_g3 0.7822
  > exp_8 audio_enhanced 0.7755 > exp_2 enhanced 0.7697 > exp_4 aug 0.7354
  > exp_10 base 0.7302 > exp_9 face_only 0.6401。
- 要点：focal 系列前三占三席，AdamW+cosine 最优；增广有害（修复前后一致）；
  纯面部单模态远逊多模态（音频贡献关键）；基础架构泛化差。
- **Phase B 候选**：exp_7 配置（afgnn_face_enhanced_focal_adamw_cosine）为首选，
  exp_6（a05_g15）备选，补跑 seeds 43~46。

## 2026-08-21：exp_9 完成

- `python src/train.py --config experiments/configs/afgnn_face_only_enhanced.yaml --seed 42`
- 17 epochs 早停，best valid AUC **0.6865** @ ep2（峰值极早）；
  **test AUC 0.6401**（acc 0.637 / precision 0.660 / recall 0.772 / F1 0.712）。
- **纯面部单模态远低于多模态**（test AUC 差 ~0.16）：音频分支贡献显著，
  多模态融合是必要的。

## 2026-08-21：exp_8 完成

- `python src/train.py --config experiments/configs/afgnn_audio_enhanced.yaml --seed 42`
- 18 epochs 早停，best valid AUC **0.7918** 出现在 **ep3**（峰值极早，之后过拟合）；
  **test AUC 0.7755**（acc 0.708 / precision 0.708 / recall 0.846 / F1 0.770）。
- 音频增强分支（delta 特征 + skip-1 边）未带来增益，且收敛/过拟合更快。

## 2026-08-21：exp_7 完成

- `python src/train.py --config experiments/configs/afgnn_face_enhanced_focal_adamw_cosine.yaml --seed 42`
- 29 epochs，best valid AUC **0.8312** @ ep14；**test AUC 0.8031**
  （acc 0.750 / precision 0.747 / recall 0.862 / F1 0.800）。
- **目前 test AUC / acc / F1 全项最高**——AdamW + cosine 调度是迄今最优组合，
  Phase B 多 seed 的首选候选。

## 2026-08-21：exp_6 完成

- `python src/train.py --config experiments/configs/afgnn_face_enhanced_focal_a05_g15.yaml --seed 42`
- 39 epochs，best valid AUC **0.8238** @ ep24；**test AUC 0.7979**
  （acc 0.712 / precision 0.718 / recall 0.829 / F1 0.770）。
- **目前 test AUC 最高**：较温和的 focal（α=0.5/γ=1.5）优于默认 α=0.25/γ=2.0（exp_1）。

## 2026-08-21：exp_5 完成

- `python src/train.py --config experiments/configs/afgnn_face_enhanced_focal_a01_g3.yaml --seed 42`
- 53 epochs，best valid AUC **0.7965** @ ep38；**test AUC 0.7822**
  （acc 0.726 / precision 0.702 / recall 0.919 / F1 0.796）。
- α=0.1/γ=3 呈高召回取向（recall 0.919 为目前最高），AUC 与 exp_1/3 接近。

## 2026-08-21：exp_4 完成

- `python src/train.py --config experiments/configs/afgnn_face_enhanced_aug.yaml --seed 42`
- 40 epochs，best valid AUC **0.8105** @ ep25；**test AUC 0.7354**
  （acc 0.698 / precision 0.712 / recall 0.805 / F1 0.756）。
- valid 尚可但 test 明显掉点（0.7354，目前最低）：landmark 增广在修复后特征上依然有害，
  与旧项目"增广有害"的注释一致。

## 2026-08-21：exp_3 完成

- `python src/train.py --config experiments/configs/afgnn_face_enhanced_concat.yaml --seed 42`
- 46 epochs，best valid AUC **0.8300** @ ep31；**test AUC 0.7827**
  （acc 0.731 / precision 0.743 / recall 0.821 / F1 0.780）。
- 单 seed 对比：concat 融合（exp_3）接近 focal（exp_1），均优于 enhanced 基准（exp_2）。
- 同日误操作：重复执行了一次 concat 训练 → 残缺 exp_4（进程被 Ctrl+Z 挂起），
  按编号规则处理后删除。

## 2026-08-21：exp_2 完成

- `python src/train.py --config experiments/configs/afgnn_face_enhanced.yaml --seed 42`
- 31 epochs，best valid AUC **0.7891** @ ep16；**test AUC 0.7697**
  （acc 0.717 / precision 0.720 / recall 0.837 / F1 0.774）。
- 单 seed 对比：focal 版（exp_1，valid 0.8335 / test 0.7857）优于 weighted_bce 基准（exp_2）。

## 2026-08-21：exp_1 完成（任务 3 首个实验）

- `python src/train.py --config experiments/configs/afgnn_face_enhanced_focal.yaml --seed 42`
- 训练 43 epochs，best valid AUC **0.8335** @ epoch 28；**test AUC 0.7857**
  （acc 0.703 / precision 0.717 / recall 0.805 / F1 0.759）。
- 新增可视化脚本 `src/scripts/plot_training_history.py`：
  `--exp_dir experiments/exp_N` 自动读取 training_history_seed*.json，
  输出 loss / valid AUC / acc / F1 三联曲线到 `<exp_dir>/training_curves.png`。
- `src/train.py` 已接入自动绘图：每次训练保存 history 后自动生成
  `training_curves.png`（绘图失败仅告警，不影响训练）。76 项测试全绿。

## 2026-08-21：数据源迁移到 D-Vlog_Raw + 数据维度建档

- 旧数据目录 `/data/ltq/DVlog/` 被删除，数据源切换为 **数据目录**
  `/data/ltq/D-Vlog_Raw/dvlog-dataset`（961 个逐样本变长特征 + `../labels.csv`）。
- 术语约定（用户定）：**工作目录** = `/home/ltq/Code/AFGNN`，
  **数据目录** = `/data/ltq/D-Vlog_Raw/dvlog-dataset`。
- `DATA_DIMENSIONS.md`（工作目录）：阶段 0 全量维度统计 + 语义拆分
  （visual 136 维 = 9 区域 landmark 分块布局，已实证；acoustic 25 维 =
  eGeMAPSv02 LLD，逐列顺序经 openSMILE config + 实测数据三重互证）。
  发现的数据质量问题：14 样本视觉/声学帧数不对齐、92 样本视觉全零。
- **processed 特征重建**（`src/scripts/build_processed_features.py`）：
  按论文方案（各模态独立截断/零填充到 T=596）生成
  `/data/ltq/D-Vlog_Raw/processed_official_features/` 9 个 .npy；
  与迁移前旧产物**逐字节完全一致**（一次性验证），sanity 与数据集加载验证通过。
- ⛔ 用户规定：此次比对后**严禁再参考 `/data/ltq/backup`** 的任何内容。
- 26 个 config + 4 个脚本（ensemble/seed 系列、generate_ablation_configs）
  的 `processed_dir` 默认值全部切到新 processed 路径。

## 2026-08-20：项目重建（任务 1 + 任务 2 完成）

**背景**：旧项目（`/home/ltq/DepressionCode/DepGNN/AFGNN`）发现坐标读取 bug，
历史实验全部作废；本项目整体搬迁代码（不含结果/checkpoint/历史），重打地基。

### 任务 1：坐标解析修复

- 新建 `src/data/landmark_layout.py` 作为唯一解析入口：
  `flat_to_coords`（(T,136)→(T,68,2)，OpenFace 分块布局 `[x_0..x_67, y_0..y_67]`）、
  `coords_to_flat`（逆变换）、`assert_landmark_layout` / `check_visual_array`（解剖学断言）。
- 修复两处误读：`build_face_graph`（graph_utils.py:484 原 `reshape(num_frames, 68, 2)`）、
  `augment_landmark_sequence`（augmentation.py 原 `reshape(T, 68, 2)` 及其写回）。
- 加载时 sanity 断言接入 `DVlogFaceDataset` / `DVlogFaceArrayDataset` 的 `__init__`。
- 实证（真实数据，train split）：
  - 数据为逐帧标准化（每帧 x/y 各自均值 0 方差 1），解剖学关系在聚合层面成立：
    眼嘴垂直 margin +0.91、眉毛左右 margin +1.12；交错误读下眼嘴顺序仅 ~1.3% 帧成立。
  - 修复后数据集加载验证通过，图节点特征逐点核对正确（node = (flat[i], flat[68+i])）。
- 新增 `tests/test_landmark_layout.py`（12 项：逐点还原、round-trip、图构建端到端、
  增广写回布局、sanity 正/反例）。

### 任务 2：实验管理体系

- 新建 `src/utils/experiment.py`（移植自 HiFAG）：exp_N 自动编号（max+1）、
  config/model_summary/run_info 落盘。
- `src/train.py` 接入：每次运行产出自包含 `experiments/exp_N/`
  （config.yaml、run_info.txt、model_summary.txt、best_seed{N}.pt、
  training_history_seed{N}.json），checkpoint 路径覆盖进 exp_N。
- `src/test.py` 接入：`--exp_dir` 支持，结果写回同一 exp_N 目录。
- 新建 `experiments/INDEX.md`（权威索引，下一个编号 exp_1）、`ReadmeForKimi.md`、本文件。
- 新增 `tests/test_experiment.py`（6 项：编号递增、缺口不回填、checkpoint 反推等）。
- **测试基线：76 项全绿**（58 原有 + 12 坐标布局 + 6 实验管理）。

### 任务 3（待做）：重跑实验

- 按 `experiments/configs/`（10 个）和 `ablation/configs/`（16 个）原配置重跑，
  seeds 42~46，由用户亲自执行，结果登记 INDEX.md。
- ⚠️ 超参是修复前错误特征上调出的，先原样重跑作对照，不调参。
