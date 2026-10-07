# AFGNN 实验索引

> 每次训练/评估完成后更新此表。exp_N 编号由 `src/utils/experiment.py`
> 运行时自动扫描分配（max+1），下表是人工维护的权威索引。
>
> **下一个可用编号：exp_104**

## 编号规则

> ⚠️ **坐标布局修复（2026-08-20 重建）**：D-Vlog visual 特征是 OpenFace 分块布局
> `[x_0..x_67, y_0..y_67]`，旧代码按交错 `reshape(T, 68, 2)` 误读，
> 重建前（/home/ltq/DepressionCode/DepGNN/AFGNN）的所有实验结果全部作废。
> 本项目（/home/ltq/Code/AFGNN）的一切 exp_N 均为修复后结果。

- 一次 `train.py` 运行 = 一个 exp_N（含单 seed）；多 seed 就是连续多个 exp_N。
- 消融实验（`ablation/configs/` 下的配置）也统一编号进 `experiments/exp_N`，
  不另开序列；登记表"目的"列注明消融。
- exp_N 目录内容：`config.yaml`、`run_info.txt`、`model_summary.txt`、
  `best_seed{N}.pt`、`training_history_seed{N}.json`、`test_{split}_results.json`。
- 若运行中断/失败，exp_N 目录是残缺的：**重跑前先删除该目录**，避免编号错位。
- `outputs/` 下的 `<expid>_<时间戳>/` 是 `run_context` 的附带日志
  （run.log + run.json + best.pt 软链），仅作日志参考，权威记录以 exp_N 为准。

## 实验登记表

| exp | 配置 | seed | 目的 | best valid AUC | test AUC | 状态/备注 |
|-----|------|------|------|----------------|----------|-----------|
| exp_1 | afgnn_face_enhanced_focal.yaml | 42 | 主配置原样重跑（修复后对照） | 0.8335 | 0.7857 | 43 epochs，best@ep28；test acc 0.703 / F1 0.759 |
| exp_2 | afgnn_face_enhanced.yaml | 42 | 主配置原样重跑（修复后对照） | 0.7891 | 0.7697 | 31 epochs，best@ep16；test acc 0.717 / F1 0.774 |
| exp_3 | afgnn_face_enhanced_concat.yaml | 42 | 主配置原样重跑（修复后对照） | 0.8300 | 0.7827 | 46 epochs，best@ep31；test acc 0.731 / F1 0.780 |
| exp_4 | afgnn_face_enhanced_aug.yaml | 42 | 主配置原样重跑（修复后对照） | 0.8105 | 0.7354 | 40 epochs，best@ep25；test acc 0.698 / F1 0.756；增广掉点明显 |
| exp_5 | afgnn_face_enhanced_focal_a01_g3.yaml | 42 | 主配置原样重跑（修复后对照） | 0.7965 | 0.7822 | 53 epochs，best@ep38；test acc 0.726 / F1 0.796 / recall 0.919（高召回取向） |
| exp_6 | afgnn_face_enhanced_focal_a05_g15.yaml | 42 | 主配置原样重跑（修复后对照） | 0.8238 | 0.7979 | 39 epochs，best@ep24；test acc 0.712 / F1 0.770 |
| exp_7 | afgnn_face_enhanced_focal_adamw_cosine.yaml | 42 | 主配置原样重跑（修复后对照） | 0.8312 | **0.8031** | 29 epochs，best@ep14；test acc 0.750 / F1 0.800，**目前 test AUC/acc/F1 全项最高** |
| exp_8 | afgnn_audio_enhanced.yaml | 42 | 主配置原样重跑（修复后对照） | 0.7918 | 0.7755 | 18 epochs，best@ep3 即峰（early stop）；test acc 0.708 / F1 0.770 |
| exp_9 | afgnn_face_only_enhanced.yaml | 42 | 主配置原样重跑（修复后对照；唯一纯面部单模态） | 0.6865 | 0.6401 | 17 epochs，best@ep2 即峰；test acc 0.637 / F1 0.712；**去掉音频掉点显著** |
| exp_10 | afgnn_base.yaml | 42 | 主配置原样重跑（修复后对照；基础多模态） | 0.8023 | 0.7302 | 57 epochs，best@ep42；test acc 0.670 / F1 0.754；valid-test 落差最大（0.072） |
| exp_11 | afgnn_face_enhanced_focal_adamw_cosine.yaml | 43 | Phase B 多 seed | 0.8265 | 0.7885 | 38 epochs，best@ep23；test acc 0.731 / F1 0.782 |
| exp_12 | afgnn_face_enhanced_focal_adamw_cosine.yaml | 44 | Phase B 多 seed | 0.8320 | 0.7825 | 57 epochs，best@ep42；test acc 0.736 / F1 0.788 |
| exp_13 | afgnn_face_enhanced_focal_adamw_cosine.yaml | 45 | Phase B 多 seed | 0.8136 | 0.7860 | 32 epochs，best@ep17；test acc 0.722 / F1 0.777 |
| exp_14 | afgnn_face_enhanced_focal_adamw_cosine.yaml | 46 | Phase B 多 seed | 0.8187 | 0.7804 | 37 epochs，best@ep22；test acc 0.731 / F1 0.797 |
| exp_15 | afgnn_face_enhanced_focal_a05_g15.yaml | 43 | Phase B 多 seed | 0.8230 | 0.7894 | 48 epochs，best@ep33；test acc 0.741 / F1 0.783 |
| exp_16 | afgnn_face_enhanced_focal_a05_g15.yaml | 44 | Phase B 多 seed | 0.7926 | 0.7594 | 18 epochs，best@ep3 即峰；test acc 0.679 / F1 0.757 |
| exp_17 | afgnn_face_enhanced_focal_a05_g15.yaml | 45 | Phase B 多 seed | 0.8140 | 0.7992 | 40 epochs，best@ep25；test acc 0.759 / F1 0.806 |
| exp_18 | afgnn_face_enhanced_focal_a05_g15.yaml | 46 | Phase B 多 seed | 0.8101 | 0.7662 | 40 epochs，best@ep25；test acc 0.703 / F1 0.779 |
| exp_19 | ablation_full.yaml | 42 | Phase C 消融基准（=exp_1 配置家族） | 0.8214 | 0.7824 | 44 epochs，best@ep29；test acc 0.726 / F1 0.775 |
| exp_20 | ablation_no_static.yaml | 42 | 消融：去静态边 | 0.8288 | 0.7742 | -0.008 vs full |
| exp_21 | ablation_no_temporal.yaml | 42 | 消融：去时间边 | 0.8179 | 0.7842 | +0.002 vs full |
| exp_22 | ablation_no_region.yaml | 42 | 消融：去区域 one-hot | 0.7984 | 0.7811 | -0.001 vs full |
| exp_23 | ablation_no_velocity.yaml | 42 | 消融：去速度特征 | 0.8172 | 0.7593 | **-0.023 vs full（唯一显著掉点）** |
| exp_24 | ablation_no_selfloop.yaml | 42 | 消融：去自环 | 0.8144 | 0.7775 | -0.005 vs full |
| exp_25 | ablation_random_graph.yaml | 42 | 消融：随机静态边 | 0.8530 | 0.7883 | +0.006 vs full |
| exp_26 | ablation_region_mask_00_face_contour.yaml | 42 | 消融：mask 面部轮廓 | 0.8246 | 0.7830 | +0.001 vs full |
| exp_27 | ablation_region_mask_01_left_eyebrow.yaml | 42 | 消融：mask 左眉 | 0.8183 | 0.7792 | -0.003 vs full |
| exp_28 | ablation_region_mask_02_right_eyebrow.yaml | 42 | 消融：mask 右眉 | 0.8359 | 0.7865 | +0.004 vs full |
| exp_29 | ablation_region_mask_03_nose_bridge.yaml | 42 | 消融：mask 鼻梁 | 0.8331 | 0.7966 | +0.014 vs full |
| exp_30 | ablation_region_mask_04_nose_bottom.yaml | 42 | 消融：mask 鼻底 | 0.8261 | 0.7883 | +0.006 vs full |
| exp_31 | ablation_region_mask_05_left_eye.yaml | 42 | 消融：mask 左眼 | 0.8257 | 0.7916 | +0.009 vs full |
| exp_32 | ablation_region_mask_06_right_eye.yaml | 42 | 消融：mask 右眼 | 0.8246 | 0.7873 | +0.005 vs full |
| exp_33 | ablation_region_mask_07_outer_mouth.yaml | 42 | 消融：mask 外唇 | 0.8296 | 0.8048 | +0.022 vs full |
| exp_34 | ablation_region_mask_08_inner_mouth.yaml | 42 | 消融：mask 内唇 | 0.8136 | 0.7871 | +0.005 vs full |
| exp_35 | ablation_full.yaml | 43 | Phase C-2 消融多 seed | 0.8179 | 0.7749 | |
| exp_36 | ablation_full.yaml | 44 | Phase C-2 消融多 seed | 0.8452 | 0.7948 | |
| exp_37 | ablation_full.yaml | 45 | Phase C-2 消融多 seed | 0.8179 | 0.8052 | |
| exp_38 | ablation_full.yaml | 46 | Phase C-2 消融多 seed | 0.8366 | 0.7733 | |
| exp_39 | ablation_no_static.yaml | 43 | Phase C-2 消融多 seed | 0.8269 | 0.8055 | |
| exp_40 | ablation_no_static.yaml | 44 | Phase C-2 消融多 seed | 0.8246 | 0.8016 | |
| exp_41 | ablation_no_static.yaml | 45 | Phase C-2 消融多 seed | 0.8390 | 0.7930 | |
| exp_42 | ablation_no_static.yaml | 46 | Phase C-2 消融多 seed | 0.7992 | 0.7603 | |
| exp_43 | ablation_no_temporal.yaml | 43 | Phase C-2 消融多 seed | 0.8452 | 0.7984 | |
| exp_44 | ablation_no_temporal.yaml | 44 | Phase C-2 消融多 seed | 0.8339 | 0.7913 | |
| exp_45 | ablation_no_temporal.yaml | 45 | Phase C-2 消融多 seed | 0.8300 | 0.8021 | |
| exp_46 | ablation_no_temporal.yaml | 46 | Phase C-2 消融多 seed | 0.8288 | 0.7781 | |
| exp_47 | ablation_no_region.yaml | 43 | Phase C-2 消融多 seed | 0.8273 | 0.7821 | |
| exp_48 | ablation_no_region.yaml | 44 | Phase C-2 消融多 seed | 0.8187 | 0.7934 | |
| exp_49 | ablation_no_region.yaml | 45 | Phase C-2 消融多 seed | 0.8320 | 0.7840 | |
| exp_50 | ablation_no_region.yaml | 46 | Phase C-2 消融多 seed | 0.8409 | 0.7814 | |
| exp_51 | ablation_no_selfloop.yaml | 43 | Phase C-2 消融多 seed | 0.8335 | 0.7772 | |
| exp_52 | ablation_no_selfloop.yaml | 44 | Phase C-2 消融多 seed | 0.8273 | 0.7806 | |
| exp_53 | ablation_no_selfloop.yaml | 45 | Phase C-2 消融多 seed | 0.8300 | 0.8034 | |
| exp_54 | ablation_no_selfloop.yaml | 46 | Phase C-2 消融多 seed | 0.8370 | 0.7563 | |
| exp_55 | ablation_random_graph.yaml | 43 | Phase C-2 消融多 seed | 0.8199 | 0.7964 | |
| exp_56 | ablation_random_graph.yaml | 44 | Phase C-2 消融多 seed | 0.8016 | 0.7925 | |
| exp_57 | ablation_random_graph.yaml | 45 | Phase C-2 消融多 seed | 0.8378 | 0.8006 | |
| exp_58 | ablation_random_graph.yaml | 46 | Phase C-2 消融多 seed | 0.8281 | 0.7702 | |
| exp_59 | ablation_no_velocity.yaml | 43 | Phase D 消融多 seed | 0.8121 | 0.7913 | |
| exp_60 | ablation_no_velocity.yaml | 44 | Phase D 消融多 seed | 0.8242 | 0.8062 | |
| exp_61 | ablation_no_velocity.yaml | 45 | Phase D 消融多 seed | 0.7556 | 0.7690 | best@ep13 早峰 |
| exp_62 | ablation_no_velocity.yaml | 46 | Phase D 消融多 seed | 0.8160 | 0.7822 | |
| exp_63 | afgnn_audio_only.yaml | 42 | Phase D 纯音频单模态 | 0.7587 | 0.7503 | best@ep4 即峰；test acc 0.703 / F1 0.749 |
| exp_64 | afgnn_audio_only.yaml | 43 | Phase D 纯音频单模态 | 0.7595 | 0.7468 | best@ep4 即峰；test acc 0.698 / F1 0.759 |
| exp_65 | afgnn_audio_only.yaml | 44 | Phase D 纯音频单模态 | 0.7680 | 0.7427 | best@ep1 即峰；test acc 0.670 / F1 0.745 |
| exp_66 | afgnn_audio_only.yaml | 45 | Phase D 纯音频单模态 | 0.7856 | 0.7181 | test acc 0.670 / F1 0.735 |
| exp_67 | afgnn_audio_only.yaml | 46 | Phase D 纯音频单模态 | 0.7984 | 0.7517 | test acc 0.679 / F1 0.754 |
| exp_68 | ablation_region_mask_00_face_contour.yaml | 43 | Phase D 区域 mask 多 seed | 0.8238 | 0.7941 | |
| exp_69 | ablation_region_mask_00_face_contour.yaml | 44 | Phase D 区域 mask 多 seed | 0.8226 | 0.7874 | |
| exp_70 | ablation_region_mask_00_face_contour.yaml | 45 | Phase D 区域 mask 多 seed | 0.8148 | 0.7906 | |
| exp_71 | ablation_region_mask_00_face_contour.yaml | 46 | Phase D 区域 mask 多 seed | 0.8265 | 0.7668 | |
| exp_72 | ablation_region_mask_01_left_eyebrow.yaml | 43 | Phase D 区域 mask 多 seed | 0.8094 | 0.8048 | |
| exp_73 | ablation_region_mask_01_left_eyebrow.yaml | 44 | Phase D 区域 mask 多 seed | 0.8526 | 0.8066 | |
| exp_74 | ablation_region_mask_01_left_eyebrow.yaml | 45 | Phase D 区域 mask 多 seed | 0.8000 | 0.7819 | |
| exp_75 | ablation_region_mask_01_left_eyebrow.yaml | 46 | Phase D 区域 mask 多 seed | 0.8261 | 0.7727 | |
| exp_76 | ablation_region_mask_02_right_eyebrow.yaml | 43 | Phase D 区域 mask 多 seed | 0.7984 | 0.7938 | |
| exp_77 | ablation_region_mask_02_right_eyebrow.yaml | 44 | Phase D 区域 mask 多 seed | 0.7973 | 0.7880 | |
| exp_78 | ablation_region_mask_02_right_eyebrow.yaml | 45 | Phase D 区域 mask 多 seed | 0.8211 | 0.8092 | |
| exp_79 | ablation_region_mask_02_right_eyebrow.yaml | 46 | Phase D 区域 mask 多 seed | 0.8405 | 0.7728 | |
| exp_80 | ablation_region_mask_03_nose_bridge.yaml | 43 | Phase D 区域 mask 多 seed | 0.8148 | 0.8073 | |
| exp_81 | ablation_region_mask_03_nose_bridge.yaml | 44 | Phase D 区域 mask 多 seed | 0.8047 | 0.8099 | |
| exp_82 | ablation_region_mask_03_nose_bridge.yaml | 45 | Phase D 区域 mask 多 seed | 0.8183 | 0.8019 | |
| exp_83 | ablation_region_mask_03_nose_bridge.yaml | 46 | Phase D 区域 mask 多 seed | 0.8288 | 0.7538 | |
| exp_84 | ablation_region_mask_04_nose_bottom.yaml | 43 | Phase D 区域 mask 多 seed | 0.8222 | 0.7940 | |
| exp_85 | ablation_region_mask_04_nose_bottom.yaml | 44 | Phase D 区域 mask 多 seed | 0.8261 | 0.8059 | |
| exp_86 | ablation_region_mask_04_nose_bottom.yaml | 45 | Phase D 区域 mask 多 seed | 0.8300 | 0.8044 | |
| exp_87 | ablation_region_mask_04_nose_bottom.yaml | 46 | Phase D 区域 mask 多 seed | 0.8234 | 0.7685 | |
| exp_88 | ablation_region_mask_05_left_eye.yaml | 43 | Phase D 区域 mask 多 seed | 0.8113 | 0.8051 | |
| exp_89 | ablation_region_mask_05_left_eye.yaml | 44 | Phase D 区域 mask 多 seed | 0.8148 | 0.8000 | |
| exp_90 | ablation_region_mask_05_left_eye.yaml | 45 | Phase D 区域 mask 多 seed | 0.8214 | 0.8014 | |
| exp_91 | ablation_region_mask_05_left_eye.yaml | 46 | Phase D 区域 mask 多 seed | 0.8125 | 0.7952 | |
| exp_92 | ablation_region_mask_06_right_eye.yaml | 43 | Phase D 区域 mask 多 seed | 0.8183 | 0.7999 | |
| exp_93 | ablation_region_mask_06_right_eye.yaml | 44 | Phase D 区域 mask 多 seed | 0.8242 | 0.7925 | |
| exp_94 | ablation_region_mask_06_right_eye.yaml | 45 | Phase D 区域 mask 多 seed | 0.8148 | 0.8057 | |
| exp_95 | ablation_region_mask_06_right_eye.yaml | 46 | Phase D 区域 mask 多 seed | 0.7984 | 0.8040 | |
| exp_96 | ablation_region_mask_07_outer_mouth.yaml | 43 | Phase D 区域 mask 多 seed | 0.8047 | 0.8023 | |
| exp_97 | ablation_region_mask_07_outer_mouth.yaml | 44 | Phase D 区域 mask 多 seed | 0.8043 | 0.8033 | |
| exp_98 | ablation_region_mask_07_outer_mouth.yaml | 45 | Phase D 区域 mask 多 seed | 0.8433 | 0.8024 | |
| exp_99 | ablation_region_mask_07_outer_mouth.yaml | 46 | Phase D 区域 mask 多 seed | 0.8109 | 0.7906 | |
| exp_100 | ablation_region_mask_08_inner_mouth.yaml | 43 | Phase D 区域 mask 多 seed | 0.8269 | 0.8036 | |
| exp_101 | ablation_region_mask_08_inner_mouth.yaml | 44 | Phase D 区域 mask 多 seed | 0.8168 | 0.7971 | |
| exp_102 | ablation_region_mask_08_inner_mouth.yaml | 45 | Phase D 区域 mask 多 seed | 0.8246 | 0.8067 | |
| exp_103 | ablation_region_mask_08_inner_mouth.yaml | 46 | Phase D 区域 mask 多 seed | 0.8534 | 0.7732 | |

## Phase A 小结（seed 42，主配置 10 个）

test AUC 排名：exp_7 adamw_cosine **0.8031** > exp_6 focal_a05_g15 0.7979 >
exp_1 focal 0.7857 > exp_3 concat 0.7827 > exp_5 focal_a01_g3 0.7822 >
exp_8 audio_enhanced 0.7755 > exp_2 enhanced 0.7697 > exp_4 aug 0.7354 >
exp_10 base 0.7302 > exp_9 face_only 0.6401。

结论：focal 系列整体占优（前三占三席），AdamW+cosine 最优；增广有害；
纯面部单模态远逊多模态；基础架构（16 帧/4ch）泛化最差之一。

## 多 seed 汇总（seeds 42~46）

| 配置 | test AUC (mean ± std) | valid AUC 范围 | 备注 |
|------|----------------------|----------------|------|
| afgnn_face_enhanced_focal_adamw_cosine.yaml | **0.7881 ± 0.0089** | [0.8136, 0.8320] | exp_7/11~14；五个 seed 全部 ≥ 0.780，最稳最优。**5 种子集成：test AUC 0.7990，acc 0.726 / F1 0.788 / recall 0.878**（`experiments/results/seed_ensemble_face_enhanced_focal_adamw_cosine_seed42-46_test.json`） |
| afgnn_face_enhanced_focal_a05_g15.yaml | 0.7824 ± 0.0185 | [0.7926, 0.8238] | exp_6/15~18；方差大（0.759~0.799 波动） |

## 消融多 seed 汇总（seeds 42~46，test AUC）

| 消融配置 | test AUC (mean ± std) | Δ vs full | 结论 |
|----------|----------------------|-----------|------|
| ablation_full（基准，exp_19/35~38） | 0.7861 ± 0.0136 | — | |
| ablation_no_temporal（exp_21/43~46） | 0.7908 ± 0.0099 | +0.0047 | 噪声内 |
| ablation_random_graph（exp_25/55~58） | 0.7896 ± 0.0118 | +0.0035 | 噪声内 |
| ablation_no_static（exp_20/39~42） | 0.7869 ± 0.0192 | +0.0008 | 噪声内 |
| ablation_no_region（exp_22/47~50） | 0.7844 ± 0.0052 | -0.0017 | 噪声内 |
| ablation_no_selfloop（exp_24/51~54） | 0.7790 ± 0.0167 | -0.0071 | 噪声内（最大跌幅，但仍 < 1 std） |
| ablation_no_velocity（exp_23/59~62） | 0.7816 ± 0.0184 | -0.0045 | 噪声内（单 seed 42 曾 -0.023，多 seed 不成立） |

**结论**：在 961 样本规模下，面部图的全部结构组件（静态边/时间边/区域编码/自环/
解剖学图结构）**以及速度特征**的贡献均小于种子间方差，**旧项目"去掉即显著掉点"
的消融结论在修复后特征上不成立**。no_velocity 的 seed-42 单点跌幅（-0.023）经
Phase D 多 seed 验证为种子噪声（5 seeds 均值差仅 -0.0045）。

## 单模态汇总（seeds 42~46，test AUC）

| 配置 | test AUC (mean ± std) | 备注 |
|------|----------------------|------|
| afgnn_audio_only.yaml（exp_63~67） | 0.7419 ± 0.0138 | 纯音频；普遍早峰（3/5 在 ep≤4 见顶） |
| afgnn_face_only_enhanced.yaml（exp_9，仅 seed 42） | 0.6401（单 seed） | 纯面部 |
| 多模态最优（afgnn_face_enhanced_focal_adamw_cosine） | **0.7881 ± 0.0089** | 融合较纯音频 +0.046、较纯面部 +0.148 |

## 区域 mask 消融多 seed 汇总（seeds 42~46，test AUC，基准 full = 0.7861 ± 0.0136）

| 区域 | test AUC (mean ± std) | Δ vs full | 结论 |
|------|----------------------|-----------|------|
| 00 面部轮廓（exp_26/68~71） | 0.7844 ± 0.0107 | -0.0017 | 噪声内 |
| 01 左眉（exp_27/72~75） | 0.7890 ± 0.0156 | +0.0029 | 噪声内 |
| 02 右眉（exp_28/76~79） | 0.7901 ± 0.0132 | +0.0039 | 噪声内 |
| 03 鼻梁（exp_29/80~83） | 0.7939 ± 0.0230 | +0.0078 | 噪声内 |
| 04 鼻底（exp_30/84~87） | 0.7922 ± 0.0151 | +0.0061 | 噪声内 |
| 05 左眼（exp_31/88~91） | 0.7987 ± 0.0053 | +0.0125 | 噪声内 |
| 06 右眼（exp_32/92~95） | 0.7979 ± 0.0078 | +0.0118 | 噪声内 |
| 07 外唇（exp_33/96~99） | 0.8007 ± 0.0057 | +0.0146 | 噪声内（最大正差，但 < 1.1 std） |
| 08 内唇（exp_34/100~103） | 0.7935 ± 0.0136 | +0.0074 | 噪声内 |

**结论**：9 个面部区域逐一 mask 后 5 seeds 均值差全部落在 ±0.015 以内，
无任一区域表现出超过种子噪声的不可替代性（单 seed 下 exp_33 外唇 +0.022
的现象在多 seed 下不成立）。
