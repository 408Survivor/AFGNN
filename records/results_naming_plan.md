# results/ 文件命名统一方案(待执行)

> 记录日期:2026-07-01
> 记录人:Kimi Code CLI
> 背景:`experiments/results/` 下命名混用 `history_ / test_ / training_history_ / seed_ /
> ensemble_` 多套前缀,且有疑似默认产物。此为当时「先出方案不改」的方案存档,
> **尚未对现有文件执行重命名**。
>
> 说明:P0-3(per-run 输出目录)已让**新运行**自动留痕于 `outputs/<expid>_<ts>/`,
> 命名问题主要针对 `experiments/results/` 里的**历史文件**;是否清理由你决定。

---

## 统一约定

实验 id:`<expid> = checkpoint 文件名去 afgnn_ 前缀与 _best 后缀`(见 GUIDE.md §2)。

| 类型 | 命名 |
|---|---|
| 训练历史 | `history_<expid>.json` / `history_<expid>_seed<N>.json` |
| 测试指标 | `test_<expid>.json` / `test_<expid>_seed<N>.json` |
| 集成结果 | `ensemble_<expid>_seed<a-b>.json`(显式写成员种子) |
| 种子稳定性 | `seedstab_<expid>_seed<列表>.json` |

---

## 具体处理项

### a) 两个跑偏的默认产物 → 删除或归位(已核实)
`experiments/results/training_history.json` 与 `test_test_results.json`(时间戳 06-25 23:39,
内部 config 指向 `ablation/ablation_region_mask_08_inner_mouth.yaml`)是一次**消融运行**
把默认输出误写进 `experiments/results/`;`ablation/results/` 已有正名版本。核对内容一致后可删。

### b) 训练历史两套前缀不统一
`training_history_seed42~46.json`(默认名未改)→ `history_face_enhanced_focal_seed42.json` …
(与 focal 的 seed checkpoint 对齐)。

### c) 集成/稳定性命名统一
- `seed_stability_42-43-44_test_results.json` → `seedstab_face_enhanced_focal_seed42-44.json`
- `seed_stability_42-43-44-45-46_test_results.json` → `seedstab_face_enhanced_focal_seed42-46.json`
- `seed_ensemble_42-44_test_results.json` → `ensemble_face_enhanced_focal_seed42-44.json`(42-46 同理)
- `ensemble_test_results.json` / `seed_ensemble_test_results.json`:**未写清成员种子,需确认后再定名**。

---

## 需拍板项(未定)

1. `test_face_audio_baseline.json`(06-22 20:30,f1=.774)与 `test_face_audio.json`
   (06-22 12:57,f1=.758)**不是重复**(时间与指标都不同);"baseline" 指什么需确认。
2. 有 checkpoint 却缺 `test_`/`history_` 结果的配置:`afgnn_face_enhanced_aug`、
   `focal_a01_g3`、`focal_a05_g15`、`focal_adamw_cosine`、`audio_enhanced`——是缺测还是未归档,需确认。

> 执行前务必确认没有脚本以旧文件名硬引用(如可视化脚本读取特定 json),避免改名后断链。
