# 工程重构与规范化 — 修改记录(2026-07-01)

> 记录日期:2026-07-01
> 记录人:Kimi Code CLI
> 背景:目录/文件组织与可扩展性梳理。产出规范文档、修复孤儿 config 溯源、
> 并落地工程重构 P0-1 / P0-2 / P0-3 与 P2-9(部分)。**未改动任何模型逻辑与实验结果。**
> 本文件为本次会话的文件级合并 changelog;各项「为什么/怎么验证」的详情见
> `records/engineering_review.md`。本项目无 git,故以本文件作为变更记录。

---

## 1. 新增文件

| 文件 | 用途 |
|---|---|
| `requirements.txt` | 依赖锁定(DVlog 实际版本)+ dev 用 pytest==8.4.2 |
| `GUIDE.md` | 实验文件命名/组织规范、DVlog 环境、`outputs/`、测试运行 |
| `src/utils/builders.py` | `load_config` / `build_model` / `build_loaders` 工厂(P0-1) |
| `src/utils/run_context.py` | per-run 输出目录 `outputs/<expid>_<ts>/`(P0-3) |
| `tests/conftest.py` | 把 `src/` 加入 import 路径 |
| `tests/test_builders.py` | 每个 config 都能 build_model + 结构不变量 |
| `tests/test_checkpoint_config.py` | checkpoint 内嵌 config 往返 + 向后兼容 |
| `tests/test_run_context.py` | run 目录 / 清单 / 软链 冒烟测试 |
| `records/checkpoint_config_mapping.md` | 孤儿 checkpoint 溯源表 |
| `records/engineering_review.md` | 工程审查 + P0/P2 落地状态与验证 |
| `records/P0-3_plan.md` | P0-3 方案(已实施,顶部标注) |
| `records/results_naming_plan.md` | results 命名统一方案(待执行) |
| `records/refactor_changelog_2026-07-01.md` | 本文件 |

## 2. 改动文件(仅工程,不涉模型逻辑)

| 文件 | 改动 |
|---|---|
| `src/train.py` | 改用 builders;传 `config=cfg` 存入 checkpoint;接入 `start_run/finalize`;docstring/描述去 face-only |
| `src/test.py` | 改用 builders;先加载 checkpoint→优先用内嵌 config 重建(向后兼容);接入 run 目录;docstring/描述修正 |
| `src/train_combined.py` | 改用 builders;传 `config=cfg`;接入 run 目录 |
| `src/seed_ensemble_test.py` | 改用 builders(删本地 load_config/build_model 重复) |
| `src/seed_stability_test.py` | 改用 builders(同上) |
| `src/ensemble_test.py` | 改用共享 `load_config`(模型构造逻辑因结构特殊保留) |
| `src/utils/trainer.py` | `train_model` 新增 `config` 参数并写入 checkpoint;`import os` 提到模块顶部 |
| `records/STATUS.md` | 「详细记录」新增指向本次工程文档的指针 |

## 3. 环境变更

- 在 `DVlog` conda 环境安装 `pytest==8.4.2`(经用户同意),用于跑 `tests/`。

## 4. 完成情况

- **P0-1** ✅ builders 抽取(消除 6 脚本重复);已验证 build_model/build_loaders 与旧内联逐位等价。
- **P0-2** ✅ checkpoint 内嵌 config,test.py 自描述加载 + 向后兼容。
- **P0-3** ✅ 方案 A:附加式 `outputs/<expid>_<ts>/`(config 快照 + run.log + history + run.json + best.pt 软链),canonical 路径与读取方零改动。
- **P2-9** ◑ docstring/描述漂移与 `import os` smell 已清;**cudnn 确定性 + DataLoader generator 播种未做**(会改训练数值/性能,待定)。
- **未做**:P1-4(打包)、P1-5(registry)、P1-6(config 继承)、P2-7(日志/TensorBoard)、P2-8 扩展测试;results 历史文件重命名(方案见 `results_naming_plan.md`,待执行)。

## 5. 验证

- `pytest tests/ -q` → **58 passed**。
- 6 脚本编译 + 导入通过;`build_model` 三配置 state_dict 与旧构造一致;真实数据上 `build_loaders` 与旧调用一致;真实 config+checkpoint 端到端确认 run 目录产物与软链正常。
