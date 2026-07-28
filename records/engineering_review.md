# AFGNN 工程与可扩展性审查

> 对照成熟可扩展的深度学习项目实践(PyTorch Lightning、Hydra / OmegaConf、
> mmengine / detectron2 Registry、src-layout 打包)对本项目做的工程审查。
> **范围:仅工程结构与可扩展性,不涉及模型本身的改动。**
> 审查时间:2026-07-01。证据均指向具体文件/行号,便于后续动手。

---

## 总体判断

模型代码本身分层清晰:`models/`(afgnn / face_gnn / audio_gnn / fusion / graph_utils)、
`data/`、`utils/` 职责分明,`AFGNN` 用 `use_face` / `use_audio` flag 支持三种模态组合,
写得不错。

问题几乎全集中在**「配置 → 对象」的胶水层和实验管理**上——而这些正是此前那批
孤儿 checkpoint(见 `records/checkpoint_config_mapping.md`)、结果文件命名混乱的
**共同根因**。下文按投入产出比(ROI)分 P0 / P1 / P2 三档排列。

---

## P0:最该修的三件(直接解决可追溯 + 消除重复)

### P0-1. 「config→对象」的构造代码被复制了 5~6 份  ✅ 已完成(2026-07-01)

> **落地情况**:已新建 `src/utils/builders.py`,提供 `load_config` / `build_model(cfg, device)` /
> `build_loaders(cfg, *, data_dir, batch_size, num_workers, augment=False, worker_init_fn=None)`;
> `train.py` / `test.py` / `train_combined.py` / `seed_ensemble_test.py` / `seed_stability_test.py`
> 已全部改用 builders,`ensemble_test.py` 因结构特殊仅共享 `load_config`。
> 已验证:6 脚本编译与导入通过;`build_model` 对 focal/face_only/concat 三配置的 state_dict
> 键·形状·参数量与旧内联构造完全一致;真实数据上 `build_loaders` 的 dataset 配置·样本数·
> 图形状与旧调用完全一致。行为不变。

**现状**
- `AFGNN(...)` 构造(20+ 个 `cfg["model"].get(...)`)在以下 6 个文件里**逐字重复**:
  `train.py`(L182-201)、`test.py`(L101-120)、`train_combined.py`、`ensemble_test.py`、
  `seed_ensemble_test.py`、`seed_stability_test.py`。
- `get_dvlog_face_loaders(...)`(30+ kwargs)重复出现在 6 个文件。
- `load_config(config_path)` 在 5 个文件里各定义了一遍。

**后果**:模型/数据每新增一个超参,要同步修改 5~6 处;漏改一处就静默回退默认值,
跑出一个「看起来对、其实参数错」的实验。

**成熟做法**:集中式工厂函数(Lightning 的 `LightningModule` / `DataModule`、
mmengine 的 `build_from_cfg`)。

**建议**:新增 `src/utils/builders.py`,提供
`load_config(path)`、`build_model(cfg)`、`build_loaders(cfg, splits)`;
所有脚本收敛成一行 `model = build_model(cfg)` / `loaders = build_loaders(cfg, [...])`。
**这是当前最高 ROI 的重构**,能删掉数百行重复,并让新增超参只改一处。

### P0-2. checkpoint 不存超参 → 加载必须手配 yaml(孤儿 config 的病根)  ✅ 已完成(2026-07-01)

> **落地情况**:`utils.trainer.train_model` 新增 `config` 参数并把整份 cfg 写进 checkpoint
> (`checkpoint["config"]`);`train.py` / `train_combined.py` 传入 `config=cfg`。
> `test.py` 改为先加载 checkpoint,优先用内嵌 config 重建模型与数据加载,
> 老 checkpoint(无 config)回退到 `--config` yaml,向后兼容。
> 已验证:26 个配置全部可 build;`weights_only=True` 能读回内嵌 dict config 并重建加载;
> 无 config 时回退分支正确。新增 `tests/` 冒烟测试(P2-8 轻量版)作为回归网。

**现状**
- `trainer.py` L237-248 保存的 dict 只有
  `epoch / model_state_dict / optimizer_state_dict / best_score / early_stopping_metric / best_threshold*`,
  **不含任何配置**。
- `test.py` L101-120 因此必须把整套架构参数重新用 yaml 喂一遍,才能 `load_state_dict`。
- 一旦 config 丢失,checkpoint 就无法复现——这正是 10 个孤儿 checkpoint 的成因。

**成熟做法**:Lightning `save_hyperparameters()` 把超参随 checkpoint 一起持久化。

**建议**:在 `trainer.py` 保存处加 `"config": cfg`(以及可选的 git commit / 时间戳),
让 `test.py` / 各评测脚本能**直接从 checkpoint 重建模型**,配置永远跟着权重走,
从机制上根除「配置丢失无法复现」。

### P0-3. 没有 per-run 输出目录,默认文件名互相覆盖  ✅ 已完成(2026-07-01,方案 A)

> **落地情况**:新增 `src/utils/run_context.py`(`resolve_expid` / `start_run` / `RunContext.finalize`);
> `train.py` / `test.py` / `train_combined.py` 每次运行自动建 `outputs/<expid>_<时间戳>/`,
> 内含 config 快照、run.log(stdout/stderr tee)、history.json、run.json 清单、best.pt(软链到
> canonical checkpoint)。采**附加式方案 A**:canonical checkpoint 仍在 `experiments/checkpoints/`,
> test/seed/ensemble 读取方式零改动;`outputs/` 为可清理的运行留痕。默认选项:软链 checkpoint、
> history 两处都写、GUIDE 注明可清理。已验证:pytest 58 passed(含 run_context 3 项),
> 并对真实 config+checkpoint 端到端确认 run 目录五件产物与软链正常。

**现状**
- `train.py` L263 永远先写 `training_history.json`,`test.py` L147 永远先写
  `test_test_results.json`,再靠人手动改名。此前两个跑偏的重复文件
  (`training_history.json` / `test_test_results.json`,内容属某次消融)就是这么产生的。
- 所有 checkpoint / 结果平铺进共享目录,靠文件名区分,极易碰撞。

**成熟做法**:Hydra 默认每次运行创建独立 `outputs/<name>_<timestamp>/`,
内含 config 快照 + checkpoint + metrics + 日志。

**建议**:每次运行自动建 `outputs/<expid>_<timestamp>/`,放入
**config 快照 + checkpoint + metrics.json + train.log**。一次性同时解决
命名混乱、config 溯源、结果归档三个问题。

---

## P1:结构与可扩展性

### P1-4. `src/` 不是一个包,靠 CWD 隐式上 `sys.path`

**现状**:无 `src/__init__.py`;`data/`、`models/`、`utils/` 三个 `__init__.py` 全为空(0 字节);
`from data.xxx import ...` 只有在从仓库根 `python src/train.py` 时才成立,
notebook 或其他位置 import 会断。

**成熟做法**:src-layout + `pyproject.toml` + `pip install -e .`,统一绝对导入。

**建议**:加最小 `pyproject.toml`,把 `src` 变成可安装包(如 `afgnn`),
导入改为 `from afgnn.models import AFGNN`。这样测试、可视化、外部脚本都能稳定 import。

### P1-5. 组件选择全是 if/elif 硬编码

**现状**:`train.py` 的 `build_scheduler`(plateau/step/cosine)、`build_criterion`
(focal/weighted_bce/bce)、优化器选择,以及 `models/fusion.py` 的 `build_fusion`,
都是 if/elif 链,新增一种就得改分支。

**成熟做法**:mmengine / detectron2 的 Registry(注册表 + 名字映射)。

**建议**:当前规模属**可选优化**,不是硬伤;但若会持续增加融合/损失变体,
引入轻量 registry 能做到「加组件只加注册、不动核心」。

### P1-6. 10 个 config 约 90% 内容重复,无继承机制

**现状**:`afgnn_face_only.yaml` 与 `afgnn_face_enhanced.yaml` 仅差 35 行却整份复制;
每加一个全局参数要改 10 份配置。

**成熟做法**:Hydra `defaults` / mmengine `_base_` 的配置继承。

**建议**:抽 `configs/_base_.yaml`,子配置只写 override;过渡期用 YAML anchor 也能缓解。

---

## P2:研究项目的基础设施缺口

### P2-7. 全靠 `print`,无日志 / 实验跟踪

**现状**:`trainer.py` / `train.py` 全程 `print`,无 `logging`、无 TensorBoard / CSV / W&B。
批量跑实验时终端刷屏、无法回溯、无法横向对比曲线。

**建议**:至少接 `logging` + TensorBoard(`SummaryWriter`)或轻量 CSVLogger 落
`metrics.csv`,与 P0-3 的 per-run 目录天然配合。

### P2-8. 零单元测试、零 CI

**现状**:`src/` 下没有任何 pytest(`ensemble_test.py` / `seed_*_test.py` 是评测脚本,
不是单测);重构 P0-1 / P1-4 时没有回归保护网。

**建议**:先加 3~5 个 smoke test(模型 forward 输出形状、dataset 出图形状、
config 能 build 出 model),放 `tests/` + 本地一条 `pytest` 即可,
内网服务器可不上 GitHub CI。

### P2-9. 复现性未拉满 + 文档漂移(小)  ◑ 部分完成(2026-07-01)

> **落地情况**:文档漂移与 code smell 已修——`train.py` / `test.py` 的 docstring 和
> argparse description 已去掉过时的 "face-only"(改为通用多模态说明);`trainer.py`
> 函数内的 `import os` 已提到模块顶部。README 经核查无 "face-only" 表述,无需改。
> **刻意保留未做**:cudnn 确定性(`torch.backends.cudnn.deterministic`)与 DataLoader
> `generator` 播种——它们会改变训练数值/性能,且不影响已存 checkpoint,留待需要严格
> 复现时再由用户决定开启。

- `train.py` L129-134 设了 seed,但未设 `torch.backends.cudnn.deterministic`;
  DataLoader 的 `generator` 未播种(仅 `worker_init_fn`)。  ← 未做(见上)
- ~~`train.py` / `test.py` 的 docstring 与 argparse description 仍写 "face-only"~~ ✅ 已修
- ~~`trainer.py` 在函数内部 `import os`,小 code smell~~ ✅ 已修

---

## 如果只做三件事(推荐顺序)

1. ~~**抽 `build_model` / `build_loaders` / `load_config` 到 builders 模块**——消除 6 个脚本的重复(P0-1)。~~ ✅ 已完成(2026-07-01)
2. ~~**checkpoint 内存 `config`,让 test / 加载不再手配 yaml**——根治孤儿 config(P0-2)。~~ ✅ 已完成(2026-07-01)
3. ~~**每次运行落到 `outputs/<expid>_<ts>/` 并快照 config**——治命名 / 归档 / 溯源(P0-3)。~~ ✅ 已完成(2026-07-01,方案 A)

这三项与此前「命名与溯源」问题是同一个病的不同症状:从「运行时自动留痕」这一侧根治,
比事后靠 `GUIDE.md` 约束人更可靠。三项改动范围均可控,建议先做 P0-1(纯重构、行为不变),
配 P2-8 的 smoke test 作回归网,再推进 P0-2、P0-3。
