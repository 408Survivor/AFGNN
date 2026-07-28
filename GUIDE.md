# AFGNN 实验文件命名与组织规范 (GUIDE)

> 本文件为**以后新增实验**提供统一的文件放置与命名规范,目的是让每个
> checkpoint / 结果都能追溯到产生它的配置。请在开跑新实验前对照本规范。
> 现有历史文件不强制回改;新文件一律遵循本规范。

> ⚠️ **运行任何脚本前先 `conda activate DVlog`**(所有训练/测试/可视化都在此环境跑,详见第 5 节)。

---

## 0. 一条铁律:配置必须留快照

从近期改动起,`src/train.py` 训练出的 `.pt` 会**自动内嵌整份 config**
(`checkpoint["config"]`),`src/test.py` 会优先用它重建模型/数据、无需再手配 yaml。
但这只覆盖**新训练**的 checkpoint——历史 `.pt` 不含 config;而 `history_*.json` /
`test_*.json` 里的 `config`、`checkpoint` 字段仍是 `train.py` 的默认值、**不能用来溯源**。

因此下面的铁律依然成立(config 文件仍是启动来源和人类可读记录):

- **每个实验必须有一个独立的 `.yaml` 配置文件**,且**运行时用 `--config` 指向它**。
- **禁止**原地反复修改同一个 config(如 `afgnn_face_only.yaml`)再跑不同实验——
  这正是目前 10 个孤儿 checkpoint 配置丢失的原因(见 `records/checkpoint_config_mapping.md`)。
- 新变体 = 复制一份 config → 改名 → 改参数 → 再跑。

---

## 1. 目录归属:实验放哪里

| 目录 | 用途 |
|---|---|
| `experiments/` | 主线实验(架构、模态、融合、损失、种子等) |
| `ablation/`    | 消融实验(去掉某模块 / region mask 等) |

两者结构相同,各含三个子目录,**三者文件名必须成套对应**:

```
<experiments|ablation>/
├── configs/       # 每个实验一个 .yaml
├── checkpoints/   # 训练产出的 .pt
└── results/       # history_*.json + test_*.json 等
```

> 注意:跑消融实验时,`--output_dir` 和 config 里的 `checkpoint_path` 要指向
> `ablation/` 下,**不要**让默认输出(`experiments/results/training_history.json`)
> 落到 `experiments/`,否则会产生跑偏的重复文件。

### `outputs/`:每次运行的自动留痕(P0-3)

除上面的 canonical 位置外,`src/train.py` / `test.py` / `train_combined.py` 每次运行会
**自动**建一个独立目录 `outputs/<expid>_<时间戳>/`,把这一次运行聚到一处:

```
outputs/face_enhanced_focal_20260701_161230/
├── config.yaml    # 本次实际使用的 config 快照
├── run.log        # 完整 stdout/stderr
├── history.json   # 训练曲线(训练运行)
├── run.json       # 清单:kind/时间/命令行/checkpoint/指标
└── best.pt        # 软链到 canonical checkpoint(训练运行)
```

- 这是**纯附加**层:canonical checkpoint 仍在 `experiments/checkpoints/`,
  test/seed/ensemble 的读取方式不变。
- `outputs/` 是**可丢弃的运行留痕,不作为论文资产**,可定期清理;
  权威产物仍以 `experiments/` / `ablation/` 下的为准。
- `best.pt` 是软链——移动或删除 canonical checkpoint 会使其失效,属正常。

---

## 2. 实验 id (`<expid>`) 的定义

`<expid>` 是贯穿三类文件的唯一标识:

```
<expid> = checkpoint 文件名去掉 "afgnn_" 前缀和 "_best" 后缀
```

例:`afgnn_face_enhanced_focal_best.pt` → `<expid>` = `face_enhanced_focal`。

命名 `<expid>` 的约定:

- 全小写,用下划线分词;从「基座 → 修改点」由粗到细排列。
- 用语义清晰的短词描述变体,不要用日期/序号当版本。
  - 好:`face_audio_crossattn_audio64`、`face_enhanced_focal_a01_g3`
  - 差:`face_v2`、`exp3`、`test_new`
- 关键超参进名字时用 `键值` 拼接:`audio64`(音频64帧)、`k2`(motion k=2)、
  `a01_g3`(focal α=0.1/γ=3.0)。

---

## 3. 三类文件的命名规则

### 3.1 Config (`configs/`)

```
<expid>.yaml
```
文件名去掉前后缀后应与 checkpoint 完全对应。config 顶部用注释写清该实验相对
基线改了什么(现有 config 已是此风格,照抄)。

### 3.2 Checkpoint (`checkpoints/`)

```
afgnn_<expid>_best.pt          # 常规最佳权重
afgnn_<expid>_best_seed<N>.pt  # 带随机种子的复现实验
```

- `afgnn_` 前缀 + `_best` 后缀由训练流程约定,保持一致。
- **种子后缀由 `src/train.py --seed <N>` 自动追加**(见 L241-243),不要手改。
- 同一 config 跑多个种子时,共用一个 `.yaml`,靠 `_seed<N>` 区分 checkpoint。

### 3.3 结果 (`results/`)

| 类型 | 命名 | 说明 |
|---|---|---|
| 训练历史 | `history_<expid>.json` | 单模型每 epoch 指标 |
| 训练历史(种子) | `history_<expid>_seed<N>.json` | 对应种子 checkpoint |
| 测试指标 | `test_<expid>.json` | 单模型 test 集指标 |
| 测试指标(种子) | `test_<expid>_seed<N>.json` | |
| 集成结果 | `ensemble_<expid>_seed<a-b>.json` | 多模型平均/加权,**显式写成员种子范围** |
| 种子稳定性 | `seedstab_<expid>_seed<列表>.json` | 例 `seedstab_face_enhanced_focal_seed42-46.json` |

要点:

- **不要留默认名**:`train.py` 默认输出 `training_history.json`、`test.py` 默认
  `test_test_results.json`。跑完后**立即重命名**成 `history_<expid>.json` /
  `test_<expid>.json`,或通过 `--output_dir` / config 直接指定正确文件名。
- 聚合类文件(ensemble / seedstab)**必须在文件名里写清参与的种子**,不要用
  `ensemble_test_results.json` 这种无成员信息的名字。
- 一个 `<expid>` 理想情况下应同时有 config、checkpoint、`history_`、`test_` 四件,
  跑完自查是否配套齐全。

---

## 4. 命名对照速查(以 `face_enhanced_focal` 为例)

```
experiments/configs/afgnn_face_enhanced_focal.yaml
experiments/checkpoints/afgnn_face_enhanced_focal_best.pt
experiments/checkpoints/afgnn_face_enhanced_focal_best_seed42.pt   # --seed 42
experiments/results/history_face_enhanced_focal.json
experiments/results/test_face_enhanced_focal.json
experiments/results/history_face_enhanced_focal_seed42.json
experiments/results/test_face_enhanced_focal_seed42.json
experiments/results/ensemble_face_enhanced_focal_seed42-46.json
experiments/results/seedstab_face_enhanced_focal_seed42-46.json
```

---

## 5. 环境与复现

- **所有训练、测试、可视化脚本都必须在 `DVlog` 这个 conda 环境下运行**,
  跑任何脚本前先激活它:

  ```bash
  conda activate DVlog
  ```

  该环境为 Python 3.10,依赖版本见根目录 `requirements.txt`(锁定 `DVlog`
  环境实际版本)。不要在 base 或其他环境里跑,也不要在此环境外新装/升级包;
  新增依赖后请同步更新 `requirements.txt`。
- 新增实验的标准流程:
  1. `conda activate DVlog`;
  2. 复制并改写 `configs/<expid>.yaml`;
  3. 在 config 里设好 `checkpoint_path` 指向对应目录的 `afgnn_<expid>_best.pt`;
  4. `python src/train.py --config <上面的yaml>`(需多种子加 `--seed`);
  5. `python src/test.py --config <同一yaml> --split test`;
  6. 把默认输出重命名为 `history_<expid>.json` / `test_<expid>.json`。

---

## 6. 冒烟测试(可选)

`tests/` 下有一组不依赖数据、纯 CPU 的冒烟测试,作为重构的回归网:
- `test_builders.py`:每个 `experiments/` 与 `ablation/` config 都能被
  `build_model` 构造出结构自洽的模型;
- `test_checkpoint_config.py`:checkpoint 能内嵌 config 且 `weights_only=True`
  可读回、并据此重建加载(对应 `test.py` 的自描述加载路径)。

运行(pytest 已装入 `DVlog` 环境):

```bash
conda activate DVlog
pytest tests/ -q            # 从仓库根运行;conftest.py 会把 src/ 加入 import 路径
```

改动 `src/utils/builders.py`、模型结构或 checkpoint 字段后,建议先跑一遍。

---

## 7. 其他约定

- **paper/**、**visualization/** 目录不在本规范约束内;paper 内容按现状维护。
- 论文/笔记类文档放 `records/`;可视化脚本与产图放 `visualization/`。
- 已知历史遗留问题(孤儿 checkpoint、命名不统一)记录在
  `records/checkpoint_config_mapping.md`,新实验避免重蹈即可。
