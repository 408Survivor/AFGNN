# P0-3 方案:per-run 输出目录 + 自动快照(待决策)

> ✅ **已实施(2026-07-01):采纳方案 A + 推荐默认**(软链 checkpoint、history 两处都写、
> `outputs/` 可清理、接入 train/test/train_combined)。落地细节见 `records/engineering_review.md`
> 的 P0-3 条目与 `src/utils/run_context.py`。以下为当初的评审方案,保留备查。

> 目的:给你一份可评审的方案,决定**是否做、怎么做**。本文件只是设计,不含代码改动。
> 关联:P0-1(builders)、P0-2(checkpoint 内嵌 config)已完成。

---

## 1. 背景:P0-2 之后,P0-3 还剩多少价值

P0-3 最初要解决三件事,其中一件已被 P0-2 解决:

| 原始痛点 | 现状 |
|---|---|
| config 溯源(孤儿 checkpoint) | ✅ 已由 P0-2 解决(config 内嵌进 `.pt`) |
| 默认文件名互相覆盖(`training_history.json` / `test_test_results.json`) | ❌ 仍在,靠 GUIDE 人工重命名约束 |
| 每次运行的产物散乱、无独立日志/归档 | ❌ 仍在 |

所以 P0-3 现在的净价值是**「运行产物的组织 + 自动留痕 + 独立日志」**,不再是 config 溯源。价值仍在,但没那么紧迫。

---

## 2. 现状的耦合点(决定改动面)

`cfg["training"]["checkpoint_path"]` 是 checkpoint 落盘/读取的**唯一锚点**,被 5 处引用:
- 写:`train.py:188`、`train_combined.py:319`
- 读:`test.py:60`(默认)、`seed_ensemble_test.py:87`、`seed_stability_test.py:92`

此外 `ensemble_test.py:60-95` 硬编码了 5 条 `experiments/checkpoints/*.pt` 路径。

**结论**:只要 checkpoint 仍落在 `checkpoint_path`,所有读取方零改动;一旦把 checkpoint 挪进 per-run 目录,这 5+5 处全要改。这是两种方案的分水岭。

---

## 3. 两种方案

### 方案 A(推荐):附加式 run 目录,canonical 路径不变

**核心**:保持现有落盘行为 100% 不变(checkpoint 仍写 `checkpoint_path`),**额外**为每次运行建一个
`outputs/<expid>_<timestamp>/`,把「这次运行的完整留痕」聚到一处:

```
outputs/
  face_enhanced_focal_20260701_161230/
    config.yaml        # cfg 快照(yaml.safe_dump)
    run.json           # 清单: expid, 时间, 命令行, checkpoint 路径, best_score, 最终指标
    history.json       # 训练曲线(= 原 training_history 内容)
    run.log            # 该次运行的完整 stdout/stderr
    best.pt            # 指向/复制 canonical checkpoint(见决策点 D2)
```

- **读取方(test/seed/ensemble)完全不用改**——canonical checkpoint 还在老位置。
- GUIDE 现有约定继续有效;`outputs/` 是纯增量的溯源层。
- eval 运行(test.py)也建 run 目录,存 `run.json`+`run.log`+指标(它不产 checkpoint)。

**新增**:`src/utils/run_context.py`
- `resolve_expid(cfg, seed=None)`:从 `checkpoint_path` 文件名推 expid(去 `afgnn_` 前缀、`_best.pt` 后缀),有 seed 加 `_seed{N}`。
- `start_run(cfg, kind, expid=None, base="outputs")`:建目录、`yaml.safe_dump` 写 `config.yaml`、开日志 tee 到 `run.log`,返回 `RunContext`。
- `RunContext.finalize(checkpoint_path=None, history=None, metrics=None, extra=None)`:写 `history.json` / `run.json`,并把 checkpoint 复制或软链进 run 目录。

**改动清单(小、局部)**:
- `train.py`:开头 `run = start_run(cfg, "train", seed=args.seed)`;结尾 `run.finalize(checkpoint_path=save_path, history=history)`。约 3~4 行。
- `test.py`:`run = start_run(run_cfg, "eval")`;结尾 `run.finalize(metrics=metrics, checkpoint_path=args.checkpoint)`。约 3 行。
- `train_combined.py`:同 train。可选。
- seed/ensemble 脚本:可选加 `run.finalize(metrics=...)`,不改也行。
- `trainer.py` / `builders.py`:**不动**。

**风险**:极低(纯附加)。**工作量**:约半天(含 run_context + 2~3 脚本接入 + smoke test)。

**缺点**:canonical 产物仍平铺在 `experiments/{checkpoints,results}/`,「默认文件名覆盖」问题只是被 run 目录**旁路留痕**,没根除。

---

### 方案 B:run 目录成为 canonical,彻底重组

**核心**:`outputs/<expid>_<ts>/best.pt` 成为唯一权威 checkpoint;`experiments/{checkpoints,results}` 逐步弃用。

**必须改**:
- train/train_combined:写进 run 目录。
- test/seed_ensemble/seed_stability:改为用 `--run-dir`(或「按 expid 找最近一次 run」的解析器)定位 checkpoint,替换 `cfg["training"]["checkpoint_path"]` 逻辑。
- ensemble_test:5 条硬编码路径改成 run 目录/manifest 解析。
- GUIDE:命名规范整段重写(checkpoint_path 不再是锚点)。
- 兼容:老 checkpoint 仍在 `experiments/checkpoints/`,需保留「传统路径」回退分支,否则历史模型/论文复现全断。

**风险**:高(触及全部读取方 + 硬编码 + 已有约定 + 历史资产)。**工作量**:2~3 天且回归面大。**价值**:组织性最好、无文件名碰撞。

---

## 4. 建议

**先做方案 A**。理由:P0-2 已经把最关键的 config 溯源解决了,方案 A 用极低风险拿到「每次运行独立留痕 + 日志 + 指标归档」的主要收益;方案 B 的额外收益(canonical 重组)在当前阶段不值那份侵入性和回归风险。等 A 用一段时间、确实需要彻底告别 `experiments/` 平铺时,再评估 B。

---

## 5. 需要你拍板的决策点

- **D1 是否做 P0-3**:做 A / 做 B / 暂不做(P0-2 已够,靠 GUIDE 约束命名)。
- **D2 run 目录里的 checkpoint 用复制还是软链**:复制(占空间,~几 MB/个,自包含)/ 软链(省空间,但移动 canonical 会断链)。默认建议**软链**。
- **D3 history/结果是否重定向进 run 目录**:
  - (a) 只在 run 目录留,不再写 `experiments/results/`(消除碰撞,但破坏现有「results/ 汇总」习惯);
  - (b) 两处都写(零破坏,略冗余,**推荐**)。
- **D4 `outputs/` 是否加入清理策略**:是否在 GUIDE 注明「outputs/ 为运行留痕,可定期清理,不作为论文资产」。
- **D5 seed/ensemble 脚本是否也接入**(方案 A 下为可选)。

给我 D1~D5 的选择(或直接说「A + 推荐默认」),我就按定稿实现,并配 smoke test。
