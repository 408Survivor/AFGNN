# BIBM 2026 投稿版本（Overleaf 最终版）

**请把 Overleaf 上下载的最终投稿项目文件放在这个目录下。**

操作方式（Overleaf 项目页 → Menu → Download → Source，得到 zip）：

```bash
# 解压后把里面的文件（tex / bib / figures 等）直接放进本目录
unzip ~/Downloads/<overleaf项目>.zip -d paper/submitted_bibm2026/
```

要求：

- 放进去之后**不要再修改**——这是录用稿的对照基线。
- 后续的 camera-ready 修改会从这里复制一份到 `paper/camera_ready/` 再进行。

## 目录约定

| 目录 | 用途 |
|---|---|
| `paper/submitted_bibm2026/` | 本目录，最终投稿版（冻结） |
| `paper/overleaf_upload/` | 本地早期打包的 Overleaf 快照（冻结，仅存档） |
| `paper/archive/drafts/` | V1/V2/V3 等历史草稿（冻结） |
| `paper/archive/notes/` | 写作笔记、核对记录（冻结） |
| `paper/camera_ready/` | 修改稿工作区（待创建） |

以上冻结状态对应 git tag：`bibm2026-submitted`。
