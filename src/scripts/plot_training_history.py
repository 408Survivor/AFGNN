"""绘制训练曲线：读取 exp_N 目录下的 training_history_seed*.json，输出 PNG。

用法：
    python src/scripts/plot_training_history.py --exp_dir experiments/exp_1
    python src/scripts/plot_training_history.py experiments/exp_1/training_history_seed42.json

输出：<exp_dir>/training_curves.png（或 JSON 同目录）
"""

import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_histories(paths):
    """返回 [(label, history_list), ...]"""
    out = []
    for p in paths:
        with open(p) as f:
            data = json.load(f)
        history = data["history"] if isinstance(data, dict) and "history" in data else data
        label = os.path.basename(p).replace("training_history_", "").replace(".json", "")
        out.append((label, history))
    return out


def plot(histories, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    for label, history in histories:
        epochs = [e["epoch"] for e in history]
        axes[0].plot(epochs, [e["train_loss"] for e in history], label=f"{label} train")
        axes[0].plot(epochs, [e["loss"] for e in history], "--", label=f"{label} valid")
        axes[1].plot(epochs, [e["auc"] for e in history], label=label)
        axes[2].plot(epochs, [e["accuracy"] for e in history], label=f"{label} acc")
        axes[2].plot(epochs, [e["f1"] for e in history], "--", label=f"{label} f1")

        best = max(history, key=lambda e: e["auc"])
        axes[1].scatter([best["epoch"]], [best["auc"]], marker="*", s=120, zorder=5)
        axes[1].annotate(f"best {best['auc']:.4f} @ep{best['epoch']}",
                         (best["epoch"], best["auc"]), textcoords="offset points",
                         xytext=(6, -14), fontsize=8)

    axes[0].set_title("Loss")
    axes[1].set_title("Valid AUC")
    axes[2].set_title("Valid Accuracy / F1")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"saved: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*", help="training_history JSON 路径（可多个）")
    ap.add_argument("--exp_dir", help="exp_N 目录，自动找其中的 training_history_seed*.json")
    ap.add_argument("--out", help="输出 PNG 路径（默认写到 exp_dir 或首个 JSON 同目录）")
    args = ap.parse_args()

    paths = list(args.paths)
    if args.exp_dir:
        paths.extend(sorted(glob.glob(os.path.join(args.exp_dir, "training_history_seed*.json"))))
    if not paths:
        ap.error("请提供 --exp_dir 或至少一个 JSON 路径")

    if args.out:
        out_path = args.out
    elif args.exp_dir:
        out_path = os.path.join(args.exp_dir, "training_curves.png")
    else:
        out_path = os.path.join(os.path.dirname(paths[0]), "training_curves.png")

    plot(load_histories(paths), out_path)


if __name__ == "__main__":
    main()
