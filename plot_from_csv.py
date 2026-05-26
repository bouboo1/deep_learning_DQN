"""从已有 metrics.csv 生成训练曲线图。

用法：
    python 强化学习/plot_from_csv.py --worker-csv path/to/worker/metrics.csv --requester-csv path/to/requester/metrics.csv
    python 强化学习/plot_from_csv.py --run-dir 强化学习/runs_vis/double_dqn_lr0.001_seed42_xxx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as _fm
import pandas as pd

_CJK_FONTS = ["Arial Unicode MS", "Heiti TC", "STHeiti", "SimHei", "PingFang SC"]
_available = {f.name for f in _fm.fontManager.ttflist}
for _fn in _CJK_FONTS:
    if _fn in _available:
        matplotlib.rcParams["font.family"] = _fn
        break
matplotlib.rcParams["axes.unicode_minus"] = False


def plot_single_csv(df: pd.DataFrame, reward_type: str, out_dir: Path, title_suffix: str = "") -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"DQN 训练曲线 - {reward_type} 目标{title_suffix}", fontsize=13, fontweight="bold")

    x = df["epoch"] if "epoch" in df.columns else range(len(df))

    ax = axes[0, 0]
    if "train_hit_rate" in df.columns:
        ax.plot(x, df["train_hit_rate"], "b-o", label="训练", linewidth=2)
    if "eval_hit_rate" in df.columns:
        ax.plot(x, df["eval_hit_rate"], "r-s", label="验证", linewidth=2)
    ax.set_title("Hit Rate（命中率）")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Hit Rate")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(1, df.get("train_hit_rate", pd.Series([0])).max() * 1.2))

    ax = axes[0, 1]
    if "train_avg_reward" in df.columns:
        ax.plot(x, df["train_avg_reward"], "b-o", label="训练", linewidth=2)
    if "eval_avg_reward" in df.columns:
        ax.plot(x, df["eval_avg_reward"], "r-s", label="验证", linewidth=2)
    ax.set_title("平均奖励（Avg Reward）")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Avg Reward")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    if "loss" in df.columns:
        ax.plot(x, df["loss"], "g-o", linewidth=2)
    ax.set_title("训练损失（Loss）")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    if "epsilon" in df.columns:
        ax.plot(x, df["epsilon"], "m-o", linewidth=2)
    ax.set_title("探索率（Epsilon）")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Epsilon")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    out_path = out_dir / f"training_curve_{reward_type}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[保存] {out_path}")


def plot_comparison(worker_df: pd.DataFrame, requester_df: pd.DataFrame, out_dir: Path, title_suffix: str = "") -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(f"Worker vs Requester 对比{title_suffix}", fontsize=13, fontweight="bold")

    xw = worker_df["epoch"] if "epoch" in worker_df.columns else range(len(worker_df))
    xr = requester_df["epoch"] if "epoch" in requester_df.columns else range(len(requester_df))

    ax = axes[0, 0]
    ax.plot(xw, worker_df["train_hit_rate"], "b-o", label="Worker (训练)", linewidth=2)
    ax.plot(xr, requester_df["train_hit_rate"], "r-s", label="Requester (训练)", linewidth=2)
    ax.set_title("训练集 Hit Rate")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Hit Rate")
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_ylim(0, 1)

    ax = axes[0, 1]
    ax.plot(xw, worker_df["eval_hit_rate"], "b-o", label="Worker (验证)", linewidth=2)
    ax.plot(xr, requester_df["eval_hit_rate"], "r-s", label="Requester (验证)", linewidth=2)
    ax.set_title("验证集 Hit Rate")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Hit Rate")
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_ylim(0, 1)

    ax = axes[1, 0]
    ax.plot(xw, worker_df["loss"], "b-o", label="Worker", linewidth=2)
    ax.plot(xr, requester_df["loss"], "r-s", label="Requester", linewidth=2)
    ax.set_title("训练损失")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(xw, worker_df["eval_avg_reward"], "b-o", label="Worker", linewidth=2)
    ax.plot(xr, requester_df["eval_avg_reward"], "r-s", label="Requester", linewidth=2)
    ax.set_title("验证集平均奖励")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Avg Reward")
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = out_dir / "comparison_worker_vs_requester.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[保存] {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=str, default="",
                        help="包含 worker/ 和 requester/ 子目录的实验目录")
    parser.add_argument("--worker-csv", type=str, default="",
                        help="worker metrics.csv 路径")
    parser.add_argument("--requester-csv", type=str, default="",
                        help="requester metrics.csv 路径")
    parser.add_argument("--output-dir", type=str, default="",
                        help="图片输出目录（默认与 csv 同目录）")
    args = parser.parse_args()

    worker_csv = None
    requester_csv = None

    if args.run_dir:
        run_dir = Path(args.run_dir)
        w = run_dir / "worker" / "metrics.csv"
        r = run_dir / "requester" / "metrics.csv"
        if w.exists():
            worker_csv = w
        if r.exists():
            requester_csv = r
        out_dir = run_dir
    else:
        if args.worker_csv:
            worker_csv = Path(args.worker_csv)
        if args.requester_csv:
            requester_csv = Path(args.requester_csv)
        out_dir = Path(args.output_dir) if args.output_dir else (worker_csv or requester_csv).parent

    out_dir.mkdir(parents=True, exist_ok=True)

    worker_df = pd.read_csv(worker_csv) if worker_csv and worker_csv.exists() else None
    requester_df = pd.read_csv(requester_csv) if requester_csv and requester_csv.exists() else None

    if worker_df is not None:
        plot_single_csv(worker_df, "worker", out_dir)
    if requester_df is not None:
        plot_single_csv(requester_df, "requester", out_dir)
    if worker_df is not None and requester_df is not None:
        plot_comparison(worker_df, requester_df, out_dir)

    print("完成！")


if __name__ == "__main__":
    main()
