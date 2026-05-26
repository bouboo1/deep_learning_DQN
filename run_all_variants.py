"""运行 DQN / Double DQN / Dueling DQN 三种变体的对比实验，生成汇总图。

用法：
    python 强化学习/run_all_variants.py --epochs 5
    python 强化学习/run_all_variants.py --epochs 3 --train-limit 10000 --val-limit 3000
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as _fm

_CJK_FONTS = ["Arial Unicode MS", "Heiti TC", "STHeiti", "SimHei", "PingFang SC"]
_available = {f.name for f in _fm.fontManager.ttflist}
for _fn in _CJK_FONTS:
    if _fn in _available:
        matplotlib.rcParams["font.family"] = _fn
        break
matplotlib.rcParams["axes.unicode_minus"] = False


VARIANTS = ["dqn", "double_dqn", "dueling_dqn"]
VARIANT_LABELS = {
    "dqn": "DQN",
    "double_dqn": "Double DQN",
    "dueling_dqn": "Dueling DQN",
}
COLORS = {
    "dqn": "#1f77b4",
    "double_dqn": "#ff7f0e",
    "dueling_dqn": "#2ca02c",
}


def run_variant(variant: str, args: argparse.Namespace, out_dir: Path) -> dict:
    """调用 visualize_train.py 训练单个变体，返回 summary.json 内容。"""
    variant_dir = out_dir / variant
    variant_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(Path(__file__).parent / "visualize_train.py"),
        "--variant", variant,
        "--epochs", str(args.epochs),
        "--lr", str(args.lr),
        "--output-dir", str(variant_dir),
        "--seed", str(args.seed),
    ]
    if args.train_limit > 0:
        cmd += ["--train-limit", str(args.train_limit)]
    if args.val_limit > 0:
        cmd += ["--val-limit", str(args.val_limit)]
    if args.max_steps_per_epoch > 0:
        cmd += ["--max-steps-per-epoch", str(args.max_steps_per_epoch)]
    if args.max_eval_steps:
        cmd += ["--max-eval-steps", str(args.max_eval_steps)]

    print(f"\n{'='*60}")
    print(f"运行变体: {VARIANT_LABELS[variant]}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"[错误] {variant} 训练失败")
        return {}

    # 找到最新的 summary.json
    summaries = sorted(variant_dir.glob("*/summary.json"), key=lambda p: p.stat().st_mtime)
    if summaries:
        with summaries[-1].open(encoding="utf-8") as f:
            return json.load(f)
    return {}


def plot_variant_comparison(all_results: dict, out_dir: Path) -> None:
    """生成三种变体的对比柱状图。"""
    variants = list(all_results.keys())
    labels = [VARIANT_LABELS[v] for v in variants]
    colors = [COLORS[v] for v in variants]

    worker_hits = [all_results[v].get("worker_test", {}).get("eval_hit_rate", 0) for v in variants]
    requester_hits = [all_results[v].get("requester_test", {}).get("eval_hit_rate", 0) for v in variants]
    worker_val = [all_results[v].get("worker_val_best_hit", 0) for v in variants]
    requester_val = [all_results[v].get("requester_val_best_hit", 0) for v in variants]

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle("三种 DQN 变体对比（测试集 Hit Rate）", fontsize=14, fontweight="bold")

    x = range(len(variants))
    width = 0.35

    ax = axes[0]
    bars1 = ax.bar([i - width/2 for i in x], worker_hits, width, label="测试集", color=colors, alpha=0.85)
    bars2 = ax.bar([i + width/2 for i in x], worker_val, width, label="验证集最佳", color=colors, alpha=0.5, hatch="//")
    ax.set_title("Worker 目标（最大化参与者利益）")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Hit Rate")
    ax.set_ylim(0, max(max(worker_hits + worker_val) * 1.3, 0.1))
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.002, f"{h:.3f}", ha="center", va="bottom", fontsize=9)

    ax = axes[1]
    bars3 = ax.bar([i - width/2 for i in x], requester_hits, width, label="测试集", color=colors, alpha=0.85)
    bars4 = ax.bar([i + width/2 for i in x], requester_val, width, label="验证集最佳", color=colors, alpha=0.5, hatch="//")
    ax.set_title("Requester 目标（最大化请求者利益）")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Hit Rate")
    ax.set_ylim(0, max(max(requester_hits + requester_val) * 1.3, 0.1))
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    for bar in bars3:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.002, f"{h:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    out_path = out_dir / "variant_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[保存] 变体对比图: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--val-limit", type=int, default=0)
    parser.add_argument("--max-steps-per-epoch", type=int, default=0)
    parser.add_argument("--max-eval-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=VARIANTS,
                        help="要运行的变体列表")
    parser.add_argument("--output-dir", default="强化学习/runs_all_variants")
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / f"compare_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"多变体对比实验")
    print(f"变体: {args.variants}")
    print(f"Epochs: {args.epochs}  LR: {args.lr}")
    print(f"输出目录: {out_dir}")
    print(f"{'='*60}")

    all_results = {}
    for variant in args.variants:
        summary = run_variant(variant, args, out_dir)
        if summary:
            all_results[variant] = summary

    if len(all_results) >= 2:
        plot_variant_comparison(all_results, out_dir)

    final_summary = {
        "variants": args.variants,
        "epochs": args.epochs,
        "lr": args.lr,
        "results": all_results,
    }
    with (out_dir / "all_variants_summary.json").open("w", encoding="utf-8") as f:
        json.dump(final_summary, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print("所有变体实验完成！")
    for v, r in all_results.items():
        wh = r.get("worker_test", {}).get("eval_hit_rate", 0)
        rh = r.get("requester_test", {}).get("eval_hit_rate", 0)
        print(f"  {VARIANT_LABELS[v]:15s}  Worker Hit={wh:.4f}  Requester Hit={rh:.4f}")
    print(f"输出目录: {out_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
