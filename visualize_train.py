"""实时可视化训练脚本：同时训练 worker 和 requester 两个目标，并生成对比图。

用法（在项目根目录 /Users/bouboo/Documents/强化学习 下运行）：
    python 强化学习/visualize_train.py --epochs 5 --variant double_dqn
    python 强化学习/visualize_train.py --epochs 5 --variant dueling_dqn --lr 0.0001
    python 强化学习/visualize_train.py --epochs 3 --train-limit 5000 --val-limit 1000
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as _fm
    # 优先使用系统中文字体，避免方块乱码
    _CJK_FONTS = ["Arial Unicode MS", "Heiti TC", "STHeiti", "SimHei", "PingFang SC"]
    _available = {f.name for f in _fm.fontManager.ttflist}
    for _fn in _CJK_FONTS:
        if _fn in _available:
            matplotlib.rcParams["font.family"] = _fn
            break
    matplotlib.rcParams["axes.unicode_minus"] = False
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[警告] matplotlib 未安装，跳过可视化。pip install matplotlib")

try:
    import torch
except ModuleNotFoundError as exc:
    raise SystemExit("PyTorch 未安装。请先运行: pip install torch numpy") from exc

from env import CrowdRecEnv, load_split
from DQN.agent import DQNAgent, make_agent_config
from DQN.metrics import MetricLogger


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def maybe_limit(data: list, limit: int) -> list:
    return data[:limit] if limit > 0 else data


def evaluate_policy(agent, data, reward_type, max_steps=None):
    env = CrowdRecEnv(data, reward_type=reward_type, max_candidates=agent.config.state_shape[0])
    state = env.reset(shuffle=False)
    done = False
    total_reward = 0.0
    hits = 0
    invalid = 0
    steps = 0
    while not done:
        action = agent.act(state, evaluate=True)
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        hits += int(info["hit"])
        invalid += int(not info["valid_action"])
        steps += 1
        if max_steps and steps >= max_steps:
            break
        if next_state is not None:
            state = next_state
    n = steps or 1
    return {
        "eval_steps": float(steps),
        "eval_avg_reward": total_reward / n,
        "eval_hit_rate": hits / n,
        "eval_invalid_action_rate": invalid / n,
    }


def train_one_objective(reward_type, train_data, val_data, args, run_dir):
    """训练单个目标，返回每 epoch 的指标列表和最终 agent。"""
    max_candidates = max(
        max(len(s["candidate_projects"]) for s in train_data),
        max(len(s["candidate_projects"]) for s in val_data),
    )
    train_env = CrowdRecEnv(
        train_data, reward_type=reward_type,
        max_candidates=max_candidates, seed=args.seed,
    )
    first_state = train_env.reset(shuffle=False)
    state_shape = tuple(first_state["features"].shape)
    action_dim = int(train_env.spec.action_dim)

    agent_config = make_agent_config(
        state_shape=state_shape,
        action_dim=action_dim,
        variant=args.variant,
        lr=args.lr,
        hidden_dims=tuple(args.hidden_dims),
        gamma=args.gamma,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        min_replay_size=args.min_replay_size,
        target_update_interval=args.target_update_interval,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_steps=args.epsilon_decay_steps,
        aux_ce_weight=args.aux_ce_weight,
        seed=args.seed,
    )
    agent = DQNAgent(agent_config)
    logger = MetricLogger(run_dir)

    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {**vars(args), "reward_type": reward_type},
            f, ensure_ascii=False, indent=2, default=str,
        )

    history = []
    best_val_reward = -float("inf")
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        state = train_env.reset(shuffle=True)
        done = False
        total_reward = 0.0
        hits = 0
        invalid = 0
        losses = []
        steps = 0

        while not done:
            action = agent.act(state, evaluate=False)
            next_state, reward, done, info = train_env.step(action)
            agent.remember(
                state, action, reward, next_state, done,
                positive_action=int(info["positive_index"]),
            )
            loss = agent.learn()
            if loss is not None:
                losses.append(loss)
            total_reward += reward
            hits += int(info["hit"])
            invalid += int(not info["valid_action"])
            steps += 1
            global_step += 1
            if args.max_steps_per_epoch and steps >= args.max_steps_per_epoch:
                break
            if next_state is not None:
                state = next_state

        eval_stats = evaluate_policy(
            agent, val_data, reward_type=reward_type,
            max_steps=args.max_eval_steps,
        )
        n = steps or 1
        row = {
            "epoch": epoch,
            "global_step": global_step,
            "epsilon": agent.epsilon,
            "train_avg_reward": total_reward / n,
            "train_hit_rate": hits / n,
            "train_invalid_action_rate": invalid / n,
            "loss": float(np.mean(losses)) if losses else 0.0,
            **eval_stats,
        }
        logger.record(row)
        logger.write_csv()
        history.append(row)

        if row["eval_avg_reward"] > best_val_reward:
            best_val_reward = row["eval_avg_reward"]
            agent.save(run_dir / "best_model.pt", extra={"epoch": epoch, "metrics": row})

        print(
            f"  [{reward_type:10s}] epoch={epoch}/{args.epochs}"
            f"  train_hit={row['train_hit_rate']:.4f}"
            f"  val_hit={row['eval_hit_rate']:.4f}"
            f"  loss={row['loss']:.4f}"
            f"  eps={row['epsilon']:.3f}"
        )

    agent.save(run_dir / "last_model.pt", extra={"epoch": args.epochs})
    return history, agent


def evaluate_on_test(agent, test_data, reward_type, run_dir):
    """在测试集上评估，保存结果。"""
    stats = evaluate_policy(agent, test_data, reward_type=reward_type)
    result = {"reward_type": reward_type, **stats}
    with (run_dir / "test_result.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(
        f"  [{reward_type:10s}] TEST"
        f"  hit_rate={stats['eval_hit_rate']:.4f}"
        f"  avg_reward={stats['eval_avg_reward']:.4f}"
    )
    return stats


def plot_comparison(worker_history, requester_history, out_dir, variant, lr):
    """生成 worker vs requester 对比图，保存到 out_dir。"""
    if not HAS_MPL:
        return

    epochs_w = [r["epoch"] for r in worker_history]
    epochs_r = [r["epoch"] for r in requester_history]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        f"众包任务推荐 DQN 训练对比\n模型: {variant}  学习率: {lr}",
        fontsize=13, fontweight="bold",
    )

    # 1. 训练 hit rate
    ax = axes[0, 0]
    ax.plot(epochs_w, [r["train_hit_rate"] for r in worker_history],
            "b-o", label="Worker (训练)", linewidth=2)
    ax.plot(epochs_r, [r["train_hit_rate"] for r in requester_history],
            "r-s", label="Requester (训练)", linewidth=2)
    ax.set_title("训练集 Hit Rate（命中率）")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Hit Rate")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    # 2. 验证 hit rate
    ax = axes[0, 1]
    ax.plot(epochs_w, [r["eval_hit_rate"] for r in worker_history],
            "b-o", label="Worker (验证)", linewidth=2)
    ax.plot(epochs_r, [r["eval_hit_rate"] for r in requester_history],
            "r-s", label="Requester (验证)", linewidth=2)
    ax.set_title("验证集 Hit Rate（命中率）")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Hit Rate")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    # 3. 训练损失
    ax = axes[1, 0]
    ax.plot(epochs_w, [r["loss"] for r in worker_history],
            "b-o", label="Worker", linewidth=2)
    ax.plot(epochs_r, [r["loss"] for r in requester_history],
            "r-s", label="Requester", linewidth=2)
    ax.set_title("训练损失 (Loss)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. 验证平均奖励
    ax = axes[1, 1]
    ax.plot(epochs_w, [r["eval_avg_reward"] for r in worker_history],
            "b-o", label="Worker", linewidth=2)
    ax.plot(epochs_r, [r["eval_avg_reward"] for r in requester_history],
            "r-s", label="Requester", linewidth=2)
    ax.set_title("验证集平均奖励 (Avg Reward)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Avg Reward")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = out_dir / "comparison_worker_vs_requester.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[可视化] 对比图已保存: {out_path}")


def plot_single(history, reward_type, out_dir, variant, lr):
    """为单个目标生成详细训练曲线图。"""
    if not HAS_MPL:
        return

    epochs = [r["epoch"] for r in history]
    color = "steelblue" if reward_type == "worker" else "tomato"
    label_cn = "参与者（Worker）" if reward_type == "worker" else "请求者（Requester）"

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(
        f"{label_cn} 目标训练曲线\n模型: {variant}  学习率: {lr}",
        fontsize=12, fontweight="bold",
    )

    ax = axes[0]
    ax.plot(epochs, [r["train_hit_rate"] for r in history],
            color=color, linestyle="-", marker="o", label="训练", linewidth=2)
    ax.plot(epochs, [r["eval_hit_rate"] for r in history],
            color=color, linestyle="--", marker="s", label="验证", linewidth=2)
    ax.set_title("Hit Rate（命中率）")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Hit Rate")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    ax = axes[1]
    ax.plot(epochs, [r["loss"] for r in history],
            color=color, marker="o", linewidth=2)
    ax.set_title("训练损失 (Loss)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(epochs, [r["epsilon"] for r in history],
            color="gray", marker="o", linewidth=2)
    ax.set_title("探索率 (Epsilon)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Epsilon")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    out_path = out_dir / f"training_curve_{reward_type}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[可视化] {reward_type} 训练曲线已保存: {out_path}")


def plot_test_bar(worker_test, requester_test, out_dir, variant, lr):
    """生成测试集结果柱状图。"""
    if not HAS_MPL:
        return

    metrics = ["eval_hit_rate", "eval_avg_reward"]
    labels_cn = ["Hit Rate（命中率）", "Avg Reward（平均奖励）"]
    w_vals = [worker_test[m] for m in metrics]
    r_vals = [requester_test[m] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width / 2, w_vals, width, label="Worker（参与者）", color="steelblue", alpha=0.85)
    bars2 = ax.bar(x + width / 2, r_vals, width, label="Requester（请求者）", color="tomato", alpha=0.85)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=9)

    ax.set_title(f"测试集结果对比\n模型: {variant}  学习率: {lr}", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels_cn)
    ax.set_ylabel("指标值")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, max(max(w_vals), max(r_vals)) * 1.2)

    plt.tight_layout()
    out_path = out_dir / "test_results_bar.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[可视化] 测试集柱状图已保存: {out_path}")


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", choices=["dqn", "double_dqn", "dueling_dqn"], default="double_dqn",
                        help="DQN 变体")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    parser.add_argument("--epochs", type=int, default=5, help="训练 epoch 数")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--buffer-size", type=int, default=50_000)
    parser.add_argument("--min-replay-size", type=int, default=1_000)
    parser.add_argument("--target-update-interval", type=int, default=500)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay-steps", type=int, default=20_000)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[256, 128])
    parser.add_argument("--aux-ce-weight", type=float, default=1.0,
                        help="辅助交叉熵损失权重，0 表示纯 DQN")
    parser.add_argument("--train-limit", type=int, default=0, help="限制训练样本数（0=全量）")
    parser.add_argument("--val-limit", type=int, default=0, help="限制验证样本数（0=全量）")
    parser.add_argument("--max-steps-per-epoch", type=int, default=0)
    parser.add_argument("--max-eval-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="强化学习/runs_vis", help="输出目录")
    parser.add_argument("--skip-requester", action="store_true", help="只训练 worker 目标")
    parser.add_argument("--skip-worker", action="store_true", help="只训练 requester 目标")
    return parser


def main():
    args = build_arg_parser().parse_args()
    set_seed(args.seed)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.variant}_lr{args.lr:g}_seed{args.seed}_{ts}"
    out_dir = Path(args.output_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"实验目录: {out_dir}")
    print(f"模型变体: {args.variant}  学习率: {args.lr}  Epochs: {args.epochs}")
    print(f"{'='*60}\n")

    print("[数据] 加载数据集...")
    train_data = maybe_limit(load_split("train"), args.train_limit)
    val_data = maybe_limit(load_split("val"), args.val_limit)
    test_data = load_split("test")
    print(f"  训练: {len(train_data)} 条  验证: {len(val_data)} 条  测试: {len(test_data)} 条")

    worker_history = []
    requester_history = []
    worker_test = {}
    requester_test = {}

    # ---- 训练 Worker 目标 ----
    if not args.skip_worker:
        print(f"\n{'='*60}")
        print("目标 1：最大化参与者（Worker）利益")
        print(f"{'='*60}")
        worker_dir = out_dir / "worker"
        worker_dir.mkdir(exist_ok=True)
        worker_history, worker_agent = train_one_objective(
            "worker", train_data, val_data, args, worker_dir,
        )
        print("\n[测试集评估 - Worker]")
        worker_test = evaluate_on_test(worker_agent, test_data, "worker", worker_dir)
        plot_single(worker_history, "worker", out_dir, args.variant, args.lr)

    # ---- 训练 Requester 目标 ----
    if not args.skip_requester:
        print(f"\n{'='*60}")
        print("目标 2：最大化请求者（Requester）利益")
        print(f"{'='*60}")
        requester_dir = out_dir / "requester"
        requester_dir.mkdir(exist_ok=True)
        requester_history, requester_agent = train_one_objective(
            "requester", train_data, val_data, args, requester_dir,
        )
        print("\n[测试集评估 - Requester]")
        requester_test = evaluate_on_test(requester_agent, test_data, "requester", requester_dir)
        plot_single(requester_history, "requester", out_dir, args.variant, args.lr)

    # ---- 对比图 ----
    if worker_history and requester_history:
        plot_comparison(worker_history, requester_history, out_dir, args.variant, args.lr)
        plot_test_bar(worker_test, requester_test, out_dir, args.variant, args.lr)

    # ---- 汇总 ----
    summary = {
        "run_name": run_name,
        "variant": args.variant,
        "lr": args.lr,
        "epochs": args.epochs,
        "worker_test": worker_test,
        "requester_test": requester_test,
        "worker_val_best_hit": max((r["eval_hit_rate"] for r in worker_history), default=0),
        "requester_val_best_hit": max((r["eval_hit_rate"] for r in requester_history), default=0),
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print("实验完成！结果汇总：")
    if worker_test:
        print(f"  Worker   测试 Hit Rate: {worker_test.get('eval_hit_rate', 0):.4f}")
    if requester_test:
        print(f"  Requester 测试 Hit Rate: {requester_test.get('eval_hit_rate', 0):.4f}")
    print(f"  输出目录: {out_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
