"""Requester-side DQN training for CrowdRecEnv."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "PyTorch is required. Install it with: pip install torch"
    ) from exc

from env import CrowdRecEnv, load_split


@dataclass
class Transition:
    state: np.ndarray
    action_mask: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray | None
    next_action_mask: np.ndarray | None
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = int(capacity)
        self.data: List[Transition] = []
        self.pos = 0

    def __len__(self) -> int:
        return len(self.data)

    def add(self, transition: Transition) -> None:
        if len(self.data) < self.capacity:
            self.data.append(transition)
        else:
            self.data[self.pos] = transition
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.data, batch_size)


class QNetwork(nn.Module):
    def __init__(self, feature_dim: int, hidden_sizes: Tuple[int, int]) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = feature_dim
        for size in hidden_sizes:
            layers.append(nn.Linear(in_dim, size))
            layers.append(nn.ReLU())
            in_dim = size
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch, max_candidates, feature_dim = features.shape
        x = features.reshape(batch * max_candidates, feature_dim)
        q = self.mlp(x)
        return q.reshape(batch, max_candidates)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def masked_argmax(q_values: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
    masked_q = q_values.clone()
    masked_q[action_mask == 0] = -1e9
    return masked_q.argmax(dim=1)


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    train_data = load_split("train")
    env = CrowdRecEnv(
        train_data,
        reward_type=args.reward_type,
        requester_quality_weight=args.quality_weight,
        requester_urgency_weight=args.urgency_weight,
        requester_popularity_weight=args.popularity_weight,
        seed=args.seed,
    )

    feature_dim = env.spec.feature_dim
    q_net = QNetwork(feature_dim, (args.hidden_size, args.hidden_size)).to(device)
    target_net = QNetwork(feature_dim, (args.hidden_size, args.hidden_size)).to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(q_net.parameters(), lr=args.lr)
    replay = ReplayBuffer(args.buffer_size)

    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.save_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    loss_log: List[Tuple[int, float]] = []
    epoch_log: List[Tuple[int, float, float]] = []
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        state = env.reset(shuffle=True)
        done = False
        total_reward = 0.0
        hits = 0
        steps = 0

        while not done:
            epsilon = max(
                args.eps_end,
                args.eps_start - global_step * (args.eps_start - args.eps_end) / args.eps_decay,
            )

            if random.random() < epsilon:
                action = env.sample_random_action(state)
            else:
                with torch.no_grad():
                    features = torch.as_tensor(state["features"], dtype=torch.float32, device=device)
                    mask = torch.as_tensor(state["action_mask"], dtype=torch.float32, device=device)
                    q_values = q_net(features.unsqueeze(0))
                    action = int(masked_argmax(q_values, mask.unsqueeze(0)).item())

            next_state, reward, done, info = env.step(action)
            replay.add(
                Transition(
                    state=state["features"],
                    action_mask=state["action_mask"],
                    action=action,
                    reward=reward,
                    next_state=None if next_state is None else next_state["features"],
                    next_action_mask=None if next_state is None else next_state["action_mask"],
                    done=done,
                )
            )

            total_reward += reward
            hits += int(info["hit"])
            steps += 1
            global_step += 1
            state = next_state if next_state is not None else state

            if len(replay) >= args.start_learning and global_step % args.update_every == 0:
                batch = replay.sample(args.batch_size)
                batch_state = torch.as_tensor(
                    np.stack([b.state for b in batch]), dtype=torch.float32, device=device
                )
                batch_mask = torch.as_tensor(
                    np.stack([b.action_mask for b in batch]), dtype=torch.float32, device=device
                )
                batch_action = torch.as_tensor(
                    [b.action for b in batch], dtype=torch.int64, device=device
                )
                batch_reward = torch.as_tensor(
                    [b.reward for b in batch], dtype=torch.float32, device=device
                )
                batch_done = torch.as_tensor(
                    [b.done for b in batch], dtype=torch.float32, device=device
                )

                q_values = q_net(batch_state)
                q_values[batch_mask == 0] = -1e9
                q_selected = q_values.gather(1, batch_action.unsqueeze(1)).squeeze(1)

                next_state_batch = [b.next_state for b in batch]
                if all(s is None for s in next_state_batch):
                    target_q = batch_reward
                else:
                    next_state_arr = np.stack(
                        [s if s is not None else np.zeros_like(batch_state[0].cpu().numpy()) for s in next_state_batch]
                    )
                    next_mask_arr = np.stack(
                        [
                            b.next_action_mask if b.next_action_mask is not None else np.zeros_like(batch_mask[0].cpu().numpy())
                            for b in batch
                        ]
                    )
                    next_state_tensor = torch.as_tensor(next_state_arr, dtype=torch.float32, device=device)
                    next_mask_tensor = torch.as_tensor(next_mask_arr, dtype=torch.float32, device=device)

                    with torch.no_grad():
                        next_q = target_net(next_state_tensor)
                        next_q[next_mask_tensor == 0] = -1e9
                        max_next_q = next_q.max(dim=1).values
                        target_q = batch_reward + (1.0 - batch_done) * args.gamma * max_next_q

                loss = nn.SmoothL1Loss()(q_selected, target_q)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q_net.parameters(), args.grad_clip)
                optimizer.step()

                loss_log.append((global_step, float(loss.item())))

            if global_step % args.target_update == 0:
                target_net.load_state_dict(q_net.state_dict())

            if args.max_steps_per_epoch and steps >= args.max_steps_per_epoch:
                break

        avg_reward = total_reward / steps if steps else 0.0
        hit_rate = hits / steps if steps else 0.0
        epoch_log.append((epoch, avg_reward, hit_rate))

        print(
            f"epoch={epoch} steps={steps} avg_reward={avg_reward:.4f} hit_rate={hit_rate:.4f} "
            f"epsilon={epsilon:.3f} buffer={len(replay)}"
        )

    np.savetxt(
        os.path.join(run_dir, "loss_history.csv"),
        np.asarray(loss_log),
        delimiter=",")
    np.savetxt(
        os.path.join(run_dir, "epoch_metrics.csv"),
        np.asarray(epoch_log),
        delimiter=",")

    torch.save(q_net.state_dict(), os.path.join(run_dir, "q_network.pt"))
    torch.save(target_net.state_dict(), os.path.join(run_dir, "target_network.pt"))

    if args.plot:
        try:
            import matplotlib.pyplot as plt

            if loss_log:
                steps, losses = zip(*loss_log)
                plt.figure(figsize=(8, 4))
                plt.plot(steps, losses, linewidth=1.0)
                plt.xlabel("step")
                plt.ylabel("loss")
                plt.title("DQN Loss")
                plt.tight_layout()
                plt.savefig(os.path.join(run_dir, "loss_curve.png"), dpi=150)
                plt.close()
        except ImportError:
            print("matplotlib is not installed; skip plot.")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--buffer-size", type=int, default=20000)
    parser.add_argument("--start-learning", type=int, default=1000)
    parser.add_argument("--update-every", type=int, default=4)
    parser.add_argument("--target-update", type=int, default=1000)
    parser.add_argument("--max-steps-per-epoch", type=int, default=0)

    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--grad-clip", type=float, default=5.0)

    parser.add_argument("--eps-start", type=float, default=1.0)
    parser.add_argument("--eps-end", type=float, default=0.05)
    parser.add_argument("--eps-decay", type=int, default=20000)

    parser.add_argument("--reward-type", type=str, default="requester_urgency")
    parser.add_argument("--quality-weight", type=float, default=1.0)
    parser.add_argument("--urgency-weight", type=float, default=0.5)
    parser.add_argument("--popularity-weight", type=float, default=0.0)

    parser.add_argument("--save-dir", type=str, default="runs/requester_dqn")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--plot", action="store_true")

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.max_steps_per_epoch <= 0:
        args.max_steps_per_epoch = 0
    train(args)


if __name__ == "__main__":
    main()
