"""众包任务推荐的强化学习环境。

这个模块把预处理好的 worker-project 匹配样本，包装成一个
类似 Gym 的环境。代码故意不依赖 gym/gymnasium，方便 DQN 同学直接调用。
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


DEFAULT_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data_processed",
)

# 模型输入特征的固定顺序。DQN 看到的是矩阵，必须保证每一列含义稳定。
DEFAULT_FEATURE_ORDER = [
    "worker_quality",
    "worker_history_count",
    "worker_active_days",
    "project_category",
    "project_sub_category",
    "project_industry",
    "project_entry_count",
    "project_duration_days",
    "category_match",
    "industry_match",
    "project_popularity",
    "worker_category_count",
    "remaining_days",
]

SPLIT_FILES = {
    "train": "enhanced_train.pkl",
    "val": "enhanced_val.pkl",
    "test": "enhanced_test.pkl",
}
WORKER_EPISODES_FILE = "worker_episodes.pkl"


@dataclass(frozen=True)
class EnvSpec:
    """给模型代码使用的静态维度信息。"""

    max_candidates: int
    feature_dim: int
    feature_order: Tuple[str, ...]

    @property
    def state_shape(self) -> Tuple[int, int]:
        # state["features"] 的形状：[最大候选任务数, 每个候选任务的特征数]
        return self.max_candidates, self.feature_dim

    @property
    def action_dim(self) -> int:
        # DQN 输出 Q 值的长度。每个位置对应一个候选任务下标。
        return self.max_candidates


def load_split(split: str, data_dir: str = DEFAULT_DATA_DIR) -> List[Dict[str, Any]]:
    """读取一个预处理后的数据划分。

    Args:
        split: "train"、"val"、"test" 之一；也可以直接传 pkl 文件名。
        data_dir: 存放 enhanced_train/val/test.pkl 的目录。
    """

    # 支持 load_split("train")，也支持 load_split("enhanced_train.pkl")。
    filename = SPLIT_FILES.get(split, split)
    path = os.path.join(data_dir, filename)
    with open(path, "rb") as f:
        return pickle.load(f)


def load_all_splits(data_dir: str = DEFAULT_DATA_DIR) -> Dict[str, List[Dict[str, Any]]]:
    """一次性读取 train/val/test 三个划分。"""

    return {split: load_split(split, data_dir) for split in SPLIT_FILES}


def load_worker_episodes(data_dir: str = DEFAULT_DATA_DIR) -> Dict[int, List[Dict[str, Any]]]:
    """读取按 worker_id 分组的 episode 数据。"""

    path = os.path.join(data_dir, WORKER_EPISODES_FILE)
    with open(path, "rb") as f:
        return pickle.load(f)


class CrowdRecEnv:
    """一个类似 Gym 的众包任务推荐环境。

    每条样本视为一个决策点：一个 worker 到达平台，智能体从候选 project
    里推荐一个，环境返回奖励，然后推进到按时间排序的下一条样本。
    """

    def __init__(
        self,
        data: Sequence[Mapping[str, Any]],
        reward_type: str = "worker",
        max_candidates: Optional[int] = None,
        feature_order: Optional[Sequence[str]] = None,
        invalid_action_penalty: float = -1.0,
        requester_quality_weight: float = 1.0,
        requester_urgency_weight: float = 0.5,
        requester_popularity_weight: float = 0.0,
        hybrid_alpha: float = 0.5,
        seed: Optional[int] = None,
    ) -> None:
        if not data:
            raise ValueError("CrowdRecEnv requires at least one sample.")

        self.data = list(data)
        self.reward_type = reward_type
        self.feature_order = tuple(feature_order or DEFAULT_FEATURE_ORDER)
        # 候选任务数不是完全固定的，因此用 split 内最大候选数作为 padding 长度。
        self.max_candidates = max_candidates or max(
            len(sample["candidate_projects"]) for sample in self.data
        )
        # 模型如果选到了 padding 位置，就给这个惩罚。
        self.invalid_action_penalty = float(invalid_action_penalty)
        # 请求者奖励的两个权重：worker 质量和 project 流行度。
        self.requester_quality_weight = float(requester_quality_weight)
        self.requester_urgency_weight = float(requester_urgency_weight)
        self.requester_popularity_weight = float(requester_popularity_weight)
        # 混合奖励中，worker_reward 的占比；1 - alpha 是 requester_reward 占比。
        self.hybrid_alpha = float(hybrid_alpha)
        self.rng = np.random.default_rng(seed)

        if self.max_candidates < 1:
            raise ValueError("max_candidates must be positive.")
        if not 0.0 <= self.hybrid_alpha <= 1.0:
            raise ValueError("hybrid_alpha must be in [0, 1].")

        self.spec = EnvSpec(
            max_candidates=self.max_candidates,
            feature_dim=len(self.feature_order),
            feature_order=self.feature_order,
        )
        self.index = 0
        self.done = False

    def reset(self, shuffle: bool = False) -> Dict[str, np.ndarray]:
        """重置环境，返回第一条样本对应的初始状态。

        训练时可以设置 shuffle=True 打乱样本顺序。验证和测试一般保持 False，
        这样实验结果更容易复现。
        """

        if shuffle:
            self.rng.shuffle(self.data)
        self.index = 0
        self.done = False
        return self._build_state(self.data[self.index])

    def step(self, action: int) -> Tuple[Optional[Dict[str, np.ndarray]], float, bool, Dict[str, Any]]:
        """执行一个动作，并推进到下一条样本。

        Args:
            action: 模型选择的候选任务下标。合法动作满足
                state["action_mask"][action] == 1。

        Returns:
            next_state, reward, done, info。done=True 时 next_state 为 None。
        """

        if self.done:
            raise RuntimeError("step() called after episode is done. Call reset().")

        sample = self.data[self.index]
        state = self._build_state(sample)
        action_int = int(action)
        # action 必须落在 action_mask 允许的位置，不能选 padding 或过期任务。
        valid_action = (
            0 <= action_int < self.max_candidates
            and state["action_mask"][action_int] > 0
        )

        if valid_action:
            reward = self._compute_reward(sample, action_int)
        else:
            reward = self.invalid_action_penalty

        info = self._build_info(sample, action_int, valid_action, reward)

        # 本环境把每条样本当作一个时间步，所以 step 后直接移动到下一条样本。
        self.index += 1
        self.done = self.index >= len(self.data)
        next_state = None if self.done else self._build_state(self.data[self.index])
        return next_state, float(reward), self.done, info

    def sample_random_action(self, state: Mapping[str, np.ndarray]) -> int:
        """随机采样一个合法动作，可用于 smoke test 或 epsilon 探索。"""

        # action_mask 为 1 的位置才是真实候选任务。
        legal_actions = np.flatnonzero(state["action_mask"] > 0)
        if len(legal_actions) == 0:
            raise RuntimeError("No legal action is available in the current state.")
        return int(self.rng.choice(legal_actions))

    def iter_feature_vectors(self, sample: Mapping[str, Any]) -> Iterable[List[float]]:
        """按固定特征顺序，把每个候选任务的 dict 特征转换成 list。"""

        for feat in sample["candidate_features"]:
            yield [float(feat[name]) for name in self.feature_order]

    def _build_state(self, sample: Mapping[str, Any]) -> Dict[str, np.ndarray]:
        """把原始样本转换成模型可用的 state 字典。"""

        # 先创建全 0 矩阵；真实候选任务写在前 n 行，其余行保持 0 作为 padding。
        features = np.zeros(self.spec.state_shape, dtype=np.float32)
        action_mask = np.zeros(self.max_candidates, dtype=np.float32)

        vectors = list(self.iter_feature_vectors(sample))
        if len(vectors) > self.max_candidates:
            raise ValueError(
                f"Sample has {len(vectors)} candidates, larger than max_candidates={self.max_candidates}."
            )

        n = len(vectors)
        features[:n, :] = np.asarray(vectors, dtype=np.float32)
        for i in range(n):
            if self._candidate_before_deadline(sample, i):
                action_mask[i] = 1.0

        return {
            "features": features,
            "action_mask": action_mask,
            "worker_id": np.asarray(sample["worker_id"], dtype=np.int64),
            "timestamp_index": np.asarray(self.index, dtype=np.int64),
        }

    def _compute_reward(self, sample: Mapping[str, Any], action: int) -> float:
        """根据当前样本、动作和 reward_type 计算奖励。"""

        # positive_index 是历史数据中 worker 真实选择的项目位置。
        hit = action == int(sample["positive_index"])
        if not hit:
            return 0.0

        # 参与者目标：命中真实选择就给 1。
        worker_reward = 1.0
        # 请求者目标：命中后还要考虑 worker 质量等因素。
        requester_reward = self._requester_reward(sample, action)

        if self.reward_type == "worker":
            return worker_reward
        if self.reward_type == "requester":
            return requester_reward
        if self.reward_type == "requester_urgency":
            return self._requester_urgency_reward(sample, action)
        if self.reward_type == "hybrid":
            return self.hybrid_alpha * worker_reward + (1.0 - self.hybrid_alpha) * requester_reward

        raise ValueError(
            f"Unknown reward_type={self.reward_type!r}. "
            "Use 'worker', 'requester', 'requester_urgency', or 'hybrid'."
        )

    def _requester_reward(self, sample: Mapping[str, Any], action: int) -> float:
        """请求者视角的奖励：希望高质量 worker 完成项目。"""

        feat = sample["candidate_features"][action]
        # 注意这里的 worker_quality 是预处理后的标准化特征，可能为负。
        quality = float(feat["worker_quality"])
        popularity = float(feat.get("project_popularity", 0.0))
        return (
            self.requester_quality_weight * quality
            + self.requester_popularity_weight * popularity
        )

    def _requester_urgency_reward(self, sample: Mapping[str, Any], action: int) -> float:
        """请求者紧迫度奖励：高质量 worker 优先推荐给更紧急的项目。"""

        feat = sample["candidate_features"][action]
        quality = float(feat["worker_quality"])
        popularity = float(feat.get("project_popularity", 0.0))
        urgency = self._compute_urgency(sample, action)
        return (
            self.requester_quality_weight * quality
            + self.requester_urgency_weight * urgency
            + self.requester_popularity_weight * popularity
        )

    def _compute_urgency(self, sample: Mapping[str, Any], action: int) -> float:
        """根据 remaining_days 或 deadline 计算紧迫度。"""

        feat = sample["candidate_features"][action]
        project = sample["candidate_projects"][action]

        if "remaining_days" in feat:
            remaining_days = float(feat["remaining_days"])
            return float(1.0 / (1.0 + np.exp(remaining_days)))

        deadline = self._lookup_candidate_value(project, feat, "deadline")

        remaining_days = None
        if deadline is not None:
            deadline_dt = self._parse_datetime(deadline)
            timestamp_dt = self._parse_datetime(sample.get("timestamp"))
            if deadline_dt is not None and timestamp_dt is not None:
                remaining_days = (deadline_dt - timestamp_dt).total_seconds() / 86400.0

        if remaining_days is None:
            remaining_days = float(feat.get("project_duration_days", 0.0))

        remaining_days = max(float(remaining_days), 0.0)
        return float(1.0 / (1.0 + np.exp(remaining_days)))

    def _candidate_before_deadline(self, sample: Mapping[str, Any], index: int) -> bool:
        """检查候选任务是否还在截止时间前；字段缺失时默认可选。"""

        project = sample["candidate_projects"][index]
        feat = sample["candidate_features"][index]
        remaining_days = feat.get("remaining_days")
        if remaining_days is not None:
            return float(remaining_days) > 0

        deadline = self._lookup_candidate_value(project, feat, "deadline")
        if deadline is None:
            return True

        deadline_dt = self._parse_datetime(deadline)
        timestamp_dt = self._parse_datetime(sample.get("timestamp"))
        if deadline_dt is None or timestamp_dt is None:
            return True
        return deadline_dt > timestamp_dt

    @staticmethod
    def _lookup_candidate_value(project: Any, feat: Mapping[str, Any], name: str) -> Any:
        """从 candidate_project 或 candidate_feature 中读取同名字段。"""

        if isinstance(project, Mapping) and name in project:
            return project[name]
        return feat.get(name)

    @staticmethod
    def _parse_datetime(value: Any) -> Optional[datetime]:
        """兼容 datetime、date、时间戳字符串等常见时间格式。"""

        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        if hasattr(value, "to_pydatetime"):
            return value.to_pydatetime()
        if hasattr(value, "year") and hasattr(value, "month") and hasattr(value, "day"):
            return datetime(value.year, value.month, value.day)
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(value)
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            if text.endswith("Z"):
                text = text[:-1] + "+00:00"
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y/%m/%d %H:%M:%S", "%Y/%m/%d"):
                try:
                    return datetime.strptime(text, fmt)
                except ValueError:
                    pass
            try:
                parsed = datetime.fromisoformat(text)
                return parsed.replace(tzinfo=None) if parsed.tzinfo else parsed
            except ValueError:
                return None
        return None

    def _build_info(
        self,
        sample: Mapping[str, Any],
        action: int,
        valid_action: bool,
        reward: float,
    ) -> Dict[str, Any]:
        """构造调试和评估用的额外信息。

        这些字段不直接作为模型输入，主要给测试同学统计 hit_rate、平均 reward、
        非法动作率，以及检查模型到底推荐了哪个 project。
        """

        positive_index = int(sample["positive_index"])
        chosen_project = (
            sample["candidate_projects"][action] if valid_action else None
        )
        chosen_features = (
            sample["candidate_features"][action] if valid_action else None
        )

        return {
            "sample_index": self.index,
            "worker_id": sample["worker_id"],
            "timestamp": sample["timestamp"],
            "num_candidates": len(sample["candidate_projects"]),
            "action": action,
            "valid_action": valid_action,
            "chosen_project": chosen_project,
            "positive_project": sample["positive_project"],
            "positive_index": positive_index,
            "hit": valid_action and action == positive_index,
            "reward": float(reward),
            "chosen_features": chosen_features,
        }
