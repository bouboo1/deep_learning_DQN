# 众包任务推荐强化学习环境

环境用途：把 `data_processed` 里的预处理数据包装成 DQN 可以交互的形式。

## 文件说明

- `crowd_env.py`：核心环境代码，包含 `CrowdRecEnv`、`load_split()`、`load_all_splits()`、`load_worker_episodes()`。
- `demo_random.py`：随机合法动作的 smoke test，用来确认环境可以跑通。
- `__init__.py`：方便从 `env` 包导入。

## 环境定义

每条样本表示一次 worker 到达平台后的推荐决策：

- `state`：候选 project 的特征矩阵。
- `action`：推荐第几个候选 project。
- `reward`：根据推荐结果计算奖励。
- `next_state`：时间顺序上的下一条 worker 到达样本。
- `done`：是否跑完整个 split。

环境接口类似 Gym，但不依赖 Gym：

```python
from env import CrowdRecEnv, load_split

train_data = load_split("train")
env = CrowdRecEnv(train_data, reward_type="worker")

state = env.reset()
done = False

while not done:
    action = env.sample_random_action(state)
    next_state, reward, done, info = env.step(action)
    state = next_state
```

## State 格式

`reset()` 和 `step()` 返回的 state 是一个字典：

```python
{
    "features": np.ndarray,      # shape = [max_candidates, feature_dim]
    "action_mask": np.ndarray,   # shape = [max_candidates]
    "worker_id": np.ndarray,
    "timestamp_index": np.ndarray
}
```

当前数据中候选任务数不是完全固定的，所以环境会 padding 到 `max_candidates`。默认 `max_candidates` 会自动取当前 split 里的最大候选数，目前一般是 21。

模型选动作时必须使用 `action_mask` 屏蔽非法 action：

```python
valid = state["action_mask"] == 1
```

特征顺序固定为：

```text
worker_quality
worker_history_count
worker_active_days
project_category
project_sub_category
project_industry
project_entry_count
project_duration_days
category_match
industry_match
project_popularity
worker_category_count
remaining_days
```

因此默认：

```text
state["features"].shape = [20, 13]
action_dim = 20
```

如果某条样本只有 5 个候选任务，那么前 5 行是有效候选，其余行全 0，`action_mask` 后 16 位为 0。

如果候选任务带有动态剩余时间字段 `remaining_days`，环境会同时用它过滤动作：

```text
remaining_days <= 0 时，对应 action_mask 位置为 0
remaining_days 缺失时，如果有 deadline，则用 deadline 和 timestamp 判断
remaining_days 和 deadline 都缺失时，默认该候选任务仍可推荐
```

## Action 定义

`action` 是候选 project 的下标：

```text
action ∈ [0, num_candidates - 1]
```

例如 `action = 3` 表示推荐当前样本的第 4 个候选 project。

如果模型选到 padding 位置，环境会返回 `invalid_action_penalty`，默认是 `-1.0`。

如果模型选到已过截止时间过滤条件的任务，也会被视为非法 action，并返回同样的惩罚。

## Reward 定义

环境内置三种奖励，供“参与者”和“请求者”方向复用。

### 参与者利益

```python
env = CrowdRecEnv(train_data, reward_type="worker")
```

默认奖励：

```text
reward = 1, 如果 action == positive_index
reward = 0, 否则
```

含义：推荐命中 worker 历史中真实选择的任务，就认为满足参与者兴趣。

### 请求者利益

```python
env = CrowdRecEnv(train_data, reward_type="requester")
```

默认奖励：

```text
reward = worker_quality, 如果 action == positive_index
reward = 0, 否则
```

含义：请求者希望任务被真实选择，并且由质量更高的 worker 完成。

注意：这里的 `worker_quality` 来自预处理后的标准化特征，可能出现负数。负数表示低于训练集均值，不是数据错误。

可以加入项目流行度权重：

```python
env = CrowdRecEnv(
    train_data,
    reward_type="requester",
    requester_quality_weight=1.0,
    requester_popularity_weight=0.2,
)
```

这时命中奖励为：

```text
worker_quality + 0.2 * project_popularity
```

### 请求者紧迫度利益

```python
env = CrowdRecEnv(
    train_data,
    reward_type="requester_urgency",
    requester_quality_weight=1.0,
    requester_urgency_weight=0.5,
    requester_popularity_weight=0.0,
)
```

默认命中奖励为：

```text
worker_quality + 0.5 * urgency
```

如果候选任务数据里有 `remaining_days`，环境会优先使用它计算紧迫度。A 同学新版预处理脚本保留的是原始剩余天数，因此 `remaining_days > 0` 可以直接表示任务仍在 deadline 前：

```text
urgency = 1 / (1 + exp(remaining_days))
```

如果旧数据没有 `remaining_days`，但有 `deadline`，环境会用 `sample["timestamp"]` 到 deadline 的剩余天数动态计算；如果两者都没有，则回退使用 `project_duration_days`，保证旧数据仍能运行。这个奖励类型只是在环境里可用；如果要通过训练脚本命令行直接使用，还需要模型训练脚本把 `requester_urgency` 加进自己的 `--reward-type` 参数。

### 混合目标

```python
env = CrowdRecEnv(train_data, reward_type="hybrid", hybrid_alpha=0.5)
```

奖励：

```text
0.5 * worker_reward + 0.5 * requester_reward
```

这个模式可以作为综合平台收益的补充实验。

注意：step 级 `R_t` 日志属于训练循环输出，不在 `env` 目录内实现。环境每次 `step()` 已经返回 `reward` 和 `info["reward"]`，训练脚本可以据此写入 step 级日志或 CSV。

## 参与者/请求者同学如何接入自己的设计

参与者和请求者同学设计完自己的“状态、行为、奖励”以后，不一定要重新写一套环境。建议按改动大小分三种情况处理。

### 情况一：只改奖励权重

如果状态仍然使用 `state["features"]`，动作仍然是“从候选 project 中选一个”，只是奖励公式里的权重不同，那么不需要写新的代码文件，也不需要改 `crowd_env.py`。

直接在创建环境时传参数：

```python
from env import CrowdRecEnv, load_split

train_data = load_split("train")

env = CrowdRecEnv(
    train_data,
    reward_type="requester",
    requester_quality_weight=1.0,
    requester_popularity_weight=0.2,
)
```

参与者方向通常可以直接用：

```python
env = CrowdRecEnv(train_data, reward_type="worker")
```

请求者方向通常可以先用：

```python
env = CrowdRecEnv(train_data, reward_type="requester")
```

如果想做平台综合目标：

```python
env = CrowdRecEnv(train_data, reward_type="hybrid", hybrid_alpha=0.6)
```

这里 `hybrid_alpha=0.6` 表示参与者奖励占 60%，请求者奖励占 40%。

### 情况二：奖励公式变复杂

如果参与者或请求者同学设计了新的奖励公式，例如：

- 参与者：命中真实选择之外，还考虑 `category_match`、`industry_match`。
- 请求者：考虑 `worker_quality`、`project_popularity`、项目是否缺回答。
- 想把负的标准化 `worker_quality` 映射成非负奖励。

这时可以选择两种做法。

第一种做法：直接在 `crowd_env.py` 里加一个新的 `reward_type`。适合公式很短的情况。

需要改 `_compute_reward()`：

```python
if self.reward_type == "worker_match":
    return self._worker_match_reward(sample, action)
```

然后在 `CrowdRecEnv` 类里新增函数：

```python
def _worker_match_reward(self, sample, action):
    feat = sample["candidate_features"][action]
    return 1.0 + 0.2 * feat["category_match"] + 0.2 * feat["industry_match"]
```

使用时：

```python
env = CrowdRecEnv(train_data, reward_type="worker_match")
```

第二种做法：新建一个奖励文件，例如 `env/rewards.py`。适合公式较多、参与者和请求者都要反复实验的情况。

可以新建：

```text
env/rewards.py
```

写：

```python
def worker_interest_reward(sample, action):
    feat = sample["candidate_features"][action]
    hit = action == int(sample["positive_index"])
    if not hit:
        return 0.0
    return 1.0 + 0.2 * feat["category_match"] + 0.2 * feat["industry_match"]


def requester_quality_reward(sample, action):
    feat = sample["candidate_features"][action]
    hit = action == int(sample["positive_index"])
    if not hit:
        return 0.0
    return max(0.0, float(feat["worker_quality"])) + 0.1 * feat["project_popularity"]
```

然后在 `crowd_env.py` 里导入并接入：

```python
from .rewards import worker_interest_reward, requester_quality_reward
```

在 `_compute_reward()` 中增加：

```python
if self.reward_type == "worker_interest":
    return worker_interest_reward(sample, action)
if self.reward_type == "requester_quality":
    return requester_quality_reward(sample, action)
```

使用时：

```python
env = CrowdRecEnv(train_data, reward_type="worker_interest")
env = CrowdRecEnv(train_data, reward_type="requester_quality")
```

注意：无论奖励公式怎么改，建议保持“非法 action 由环境统一处理”。也就是说，奖励函数只负责合法 action 的奖励；如果模型选到 padding 位置，`step()` 会直接给 `invalid_action_penalty`。

### 情况三：要改状态

如果参与者或请求者同学只是想从现有特征中少用几列或多用几列，优先使用 `feature_order`，不需要写新环境。

例如参与者模型只使用 worker 质量、历史、活跃天数、类别匹配、行业匹配：

```python
participant_features = [
    "worker_quality",
    "worker_history_count",
    "worker_active_days",
    "category_match",
    "industry_match",
]

env = CrowdRecEnv(
    train_data,
    reward_type="worker",
    feature_order=participant_features,
)
```

这时：

```text
state["features"].shape = [max_candidates, 5]
```

如果请求者模型想用另一组特征：

```python
requester_features = [
    "worker_quality",
    "project_entry_count",
    "project_duration_days",
    "project_popularity",
]

env = CrowdRecEnv(
    train_data,
    reward_type="requester",
    feature_order=requester_features,
)
```

只有在下面这些情况下，才需要修改 `crowd_env.py` 的 `_build_state()`：

- 要加入当前样本没有的新特征。
- 要把多个历史样本拼成序列状态。
- 要加入 project 剩余时间、当前完成进度等动态状态。
- 要返回多个输入张量，而不是一个 `features` 矩阵。

如果要改 `_build_state()`，建议仍然保留这两个字段：

```python
state["features"]
state["action_mask"]
```

这样模型和测试代码不用大改。

### 情况四：要改动作

当前动作定义是：

```text
action = 推荐第几个候选 project
```

大多数 DQN 实验都可以沿用这个动作定义，不需要改环境。

如果想把动作改成“推荐多个 project”或者“先选类别再选任务”，那就不是当前这个环境的简单参数修改了，需要改 `step()` 和模型输出：

- `step(action)` 要能接收列表或多阶段动作。
- `reward` 要根据多个推荐结果计算。
- `action_mask` 的含义也要重新定义。

建议参与者和请求者同学都保持当前动作定义：**每一步只推荐一个候选 project**。

## 是否需要写新的代码文件

一般结论：

- 只调奖励权重：不用写新文件。
- 只换使用哪些特征：不用写新文件，传 `feature_order`。
- 奖励公式比较短：可以直接在 `crowd_env.py` 里加一个函数和一个 `reward_type`。
- 奖励公式很多或参与者/请求者各自实验较多：建议新建 `env/rewards.py`。
- 状态构造大改、动作定义大改：需要改 `crowd_env.py`，必要时可以新建一个继承环境，例如 `participant_env.py` 或 `requester_env.py`。

## Info 字段

`step()` 返回的 `info` 方便测试同学统计指标：

```python
{
    "sample_index": int,
    "worker_id": int,
    "timestamp": datetime,
    "num_candidates": int,
    "action": int,
    "valid_action": bool,
    "chosen_project": int | None,
    "positive_project": int,
    "positive_index": int,
    "hit": bool,
    "reward": float,
    "chosen_features": dict | None
}
```

常用指标：

- `hit_rate = mean(info["hit"])`
- `avg_reward = mean(info["reward"])`
- 非法动作率：`mean(not info["valid_action"])`

## 跑通测试

在项目根目录运行：

```bash
python env/demo_random.py --split train --reward-type worker --max-steps 1000
```

请求者奖励：

```bash
python env/demo_random.py --split train --reward-type requester --max-steps 1000
```

如果能输出 `state_shape`、`action_dim`、`hit_rate`、`avg_reward`，说明环境可用。

## 使用建议

DQN 模型同学输入 `state["features"]`，输出长度为 `action_dim` 的 Q 值。选动作前需要把非法位置的 Q 值设成很小：

```python
q_values[state["action_mask"] == 0] = -1e9
action = q_values.argmax()
```

训练时保存 transition：

```python
(state["features"], state["action_mask"], action, reward,
 next_state["features"], next_state["action_mask"], done)
```

如果 `done=True`，`next_state` 会是 `None`。
