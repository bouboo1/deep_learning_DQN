# env/crowd_env.py 设计说明

这个文件是整个项目的核心，把预处理好的数据包装成 DQN 可以交互的环境。支持两个视角：Worker 视角和 Requester 视角。

---

## 整体思路

众包推荐本质上是一个序列决策问题：每次有 worker 到达或 project 收到新 entry，系统就要做一次推荐决策。我们把每一次决策当作一个时间步，按时间顺序串起来，形成一个"伪在线"的强化学习环境。

数据是历史日志，所以环境是离线的——动作不会真正改变后续候选集，但按时间排序保证了特征的时序合理性，不会用到未来信息。

---

## 两个视角的对比

| | Worker 视角 | Requester 视角 |
|---|---|---|
| 每步含义 | 一个 worker 到达，推荐一个 project | 一个 project 收到 entry，推荐一个 worker |
| 候选集 | 候选 project（最多 20 个） | 候选 worker（最多 20 个） |
| 特征维度 | 13 维 | 16 维 |
| 奖励目标 | 推荐命中 worker 真实选择 | 推荐高质量、多样、及时的 worker |
| 数据文件 | `enhanced_train/val/test.pkl` | `requester_train/val/test.pkl` |
| 数据目录 | `data_processed/` | `data_processed/`（同一目录） |

---

## Worker 视角（CrowdRecEnv）

### 状态设计

每个时间步的状态是一个字典：

```python
{
    "features":        # shape [20, 13]，候选 project 特征矩阵，不足 20 个用 0 padding
    "action_mask":     # shape [20]，1 表示可选，0 表示 padding 或已过期
    "worker_id":       # 当前 worker 的 ID
    "timestamp_index": # 当前样本在数据集中的位置
}
```

特征矩阵的每一行对应一个候选 project，13 个特征如下：

| 特征 | 含义 | 备注 |
|------|------|------|
| worker_quality | worker 质量评分 | z-score 归一化 |
| worker_history_count | worker 历史完成任务数 | log1p + z-score |
| worker_active_days | worker 活跃天数 | z-score |
| project_category | 项目类别 ID | 整数编码 |
| project_sub_category | 项目子类别 ID | 整数编码 |
| project_industry | 项目行业 ID | 整数编码 |
| project_entry_count | 项目历史参与人数 | z-score |
| project_duration_days | 项目持续天数 | z-score |
| category_match | worker 偏好类别与项目是否一致 | 0/1 |
| industry_match | worker 偏好行业与项目是否一致 | 0/1 |
| project_popularity | 项目流行度 | z-score |
| worker_category_count | worker 参与过的类别数 | z-score |
| remaining_days | 项目剩余天数 | env 内用 sigmoid(v/30) 压到 (0,1) |

`remaining_days` 是唯一在 env 里做二次处理的特征，原因是它的原始值范围（0~200 天）和其他 z-score 特征量纲差距太大，用 sigmoid 压缩后量纲一致。

### 动作设计

从 0 到 19 选一个整数，表示推荐候选列表里第几个 project。选到 `action_mask=0` 的位置视为非法动作，给 -1 惩罚。

### 奖励设计

命中正样本（推荐的 project 和 worker 历史真实选择一致）得 1，否则得 0。

这里的逻辑是：worker 历史上选了哪个 project，说明那个对他最有吸引力，推荐对了就是最大化了他的利益。

支持四种 `reward_type`：

| reward_type | 计算方式 |
|---|---|
| `worker` | 命中得 1，否则 0 |
| `requester` | 命中后按 project 流行度给分 |
| `requester_urgency` | 命中后按流行度 + 紧迫度给分 |
| `hybrid` | `alpha × worker奖励 + (1-alpha) × requester奖励` |

### 截止时间过滤

`_build_state` 里构建 `action_mask` 时，会检查每个候选 project 的 `remaining_days`，小于 0 的（已过期）直接置 0，模型不能选。这是处理动态性的关键——随着时间推进，过期任务自动从候选集里消失。

---

## Requester 视角（RequesterCrowdEnv）

继承自 `CrowdRecEnv`，重写了状态构建、奖励计算和 info 构建三个方法。

### 状态设计

每个时间步的状态：

```python
{
    "features":        # shape [20, 16]，候选 worker 特征矩阵
    "action_mask":     # shape [20]，project_remaining_days >= 0 才可选
    "project_id":      # 当前 project 的 ID
    "timestamp_index": # 当前样本位置
}
```

16 个特征分两组：

**Project 上下文（描述当前任务的状态）**：

| 特征 | 含义 |
|------|------|
| project_category | 项目类别 ID |
| project_sub_category | 项目子类别 ID |
| project_industry | 项目行业 ID |
| project_duration_days | 项目总持续天数 |
| project_remaining_days | 项目剩余天数 |
| project_current_entry_count | 当前已收到的 entry 数（动态更新） |
| project_hist_entry_count | 历史平均 entry 数（静态参考） |
| project_fill_rate | 饱和度 = current / hist，反映任务完成进度 |

**Worker 侧特征（描述每个候选 worker）**：

| 特征 | 含义 |
|------|------|
| worker_quality | worker 质量评分 |
| worker_history_count | 历史任务数 |
| worker_active_days | 活跃天数 |
| worker_category_match | 偏好类别与项目是否匹配 |
| worker_industry_match | 偏好行业与项目是否匹配 |
| worker_category_count | 参与过的类别数（专项能力） |
| worker_already_submitted | 是否已提交过此项目（动态更新） |
| worker_recent_activity | 最近 7 天活跃次数（动态更新） |

其中 `project_current_entry_count`、`worker_already_submitted`、`worker_recent_activity` 三个特征是动态的，在预处理时按时间顺序计算，反映每个决策时刻的真实状态。

### 动作设计

从 0 到 19 选一个整数，表示推荐候选列表里第几个 worker。

### 奖励设计

```
reward = quality_weight × (1 + urgency_weight × urgency) × worker_quality
       + diversity_weight × (1 - worker_already_submitted)
```

三个部分：

**质量项**：推荐的 worker 质量越高越好，这是主项，权重默认 1.0。

**紧急度加成**：`urgency = 1 / (1 + exp(remaining_days))`，剩余天数越少，urgency 越接近 1，quality 的有效权重就越大。设计意图是：快到 deadline 的任务更需要高质量 worker，不能随便推。

**多样性项**：如果推荐的 worker 之前没有提交过这个 project，额外加 0.3 分。避免系统总是推同一批高质量 worker，让更多 worker 有机会参与。

默认权重：`quality_weight=1.0`，`urgency_weight=0.5`，`diversity_weight=0.3`，可在初始化时调整。

---

## EnvSpec

两个环境都有一个 `spec` 属性，给模型代码用：

```python
env.spec.state_shape   # (max_candidates, feature_dim)，即 (20, 13) 或 (20, 16)
env.spec.action_dim    # max_candidates，即 20
env.spec.feature_order # 特征名列表，按顺序对应特征矩阵的列
```

DQN 网络的输入层维度从 `env.spec.state_shape` 读取，不要硬编码。

---

## 数据加载

```python
from env import load_split, load_requester_split

# Worker 视角
train_data = load_split("train")          # data_processed/enhanced_train.pkl
val_data   = load_split("val")
test_data  = load_split("test")

# Requester 视角
req_train  = load_requester_split("train")  # data_processed/requester_train.pkl
req_val    = load_requester_split("val")
req_test   = load_requester_split("test")
```

两个视角的数据都在 `data_processed/` 目录下，文件名不同。

---

## DQN 接入方式

```python
# 选动作前屏蔽非法位置
q_values[state["action_mask"] == 0] = -1e9
action = q_values.argmax()

# 存 transition
(state["features"], state["action_mask"], action, reward,
 next_state["features"] if next_state else None,
 next_state["action_mask"] if next_state else None,
 done)
```

`done=True` 时 `next_state` 为 `None`，需要在存 replay buffer 时做判断。

Worker 视角输入 `[20, 13]`，Requester 视角输入 `[20, 16]`，两个目标的网络输入层维度不同，不能共用同一个网络。
