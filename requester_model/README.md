# requester_model 设计说明

这个目录负责训练和评估两个视角的 DQN 模型：Worker 视角（最大化参与者利益）和 Requester 视角（最大化请求者利益）。核心脚本是 `visualize_train.py`，支持单独训练任意一个视角，也支持联合训练后生成对比图。

---

## 文件说明

- `visualize_train.py`：主训练脚本，同时支持 worker 和 requester 两个目标
- `run_all_variants.py`：批量跑 dqn / double_dqn / dueling_dqn 三个变体
- `plot_from_csv.py`：从已有 `metrics.csv` 重新生成训练曲线图

模型网络、Agent、Replay Buffer 复用 `work_model/` 里的实现，不重复定义。

---

## Worker 视角

### 问题定义

每当一个 worker 到达平台，系统从候选 project 列表里推荐一个。目标是推荐 worker 真正感兴趣、会去做的 project，最大化参与者的利益。

### 状态

特征矩阵形状 `[20, 13]`，每行是一个候选 project 的特征，包含：
- worker 自身信息：质量评分、历史任务数、活跃天数
- project 信息：类别、子类别、行业、历史参与人数、持续天数、流行度、剩余天数
- 匹配信息：类别是否对口、行业是否对口

`action_mask` 屏蔽已过期（`remaining_days < 0`）的 project，模型不能选。

### 动作

从 0 到 19 选一个整数，表示推荐候选列表里第几个 project。

### 奖励

```
reward = 1  如果推荐的 project 和 worker 历史真实选择一致（命中）
reward = 0  否则
```

逻辑：worker 历史上选了哪个，说明那个对他最有吸引力，推荐对了就是最大化了他的利益。

### 网络结构

使用 `QNetwork`（candidate_scorer 架构）：对每个候选 project 的特征向量独立过一个共享 MLP，输出一个标量 Q 值，最终得到长度为 20 的 Q 值向量。

这种设计比 FlatQNetwork（把所有候选拼在一起）更合理：推荐的本质是对每个候选打分，同一套参数对所有候选都适用，不依赖候选在列表里的位置。

### 评估指标

| 指标 | 含义 |
|------|------|
| hit_rate | 推荐命中正样本的比例，主要指标 |
| avg_reward | 平均奖励，与 hit_rate 数值相同 |
| invalid_action_rate | 选到过期/padding 位置的比例，越低越好 |

随机策略基线：`hit_rate ≈ 1/20 = 0.05`，模型需要明显高于这个值。

---

## Requester 视角

### 问题定义

每当一个 project 收到新 entry，系统从候选 worker 池里推荐最合适的 worker。目标是让 project 得到高质量、多样化的回答，最大化请求者的利益。

### 状态

特征矩阵形状 `[20, 16]`，每行是一个候选 worker 的特征，分两组：

**Project 上下文（每行都一样，描述当前任务）**：
- 类别、子类别、行业
- 总持续天数、剩余天数
- 当前已收到的 entry 数（动态）、历史平均 entry 数、当前饱和度（动态）

**Worker 侧特征（每行不同，描述各候选 worker）**：
- 质量评分、历史任务数、活跃天数
- 类别匹配、行业匹配、参与过的类别数
- 是否已提交过此 project（动态）、最近 7 天活跃次数（动态）

标注"动态"的特征在预处理时按时间顺序计算，反映每个决策时刻的真实状态，不是静态快照。

`action_mask` 用 `project_remaining_days >= 0` 判断任务是否仍在截止期内。

### 动作

从 0 到 19 选一个整数，表示推荐候选列表里第几个 worker。

### 奖励

```
reward = quality_weight × (1 + urgency_weight × urgency) × worker_quality
       + diversity_weight × (1 - worker_already_submitted)
```

三个部分：

**质量项**：推荐的 worker 质量越高越好，权重默认 1.0，是主项。

**紧急度加成**：`urgency = 1 / (1 + exp(remaining_days))`，剩余天数越少 urgency 越接近 1，quality 的有效权重动态放大。设计意图是快到 deadline 的任务更需要高质量 worker，不能随便推。

**多样性项**：推荐一个之前没有提交过此 project 的 worker，额外加 0.3 分。避免系统总是推同一批高质量 worker，让更多 worker 有机会参与，同时也符合请求者希望获得多样化回答的需求。

默认权重：`quality_weight=1.0`，`urgency_weight=0.5`，`diversity_weight=0.3`。

### 网络结构

同样使用 `QNetwork`（candidate_scorer 架构），输入维度变为 16，其余结构相同。

### 评估指标

| 指标 | 含义 | 是否主要 |
|------|------|---------|
| avg_worker_quality | 推荐 worker 的平均质量评分 | 主要 |
| diversity_rate | 推荐新 worker（未提交过）的比例 | 主要 |
| urgent_avg_quality | 紧急任务下推荐 worker 的平均质量 | 主要 |
| avg_reward | 综合奖励（质量+多样性+紧急度加权） | 用于保存 checkpoint |
| hit_rate | 推荐命中历史正样本的比例 | 参考，非主要优化目标 |
| invalid_action_rate | 非法动作比例 | 越低越好 |

hit_rate 在 requester 视角是参考指标，不是主要目标——请求者关心的不是"猜中历史上谁来了"，而是"推荐的 worker 质不质量好"。

---

## 训练指令

### 只训练 Worker 目标

```bash
python requester_model/visualize_train.py --variant double_dqn --epochs 5 --lr 1e-3 --log-interval 500 --skip-requester
```

### 只训练 Requester 目标

```bash
python requester_model/visualize_train.py --variant double_dqn --epochs 5 --lr 1e-3 --log-interval 500 --skip-worker
```

### 联合训练（两个目标都跑，生成对比图）

```bash
python requester_model/visualize_train.py --variant double_dqn --epochs 5 --lr 1e-3 --log-interval 500
```

### 三种变体批量对比

```bash
python requester_model/run_all_variants.py --epochs 5 --lr 1e-3
```

---

## 主要参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--variant` | `double_dqn` | `dqn` / `double_dqn` / `dueling_dqn` |
| `--lr` | `1e-3` | 学习率 |
| `--epochs` | `5` | 训练轮数 |
| `--batch-size` | `64` | 批大小 |
| `--buffer-size` | `50000` | 经验回放缓冲区大小 |
| `--min-replay-size` | `1000` | 开始学习前最少积累的样本数 |
| `--target-update-interval` | `500` | target network 同步间隔（步） |
| `--epsilon-start` | `1.0` | 初始探索率 |
| `--epsilon-end` | `0.05` | 最终探索率 |
| `--epsilon-decay-steps` | `20000` | 探索率线性衰减步数 |
| `--hidden-dims` | `256 128` | 隐藏层维度，可传多个值 |
| `--aux-ce-weight` | `1.0` | 辅助交叉熵损失权重（0 = 纯 TD loss） |
| `--gamma` | `0.99` | 折扣因子 |
| `--skip-worker` | — | 跳过 worker 目标，只训练 requester |
| `--skip-requester` | — | 跳过 requester 目标，只训练 worker |
| `--log-interval` | `0` | 每隔多少步打印步级指标（0=不打印） |
| `--train-limit` | `0` | 限制训练样本数（0=全量） |
| `--seed` | `42` | 随机种子 |

---

## 输出文件

每次训练在 `requester_model/runs_vis/<variant>_lr<lr>_seed<seed>_<timestamp>/` 下生成：

```
config.json          # 本次训练超参数
worker_metrics.csv   # worker 目标每 epoch 指标
requester_metrics.csv # requester 目标每 epoch 指标
worker_best.pt       # worker 目标验证集最优 checkpoint
requester_best.pt    # requester 目标验证集最优 checkpoint
training_curves.png  # 训练曲线对比图（hit_rate、reward、quality 等）
test_results.json    # 测试集最终指标
```

---

## 辅助损失（aux_ce_weight）

除了标准的 TD loss，训练时还加了一个辅助交叉熵损失：

```
loss = TD_loss + aux_ce_weight × CE_loss(Q值, 正样本索引)
```

CE loss 直接监督模型把最高 Q 值分配给历史正样本，相当于给 DQN 加了一个模仿学习的信号。在离线数据上这个信号很稳定，能加速收敛，默认权重 1.0。设为 0 则退化为纯 DQN。

---

## Double DQN 的作用

标准 DQN 用同一个网络选动作和估值，容易高估 Q 值导致训练不稳定。Double DQN 用 online network 选动作，用 target network 估值：

```
next_action = argmax(online_net(next_state))   # 选动作
next_q      = target_net(next_state)[next_action]  # 估值
target      = reward + gamma × next_q
```

两个网络分工，有效缓解 Q 值高估问题，在本项目的离线推荐数据上收敛更稳定。
