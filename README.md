# 众包任务推荐强化学习项目说明

这个项目的核心目标，是把预处理后的众包推荐数据接入强化学习环境，再用 DQN 系列方法去学习“推荐哪个候选项目”的决策策略。当前仓库里主要分成四部分：环境、单目标训练、可视化训练、以及多变体对比实验。

## 项目在做什么

训练过程不是通用的图像分类或文本生成，而是基于 `data_processed_all/data_processed` 中已经处理好的样本进行决策学习。每条样本都表示一次 worker 到达平台后的推荐决策：

- 输入是当前 worker 的候选项目特征矩阵。
- 动作是从候选项目中选一个。
- 奖励根据 `reward_type` 计算，可站在 worker、requester、混合目标等不同视角。
- 每次 `step()` 会推进到下一条样本，直到一个 split 结束。

## 主要脚本

- `env/crowd_env.py`：定义 `CrowdRecEnv` 环境和 `load_split()` 数据读取函数。
- `env/demo_random.py`：随机合法动作的 smoke test，用来确认环境是否能跑通。
- `requester_dqn.py`：Requester 侧的单目标 DQN 训练脚本。
- `visualize_train.py`：同时训练 worker 和 requester 两个目标，并保存训练曲线、测试结果和对比图。
- `run_all_variants.py`：批量运行 `dqn`、`double_dqn`、`dueling_dqn` 三个变体，并生成汇总图。
- `plot_from_csv.py`：从已有 `metrics.csv` 重新生成训练曲线。

## 如何运行

### 快速训练（验证流程）

```bash
python DQN/train.py \
  --variant double_dqn \
  --reward-type worker \
  --epochs 1 \
  --train-limit 2000
```

### 完整学习率扫描实验

```bash
python DQN/run_experiments.py \
  --variant double_dqn \
  --reward-type worker \
  --epochs 5
# 自动扫描 lr ∈ {0.01, 0.001, 0.0001}，结果写入 DQN/runs_lr_sweep/
```

### 测试集评估

```bash
python DQN/evaluate.py \
  --checkpoint DQN/runs_lr_sweep/double_dqn_worker_lr0.001_seed42/best_model.pt \
  --split test \
  --reward-type worker
```

### 可视化训练过程

```bash
python 强化学习/visualize_train.py --epochs 5 --variant double_dqn
```

### 三种变体对比

```bash
python 强化学习/run_all_variants.py --epochs 5
```

---

## 实验结果（Double DQN，Worker 奖励，5 epochs，完整训练集）

| 学习率 | 最优验证集奖励（Hit Rate） |
|--------|--------------------------|
| 0.01   | 0.218（2 epochs，未收敛） |
| 0.001  | 待补充（需重跑完整实验）   |
| 0.0001 | 待补充（需重跑完整实验）   |

> 注：lr=0.001 和 lr=0.0001 的已有结果仅基于 1000 条样本的 1 epoch，与 lr=0.01 的完整实验不可比，需在相同条件下重跑。

随机策略基线（理论值）：hit_rate ≈ 1/21 ≈ 0.048（候选集平均大小 21）

---

## 关键超参数

| 参数 | 默认值 |
|------|--------|
| batch_size | 64 |
| buffer_size | 50,000 |
| epsilon 起始 / 终止 | 1.0 / 0.05 |
| epsilon 衰减步数 | 20,000 |
| target network 更新间隔 | 500 steps |
| 隐藏层维度 | [256, 128] |
| gamma | 0.0（单步 episode，无折扣） |
| aux_ce_weight | 1.0（辅助 CE loss 权重） |

---

## 输出文件说明

每次训练在 `runs/` 或 `runs_lr_sweep/` 下生成一个子目录，包含：

```
<run_name>/
├── config.json       # 本次训练超参数
├── metrics.csv       # 每 epoch 的 loss、reward、hit_rate 等
├── summary.json      # 最优验证集指标摘要
├── best_model.pt     # 验证集最优 checkpoint
└── last_model.pt     # 最后一个 epoch checkpoint
```

---

## 当前问题与改进方向

### 成员 A — 数据清洗与特征工程

**问题 1（最高优先级）：归一化未接入环境**

`norm.json` 存在但从未被加载。`crowd_env.py` 的 `iter_feature_vectors` 直接读取原始值，导致 `worker_history_count`（均值 647，标准差 624）与 `category_match`（0/1 二值）在同一特征矩阵中共存，MLP 梯度被大数值特征主导，小特征几乎学不到东西。

需要做的：
- 在 `crowd_env.py` 的 `__init__` 里加载 `norm.json`
- 在 `iter_feature_vectors` 里对每个特征做 `(x - mean) / std`
- `project_popularity` 没有 norm 条目，单独做 min-max 归一化

**问题 2：缺少分布对比图脚本**

任务要求保存归一化前后的分布对比图，但目前没有任何绘图脚本。需要对每个特征画归一化前后的直方图（两列对比），保存为 PNG。

**问题 3：候选列表是静态的，未动态过滤已满任务**

当前候选列表在预处理时固定，没有根据 worker 到达时刻动态计算任务剩余名额。任务被选满后应将 `action_mask` 置 0。需要在预处理阶段按时间顺序计算每个任务在每个时间点的剩余名额，并写入样本数据。

**问题 4：交叉特征信息量不足**

`category_match` 和 `industry_match` 是粗粒度 0/1 匹配，可以增加：
- `worker_category_count`：worker 在该 category 的历史完成次数（专项能力）
- `worker_recent_active`：近 30 天活跃度（比 `worker_active_days` 更有区分度）
- `task_remaining_days`：`deadline - entry_created_at`（动态紧迫度，比静态 `duration_days` 更准确）

---

### 成员 B — 环境 step 函数与实时 R_t 输出

**问题 1：R_t 只有 epoch 级别，缺少 step 级别日志**

训练循环每步都有 `reward`，但只在 epoch 结束时打印汇总 JSON。需要在 `train.py` 的训练循环内加 step 级别日志：

```python
# 每 log_interval 步打印一次
if global_step % log_interval == 0:
    print(f"step={global_step} reward={reward:.4f} epsilon={agent.epsilon:.3f}")
```

或写入 `step_rewards.csv`，用于后续绘制 R_t 曲线。

**问题 2：`requester_urgency` 奖励类型未暴露在 CLI**

`crowd_env.py` 实现了 `reward_type="requester_urgency"`，但 `train.py` 的 `--reward-type` 参数只有 `["worker", "requester", "hybrid"]`，缺少 `requester_urgency`，导致该奖励类型无法通过命令行使用。

**问题 3：urgency 特征是静态的，应改为动态计算**

当前用任务总时长近似紧迫度：

```python
# 当前（静态）
urgency = 1 / (1 + exp(duration_days))
```

应改为用 worker 到达时的剩余时间：

```python
# 改进（动态）
remaining_days = (deadline - worker_arrival_time).days
urgency = 1 / (1 + exp(remaining_days))
```

需要 A 在预处理时保留 deadline 字段，B 在 `_requester_urgency_reward` 里读取 `sample["timestamp"]` 和任务 deadline 计算动态剩余天数。

**问题 4：step 函数未处理"任务已满"情况**

`valid_action` 判断只检查 `0 <= action < num_candidates`，没有检查任务剩余名额。待 A 在数据里加入动态剩余名额后，需要在 `_build_state` 的 `action_mask` 构建逻辑里加入过滤：

```python
for i, proj in enumerate(sample["candidate_projects"]):
    if proj.get("remaining_slots", 1) > 0:
        action_mask[i] = 1.0
```

**A 和 B 的协作依赖**

| B 需要的 | 依赖 A 做的 |
|---------|-----------|
| 动态 urgency 特征 | A 在预处理时保留 deadline 字段 |
| 任务剩余名额过滤 | A 按时间计算每个任务的动态剩余名额并写入数据 |
| 归一化后的特征 | A 把 norm 接入环境，B 的实验才有意义 |

---

### 成员 C & D — DQN 实现与参数调优

**问题 1（最高优先级）：三组实验条件不一致，结果不可比**

当前 `runs_lr_sweep/` 中：
- `lr=0.01`：完整训练集，5 epochs，134k steps/epoch
- `lr=0.001`：仅 1000 条样本，1 epoch，1 train step
- `lr=0.0001`：仅 1000 条样本，1 epoch，1 train step

三组实验用了不同的数据量和 epoch 数，`sweep_summary.json` 里的数字完全不可比。需要在相同条件下重跑全部三组（完整训练集，5 epochs）。

**问题 2：缺少 Loss 曲线图**

报告需要不收敛（lr=0.01）和收敛（lr=0.0001）的 Loss 曲线截图，但 `runs_lr_sweep/` 里只有 `.pt` 和 `metrics.csv`，没有任何 PNG。需要写一个绘图脚本，从 `metrics.csv` 读取 `loss` 列，画出各 lr 的 Loss 曲线并保存为 PNG。

参考命令：

```bash
python DQN/plot_loss.py \
  --runs DQN/runs_lr_sweep/double_dqn_worker_lr0.01_seed42 \
         DQN/runs_lr_sweep/double_dqn_worker_lr0.001_seed42 \
         DQN/runs_lr_sweep/double_dqn_worker_lr0.0001_seed42 \
  --output DQN/runs_lr_sweep/loss_curves.png
```

**问题 3：调参维度单一，只扫了学习率**

`run_experiments.py` 只扫描学习率，但老师要求至少 3 个维度的调参实验。建议补充：

| 调参维度 | 建议取值 | 预期效果 |
|---------|---------|---------|
| `batch_size` | 32 / 64 / 128 | 影响梯度方差和训练稳定性 |
| `epsilon_decay_steps` | 5000 / 20000 / 50000 | 影响探索充分性 |
| `aux_ce_weight` | 0.0 / 0.5 / 1.0 | 控制监督信号强度，0 为纯 DQN |
| `hidden_dims` | [128] / [256,128] / [256,256,128] | 网络容量 |

**问题 4：三种 DQN 变体未在相同条件下对比**

`run_all_variants.py` 存在但没有对比结果记录。需要在相同超参数下跑 dqn / double_dqn / dueling_dqn 三组，记录各自的 `eval_hit_rate` 和 `loss` 曲线，写入报告。

**问题 5：Dueling DQN 用的是 FlatQNetwork，与其他变体架构不一致**

`agent.py` 中 `dueling_dqn` 使用 `DuelingQNetwork`（flatten 输入），而 `dqn` 和 `double_dqn` 使用 `QNetwork`（candidate_scorer，参数共享）。两种架构的输入处理方式不同，对比实验时需要说明这一差异，或统一架构后再对比。

---

### 成员 E & 全组 — Baseline 对比与报告

**问题 1：随机策略基线未正式记录**

`demo_random.py` 存在但没有把结果写入文件。需要正式运行并记录：

```bash
python 强化学习/env/demo_random.py --split test --reward-type worker
# 预期 hit_rate ≈ 0.048（1/21）
```

将结果与 DQN 的 `eval_hit_rate` 对比，写入报告的对比表格。

**问题 2：DQN 相对随机策略的提升幅度需要重新评估**

当前 lr=0.01 的完整实验 `eval_hit_rate=0.218`，远高于随机基线 0.048，但这个结果是在**未归一化特征**上得到的。A 修复归一化后，需要重新跑实验，确认提升是否来自模型学习而非特征尺度问题。

**问题 3：报告章节"强化学习训练过程分析与不收敛问题排查"需要真实数据支撑**

该章节需要：
1. 不收敛的 Loss 曲线（lr=0.01，loss 在 2.6 附近震荡，未下降）
2. 收敛的 Loss 曲线（lr=0.0001，需重跑完整实验）
3. 不收敛原因分析：学习率过大导致 Q 值震荡，loss 无法稳定下降
4. 解决方案：降低学习率 + 归一化特征（A 的工作）

**问题 4：报告中需说明离线 RL 的局限性**

当前环境基于历史日志数据构造，动作不会真正改变后续候选集分布，更接近离线推荐实验而非完全在线交互式强化学习。报告中应明确说明这一点，将结论表述为"在历史候选集上的命中奖励提升"。

---