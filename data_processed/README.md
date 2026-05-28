# 数据预处理说明

## 1. 运行方式

```bash
#cd /Users/tangyi/Desktop/强化学习/强化学习
python process.py
```

输入：`data/` 目录下的原始数据（entry/、project/、project_list.csv、worker_quality.csv）  
输出：`data_processed/` 目录下的所有文件

---

## 2. 输出文件说明

| 文件 | 内容 |
|------|------|
| `enhanced_train.pkl` | 训练集 event 列表（按时间前 70%） |
| `enhanced_val.pkl` | 验证集 event 列表（70%~85%） |
| `enhanced_test.pkl` | 测试集 event 列表（后 15%） |
| `worker_episodes.pkl` | 按 worker 分组的完整序列，用于 episode 级 RL 训练（目前还是按照时间轴构建训练的，暂时不用这个pkl） |
| `worker_features.pkl` | worker 静态特征字典 |
| `norm.json` | 特征归一化参数（mean/std），仅从训练集计算，用了z-score归一化，对于之前说的那个可能有绝对值较大的数值引导导致训练受这些数值影响的问题，对于长尾分布的数据先做了log再归一化 |
| `category_maps.json` | 类别特征的原始值 → 连续整数映射表 |
| `info.json` | 数据集规模、特征维度等元信息 |

---

## 3. 单条样本结构（train/val/test.pkl 中每个元素）

```python
{
    "worker_id":          int,           # worker 唯一标识
    "timestamp":          datetime,      # worker 到达时刻
    "candidate_projects": List[int],     # 候选项目 ID 列表（最多 20 个）
    "candidate_features": List[Dict],    # 每个候选项目的特征向量（见第4节）
    "positive_project":   int,           # worker 历史上实际选择的项目 ID
    "positive_index":     int,           # positive_project 在 candidate_projects 中的下标
    "worker_history_at_t": List[int],    # 截止当前时刻该 worker 已完成的项目列表（最近10条）
}
```

`positive_index` 是 RL 的监督信号：`reward = 1 if action == positive_index else 0`

`worker_history_at_t` 是状态转移的关键：命中后下一步的历史会更新，使 next_state 真正依赖当前动作。

---

## 4. candidate_features 结构

`candidate_features[i]` 表示 worker 在当前时刻对第 i 个候选项目的匹配特征。除类别/二值特征和 `remaining_days` 外，连续特征已归一化：

```python
{
    # Worker 特征
    "worker_quality":        float,  # worker 历史完成质量（来自 worker_quality.csv）
    "worker_history_count":  float,  # worker 截止当前时刻的历史完成数（动态）
    "worker_active_days":    float,  # worker 活跃时间跨度（天）

    # Project 特征
    "project_category":      float,  # 项目类别（连续整数编码，见第6节）
    "project_sub_category":  float,  # 项目子类别（连续整数编码）
    "project_industry":      float,  # 项目行业（连续整数编码）
    "project_entry_count":   float,  # 项目历史参与人数
    "project_duration_days": float,  # 项目生命周期（天）

    # 匹配特征
    "category_match":        float,  # worker 偏好类别与项目类别是否匹配（0/1）
    "industry_match":        float,  # worker 偏好行业与项目行业是否匹配（0/1）

    # 统计特征
    "project_popularity":    float,  # 项目流行度 = entry_count / duration_days

    # 动态特征（新增，体现 RL 状态转移）
    "worker_category_count": float,  # worker 在该项目 category 上的历史完成次数（截止当前时刻）
    "remaining_days":        float,  # deadline - worker 到达时刻（原始天数，不归一化）
}
```

特征总维度：**13**

---

## 5. 归一化说明

除类别/二值特征和 `remaining_days` 外，连续特征做 z-score 标准化：`(x - mean) / std`

- 归一化参数仅从**训练集**计算，验证集和测试集使用相同参数，避免数据泄露
- 参数保存在 `norm.json`，格式为 `{特征名: {"mean": float, "std": float}}`
- `remaining_days` 保留原始天数，用于环境中判断任务是否过期：`remaining_days > 0` 表示仍在 deadline 前

> 注意：归一化后特征值可能为负，这是正常的，不是错误。如需还原原始值：`x_orig = x_norm * std + mean`

---

## 6. 类别特征编码说明

`project_category`、`project_sub_category`、`project_industry` 均为类别型特征，已映射为从 0 开始的连续整数，映射表保存在 `category_maps.json`。

映射规则：对所有出现的原始值排序后依次编号，保证每次运行结果一致。

```json
{
  "category_map":     {"2": 0, "3": 1, "5": 2, "6": 3, "7": 4, "9": 5, "10": 6},
  "sub_category_map": {"2": 0, "3": 1, ...},
  "industry_map":     {"Advertising": 0, "Architecture": 1, ...}
}
```

**注意**：连续整数编码只是让编号更紧凑，类别之间没有大小关系。如果模型使用 MLP，建议对这三个特征做 embedding 或 one-hot 编码后再与连续特征拼接，而不是直接当数值使用。

---

## 7. 数据集划分

按时间顺序划分，避免未来信息泄露：

| 划分 | 比例 | 说明 |
|------|------|------|
| train | 前 70% | 用于训练，归一化参数从此计算 |
| val | 70%~85% | 用于验证集评估和模型选择 |
| test | 后 15% | 最终测试，训练过程中不可见 |

---

## 8. worker_episodes.pkl 说明（目前按照时间构建，不用这个）

```python
worker_episodes: Dict[int, List[Dict]]
# key: worker_id
# value: 该 worker 按时间排序的所有 event 列表（一个完整 episode）
```

用于 episode 级别的 RL 训练：

```python
from env.crowd_env import CrowdRecEnv, load_worker_episodes

episodes = load_worker_episodes()
# 取一个 worker 的完整序列作为一个 episode
env = CrowdRecEnv(episodes[worker_id], reward_type="worker")
state = env.reset()
```

---

## 9. RL 建模说明

| 要素 | 定义 |
|------|------|
| 状态 s_t | worker 动态历史（worker_history_at_t）+ 当前候选任务特征矩阵 |
| 动作 a_t | 从候选集中选一个任务下标，`a ∈ [0, max_candidates-1]` |
| 奖励 r_t（参与者）| `1 if action == positive_index else 0` |
| 奖励 r_t（请求者）| 命中后乘以 worker_quality 和紧迫度加权 |
| 状态转移 | 命中时 worker_history_at_t 更新，worker_category_count 和 remaining_days 随时间变化 |
| episode 结束 | 该 worker 的所有访问记录走完 |

---

## 10. worker_features.pkl 说明

```python
worker_features[worker_id] = {
    "quality":            float,  # 来自 worker_quality.csv，已归一化到 [0,1]
    "history_count":      int,    # 全局历史完成数
    "active_days":        int,    # 活跃时间跨度（天）
    "favorite_category":  int,    # 历史完成最多的 category（连续整数编码后）
    "favorite_industry":  int,    # 历史完成最多的 industry（连续整数编码后）
    "category_counts":    Dict,   # 各 category 的历史完成次数
}
```
