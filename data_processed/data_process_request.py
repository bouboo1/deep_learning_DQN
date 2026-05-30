"""
众包任务推荐 RL 数据预处理脚本 —— 请求者（Requester）视角
=============================================================
输入：data/ 目录下的原始数据
输出：data_processed_requester/ 目录下的 pkl 文件

与 worker 视角预处理的核心区别：
  本脚本以 project 为中心构建数据集。

  每个 event 对应"某个 project 在某时刻收到一条新 entry"这一交互，
  系统需要从当前活跃的 worker 池中为该 project 推荐最合适的 worker。

  核心字段：
    - candidate_workers：候选 worker 列表（负样本 + 1 个正样本）
    - candidate_features：每个候选 worker 的特征向量
    - positive_worker：实际提交 entry 的 worker（正样本）
    - project_history_at_t：截止当前时刻，该 project 已收到的 worker 列表
      （体现状态转移：随推荐结果变化，project 的"被完成程度"动态更新）

  Episode 组织：
    - 按 project 序列组织，每个 project 的所有 events 按时间排序
    - 一个 episode = 一个 project 从发布到截止期间收到的所有 entry 序列
    - 供 episode 级别的 RL 训练使用（project_episodes.pkl）

奖励设计提示（训练时使用，本脚本只构建数据）：
  最大化请求者利益 → 奖励可定义为：
    r = worker_quality * quality_weight
        + (1 if worker 是新 worker else 0) * diversity_weight
        + remaining_capacity_bonus   # entry 数量不足时推荐权重更高
  具体权重在 RL 训练脚本中调节。
"""

#from __future__ import annotations

import glob
import json
import math
import os
import pickle
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from dateutil.parser import parse
from tqdm import tqdm

# ============================================================
# 路径配置（与 worker 视角保持一致，共用原始数据）
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

PROJECT_DIR   = os.path.join(DATA_DIR, "project")
ENTRY_DIR     = os.path.join(DATA_DIR, "entry")
PROJECT_LIST  = os.path.join(DATA_DIR, "project_list.csv")
WORKER_QUALITY = os.path.join(DATA_DIR, "worker_quality.csv")

# 输出到与 worker 视角相同的目录
SAVE_DIR = os.path.join(BASE_DIR, "data_processed")
os.makedirs(SAVE_DIR, exist_ok=True)

# 每个 project 的候选 worker 集大小（负样本数 + 1 个正样本）
MAX_CANDIDATES = 20
# project 已收到 worker 历史的最大记录长度（截断/padding 到固定长度）
HISTORY_LEN = 10
# 训练/验证/测试划分比例
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.85

# ============================================================
# 特征顺序（请求者视角）
# ============================================================
# 状态特征分两组：
#   Project 特征（描述当前任务本身）
#   Worker 特征（描述候选 worker）
# 拼接后作为 Q 网络的输入

FEATURE_ORDER = [
    # --- Project 侧特征（对每个候选 worker 都相同，作为"上下文"） ---
    "project_category",           # 任务类别（编码后）
    "project_sub_category",       # 任务子类别
    "project_industry",           # 任务行业
    "project_duration_days",      # 任务持续天数（发布周期长短）
    "project_remaining_days",     # 距 deadline 还剩多少天（动态紧迫度）
    "project_current_entry_count",# 截止当前时刻已收到的 entry 数（动态饱和度）
    "project_hist_entry_count",   # 数据集中该 project 最终收到的 entry 总数（静态热度参考）
    "project_fill_rate",          # 已填充率 = current / target（动态）
    # --- Worker 侧特征（描述每个候选 worker） ---
    "worker_quality",             # worker 质量评分
    "worker_history_count",       # worker 历史完成任务总数
    "worker_active_days",         # worker 活跃天数跨度
    "worker_category_match",      # worker 最擅长类别是否匹配该 project
    "worker_industry_match",      # worker 最擅长行业是否匹配该 project
    "worker_category_count",      # worker 在该 project category 上的历史次数
    "worker_already_submitted",   # 该 worker 是否已向该 project 提交过（去重信号）
    "worker_recent_activity",     # worker 最近 7 天活跃次数（短期活跃度）
]

np.random.seed(42)


# ============================================================
# 1. 加载 worker quality
# ============================================================

def load_worker_quality() -> Dict[int, float]:
    print("=" * 60)
    print("1. Load Worker Quality")
    print("=" * 60)
    wq: Dict[int, float] = {}
    with open(WORKER_QUALITY, "r", encoding="utf-8") as f:
        import csv
        for line in csv.reader(f):
            try:
                wid, q = int(line[0]), float(line[1])
                if q > 0:
                    wq[wid] = q / 100.0
            except Exception:
                continue
    print(f"  Valid workers: {len(wq)}")
    return wq


# ============================================================
# 2. 加载 project 信息
# ============================================================

def load_project_info() -> Tuple[Dict[int, Dict[str, Any]], Dict[str, Dict]]:
    print("\n" + "=" * 60)
    print("2. Load Project Info")
    print("=" * 60)

    raw_categories: set    = set()
    raw_sub_categories: set = set()
    raw_industries: set    = set()

    with open(PROJECT_LIST, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # 第一遍：收集所有类别值
    for line in lines:
        try:
            pid = int(line.strip().split(",")[0])
            pfile = os.path.join(PROJECT_DIR, f"project_{pid}.txt")
            with open(pfile, "r", encoding="utf-8") as pf:
                text = json.load(pf)
            raw_categories.add(int(text.get("category", -1)))
            raw_sub_categories.add(int(text.get("sub_category", -1)))
            raw_industries.add(text.get("industry") or "unknown")
        except Exception:
            continue

    category_map     = {v: i for i, v in enumerate(sorted(raw_categories))}
    sub_category_map = {v: i for i, v in enumerate(sorted(raw_sub_categories))}
    industry_map     = {v: i for i, v in enumerate(sorted(raw_industries))}

    # 第二遍：构建 project_info
    project_info: Dict[int, Dict[str, Any]] = {}
    for line in tqdm(lines, desc="projects"):
        try:
            pid = int(line.strip().split(",")[0])
            pfile = os.path.join(PROJECT_DIR, f"project_{pid}.txt")
            with open(pfile, "r", encoding="utf-8") as pf:
                text = json.load(pf)

            start    = parse(text["start_date"])
            deadline = parse(text["deadline"])
            duration_days = max((deadline - start).days, 1)

            # entry_count：该 project 在数据集中实际收到的历史 entry 总数
            # 作为任务热度/受欢迎程度的静态代理特征
            hist_entry_count = int(text.get("entry_count", 0))

            project_info[pid] = {
                "project_id":       pid,
                "category":         category_map[int(text.get("category", -1))],
                "sub_category":     sub_category_map[int(text.get("sub_category", -1))],
                "industry":         industry_map[text.get("industry") or "unknown"],
                "hist_entry_count": hist_entry_count,   # 静态热度参考值
                "start_date":       start,
                "deadline":         deadline,
                "duration_days":    duration_days,
            }
        except Exception:
            continue

    category_maps = {
        "category_map":     {str(k): v for k, v in category_map.items()},
        "sub_category_map": {str(k): v for k, v in sub_category_map.items()},
        "industry_map":     {str(k): v for k, v in industry_map.items()},
    }

    print(f"  Total projects: {len(project_info)}")
    print(f"  category: {len(category_map)}  sub_category: {len(sub_category_map)}  industry: {len(industry_map)}")
    return project_info, category_maps


# ============================================================
# 3. 加载 entry 交互记录
# ============================================================

def load_interactions(project_info: Dict[int, Dict]) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("3. Load Interactions")
    print("=" * 60)
    rows = []
    for pid in tqdm(project_info, desc="entries"):
        for ef in glob.glob(os.path.join(ENTRY_DIR, f"entry_{pid}_*.txt")):
            try:
                with open(ef, "r", encoding="utf-8") as f:
                    text = json.load(f)
                for item in text.get("results", []):
                    try:
                        rows.append({
                            "worker_id":  int(item["author"]),
                            "project_id": pid,
                            "timestamp":  parse(item["entry_created_at"]),
                        })
                    except Exception:
                        continue
            except Exception:
                continue

    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    print(f"  Total interactions: {len(df)}")
    return df


# ============================================================
# 4. 构建 worker 静态特征（全局统计）
# ============================================================

def build_worker_static_features(
    df: pd.DataFrame,
    worker_quality: Dict[int, float],
    project_info: Dict[int, Dict],
) -> Dict[int, Dict[str, Any]]:
    """
    与 worker 视角相同：计算每个 worker 的静态画像。
    这些特征在 project 视角里用来描述候选 worker 的能力。
    """
    print("\n" + "=" * 60)
    print("4. Build Worker Static Features")
    print("=" * 60)
    features: Dict[int, Dict[str, Any]] = {}

    for wid, grp in tqdm(df.groupby("worker_id"), desc="workers"):
        pids       = grp["project_id"].tolist()
        timestamps = grp["timestamp"].tolist()

        cat_counter: Counter = Counter()
        ind_counter: Counter = Counter()
        for pid in pids:
            if pid in project_info:
                cat_counter[project_info[pid]["category"]] += 1
                ind_counter[project_info[pid]["industry"]] += 1

        active_days = 1
        if len(timestamps) >= 2:
            ts_sorted = sorted(timestamps)
            active_days = max((ts_sorted[-1] - ts_sorted[0]).days + 1, 1)

        features[wid] = {
            "quality":            worker_quality.get(wid, 0.5),
            "history_count":      len(pids),
            "active_days":        active_days,
            "favorite_category":  cat_counter.most_common(1)[0][0] if cat_counter else -1,
            "favorite_industry":  ind_counter.most_common(1)[0][0] if ind_counter else -1,
            "category_counts":    dict(cat_counter),
            # 全部时间戳，供后续计算短期活跃度
            "all_timestamps":     sorted(timestamps),
        }

    print(f"  Workers with features: {len(features)}")
    return features


# ============================================================
# 5. 归一化工具
# ============================================================

# 不做归一化的特征（类别/二值特征）
SKIP_NORM = {
    "project_category",
    "project_sub_category",
    "project_industry",
    "worker_category_match",
    "worker_industry_match",
    "worker_already_submitted",
    # 保留原始天数，供 env 用 remaining_days > 0 做过滤
    "project_remaining_days",
    "project_fill_rate",          # 已是 [0,1]，无需 z-score
}

# 长尾计数类特征：先 log1p 再 z-score
LOG1P_FEATURES = {
    "project_duration_days",
    "project_current_entry_count",
    "project_hist_entry_count",
    "worker_history_count",
    "worker_active_days",
    "worker_category_count",
    "worker_recent_activity",
}


def compute_norm_stats(events: List[Dict]) -> Dict[str, Any]:
    """仅在训练集上计算每个特征的 mean/std，避免数据泄露。"""
    accum: Dict[str, List[float]] = defaultdict(list)
    for ev in events:
        for feat in ev["candidate_features"]:
            for k, v in feat.items():
                if k in SKIP_NORM:
                    continue
                v = float(v)
                if k in LOG1P_FEATURES:
                    v = math.log1p(max(v, 0.0))
                accum[k].append(v)
    stats = {}
    for k, vals in accum.items():
        arr = np.array(vals)
        stats[k] = {"mean": float(arr.mean()), "std": float(max(arr.std(), 1e-8))}
    return stats


def apply_norm(events: List[Dict], stats: Dict[str, Any]) -> List[Dict]:
    """原地归一化 candidate_features。"""
    for ev in events:
        for feat in ev["candidate_features"]:
            for k in list(feat.keys()):
                if k in SKIP_NORM or k not in stats:
                    continue
                v = feat[k]
                if k in LOG1P_FEATURES:
                    v = math.log1p(max(v, 0.0))
                feat[k] = (v - stats[k]["mean"]) / stats[k]["std"]
    return events


# ============================================================
# 6. 构建 RL events —— 请求者（Project）视角
# ============================================================

def build_events(
    df: pd.DataFrame,
    project_info: Dict[int, Dict],
    worker_static: Dict[int, Dict],
) -> List[Dict]:
    """
    遍历按时间排序的交互记录，以 project 为中心构建 RL event。

    每个 event 的含义：
      "时刻 t，project pid 收到一条新 entry，系统需要从活跃 worker 池中
       推荐最合适的 worker 来做这个任务（最大化请求者利益）"

    状态（State）：
      project 的静态属性 + project_history_at_t（已收到的 worker 列表）
      + 候选 worker 各自的特征

    动作（Action）：
      从候选 worker 列表中选择一个推荐

    正样本（Ground Truth）：
      实际提交了 entry 的 worker（wid_pos）

    状态转移：
      project_history_at_t 在每次 event 后追加当前 positive worker，
      使得下一个 event 的状态会因推荐结果不同而不同。

    奖励设计（本脚本不计算奖励，留给 RL 训练脚本）：
      可参考：r = worker_quality（主要项）
               + diversity_bonus（推荐新 worker）
               + urgency_bonus（deadline 越近奖励越高）
    """
    print("\n" + "=" * 60)
    print("5. Build RL Events (Project / Requester Perspective)")
    print("=" * 60)

    # 维护每个 project 的动态已接受 worker 历史
    project_dynamic_history: Dict[int, List[int]] = defaultdict(list)
    # 维护每个 project 的动态当前 entry 计数（动态饱和度）
    project_current_entry_count: Dict[int, int] = defaultdict(int)

    # 用于计算 worker 短期活跃度（最近 7 天活跃次数）的辅助结构
    # worker_id -> 按时间排序的 timestamp 列表
    worker_timestamp_list: Dict[int, List] = defaultdict(list)

    all_wids = list(worker_static.keys())
    # active_workers 是固定集合，提前算好，避免每条 event 重建
    active_workers_base = all_wids  # 全量 worker 都视为活跃

    # 7 天活跃度：用滑动窗口计数器，避免每条 event 遍历全部历史时间戳
    # worker_id -> deque of timestamps（只保留最近 7 天内的）
    from collections import deque
    worker_recent: Dict[int, deque] = defaultdict(deque)
    # 预计算每个 worker 的 7 天活跃计数（当前窗口内的数量）
    worker_recent_count: Dict[int, int] = defaultdict(int)

    events: List[Dict] = []

    for row in tqdm(df.itertuples(), total=len(df), desc="events"):
        wid_pos = row.worker_id
        pid     = row.project_id
        t       = row.timestamp

        # 更新 wid_pos 的滑动窗口（在处理本条 event 之前先更新，保证"截止当前时刻"语义）
        # 先清理过期时间戳
        dq = worker_recent[wid_pos]
        while dq and (t - dq[0]).days > 7:
            dq.popleft()
            worker_recent_count[wid_pos] -= 1

        if pid not in project_info or wid_pos not in worker_static:
            project_dynamic_history[pid].append(wid_pos)
            project_current_entry_count[pid] += 1
            # 记录时间戳后再继续
            dq.append(t)
            worker_recent_count[wid_pos] += 1
            continue

        proj = project_info[pid]

        negatives = [w for w in active_workers_base if w != wid_pos]

        if len(negatives) > MAX_CANDIDATES - 1:
            proj_cat = proj["category"]
            proj_ind = proj["industry"]

            # 批量计算负样本得分，不在 score_worker 里遍历时间戳
            scores = np.array([
                worker_static[w]["quality"]
                + 1.0 * int(worker_static[w]["favorite_category"] == proj_cat)
                + 0.5 * int(worker_static[w]["favorite_industry"] == proj_ind)
                + min(worker_recent_count[w] / 5.0, 1.0)
                + np.random.uniform(0, 0.1)
                for w in negatives
            ])
            probs = scores / scores.sum()
            chosen_idx = np.random.choice(len(negatives), size=MAX_CANDIDATES - 1, replace=False, p=probs)
            negatives = [negatives[i] for i in chosen_idx]

        final_candidates = negatives + [wid_pos]
        np.random.shuffle(final_candidates)
        positive_index = final_candidates.index(wid_pos)

        remaining_days   = max((proj["deadline"] - t).days, 0)
        curr_entry_count = project_current_entry_count[pid]
        hist_entry_count = max(proj["hist_entry_count"], 1)
        fill_rate        = min(curr_entry_count / hist_entry_count, 1.0)
        submitted_set    = set(project_dynamic_history[pid])

        candidate_features = []
        for cw in final_candidates:
            ws = worker_static[cw]
            worker_cat_count  = ws["category_counts"].get(proj["category"], 0)
            recent_7d         = worker_recent_count[cw]   # 直接查计数器，O(1)
            cat_match         = int(ws["favorite_category"] == proj["category"])
            ind_match         = int(ws["favorite_industry"] == proj["industry"])
            already_submitted = int(cw in submitted_set)

            candidate_features.append({
                "project_category":            float(proj["category"]),
                "project_sub_category":        float(proj["sub_category"]),
                "project_industry":            float(proj["industry"]),
                "project_duration_days":       float(proj["duration_days"]),
                "project_remaining_days":      float(remaining_days),
                "project_current_entry_count": float(curr_entry_count),
                "project_hist_entry_count":    float(proj["hist_entry_count"]),
                "project_fill_rate":           float(fill_rate),
                "worker_quality":              float(ws["quality"]),
                "worker_history_count":        float(ws["history_count"]),
                "worker_active_days":          float(ws["active_days"]),
                "worker_category_match":       float(cat_match),
                "worker_industry_match":       float(ind_match),
                "worker_category_count":       float(worker_cat_count),
                "worker_already_submitted":    float(already_submitted),
                "worker_recent_activity":      float(recent_7d),
            })

        # project_history_at_t：截止当前时刻已收到的 worker 列表（最近 HISTORY_LEN 条）
        history_snapshot = list(project_dynamic_history[pid])[-HISTORY_LEN:]

        events.append({
            "project_id":           pid,
            "timestamp":            t,
            "candidate_workers":    final_candidates,
            "candidate_features":   candidate_features,
            "positive_worker":      wid_pos,
            "positive_index":       positive_index,
            # 动态历史状态（体现状态转移）
            "project_history_at_t": history_snapshot,
            # 以下字段供奖励计算使用（训练脚本直接读取，无需二次查询）
            "positive_worker_quality":   float(worker_static[wid_pos]["quality"]),
            "project_remaining_days":    float(remaining_days),
            "project_fill_rate":         float(fill_rate),
        })

        # 当前交互完成后，更新动态状态
        project_dynamic_history[pid].append(wid_pos)
        project_current_entry_count[pid] += 1
        # 更新滑动窗口
        dq.append(t)
        worker_recent_count[wid_pos] += 1

    print(f"  Total events: {len(events)}")
    return events


# ============================================================
# 7. 按 project 序列组织 episodes
# ============================================================

def build_project_episodes(events: List[Dict]) -> Dict[int, List[Dict]]:
    """
    把 events 按 project_id 分组，每个 project 的所有 events 按时间排序，
    形成一个完整的 episode。

    episode 含义：
      - 一个 episode = 一个 project 从发布到截止期间收到的所有 entry 序列
      - episode 内相邻 step 之间有真正的状态转移
        （project_history_at_t 随每次推荐结果更新，fill_rate 动态变化）
      - 这是请求者视角 RL 训练的基本单元

    终止条件（供训练脚本参考）：
      - project_remaining_days == 0（到达 deadline）
      - project_fill_rate >= 1.0（entry 数量已满）
    """
    episodes: Dict[int, List[Dict]] = defaultdict(list)
    for ev in events:
        episodes[ev["project_id"]].append(ev)
    # events 已按时间排序，episode 内顺序天然正确
    return dict(episodes)


# ============================================================
# 8. 划分训练/验证/测试集
# ============================================================

def split_events(events: List[Dict]):
    """按时间顺序划分，保证测试集时间最晚（模拟真实部署场景）。"""
    n = len(events)
    train_end = int(n * TRAIN_RATIO)
    val_end   = int(n * VAL_RATIO)
    return events[:train_end], events[train_end:val_end], events[val_end:]


# ============================================================
# 主流程
# ============================================================

def main():
    worker_quality = load_worker_quality()
    project_info, category_maps = load_project_info()
    df = load_interactions(project_info)
    worker_static = build_worker_static_features(df, worker_quality, project_info)

    events = build_events(df, project_info, worker_static)

    # 先按时间划分，再做归一化（避免数据泄露）
    train_events, val_events, test_events = split_events(events)
    print(f"\n  Train: {len(train_events)}  Val: {len(val_events)}  Test: {len(test_events)}")

    # 在训练集上计算归一化参数
    print("\n" + "=" * 60)
    print("6. Normalize Features")
    print("=" * 60)
    norm_stats = compute_norm_stats(train_events)
    train_events = apply_norm(train_events, norm_stats)
    val_events   = apply_norm(val_events,   norm_stats)
    test_events  = apply_norm(test_events,  norm_stats)

    # 构建 project episodes（基于全量 events，归一化后）
    all_events = train_events + val_events + test_events
    project_episodes = build_project_episodes(all_events)
    print(f"  Project episodes: {len(project_episodes)}")

    # --------------------------------------------------------
    # 保存
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("7. Save")
    print("=" * 60)

    def save(obj, name):
        path = os.path.join(SAVE_DIR, name)
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        size = len(obj) if hasattr(obj, "__len__") else "?"
        print(f"  Saved {name}  ({size} records)")

    save(train_events,    "requester_train.pkl")
    save(val_events,      "requester_val.pkl")
    save(test_events,     "requester_test.pkl")
    save(project_episodes, "project_episodes.pkl")
    save(worker_static,   "worker_features.pkl")  # 与 worker 视角共享，方便联合训练

    # 归一化参数
    norm_path = os.path.join(SAVE_DIR, "norm.json")
    with open(norm_path, "w", encoding="utf-8") as f:
        json.dump(norm_stats, f, ensure_ascii=False, indent=2)
    print(f"  Saved norm.json")

    # 类别映射表
    maps_path = os.path.join(SAVE_DIR, "category_maps.json")
    with open(maps_path, "w", encoding="utf-8") as f:
        json.dump(category_maps, f, ensure_ascii=False, indent=2)
    print(f"  Saved category_maps.json")

    # 数据集信息（供报告引用）
    info = {
        "perspective":          "requester (project-centric)",
        "train_size":           len(train_events),
        "val_size":             len(val_events),
        "test_size":            len(test_events),
        "num_projects":         len(project_episodes),
        "max_candidates":       MAX_CANDIDATES,
        "history_len":          HISTORY_LEN,
        "feature_order":        FEATURE_ORDER,
        "feature_dim":          len(FEATURE_ORDER),
        "reward_hint": (
            "r = positive_worker_quality (main) "
            "+ diversity_bonus (新 worker) "
            "+ urgency_bonus (remaining_days 越小奖励越高)"
        ),
    }
    with open(os.path.join(SAVE_DIR, "info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print(f"  Saved info.json")

    print("\nDONE")
    print(f"Output: {SAVE_DIR}")


if __name__ == "__main__":
    main()