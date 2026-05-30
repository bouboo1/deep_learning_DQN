"""
众包任务推荐 RL 数据预处理脚本
================================
输入：data/ 目录下的原始数据
输出：data_processed/ 目录下的 pkl 文件
使用的时候注意输入输出路径可能要修改一下

与旧版预处理的核心区别：
  每个 event 新增 worker_history_at_t 字段，记录截止当前时刻
  该 worker 已完成的项目列表（按时间排序）。这样 next_state 会
  因为推荐结果不同而不同，环境具备真正的状态转移。

  同时按 worker 序列组织 episodes，输出 worker_episodes.pkl，
  供 episode 级别的 RL 训练使用。
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
# 路径配置
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

PROJECT_DIR = os.path.join(DATA_DIR, "project")
ENTRY_DIR = os.path.join(DATA_DIR, "entry")
PROJECT_LIST = os.path.join(DATA_DIR, "project_list.csv")
WORKER_QUALITY = os.path.join(DATA_DIR, "worker_quality.csv")

SAVE_DIR = os.path.join(BASE_DIR, "data_processed")
os.makedirs(SAVE_DIR, exist_ok=True)

# 每个 worker 的候选集大小（负样本数 + 1 个正样本）
MAX_CANDIDATES = 20
# worker 历史特征的最大长度（截断/padding 到固定长度）
HISTORY_LEN = 10
# 训练/验证/测试划分比例
TRAIN_RATIO = 0.70
VAL_RATIO = 0.85

FEATURE_ORDER = [
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
    # 新增：动态特征
    "worker_category_count",   # worker 在该 category 的历史完成次数
    "remaining_days",          # deadline - worker 到达时间（动态紧迫度）
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
    """返回 (project_info, category_maps)。

    category_maps 包含三个映射表，保存到 category_maps.json 供报告引用：
      - category_map:     原始 category 值 -> 连续整数
      - sub_category_map: 原始 sub_category 值 -> 连续整数
      - industry_map:     原始 industry 字符串 -> 连续整数
    """
    print("\n" + "=" * 60)
    print("2. Load Project Info")
    print("=" * 60)

    # 第一遍：收集所有出现的类别值，保证映射有序且可复现
    raw_categories: set = set()
    raw_sub_categories: set = set()
    raw_industries: set = set()

    with open(PROJECT_LIST, "r", encoding="utf-8") as f:
        lines = f.readlines()

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

    # 排序后建映射，保证每次运行结果一致
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

            project_info[pid] = {
                "project_id":    pid,
                "category":      category_map[int(text.get("category", -1))],
                "sub_category":  sub_category_map[int(text.get("sub_category", -1))],
                "industry":      industry_map[text.get("industry") or "unknown"],
                "entry_count":   int(text.get("entry_count", 0)),
                "start_date":    start,
                "deadline":      deadline,
                "duration_days": duration_days,
            }
        except Exception:
            continue

    category_maps = {
        "category_map":     {str(k): v for k, v in category_map.items()},
        "sub_category_map": {str(k): v for k, v in sub_category_map.items()},
        "industry_map":     {str(k): v for k, v in industry_map.items()},
    }

    print(f"  Total projects: {len(project_info)}")
    print(f"  category values: {len(category_map)}  sub_category: {len(sub_category_map)}  industry: {len(industry_map)}")
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
                            "worker_id": int(item["author"]),
                            "project_id": pid,
                            "timestamp": parse(item["entry_created_at"]),
                        })
                    except Exception:
                        continue
            except Exception:
                continue

    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    print(f"  Total interactions: {len(df)}")
    return df


# ============================================================
# 4. 构建 worker 静态特征（全局统计，不随时间变化）
# ============================================================

def build_worker_static_features(
    df: pd.DataFrame,
    worker_quality: Dict[int, float],
    project_info: Dict[int, Dict],
) -> Dict[int, Dict[str, Any]]:
    print("\n" + "=" * 60)
    print("4. Build Worker Static Features")
    print("=" * 60)
    features: Dict[int, Dict[str, Any]] = {}

    for wid, grp in tqdm(df.groupby("worker_id"), desc="workers"):
        pids = grp["project_id"].tolist()
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
            "quality": worker_quality.get(wid, 0.5),
            "history_count": len(pids),
            "active_days": active_days,
            "favorite_category": cat_counter.most_common(1)[0][0] if cat_counter else -1,
            "favorite_industry": ind_counter.most_common(1)[0][0] if ind_counter else -1,
            "category_counts": dict(cat_counter),
        }

    print(f"  Workers with features: {len(features)}")
    return features


# ============================================================
# 5. 计算归一化参数（仅在训练集上计算）
# ============================================================

def compute_norm_stats(events: List[Dict]) -> Dict[str, Any]:
    """从训练集 candidate_features 计算每个特征的 mean/std。
    LOG1P_FEATURES 先做 log1p 再统计，保证 apply_norm 时用的统计量一致。
    """
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


SKIP_NORM = {
    "project_category",
    "project_sub_category",
    "project_industry",
    "category_match",
    "industry_match",
    # 保留原始天数，供 env 用 remaining_days > 0 做 deadline 过滤。
    "remaining_days",
}

# 计数类长尾特征：先 log1p 再 z-score
LOG1P_FEATURES = {
    "worker_history_count",
    "worker_active_days",
    "project_entry_count",
    "project_duration_days",
    "worker_category_count",
}


def apply_norm(events: List[Dict], stats: Dict[str, Any]) -> List[Dict]:
    """原地归一化 candidate_features。
    - LOG1P_FEATURES：先 log1p 再 z-score
    - SKIP_NORM：类别/二值特征，跳过
    - 其余连续特征：直接 z-score
    """
    for ev in events:
        for feat in ev["candidate_features"]:
            for k in list(feat.keys()):
                if k in SKIP_NORM:
                    continue
                if k not in stats:
                    continue
                v = feat[k]
                if k in LOG1P_FEATURES:
                    v = math.log1p(max(v, 0.0))
                feat[k] = (v - stats[k]["mean"]) / stats[k]["std"]
    return events


# ============================================================
# 6. 构建 RL events（核心：加入 worker_history_at_t）
# ============================================================

def build_events(
    df: pd.DataFrame,
    project_info: Dict[int, Dict],
    worker_static: Dict[int, Dict],
) -> List[Dict]:
    """
    遍历按时间排序的交互记录，为每条记录构建一个 RL event。

    关键改动：
      - worker_history_at_t：截止当前时刻，该 worker 已完成的项目列表
        （不含当前这条，因为当前动作还没发生）。
      - candidate_features 新增两个动态特征：
          worker_category_count：worker 在候选项目 category 上的历史完成次数
          remaining_days：deadline - 当前时刻（动态紧迫度）
    """
    print("\n" + "=" * 60)
    print("5. Build RL Events")
    print("=" * 60)

    # 维护每个 worker 的动态历史（按时间顺序追加）
    worker_dynamic_history: Dict[int, List[int]] = defaultdict(list)

    all_pids = list(project_info.keys())
    events: List[Dict] = []

    for row in tqdm(df.itertuples(), total=len(df), desc="events"):
        wid = row.worker_id
        pid_pos = row.project_id
        t = row.timestamp

        if wid not in worker_static:
            worker_dynamic_history[wid].append(pid_pos)
            continue

        ws = worker_static[wid]

        # 当前时刻有效的候选项目（已开始且未截止）
        candidates = [
            p for p, info in project_info.items()
            if info["start_date"] <= t <= info["deadline"]
        ]
        if len(candidates) < 5:
            worker_dynamic_history[wid].append(pid_pos)
            continue
        if pid_pos not in candidates:
            candidates.append(pid_pos)

        # 采样负样本
        # 新增修改
        # 采样负样本（加权采样：deadline 临近 + category/industry 匹配优先）
        negatives = [p for p in candidates if p != pid_pos]

        if len(negatives) > MAX_CANDIDATES - 1:
            def score_candidate(p):
                info = project_info[p]
                # 紧迫度得分：remaining_days 越小分越高，+1 防止除零
                remaining = max((info["deadline"] - t).days, 0)
                urgency_score = 1.0 / (remaining + 1)
                # 相关性得分：category 匹配 +1，industry 匹配 +0.5
                relevance_score = (
                    1.0 * int(info["category"] == ws["favorite_category"])
                    + 0.5 * int(info["industry"] == ws["favorite_industry"])
                )
                # 加权总分 + 小随机扰动（避免分数完全相同时顺序固定）
                return urgency_score + relevance_score + np.random.uniform(0, 0.1)

            scores = np.array([score_candidate(p) for p in negatives])
            # 归一化为概率分布
            probs = scores / scores.sum()
            chosen_idx = np.random.choice(len(negatives), size=MAX_CANDIDATES - 1, replace=False, p=probs)
            negatives = [negatives[i] for i in chosen_idx]

        final_candidates = negatives + [pid_pos]
        np.random.shuffle(final_candidates)
        positive_index = final_candidates.index(pid_pos)

        # worker 在各 category 的历史完成次数（动态，截止当前时刻）
        hist_cat_counts: Counter = Counter()
        for hp in worker_dynamic_history[wid]:
            if hp in project_info:
                hist_cat_counts[project_info[hp]["category"]] += 1

        # 构建候选特征向量
        candidate_features = []
        for cp in final_candidates:
            info = project_info[cp]
            remaining = max((info["deadline"] - t).days, 0)
            cat_match = int(info["category"] == ws["favorite_category"])
            ind_match = int(info["industry"] == ws["favorite_industry"])
            popularity = info["entry_count"] / max(info["duration_days"], 1)
            worker_cat_count = hist_cat_counts.get(info["category"], 0)

            candidate_features.append({
                "worker_quality": ws["quality"],
                "worker_history_count": float(len(worker_dynamic_history[wid])),
                "worker_active_days": float(ws["active_days"]),
                "project_category": float(info["category"]),
                "project_sub_category": float(info["sub_category"]),
                "project_industry": float(info["industry"]),
                "project_entry_count": float(info["entry_count"]),
                "project_duration_days": float(info["duration_days"]),
                "category_match": float(cat_match),
                "industry_match": float(ind_match),
                "project_popularity": float(popularity),
                "worker_category_count": float(worker_cat_count),
                "remaining_days": float(remaining),
            })

        # worker_history_at_t：截止当前时刻已完成的项目（最近 HISTORY_LEN 条）
        history_snapshot = list(worker_dynamic_history[wid])[-HISTORY_LEN:]

        events.append({
            "worker_id": wid,
            "timestamp": t,
            "candidate_projects": final_candidates,
            "candidate_features": candidate_features,
            "positive_project": pid_pos,
            "positive_index": positive_index,
            # 新增：动态历史状态
            "worker_history_at_t": history_snapshot,
        })

        # 当前交互完成后，更新动态历史
        worker_dynamic_history[wid].append(pid_pos)

    print(f"  Total events: {len(events)}")
    return events


# ============================================================
# 7. 按 worker 序列组织 episodes（供 episode 级 RL 训练使用）
# ============================================================

def build_worker_episodes(events: List[Dict]) -> Dict[int, List[Dict]]:
    """
    把 events 按 worker_id 分组，每个 worker 的所有 events 按时间排序，
    形成一个完整的 episode。

    这是让项目"更像 RL"的关键数据结构：
      - 一个 episode = 一个 worker 在平台上的完整访问序列
      - episode 内相邻 step 之间有真正的状态转移
        （worker_history_at_t 随每次推荐结果更新）
    """
    episodes: Dict[int, List[Dict]] = defaultdict(list)
    for ev in events:
        episodes[ev["worker_id"]].append(ev)
    # 每个 worker 的 episode 已经按时间顺序（因为 events 本身按时间排序）
    return dict(episodes)


# ============================================================
# 8. 划分训练/验证/测试集
# ============================================================

def split_events(events: List[Dict]):
    n = len(events)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * VAL_RATIO)
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

    # 按时间划分（先划分再归一化，避免数据泄露）
    train_events, val_events, test_events = split_events(events)
    print(f"\n  Train: {len(train_events)}  Val: {len(val_events)}  Test: {len(test_events)}")

    # 在训练集上计算归一化参数
    print("\n" + "=" * 60)
    print("6. Normalize Features")
    print("=" * 60)
    norm_stats = compute_norm_stats(train_events)
    train_events = apply_norm(train_events, norm_stats)
    val_events = apply_norm(val_events, norm_stats)
    test_events = apply_norm(test_events, norm_stats)

    # 构建 worker episodes（基于全量 events，归一化后）
    all_events = train_events + val_events + test_events
    worker_episodes = build_worker_episodes(all_events)
    print(f"  Worker episodes: {len(worker_episodes)}")

    # 保存
    print("\n" + "=" * 60)
    print("7. Save")
    print("=" * 60)

    def save(obj, name):
        path = os.path.join(SAVE_DIR, name)
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        print(f"  Saved {name}  ({len(obj)} records)")

    save(train_events, "enhanced_train.pkl")
    save(val_events, "enhanced_val.pkl")
    save(test_events, "enhanced_test.pkl")
    save(worker_episodes, "worker_episodes.pkl")
    save(worker_static, "worker_features.pkl")

    # 保存归一化参数
    norm_path = os.path.join(SAVE_DIR, "norm.json")
    with open(norm_path, "w", encoding="utf-8") as f:
        json.dump(norm_stats, f, ensure_ascii=False, indent=2)
    print(f"  Saved norm.json")

    # 保存类别映射表
    maps_path = os.path.join(SAVE_DIR, "category_maps.json")
    with open(maps_path, "w", encoding="utf-8") as f:
        json.dump(category_maps, f, ensure_ascii=False, indent=2)
    print(f"  Saved category_maps.json")

    # 保存数据集信息
    info = {
        "train_size": len(train_events),
        "val_size": len(val_events),
        "test_size": len(test_events),
        "num_workers": len(worker_episodes),
        "max_candidates": MAX_CANDIDATES,
        "history_len": HISTORY_LEN,
        "feature_order": FEATURE_ORDER,
        "feature_dim": len(FEATURE_ORDER),
    }
    with open(os.path.join(SAVE_DIR, "info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print(f"  Saved info.json")

    print("\nDONE")
    print(f"Output: {SAVE_DIR}")


if __name__ == "__main__":
    main()
