"""
数据验证脚本
运行方式：python data_processed/check_data.py
"""

import pickle
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

BASE = Path(__file__).resolve().parent


def load(name):
    with open(BASE / name, "rb") as f:
        return pickle.load(f)


# ============================================================
# 1. 基本规模
# ============================================================
print("=" * 60)
print("1. 数据集规模")
print("=" * 60)
train = load("enhanced_train.pkl")
val   = load("enhanced_val.pkl")
test  = load("enhanced_test.pkl")
print(f"  train: {len(train)} 条")
print(f"  val:   {len(val)} 条")
print(f"  test:  {len(test)} 条")


# ============================================================
# 2. 字段完整性
# ============================================================
print("\n" + "=" * 60)
print("2. 字段完整性")
print("=" * 60)
required = ["worker_id", "timestamp", "candidate_projects",
            "candidate_features", "positive_project",
            "positive_index", "worker_history_at_t"]
s = train[0]
for field in required:
    print(f"  {field}: {'✓' if field in s else '✗ 缺失'}")


# ============================================================
# 3. 特征维度与键名
# ============================================================
print("\n" + "=" * 60)
print("3. 特征维度与键名")
print("=" * 60)
feat_keys = list(train[0]["candidate_features"][0].keys())
print(f"  特征维度: {len(feat_keys)}")
print("  特征列表:")
for k in feat_keys:
    tag = ""
    if k in ("worker_category_count", "remaining_days"):
        tag = "  ← 新增动态特征"
    print(f"    {k}{tag}")


# ============================================================
# 4. 归一化检验（训练集 mean≈0, std≈1）
# ============================================================
print("\n" + "=" * 60)
print("4. 归一化检验（训练集，mean≈0 / std≈1）")
print("=" * 60)
feat_vals = defaultdict(list)
for ev in train:
    for feat in ev["candidate_features"]:
        for k, v in feat.items():
            feat_vals[k].append(v)

print(f"  {'特征':<25} {'mean':>8} {'std':>8} {'min':>8} {'max':>8}  状态")
print("  " + "-" * 65)
for k in feat_keys:
    arr = np.array(feat_vals[k])
    if k == "remaining_days":
        ok = arr.min() >= 0
        status = "✓ 原始天数" if ok else "✗"
    else:
        ok = abs(arr.mean()) < 0.01 and abs(arr.std() - 1.0) < 0.01
        status = "✓" if ok else "✗"
    print(f"  {k:<25} {arr.mean():>8.4f} {arr.std():>8.4f} "
          f"{arr.min():>8.3f} {arr.max():>8.3f}  {status}")


# ============================================================
# 5. 时间顺序
# ============================================================
print("\n" + "=" * 60)
print("5. 时间顺序（三集合应首尾衔接）")
print("=" * 60)
for name, data in [("train", train), ("val", val), ("test", test)]:
    ts = [ev["timestamp"] for ev in data]
    ordered = all(ts[i] <= ts[i+1] for i in range(len(ts)-1))
    print(f"  {name}: {str(ts[0])[:19]} → {str(ts[-1])[:19]}  有序={'✓' if ordered else '✗'}")


# ============================================================
# 6. positive_index 合法性
# ============================================================
print("\n" + "=" * 60)
print("6. positive_index 合法性")
print("=" * 60)
for name, data in [("train", train), ("val", val), ("test", test)]:
    bad = [i for i, ev in enumerate(data)
           if ev["positive_index"] < 0
           or ev["positive_index"] >= len(ev["candidate_projects"])
           or ev["candidate_projects"][ev["positive_index"]] != ev["positive_project"]]
    print(f"  {name}: 非法样本={len(bad)}  {'✓' if not bad else '✗'}")


# ============================================================
# 7. worker_history_at_t 动态性
# ============================================================
print("\n" + "=" * 60)
print("7. worker_history_at_t 动态性（同一 worker 历史应递增）")
print("=" * 60)
worker_evs = defaultdict(list)
for ev in train:
    worker_evs[ev["worker_id"]].append(ev)

# 找一个有 4 条以上记录的 worker
sample_wid = next(wid for wid, evs in worker_evs.items() if len(evs) >= 4)
evs = worker_evs[sample_wid][:5]
print(f"  worker {sample_wid} 前5条历史长度: "
      f"{[len(e['worker_history_at_t']) for e in evs]}  (应单调不减)")
print(f"  第0条 history: {evs[0]['worker_history_at_t']}")
print(f"  第1条 history: {evs[1]['worker_history_at_t']}")
print(f"  第2条 history: {evs[2]['worker_history_at_t']}")


# ============================================================
# 8. 打印前5条完整样本
# ============================================================
print("\n" + "=" * 60)
print("8. 前5条训练样本（完整内容）")
print("=" * 60)
for i in range(5):
    ev = train[i]
    print(f"\n  --- 第{i}条 ---")
    print(f"  worker_id:           {ev['worker_id']}")
    print(f"  timestamp:           {str(ev['timestamp'])[:19]}")
    print(f"  positive_project:    {ev['positive_project']}  (index={ev['positive_index']})")
    print(f"  candidate_projects:  {ev['candidate_projects']}")
    print(f"  worker_history_at_t: {ev['worker_history_at_t']}")
    print(f"  正样本特征 (candidate_features[{ev['positive_index']}]):")
    feat = ev["candidate_features"][ev["positive_index"]]
    for k, v in feat.items():
        tag = " ← 新增" if k in ("worker_category_count", "remaining_days") else ""
        print(f"    {k:<25}: {v:>8.4f}{tag}")


# ============================================================
# 9. 新增特征说明
# ============================================================
print("\n" + "=" * 60)
print("9. 新增动态特征说明（给环境构建同学）")
print("=" * 60)
print("""
  特征名: worker_category_count
  含义:   截止当前时刻，该 worker 在候选项目所属 category 上的历史完成次数
  作用:   反映 worker 在该类别的专项能力，比 category_match(0/1) 更细粒度
  动态性: 随 worker 每次完成任务实时更新，是状态转移的组成部分

  特征名: remaining_days
  含义:   候选项目的截止时间 - worker 到达时刻（单位：天）
  作用:   反映任务的动态紧迫度，比静态的 project_duration_days 更准确
  动态性: 每个时间步都不同，越接近截止日期值越小

  注意：worker_category_count 已做 z-score 归一化；
        remaining_days 保留原始天数，用于 env 判断任务是否过期。
""")

print("=" * 60)
print("验证完成")
print("=" * 60)
