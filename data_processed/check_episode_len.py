import pickle
import numpy as np
from pathlib import Path

with open(Path(__file__).parent / "worker_episodes.pkl", "rb") as f:
    episodes = pickle.load(f)

lengths = [len(evs) for evs in episodes.values()]
lengths = np.array(lengths)

print(f"worker 总数:      {len(lengths)}")
print(f"episode 总步数:   {lengths.sum()}")
print(f"平均长度:         {lengths.mean():.1f}")
print(f"中位数:           {np.median(lengths):.1f}")
print(f"最短:             {lengths.min()}")
print(f"最长:             {lengths.max()}")
print(f"p25/p75/p90/p95:  {np.percentile(lengths,[25,75,90,95]).astype(int).tolist()}")
print()
# 分布直方图（文字版）
buckets = [1, 2, 5, 10, 20, 50, 100, 999999]
labels  = ["1", "2-4", "5-9", "10-19", "20-49", "50-99", "100+"]
for lo, hi, label in zip(buckets, buckets[1:], labels):
    cnt = ((lengths >= lo) & (lengths < hi)).sum()
    print(f"  长度 {label:>6}: {cnt:>5} 个 worker  ({cnt/len(lengths)*100:.1f}%)")
