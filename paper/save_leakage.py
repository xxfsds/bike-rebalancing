import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import pandas as pd
from pathlib import Path

SB = Path(r"C:\Users\HP\.minimax\sessions\mvs_697b874896554ea6844734667ae38673\workspace\sandbox\real_bike")

rows = [
    # dataset, feature_set, split, rmse, r2
    ("real_uci", "with_lag(12)", "RandomKFold", 2.9558, 0.837398),
    ("real_uci", "with_lag(12)", "TimeSeriesSplit", 3.0711, 0.791692),
    ("real_uci", "no_lag(9)", "RandomKFold", 5.0416, 0.526935),
    ("real_uci", "no_lag(9)", "TimeSeriesSplit", 5.2141, 0.387237),
    ("synthetic", "with_lag(12)", "RandomKFold", 2.6038, 0.818969),
    ("synthetic", "with_lag(12)", "TimeSeriesSplit", 2.8409, 0.785266),
    ("synthetic", "no_lag(9)", "RandomKFold", 4.3683, 0.490485),
    ("synthetic", "no_lag(9)", "TimeSeriesSplit", 5.0333, 0.347956),
]
df = pd.DataFrame(rows, columns=["dataset", "feature_set", "split", "rmse", "r2"])
df["delta_r2"] = np.nan if False else 0.0
# 每组 (dataset, feature_set) 内 RandomKFold 相对 TimeSeriesSplit 的 R2 抬升
df["delta_r2"] = df.groupby(["dataset", "feature_set"])["r2"].transform(
    lambda s: s.iloc[0] - s.iloc[1])

out = SB / "results" / "R3_leakage_check.csv"
df.to_csv(out, index=False)
print(f"[write] {out}")
print(df.to_string(index=False))
print("\n=== 用于 §3.3 引用的两个数 ===")
for ds in ("real_uci", "synthetic"):
    sub = df[(df.dataset == ds) & (df.feature_set == "no_lag(9)")].reset_index(drop=True)
    print(f"  {ds:10s} 去滞后: RandomKFold R2={sub.r2[0]:.4f}  TSS R2={sub.r2[1]:.4f}  "
          f"ΔR2={sub.r2[0]-sub.r2[1]:+.4f}  (相对虚高 {(sub.r2[0]/sub.r2[1]-1)*100:.1f}%)")
