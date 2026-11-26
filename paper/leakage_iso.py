"""Isolate the two leakage channels on the PANEL dataset.

Channel A (siblings): a held-out row (s,t) has its 39 sibling stations at the
        same hour t in the training set -> the model can reconstruct it.
Channel B (lag): with lag_1 / lag_24 / roll_24 present, the model largely
        ignores siblings, which MASKS channel A.

=> Compare RandomKFold vs TimeSeriesSplit with and without the lag block.
   The no-lag rows isolate channel A.
"""
import sys, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np, pandas as pd
from pathlib import Path

SB = Path(r"C:\Users\HP\.minimax\sessions\mvs_697b874896554ea6844734667ae38673\workspace\sandbox\real_bike")
SYN = Path(r"C:\Users\HP\.minimax\sessions\mvs_697b874896554ea6844734667ae38673\workspace\sandbox\bike_rebalancing")
sys.path.insert(0, str(SB))
import forecast as F
from sklearn.model_selection import KFold

CFG = {"n_estimators": 120, "max_depth": 12, "min_samples_leaf": 2}
LAGS = ["lag_1", "lag_24", "roll_24"]


def run(df, feats, tag):
    X = df[feats].to_numpy(dtype=np.float32)
    y = df["demand"].to_numpy(dtype=np.float32)
    res = {}
    for kind in ("RandomKFold", "TimeSeriesSplit"):
        if kind == "RandomKFold":
            folds = list(KFold(n_splits=5, shuffle=True, random_state=0).split(X))
        else:
            t0 = time.time()
            folds = F.make_folds(df, n_splits=5)
        scores = []
        t0 = time.time()
        for tr, te in folds:
            est = F.build_model("rf", seed=0, **CFG)
            est.fit(X[tr], y[tr])
            scores.append(F._metrics(y[te], est.predict(X[te])))
        res[kind] = {k: float(np.mean([s[k] for s in scores])) for k in ("rmse", "r2")}
        res[kind]["t"] = time.time() - t0
    r, t = res["RandomKFold"], res["TimeSeriesSplit"]
    gap = r["r2"] - t["r2"]
    print(f"\n--- {tag} ---")
    print(f"  RandomKFold     R2={r['r2']:.6f}  RMSE={r['rmse']:.4f}")
    print(f"  TimeSeriesSplit R2={t['r2']:.6f}  RMSE={t['rmse']:.4f}")
    print(f"  >>> 泄漏抬升 ΔR2 = {gap:+.4f}  ({'虚高明显' if gap > 0.05 else '被滞后特征掩盖'})")
    return gap


print("=" * 78)
print("真实 UCI 面板（最后 120 天）")
print("=" * 78)
real = pd.read_csv(SB / "data" / "clean.csv", parse_dates=["datetime"])
real = F.canonical(real[real["datetime"] >= real["datetime"].max() - pd.Timedelta(days=120)])
g1 = run(real, F.FEATURES, "含滞后特征 (12 维)")
g2 = run(real, [f for f in F.FEATURES if f not in LAGS], "去掉滞后特征 (9 维) —— 隔离同站点兄弟行泄漏")

print("\n" + "=" * 78)
print("合成数据面板")
print("=" * 78)
syn = F.load_clean()
g3 = run(syn, F.FEATURES, "含滞后特征 (12 维)")
g4 = run(syn, [f for f in F.FEATURES if f not in LAGS], "去掉滞后特征 (9 维)")

print("\n" + "=" * 78)
print("§3.3 可引用的结论")
print("=" * 78)
print(f"  真实数据 含滞后 : ΔR2 = {g1:+.4f}")
print(f"  真实数据 去滞后 : ΔR2 = {g2:+.4f}   <- 同站点兄弟行泄漏的净效应")
print(f"  合成数据 含滞后 : ΔR2 = {g3:+.4f}")
print(f"  合成数据 去滞后 : ΔR2 = {g4:+.4f}")
