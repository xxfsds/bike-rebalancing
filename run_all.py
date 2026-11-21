# -*- coding: utf-8 -*-
"""run_all.py —— 一键复现入口（M5 负责）

    python run_all.py                 # 完整实验（默认配置，约 5–8 分钟）
    python run_all.py --fast          # 快速冒烟（1 个种子、2 折、小预算，约 1 分钟）
    python run_all.py --skip-e7       # 跳过可扩展性实验
    python run_all.py --seed 0        # 只用单个种子

流程（对应论文的实验编号）
--------------------------
L0  data_pipeline.run      -> clean.csv, stations.csv, E1 聚类
L1  forecast.run           -> E2 基线, E3 主实验, E4 消融, predictions.csv
L2  routing.build_cost_matrix -> cost_matrix.npy, R1 A* 统计
    optimize.run_e5        -> E5 三算法对比 + 收敛轨迹
L3  optimize.run_e6        -> E6 耦合敏感性
    optimize.run_e7        -> E7 可扩展性
    report.run_allplots    -> Fig.1–Fig.6 + Tab.1–Tab.4
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

import config as C
import data_pipeline as DP
import forecast as FC
import optimize as OP
import report as RP
import routing as RT


def banner(msg: str):
    print("\n" + "=" * 72)
    print(msg)
    print("=" * 72)


def main():
    ap = argparse.ArgumentParser(description="T1 bike rebalancing pipeline")
    ap.add_argument("--seed", type=int, default=None, help="只跑指定种子")
    ap.add_argument("--stations", type=int, default=C.N_STATIONS)
    ap.add_argument("--days", type=int, default=C.DAYS)
    ap.add_argument("--budget", type=int, default=C.OPT_BUDGET, help="优化评估预算（三算法共用）")
    ap.add_argument("--fast", action="store_true", help="快速冒烟模式")
    ap.add_argument("--skip-e7", action="store_true")
    ap.add_argument("--reuse-clean", action="store_true",
                    help="复用 data/clean.csv（真实数据已就位时用），不重新生成合成数据")
    args = ap.parse_args()

    C.ensure_dirs()
    seeds = [args.seed] if args.seed is not None else ([0] if args.fast else C.SEEDS)
    budget = 600 if args.fast else args.budget
    t_start = time.time()
    manifest = {"seeds": seeds, "stations": args.stations, "days": args.days,
                "budget": budget, "fast": args.fast}

    # ---------------- L0 ----------------
    banner("L0  data layer  (E1 clustering)")
    if args.reuse_clean and (C.DATA_DIR / "clean.csv").exists():
        # 复用已有 clean.csv（真实数据路径）：只重建站点表与 E1，绝不调用
        # DP.run/save_outputs，否则会用合成数据覆盖掉真实 clean.csv。
        df = pd.read_csv(C.DATA_DIR / "clean.csv", parse_dates=["datetime"])
        stations = DP.station_table(df)
        _, labels, ks = DP.cluster_stations(DP.station_profiles(df), seed=0)
        stations["cluster"] = labels
        stations.round(4).to_csv(C.DATA_DIR / "stations.csv", index=False)
        stations[["station_id", "cluster"]].to_csv(C.RES_DIR / "E1_cluster.csv", index=False)
        ks.to_csv(C.RES_DIR / "E1_k_selection.csv", index=False)
        print(f"[L0] 复用已有 clean.csv：{len(df)} 行 / {len(stations)} 站点")
    else:
        df, stations, ks = DP.run(n_stations=args.stations, days=args.days, seed=0)
    best_k = int(ks.loc[ks["silhouette"].idxmax(), "k"])
    print(f"[L0] 数据 {len(df)} 行 / {len(stations)} 站点；最优簇数 k={best_k}")

    # ---------------- L1 ----------------
    banner("L1  forecast layer  (E2 / E3 / E4)")
    folds = FC.make_folds(df, n_splits=2 if args.fast else C.CV_SPLITS)
    e2 = FC.run_e2_baseline(df, seeds=seeds, folds=folds)
    e3, cfgs = FC.run_e3_main(df, seeds=seeds, folds=folds, fast=args.fast)
    e4 = FC.run_e4_ablation(df, cfgs["rf"], folds=folds)
    FC.feature_importance(df, cfgs["rf"])
    pred = FC.predict_all(df, {**cfgs, "linear": {}}, seed=0)

    # ---------------- L2 ----------------
    banner("L2  optimization layer  (A* -> E5)")
    cost, astar_stats, _, _ = RT.build_cost_matrix(stations)
    ctx = OP.build_context(pred, stations, cost, seed=0)
    e5, traces = OP.run_e5(ctx, seeds=seeds, runs=2 if args.fast else 3, budget=budget)
    tab3 = OP.summarise_e5(e5)
    np.savez(C.RES_DIR / "E5_traces.npz",
             **{k: np.array([np.asarray(h, dtype=float) for h in v], dtype=object)
                for k, v in traces.items()})

    # ---------------- L3 ----------------
    banner("L3  evaluation layer  (E6 / E7 / figures)")
    e6 = OP.run_e6(pred, stations, cost, seed=0, budget=budget)
    e7 = None
    if not args.skip_e7:
        e7 = OP.run_e7(sizes=(10, 30, 50), budget=600 if args.fast else 1000, seed=0)

    banner("figures and tables")
    RP.run_allplots(traces)

    # ---------------- 摘要 ----------------
    tab1 = pd.read_csv(C.TAB_DIR / "tab1_forecast.csv") if (C.TAB_DIR / "tab1_forecast.csv").exists() else None
    banner(f"done in {time.time() - t_start:.1f} s")
    if tab1 is not None:
        print("[摘要] 预测最佳:", tab1.sort_values("r2", ascending=False).iloc[0].to_dict())
    print("[摘要] 优化 Tab.3:\n", tab3.to_string(index=False))
    print("[摘要] 敏感性 Tab.:\n", e6.to_string(index=False))
    if e7 is not None:
        print("[摘要] 可扩展性:\n", e7.to_string(index=False))

    manifest.update({
        "best_model": None if tab1 is None else str(tab1.sort_values("r2", ascending=False).iloc[0]["model"]),
        "best_r2": None if tab1 is None else float(tab1["r2"].max()),
        "elapsed_s": round(time.time() - t_start, 1),
        "n_jobs": C.N_JOBS,
        "note": "N_JOBS=1 时 train_time 偏慢；本机可改 config.N_JOBS=-1",
    })
    (C.RES_DIR / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[manifest] {C.RES_DIR / 'run_manifest.json'}")


if __name__ == "__main__":
    main()
