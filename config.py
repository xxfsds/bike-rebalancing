# -*- coding: utf-8 -*-
"""config.py —— 全局常量与路径。

R2 铁律的落地点：所有随机种子、指标口径、路径常量只在这里定义，
任何人不得在自己模块里私设 seed 或自写指标函数。
"""
from pathlib import Path

# ---------- 路径 ----------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
RES_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures"
TAB_DIR = ROOT / "tables"

# ---------- 实验规模 ----------
N_STATIONS = 40          # 站点数（10/30/50 用于 E7 可扩展性实验）
DAYS = 60                # 模拟天数（真实数据时由数据决定）
START_DATE = "2026-09-01"

# ---------- 复现 ----------
SEEDS = [0, 1, 2]        # 所有实验统一用这三个种子
CV_SPLITS = 5            # TimeSeriesSplit 折数
TEST_SIZE = 0.2          # 最后 20% 作为测试段

# ---------- 预测 ----------
LAGS = [1, 24]           # 滞后特征：上一小时、上一同日
RF_GRID = [
    {"n_estimators": 120, "max_depth": None, "min_samples_leaf": 1},
    {"n_estimators": 120, "max_depth": 12, "min_samples_leaf": 2},
    {"n_estimators": 200, "max_depth": 20, "min_samples_leaf": 1},
]
MLP_GRID = [
    {"hidden_layer_sizes": (64, 32), "alpha": 1e-4},
    {"hidden_layer_sizes": (32, 16), "alpha": 1e-3},
]

# ---------- 调度（L2 优化层） ----------
# 规模经 tune_once.py 实测选定：n_visit=25 / cap=8 / fleet=3 时，
# 三种元启发式相对最近邻基线分别改进约 +10% / +21% / +13%，差异可分辨且有故事可写。
N_VISIT = 25             # 单车调度行程中访问的站点数（问题规模）
FLEET = 3                # 调度车辆数
VEH_CAP = 8              # 单车装载上限（辆）—— 容量越紧，访问顺序越关键
LAMBDA_PENALTY = 2.0     # 每单位未满足需求的惩罚系数（论文中须写明）
# 一次出车最多能服务的站点数（时间/人力预算）。这一条让「预测质量」真正影响结果：
# 预测决定了该先服务哪些站点，选错站点就等于把有限的出车配额浪费掉。
K_SERVED = 25          # 默认等于 N_VISIT（不启用配额）；调到 12 左右可做"选点+路由"拓展实验
OPT_BUDGET = 3000        # 【公平对比】每个算法的评估次数预算，三算法必须相同
TABU_A = 0.6             # 禁忌长度 L = a * Nc + Random(g)
TABU_G = 10

GA_POP = 40
GA_TOURNAMENT = 3
GA_PX = 0.9              # 交叉概率
GA_PM = 0.2              # 变异概率

# ---------- 并行 ----------
# 本机可设为 -1 全核加速；某些受限/沙箱环境创建进程间管道会被拒绝，此时必须为 1。
# 报告里如写"训练耗时"，务必注明 N_JOBS 取值，否则不同机器不可比。
import os as _os  # noqa: E402
N_JOBS = int(_os.environ.get("BR_N_JOBS", 1))   # 默认 1（受限环境必须为 1）；参考机跑论文数字时用 BR_N_JOBS=-1
SA_T0 = 50.0             # 初温
SA_TF = 0.05             # 终温
SA_COOL = 0.997

# ---------- 绘图（IEEE 规范） ----------
FIG_SINGLE = (3.5, 2.6)   # 单栏图宽 3.5 in
FIG_DOUBLE = (7.16, 3.0)  # 双栏图宽 7.16 in
DPI = 300

# ---------- 指标口径（唯一实现，禁止各写一份） ----------
METRIC_DEF = {
    "rmse": "sqrt(mean((y - yhat)^2))",
    "mae": "mean(|y - yhat|)",
    "r2": "1 - SS_res / SS_tot",
}


def ensure_dirs():
    for d in (DATA_DIR, RAW_DIR, RES_DIR, FIG_DIR, TAB_DIR):
        d.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ensure_dirs()
    print("ROOT      :", ROOT)
    print("seeds     :", SEEDS)
    print("stations  :", N_STATIONS, "| days:", DAYS)
    print("opt budget:", OPT_BUDGET, "evals (三算法共用)")
    print("dirs ok ->", DATA_DIR, RES_DIR, FIG_DIR, TAB_DIR)
