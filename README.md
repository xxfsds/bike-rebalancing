# T1 · Bike-Sharing 「Demand Forecasting → Dispatch Optimization」闭环系统

> AI2013 Introduction to AI — Group AI Survey Paper 的可运行代码骨架
> 对应论文：Methodology（第 3 节）+ Implementation（第 4 节）+ Results and Analysis（第 5 节）

这套代码把课程里的 **6 个模块**串成一条可复现的流水线：

| 层 | 作用 | 课程出处 | 负责人 |
|---|---|---|---|
| L0 数据层 | 清洗 / 特征 / 归一化 / 离散化 / 站点聚类（E1） | W9 数据挖掘 | M2 |
| L1 预测层 | 线性基线、随机森林、FFBPN（E2 / E3 / E4） | W5b 回归、W7 机器学习 | M3 |
| L2 优化层 | A\* 代价矩阵（W2–3）+ CSP 约束（W4）+ GA/SA/Tabu（E5 / E7） | W2–3、W4、W5a | M4 |
| L3 评估层 | 耦合敏感性 E6、全部图表、一键复现 | W9 可视化 | M5 |
| 全局 | 架构图、接口契约、论文整合 | — | M1 |
| 写作 | 文献综述、格式、查重 | W11 / 学术写作 | M6 |

---

## 1. 快速开始

```bash
cd bike_rebalancing
pip install -r requirements.txt

python run_all.py            # 完整实验，约 5–10 分钟
python run_all.py --fast     # 冒烟测试：1 个种子、2 折、小预算，约 30 秒
```

跑完你会得到：

```
figures/fig1_architecture.pdf|png   系统架构图（M1 可直接改）
figures/fig2_clusters.pdf|png       站点分群 + k 选择（E1）
figures/fig3_forecast.pdf|png       预测效果：时序 + 散点（E3）
figures/fig4_convergence.pdf|png    GA/SA/Tabu 收敛曲线（E5）
figures/fig5_sensitivity.pdf|png    预测误差 → 调度成本（E6）★论文亮点
figures/fig6_scalability.pdf|png    A* 与 GA 的运行时间随规模变化（E7）
tables/tab1_forecast.csv            预测模型对比
tables/tab2_ablation.csv            特征消融
tables/tab3_optimize.csv            优化算法对比（均值±标准差）
tables/tab4_literature_TEMPLATE.csv 文献对比表模板（M6 填）
results/*.csv                       全部原始实验数字（写论文只从这里读数）
```

**一条铁律**：论文里的每个数字都必须能在 `results/*.csv` 里找到。禁止手抄、禁止改数。

---

## 2. 目录与模块

```
bike_rebalancing/
├── config.py          全局常量：种子 SEEDS=[0,1,2]、预算、路径、IEEE 图尺寸
├── data_pipeline.py   L0：synthesize / load_uci_hour / clean / 特征 / E1 聚类
├── forecast.py        L1：make_folds / build_model / E2 / E3 / E4 / predict_all
├── routing.py         L2-a：带障碍网格 + A* + 站点代价矩阵
├── optimize.py        L2-b：Ctx / evaluate（唯一目标函数）/ greedy / ga / sa / tabu / E5 / E6 / E7
├── report.py          L3：Fig.1–Fig.6、Tab.1–Tab.4
├── run_all.py         一键复现入口（--seed / --stations / --days / --budget / --fast）
├── data/raw/          放真实数据（hour.csv）
├── data/              中间产物：clean.csv / stations.csv / predictions.csv / cost_matrix.npy
├── results/           实验原始数字
├── figures/           论文图（PDF + PNG，300 dpi）
└── tables/            论文表
```

---

## 3. 换成真实数据（4.1 节要写清）

默认用**合成数据**，所以第一天就能跑通全链路。拿到真实数据后：

1. 把 UCI Bike Sharing 的 `hour.csv` 放进 `data/raw/`；
2. 在 `run_all.py` 里把 L0 那一行改成：

```python
df, stations, ks = DP.run(n_stations=args.stations, days=args.days,
                           seed=0, raw_path=C.RAW_DIR / "hour.csv")
```

⚠️ **必须在论文里如实交代**：UCI Hour 数据集是**全市聚合**的逐小时计数，没有站点维度。
`load_uci_hour()` 按各站点权重做多项式摊分（"合成站点化"），只用于验证系统链路。
若组内能拿到城市级站点数据（Citi Bike / 本地共享单车开放数据），应替换该函数，
并在 4.1 节写明数据来源、时间范围、站点数。

---

## 4. 接口契约（改之前先问 M1）

### 文件

| 路径 | 字段 | 生产者 → 消费者 |
|---|---|---|
| `data/clean.csv` | datetime, station_id, hour, dow, is_weekend, season, weather, temp, hum, windspeed, is_rush, lag_1, lag_24, roll_24, demand, demand_level, *_n | M2 → M3/M4/M5 |
| `data/stations.csv` | station_id, x, y, base_demand, cluster | M2 → M4/M5 |
| `data/predictions.csv` | datetime, station_id, y_true, y_pred_linear, y_pred_rf, y_pred_ffbpn | M3 → M4/M5 |
| `data/cost_matrix.npy` | N×N 对称矩阵（A\* 最短路代价） | M4 → M5 |
| `results/E1..E7_*.csv` | 见 `report.py` 顶部注释 | M2–M5 → M5 |

### 函数

```python
data_pipeline.build_clean(n_stations, days, seed, raw_path) -> DataFrame
data_pipeline.station_table(df) -> DataFrame
data_pipeline.cluster_stations(profiles, k_range, seed) -> (k, labels, ks_df)

forecast.make_folds(df, n_splits) -> [(train_mask, test_mask), ...]
forecast.build_model(name, seed, **cfg) -> sklearn estimator
forecast.evaluate_cv(df, name, seed, folds, **cfg) -> (metrics, seconds, per_fold)

routing.build_cost_matrix(stations, seed, obstacle_p, save) -> (matrix, stats, blocked, cells)
routing.astar(blocked, start, goal) -> (path, cost, expansions)

optimize.build_context(pred_df, stations, cost, n_visit, seed, select_ids) -> Ctx
optimize.Ctx.evaluate(perm) -> {"cost", "travel", "unmet", "perm"}   # 唯一目标函数
optimize.ga / sa / tabu (ctx, budget, seed, trace) -> (solution, history)
optimize.run_e5 / run_e6 / run_e7
```

> **三算法必须共用 `Ctx.evaluate`**。谁另写一个 cost 函数，E5 的对比就作废。

---

## 5. 实验编号 ↔ 论文位置

| 编号 | 内容 | 产出 | 论文位置 |
|---|---|---|---|
| E1 | K-means 站点分群 + k 选择（肘部法 + 轮廓系数） | Fig.2 | 4.2 / 5.1 |
| E2 | 线性 / Ridge / Lasso 基线 | Tab.1 | 5.2 |
| E3 | 随机森林 vs FFBPN（时序交叉验证） | Tab.1、Fig.3 | 5.2 |
| E4 | 特征消融（去天气 / 去时间 / 仅滞后） | Tab.2 | 5.3 |
| R1 | A\* 绕行系数与扩展节点数 | `R1_astar_stats.csv` | 3.4 |
| E5 | GA vs SA vs Tabu vs 贪心（同预算） | Fig.4、Tab.3 | 5.4 ★ |
| E6 | 预测误差 → 调度成本（决策用带噪预测，评估用真值） | Fig.5 | 5.5 ★ |
| E7 | 站点数 10/30/50 的运行时间 | Fig.6 | 5.6 |

---

## 6. 方法学上的三个决定（审稿会问，先写进 3.3 / 3.4）

1. **时序交叉验证**：`make_folds` 先对**唯一时间戳**做 `TimeSeriesSplit`，再映射回行。
   若直接对行切分，同一小时的不同站点会同时出现在训练与测试集 —— 数据泄漏，R² 会虚高。
2. **GA 用 OX 交叉**：排列编码下，单点交叉会产生重复/缺失站点的**非法子代**
   （课件 Week5a "Problems with Crossover" 讲的就是这个）。OX 保证子代仍是合法排列。
3. **公平对比**：三算法同一初始解（最近邻暖启动）、同一 `evaluate`、同一评估次数预算
   `C.OPT_BUDGET`。Tabu 禁忌长度按课件公式 `L = a×Nc + Random(g)`。

---

## 7. 已知限制（写进 Limitations，别装作没有）

- 调度模型是**简化 VRP**：车辆按排列顺序连续访问，不做多趟往返、不含时间窗的实际时间推演。
- 合成数据的 R² 天然偏高，**换成真实数据后指标会下降**，这是正常的。
- `N_JOBS = 1`：某些受限环境禁止子进程管道；本机可改 `config.N_JOBS = -1` 提速，
  但论文写 `train_time` 时必须注明并行设置，否则不同机器不可比。
- Lasso 在高相关特征下解不稳定，仅作基线。

## 8. 常见坑

| 现象 | 原因 | 处理 |
|---|---|---|
| `PermissionError: [WinError 5]` 建进程池 | 沙箱/安全软件禁止命名管道 | `config.N_JOBS = 1` |
| 中文图例变方框 | 缺中文字体 | 图内**一律用英文**（已经是这么做的） |
| `matplotlib` 首次运行慢 | 建字体缓存 | 正常，第二次即快 |
| `run_all.py --fast` 与完整结果不一致 | fast 改了种子数/折数/预算 | 论文数字**只能**用不带 `--fast` 的结果 |

---

## 9. E6 的现状与正确读法（重要，别照着想象写）

E6 在 2026-11-23 换过一次设计，**旧结论（负结果）已经作废**，理由见下。

### 9.1 旧设计为什么必然是平线（不是"影响被削弱"，是"结构上进不去"）

`optimize.py` 的 `build_context` 只吃**每个站点的平均预测**：

```python
pred = pred_df.groupby("station_id", observed=True)[pred_col].mean()   # optimize.py:199
target = np.round(pred.reindex(ids).fillna(need.reindex(ids)).to_numpy())  # optimize.py:204
```

旧 E6 是**逐小时（站点×小时，逐行）独立加噪**，测试段每站约 288 个小时点，
独立噪声进入站点均值后按 1/√288 ≈ 1/17 缩水：20% 档 × 0.059 ≈ 1.2%，再被 `round` 取整吸收。
所以"±30% 噪声对成本没影响"不是模型瓶颈问题，而是**噪声根本没走到目标函数里**。

### 9.2 新设计为什么有信号（三条设计）

1. **站点级因子**：每站每轮只抽一个乘性因子 `1+ε_s`，全小时共用，站点均值精确变成 `mean×(1+ε_s)`，不再被平均掉；
2. **固定 depot**：四档共用"完美预测"决定的 depot（`optimize.py:550`），否则噪声会连 depot 和候选站点集合一起换掉；
3. **配对差分**：同一轮 r 的 0% 与 lv 档共用 `seed`、候选站点、库存、GA 种子，主指标是同轮成本差 `delta_pct`（方差远小于"各档独立取均值再相减"）。

结果（40 站真实数据、budget=3000、20 轮配对）：0% → +0.00%，5% → **+3.73%**，10% → **+7.49%**，
20% → **+18.32%**，单调上升，三档 95% 置信区间都不跨 0。

旧法可以在同一个函数里复现：`run_e6(..., noise_mode="hourly")`，用于论文里做"两种注入方式"的消融对照。

### 9.3 怎么读 Fig.5

横轴是注入的站点级误差水平，纵轴是**同一轮内相对 0% 的调度成本变化**（不是相对完美预测）。
误差线建议画 ±1 SE（`delta_std/√rounds`），因为主指标是"配对均值"；20 轮时 5% 档 95% CI 约 ±2.4%。

> 另外：合成数据上 E1 的最优簇数稳定在 `k*=2`（站点之间主要差别在规模，而规模已被归一化掉）。
> 换成真实站点数据后通常会出现 3–4 个簇。写论文时说明这是**合成数据的性质**即可，
> 方法本身（肘部法 + 轮廓系数）是正确的。

