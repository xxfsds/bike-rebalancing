# M3 交接说明 —— 预测层（L1 Prediction Layer）

> 交付人：M3（预测建模）　｜　日期：2026-11-21
> 数据集：UCI Bike Sharing（hour.csv 摊分到 40 站点，17,379 小时戳，**695,160 行**，2011-01-01 → 2012-12-31）
> 运行环境：Python 3.11.9 / scikit-learn 1.9.1 / numpy 1.26.4 / `N_JOBS = -1`（Intel i9-14900HX, 24C/32T）

---

## 0. 一句话

M3 的代码、实验、图表、正文**全部完成**，真实 UCI 数据上的 E2/E3/E4/predictions 已跑通并落盘。
但**全组的数字还需要 M1 统一重跑一次**，原因是切数据源 + 三个复现性补丁，见 §3。

---

## 1. 交付物清单

| 文件 | 内容 | 状态 |
|---|---|---|
| `results/E2_baseline.csv` | linear/ridge/lasso × 3 seed | ✅ 真实数据 |
| `results/E3_main.csv` | rf / ffbpn × 3 seed（含 config 列） | ✅ 真实数据 |
| `results/E4_ablation.csv` | full / no_weather / no_time / lag_only | ✅ 真实数据 |
| `results/R2_feat_importance.csv` | 12 个特征的重要性 | ✅ 真实数据 |
| `results/R3_leakage_check.csv` | **新增**：随机切分 vs 时序切分的泄漏量化 | ✅ |
| `data/predictions.csv` | 139,040 行（最后 20% 测试段），0 NaN | ✅ 真实数据 |
| `tables/tab1_forecast.csv` | 模型对比 | ✅ 真实数据 |
| `tables/tab2_ablation.csv` | 消融 | ✅ 真实数据 |
| `figures/fig3_forecast.pdf\|png` | 预测效果双栏图 | ✅ 真实数据 |
| `paper/sections/3.3-4.3_forecast_layer.md` | **§3.3 / §4.3 正文**（中文脚手架 + 英文成稿） | ✅ |
| `M3_patches.patch` | 三个代码补丁（`git apply` 已验证） | ✅ |

---

## 2. 真实数据 vs 合成数据：结论方向没变，但机理变了

| 模型 | 合成 R² | 真实 R² | Δ |
|---|---|---|---|
| **rf** | 0.856 | **0.785** | −0.071 |
| ffbpn | 0.838 | 0.774 | −0.064 |
| linear / ridge | 0.811 | 0.673 | −0.138 |
| lasso | 0.803 | 0.635 | −0.168 |

**最优模型仍是 RF**，与合成数据一致，§5.2 的结论不用改。

### 2.1 消融的故事完全反过来了 —— 这是最值得写进论文的发现

| 特征集 | ΔR²（合成） | ΔR²（真实） | 解读 |
|---|---|---|---|
| no_weather | −0.026 | **−0.002** | **天气几乎不携带信息** |
| no_time | −0.040 | **−0.113** | **日历/时段成为主要信息源** |
| lag_only | −0.079 | **−0.118** | 只靠自相关不够 |

真实 UCI 数据上，**去掉天气特征几乎没有损失（ΔR² = −0.002）**，而合成数据上是 −0.026。
反过来，去掉时间特征的代价从 −0.040 涨到 −0.113。

### 2.2 特征重要性重排

| 合成 Top5 | 真实 Top5 |
|---|---|
| lag_24 0.794 | **lag_1 0.625** |
| lag_1 0.049 | lag_24 0.196 |
| weather 0.039 | hour 0.079 |
| hour 0.039 | roll_24 0.044 |
| roll_24 0.036 | dow 0.017 |

合成数据上 `lag_24` 独占 79% 权重；真实数据上 **`lag_1`（上一小时）才是第一**，且 `weather` 直接跌出 Top-5。
这个重排和 §2.1 的消融结论互相印证，是同���件事的两个侧面。

> **写作建议**：§5.3 可以据此展开一个诚实的观察——"在真实 UCI 数据上，
> 天气变量对逐小时需求几乎没有增量解释力，日历变量与短期自相关才是主要驱动"。
> 这比单纯堆 R² 更有分量，也顺带说明了为什么合成数据会让天气看起来比实际更有用。

---

## 3. 三个代码补丁（`M3_patches.patch`，`git apply --check` 已通过）

### 补丁 A：`forecast.py` 固定唯一行序（`canonical()`）
**问题**：`forecast.py` 独立跑时按 `(datetime, station_id)` 排序（时间优先），
`run_all.py` 则直接把 `data_pipeline.run()` 的 DataFrame 喂进去（站点优先）。
行序不同 ⇒ 树的 bootstrap 抽到不同记录 ⇒ **同一 seed、同样的超参训出不同的模型**。

**实测**：57,600 行、seed=0、超参完全相同，仅行序不同 →
RF 预测在 **100% 的行**上都不同，max|diff| = 4.74、mean|diff| = 0.21。
该差异经 `optimize.build_context` 的 `np.round()` 放大后（40 个站点里有 6 个的预测均值
距 .5 边界 < 0.05），会改写 M4 的 E5 站点选择与总成本。

**修复**：新增 `canonical()`，`load_clean` / `evaluate_cv` / `predict_all` 三个入口统一过一遍。
**验证**：加补丁后 RF 与 FFBPN 预测 **max|diff| = 0.000e+00**。

### 补丁 B：`forecast.py` 超参选择加 Occam 决胜（`SELECT_MARGIN = 0.005`）
**问题**：两个 MLP 配置的 R² 只差 0.003~0.004，落在环境噪声内 → 选出的配置随机器翻转。
实测：本机选 `(64,32)/1e-4`，原机选 `(32,16)/1e-3`，E3 的 ffbpn 整行数字跟着变。

**修复**：多个配置的 R² 差距 < 0.005 时，按 Occam 原则取结构更简单的那个。
**验证**：真实数据上 RF 三个配置 0.7888 / 0.7916 / 0.7910 全部落在阈值内 →
稳定固定为 (120 棵, depth 12, min_leaf 2)，与原机结果一致但现在跨机器确定。

### 补丁 C：`run_all.py` 新增 `--reuse-clean`（M1/M5 的坑）
**问题**：`run_all.py` 写死 `DP.run(raw_path=None)` → 每次都用**合成数据覆盖** `data/clean.csv`。
把真实 clean.csv 放进去再跑 `run_all.py`，真实数据会被立刻销毁。

**修复**：`--reuse-clean` 复用已有 `data/clean.csv`，只重建 stations.csv 与 E1，
**不调用 `save_outputs`**。验证：clean.csv 的 SHA256 与字节数完全未变。

---

## 4. §3.3 / §4.3 正文已写好

文件：`paper/sections/3.3-4.3_forecast_layer.md`
**由 `paper/build_m3_sections.py` 从 `results/*.csv` 自动生成**，满足项目"论文数字只从 CSV 取"的铁律。

含三部分：中文脚手架（写作时对照）→ 英文 IEEE 成稿（可直接粘贴）→ 数字来源对照表。

### §3.3 的核心论证（新写进去的）
用 `R3_leakage_check.csv` 量化了面板数据的时间泄漏：

| 数据集 | 特征集 | RandomKFold | TimeSeriesSplit | 泄漏抬升 |
|---|---|---|---|---|
| 真实 UCI | 含滞后(12) | 0.8374 | 0.7917 | +0.046 |
| 真实 UCI | **去滞后(9)** | 0.5269 | 0.3872 | **+0.140** |
| 合成 | 含滞后(12) | 0.8190 | 0.7853 | +0.034 |
| 合成 | **去滞后(9)** | 0.4905 | 0.3480 | **+0.143** |

滞后特征会掩盖同站点兄弟行的泄漏；隔离出来后随机切分把 R² 虚高 **+0.14（相对虚高 36–41%）**。
这比"我们用了 TimeSeriesSplit 所以是对的"有说服力得多。

---

## 5. 需要 M1 / 其它角色配合的事

| 谁 | 要做什么 | 为什么 |
|---|---|---|
| **M1** | 决定论文主数据源：真实 UCI（推荐）还是合成 | 执行预案 M2 任务①本来就要 UCI；合成只是降级预案 |
| **M1** | 采纳 `M3_patches.patch` 三个补丁 | 不采纳则 E3/E5 数字仍不可复现 |
| **M1** | 改 `run_all.py` 的 L0 用 `--reuse-clean`，别再用合成数据 | 否则真实数据会被覆盖 |
| **M2** | 用真实 clean.csv 重建 stations.csv + E1（已备好方法，**不要调 `DP.run()`**） | E1 在真实数据上 k*=2 但轮廓系数只有 0.181（合成 0.516），簇分布 13:27，§4.2 必须如实交代 |
| **M4** | E5/E6 必须重跑 | 每站装载量来自 base_demand，真实数据下均值从 8.96 降到 4.74，调度实例变了 |
| **M4** | `cost_matrix.npy` **不用**重建 | `build_cost_matrix` 只用 x/y，坐标与数据无关（已验证新旧坐标完全相同） |
| **M5** | 全部图重出 | Fig.3/Fig.5 等依赖新的 predictions.csv 与 E5/E6 |
| **M6** | §3.3/§4.3 需要的引用编号 | 正文里 `[n]` 待替换 |

---

## 6. 复现命令

```powershell
cd bike_rebalancing
# 1) 把真实 clean.csv 放进 data/ （695,160 行，表头与合成版完全一致）
# 2) 应用补丁
git apply ..\M3_patches.patch
# 3) 重建站点表与 E1（不要用 DP.run()，它会用合成数据覆盖 clean.csv）
python -c "import sys,pandas as pd,config as C,data_pipeline as DP; df=pd.read_csv(C.DATA_DIR/'clean.csv',parse_dates=['datetime']); st=DP.station_table(df); _,lab,ks=DP.cluster_stations(DP.station_profiles(df),seed=0); st['cluster']=lab; st.round(4).to_csv(C.DATA_DIR/'stations.csv',index=False); st[['station_id','cluster']].to_csv(C.RES_DIR/'E1_cluster.csv',index=False); ks.to_csv(C.RES_DIR/'E1_k_selection.csv',index=False)"
# 4) 跑预测层（约 38 分钟，i9-14900HX / N_JOBS=-1）
python forecast.py
# 5) 出图出表
python report.py
```

---

## 7. 已知限制（M3 自报，请写进论文的 Limitations）

1. **真实数据是"站点化"后的 UCI**：UCI hour 是全市聚合计数，没有站点维度，
   按各站权重做多项式摊分得到。这是 §4.1 必须如实交代的，否则会被质疑。
2. **E1 聚类信息量低**：真实数据上 k*=2 但轮廓系数仅 0.181，簇分布 13:27。
   原因是各站只是规模不同、形状几乎相同，`station_profiles` 又按站归一化掉了规模。
3. **线性与 Ridge 数值相同**：不是 bug。Ridge α=1 在 48,000+ 训练样本、12 特征下
   预测只差 9e-4，四舍五入到 4 位小数就消失了。建议 §4.3 解释一句，或把 α 调大。
4. **存在负预测**：linear 5,076 行、ffbpn 1,970 行（最低 −1.25）。需求物理非负，
   可在 §4.3 提一句或做 clip（会改变指标，需重跑）。
5. **测试段指标高于交叉验证均值**：Fig.3 的测试段 RF R²=0.830，而 Tab.1 的五折均值是 0.785。
   因为测试段是最后 20% 时间窗、训练集更大，且早期折训练数据少。**图注不要与 Tab.1 的数字混用**。
6. **FFBPN 训练慢且对环境敏感**：本机 MLP 单线程，3 个种子 594 秒；原机约 14 秒。
   `train_time` 不可跨机器比较，论文里必须注明 `N_JOBS` 与 numpy 版本。
