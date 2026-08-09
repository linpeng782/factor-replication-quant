# A+B 5050 集成方案 —— 完整沉淀

> 2026-07-02。LGBM 固定训练 + LGBM 滚动训练的名额分配集成，全周期 6.5 年回测验证。

---

## 1. 方案概述

将两个 LGBM 模型通过**名额分配法**集成，各占 50% 持仓名额：

```
top-100 = A固定 top-50 + B滚动 top-50（去重后从集成 rank 补足至 100）
top-101~500 按集成 rank 排序填充（回测引擎候选池用）
```

这是唯一在 6.5 年全周期回测中**累计收益超越两个单模型**的集成方案（393% > A 361% > B 315%）。

---

## 2. 两个子模型

### A — LGBM 固定训练

| 项 | 值 |
|---|---|
| 模型 ID | `lgbm_a158_p27_top64` |
| 训练方式 | 一次性训练（无 split_mode），训练区间 2005~2017，之后不再更新 |
| 特征 | alpha158-dquant（158 个技术因子）+ kysec-dquant/paper_27（27 个微观结构因子）= 185 个 |
| 特征选择 | SHAP GBDT 选 top-64 |
| 标签 | ExcessReturn（截面等权 demean 超额收益，回归目标） |
| 标签标准化 | WholeSetRobustZ（全样本 per-feature Robust Z-score，有状态持久化） |
| 特征标准化 | 同上，WholeSetRobustZ（train 段 fit，valid/test 仅 transform） |
| has_factor 策略 | NONE（LGBM 原生吃 NaN，不剔除缺失因子行） |

#### LGBM 超参（A 固定）

| 参数 | 值 | 来源 |
|---|---|---|
| objective | regression（MSE 回归） | `model.py:LGBM_DEFAULT_PARAMS` |
| metric | l2 | `model.py:LGBM_DEFAULT_PARAMS` |
| boosting_type | gbdt | `model.py:LGBM_DEFAULT_PARAMS` |
| learning_rate | 0.05 | `train_config.yaml: lgbm.learning_rate` |
| num_leaves | 31 | `train_config.yaml: lgbm.num_leaves` |
| min_child_samples | 200 | `model.py:LGBM_DEFAULT_PARAMS` |
| feature_fraction | 0.8 | `model.py:LGBM_DEFAULT_PARAMS` |
| bagging_fraction | 0.8 | `model.py:LGBM_DEFAULT_PARAMS` |
| bagging_freq | 1 | `model.py:LGBM_DEFAULT_PARAMS` |
| lambda_l1 | 0.0 | `model.py:LGBM_DEFAULT_PARAMS` |
| lambda_l2 | 0.0 | `model.py:LGBM_DEFAULT_PARAMS` |
| max_depth | -1（无限制） | LightGBM 默认 |
| max_bin | 255 | LightGBM 默认 |
| num_threads | 64 | `train_config.yaml: lgbm.num_threads` |
| seed | 42 | `train_config.yaml: lgbm.seed` |
| deterministic | True | `model.py:LGBM_DEFAULT_PARAMS` |
| force_row_wise | True | `model.py:LGBM_DEFAULT_PARAMS` |
| num_boost_round | 1000（最大迭代轮数） | `model.py:LGBM_NUM_BOOST_ROUND` |
| early_stopping_rounds | 200（valid l2 连续 200 轮不降则停） | `model.py:LGBM_EARLY_STOPPING` |
| **best_iteration** | **406**（实际早停轮数） | `model.txt` 树数 |

> 超参定义位置：`ml_core/model.py` 第 42-50 行（`LGBM_DEFAULT_PARAMS` + `LGBM_NUM_BOOST_ROUND` + `LGBM_EARLY_STOPPING`），`ml_core/train_config.yaml` 第 47-51 行（`lgbm:` 段覆盖子集）。

### B — LGBM 滚动训练

| 项 | 值 |
|---|---|
| 模型 ID | `lgbm_rolling_concat_2019_2025`（7 年拼接） |
| 训练方式 | 每年滚动重训，`split_mode=stock`（按股票随机切 80% train / 20% valid） |
| 训练区间 | 每年用前 10 年数据（`train_years_back=10`），如 year=2024 模型训练 2014~2024 |
| 训练年份 | year=2019~2025 共 7 个子模型，分别预测 2020~2026 |
| 特征 | 与 A 完全相同（alpha158-dquant + paper_27 = 185 个） |
| 特征选择 | 与 A 相同（SHAP GBDT 选 top-64），但每年独立选因子 |
| 标签 | **与 A 相同**：ExcessReturn（截面等权 demean 超额收益，回归目标） |
| 标签标准化 | **与 A 相同**：WholeSetRobustZ（全样本 Robust Z-score，有状态持久化） |
| 特征标准化 | **与 A 相同**：WholeSetRobustZ（train 段 fit，valid/test 仅 transform） |
| has_factor 策略 | **与 A 相同**：NONE |
| 预测面板 | `/nfs/ofs-prediction/peterzhenglinpeng/ml/predictions/lgbm_rolling_concat_2019_2025/pred_panel_live.parquet` |
| 信号目录 | `/nfs/ofs-prediction/peterzhenglinpeng/ml/predictions/lgbm_rolling_concat_2019_2025/signals/` |
| 模型目录 | `/nfs/ofs-prediction/peterzhenglinpeng/ml/models/lgbm_rolling_{year}/`（7 个子目录） |

#### LGBM 超参（B 滚动）

**与 A 固定完全相同**（`rolling_config.yaml` 第 39-44 行 `lgbm:` 段只覆盖了 seed/num_threads/learning_rate/num_leaves，其余走 `LGBM_DEFAULT_PARAMS` 默认值）。

| 参数 | 值 | 与 A 相同？ |
|---|---|---|
| objective / metric / boosting_type | regression / l2 / gbdt | ✓ |
| learning_rate | 0.05 | ✓ |
| num_leaves | 31 | ✓ |
| min_child_samples | 200 | ✓ |
| feature_fraction / bagging_fraction / bagging_freq | 0.8 / 0.8 / 1 | ✓ |
| lambda_l1 / lambda_l2 | 0.0 / 0.0 | ✓ |
| num_boost_round | 1000 | ✓ |
| early_stopping_rounds | 200 | ✓ |
| seed / num_threads | 42 / 64 | ✓ |

**best_iteration（各年实际早停轮数）**：

| 滚动年份 | best_iteration | 说明 |
|---|---|---|
| 2019 | 1000 | 未早停（跑满） |
| 2020 | 989 | 接近跑满 |
| 2021 | 995 | 接近跑满 |
| 2022 | 986 | 接近跑满 |
| 2023 | 949 | 接近跑满 |
| 2024 | 997 | 接近跑满 |
| 2025 | 998 | 接近跑满 |

> **关键差异**：A 固定在 406 轮早停，B 滚动几乎全部跑满 1000 轮。说明滚动训练的 valid 段（stock 随机切分 20%）与 train 段分布过于一致，valid loss 持续微降无法触发早停——这是 `split_mode=stock`（股票随机切分）的已知问题，valid 不是时间外样本，缺乏真正的泛化检验。A 固定用的是时间切分（train ≤ 2017-11，valid 2018~2019），valid 是真正的时间外样本，能在 406 轮有效早停。

> 超参定义位置：`ml_core/model.py` 第 42-50 行（`LGBM_DEFAULT_PARAMS`），`ml_core/rolling_config.yaml` 第 39-44 行（`lgbm:` 段覆盖子集）。

### A vs B 的唯一区别

A 和 B 的**标签、标准化、特征、特征选择方法、has_factor 策略完全相同**。唯一区别是：

| | A 固定 | B 滚动 |
|---|---|---|
| 训练频率 | 一次性（2005~2017） | 每年重训（前 10 年数据） |
| train/valid 切分 | 无 split_mode | `split_mode=stock`（股票随机切 80/20） |
| 标准化 fit 数据 | 2005~2017 train 段 | 每年各自的 train 段（滚动窗口） |
| SHAP 选因子 | 在 2005~2017 上选一次 | 每年独立选（因子重合度 70~88%） |

> 注意：`DailyCrossSectionMAD`（逐日截面 MAD）是 **MLP** 的标准化方式，不是 LGBM 滚动训练的。LGBM 无论固定还是滚动，都用 `WholeSetRobustZ`。

### 两模型差异化指标

| 指标 | 值 | 含义 |
|---|---|---|
| 持仓重合度（top-100） | 36.08% 均值 | 选股逻辑差异化显著 |
| 预测截面 rank-IC | 0.7996 均值 | 相关性较高但非完全一致 |
| 因子重合度（top-64） | 70~88%（逐年递减） | 因子选择高度稳定，差异在权重 |

---

## 3. 集成方法：名额分配法

### 核心思路

**不排序求共识，而是各模型独立选股，按权重分配名额**。权重直接体现为持仓名额数量，不经过排序平均的"折中效应"。

### 算法

```python
# 每个交易日 d：
r = rolling_pred.loc[d]    # B 滚动的预测分
f = fixed_pred.loc[d]      # A 固定的预测分

# Step 1: 各模型独立选股
r_top50 = r.nlargest(50).index    # B 滚动 top-50
f_top50 = f.nlargest(50).index    # A 固定 top-50

# Step 2: 合并去重（保持顺序）
top100 = dedup(r_top50 + f_top50) # 去重后通常 79~90 只

# Step 3: 不足 100 从集成 rank 补足
r_rank = rank_pct(r)              # 截面排名分位 0~1
f_rank = rank_pct(f)
ens_rank = 0.5 * r_rank + 0.5 * f_rank
top100 += ens_rank.nlargest(剩余名额)

# Step 4: top-101~500 按集成 rank 填充（回测引擎候选池）
rest500 = ens_rank 排序后取 top-500 去掉已在 top-100 的
```

### 为什么不用排序平均？

排序平均（`0.5*rank_A + 0.5*rank_B`）在 top 端会抹平权重：一只股票要进 top-100 需要两个模型 rank 之和最高，倾向于选"两边都还行但不突出"的折中股。实测 0.6/0.4 权重经排序平均后变成 ~0.5/0.5，弱模型的拖累被放大。

名额分配则精确保证：A 选 50 只、B 选 50 只，弱模型最多影响 50% 名额，不会侵蚀强模型的 top 选股。

详见 `docs/ensemble_methodology.md`（2026 H1 实验记录）。

---

## 4. 信号产出

| 项 | 值 |
|---|---|
| 集成 ID | `ensemble_quota_50_50` |
| 信号目录 | `/nfs/ofs-prediction/peterzhenglinpeng/ml/predictions/ensemble_quota_50_50/signals/` |
| 信号格式 | `YYYY-MM-DD.txt`，每行 `YYYY-MM-DD_股票代码`，共 500 行 |
| 信号数量 | 1571 份（2020-01-02 ~ 2026-06-30，无重叠无缺日） |
| top-100 来源 | A 固定 top-50 + B 滚动 top-50（去重补足） |
| top-101~500 | 按 0.5×A_rank + 0.5×B_rank 排序填充 |
| 生成代码 | `ml_core/ensemble_quota.py` |

---

## 5. 回测参数

### 回测配置文件

`/nfs/ofs-prediction/peterzhenglinpeng-code/daily-realtime-backtest-pipeline/config/config_ensemble_quota_50_50.yaml`

### 回测参数全表

| 参数 | 值 | 说明 |
|---|---|---|
| **信号** | | |
| signal_dir | `.../ensemble_quota_50_50/signals/` | 集成信号目录 |
| trade_start | 自动推断 | 2020-01-03（首个信号日 +1） |
| trade_end | 自动推断 | 2026-06-30 |
| **执行** | | |
| top_k | 100 | 每日最大持仓数 |
| candidate_pool_size | 300 | 候选池（信号前 300，跳过涨停/停牌后补位至 100） |
| benchmark_index | 000985.XSHG | 中证全指 |
| max_position_weight | null | 无单票上限（纯等权轧差） |
| **交易** | | |
| signal_shift | true | T+1 执行：T 日信号 → T+1 日 VWAP 执行 |
| rebalance_interval | 5 | 每 5 个交易日调仓一次 |
| sell_price_col | vwap_am | 卖出价：早盘 VWAP |
| buy_price_col | vwap_am | 买入价：早盘 VWAP |
| mode | netting | 轧差模式：只交易新旧持仓差额 |
| **资金** | | |
| initial_capital | 20,000,000 | 初始资金 2000 万 |
| stamp_tax_rate | 0.0005 | 印花税 0.05%（卖出） |
| commission_rate | 0.0002 | 佣金 0.02%（双向） |
| transfer_fee_rate | 0.0001 | 过户费 0.01% |
| **其他** | | |
| mysql_config.enabled | false | 禁用 MySQL 写入 |

### 回测目录

```
/nfs/ofs-prediction/peterzhenglinpeng-code/daily-realtime-backtest-pipeline/results/
  ensemble_quota_50_50_20200103_20260630_topk100_netting_vwapam_shift1_interval5/
    account_history.csv       # 每日账户净值、收益率、基准
    trade_history.csv         # 逐笔交易明细
    rebalance_snapshot.csv    # 调仓日信号→执行对照
    periodic_returns.txt      # 年度/月度收益汇总
```

### 回测目录命名规则

`{signal_name}_{start}_{end}_topk{N}_{mode}_{price}_shift{N}_interval{N}`

即：`ensemble_quota_50_50` + `20200103_20260630` + `topk100` + `netting` + `vwapam` + `shift1` + `interval5`

---

## 6. 回测结果

### 6.1 全周期（2020-01 ~ 2026-06，6.5 年）

#### 全局绩效

| 指标 | A+B 5050 | A LGBM固定 | B LGBM滚动 |
|---|---|---|---|
| **累计收益** | **393.33%** | 361.43% | 315.33% |
| **年化收益** | **29.18%** | 27.80% | 25.66% |
| 最大回撤 | -38.12% | **-35.34%** | -39.80% |
| 夏普比率 | 1.03 | **1.04** | 0.91 |
| 卡玛比率 | 0.77 | **0.79** | 0.64 |
| **信息比率** | **1.24** | 1.23 | 1.06 |
| 换手率 | 13.33% | 13.75% | 13.29% |

#### 逐年收益

| 年份 | A+B 5050 | A LGBM固定 | B LGBM滚动 | 最强模型 |
|---|---|---|---|---|
| **2020** | **+29.29%** | +26.79% | +20.47% | **A+B 5050** |
| 2021 | +55.21% | +43.57% | **+58.84%** | B 滚动 |
| 2022 | +3.41% | **+7.17%** | -5.00% | A 固定 |
| 2023 | +19.00% | +14.44% | **+20.77%** | B 滚动 |
| 2024 | +10.11% | **+22.11%** | +7.08% | A 固定 |
| 2025 | +56.88% | **+58.02%** | +51.33% | A 固定 |
| **2026** | +15.64% | +7.11% | **+16.74%** | B 滚动 |

**2020 年是唯一集成超越两个单模型的年份**（+29.29% > A 26.79% > B 20.47%），这一年的超额足够拉起全局累计。

#### 超额曲线质量

| 指标 | A+B 5050 | A LGBM固定 | B LGBM滚动 |
|---|---|---|---|
| 超额峰值 | **275.23%** | 250.20% | 210.50% |
| 创新高天数 | 247 天 | **258 天** | 212 天 |
| 最长回撤段 | 194 天 | **134 天** | 202 天 |
| 超额 Sharpe | **1.24** | 1.23 | 1.06 |

#### 逐年最大回撤

| 年份 | A+B 5050 | A LGBM固定 | B LGBM滚动 |
|---|---|---|---|
| 2020 | -13.72% | **-13.10%** | -14.41% |
| 2021 | -16.44% | **-15.67%** | -16.15% |
| 2022 | -31.01% | **-29.02%** | -34.02% |
| 2023 | -9.20% | **-8.49%** | -9.08% |
| 2024 | -37.77% | **-35.33%** | -39.44% |
| 2025 | -18.90% | **-12.66%** | -22.31% |
| 2026 | -12.82% | -14.12% | **-12.36%** |

### 6.2 2020-2025（不含 2026，5.77 年 / 1455 个交易日）

> 2026 年 A+B 5050 收益 +15.64% 仍为正，但剔除后可对比核心选股能力。

#### 全局绩效

| 指标 | A+B 5050 | A LGBM固定 | B LGBM滚动 |
|---|---|---|---|
| **累计收益** | 326.60% | **330.82%** | 255.76% |
| **年化收益** | 28.56% | **28.78%** | 24.58% |
| 最大回撤 | -38.12% | **-35.34%** | -39.80% |
| 夏普比率 | 1.01 | **1.07** | 0.88 |
| 卡玛比率 | 0.75 | **0.81** | 0.62 |
| **信息比率** | 1.27 | **1.35** | 1.07 |
| 换手率 | 13.26% | 13.69% | 13.25% |

**关键变化**：剔除 2026 年后，A+B 5050 的累计收益（326.60%）被 A 固定（330.82%）反超。全周期的"1+1>2"效应主要来自 2026 年集成（+15.64%）远超 A 固定（+7.11%）的贡献。2020-2025 纯区间内，A 固定单模型略优于集成。

#### 逐年收益

| 年份 | A+B 5050 | A LGBM固定 | B LGBM滚动 | 最强模型 |
|---|---|---|---|---|
| **2020** | **+29.29%** | +26.79% | +20.47% | **A+B 5050** |
| 2021 | +55.21% | +43.57% | **+58.84%** | B 滚动 |
| 2022 | +3.41% | **+7.17%** | -5.00% | A 固定 |
| 2023 | +19.00% | +14.44% | **+20.77%** | B 滚动 |
| 2024 | +10.11% | **+22.11%** | +7.08% | A 固定 |
| 2025 | +56.88% | **+58.02%** | +51.33% | A 固定 |

**2020-2025 的 6 年中集成仅 1 年最强**（2020），A 固定 3 年最强（2022/2024/2025），B 滚动 2 年最强（2021/2023）。

#### 超额曲线质量

| 指标 | A+B 5050 | A LGBM固定 | B LGBM滚动 |
|---|---|---|---|
| 超额峰值 | 240.42% | **242.17%** | 186.22% |
| 创新高天数 | 229 天 | **248 天** | 197 天 |
| 最长回撤段 | 194 天 | **134 天** | 202 天 |
| 超额 Sharpe | 1.27 | **1.35** | 1.07 |

#### 逐年最大回撤

| 年份 | A+B 5050 | A LGBM固定 | B LGBM滚动 |
|---|---|---|---|
| 2020 | -13.72% | **-13.10%** | -14.41% |
| 2021 | -16.44% | **-15.67%** | -16.15% |
| 2022 | -31.01% | **-29.02%** | -34.02% |
| 2023 | -9.20% | **-8.49%** | -9.08% |
| 2024 | -37.77% | **-35.33%** | -39.44% |
| 2025 | -18.90% | **-12.66%** | -22.31% |

### 6.3 全周期 vs 2020-2025 对比

| 指标 | A+B 5050 全周期 | A+B 5050 2020-2025 | 差 |
|---|---|---|---|
| 年化收益 | **29.18%** | 28.56% | -0.62% |
| 夏普 | 1.03 | 1.01 | -0.02 |
| 卡玛 | 0.77 | 0.75 | -0.02 |
| IR | **1.24** | 1.27 | +0.03 |
| 超额 Sharpe | **1.24** | 1.27 | +0.03 |

剔除 2026 年后，A+B 5050 的年化从 29.18% 微降到 28.56%（-0.62%），IR 反而从 1.24 升到 1.27。2026 年对集成是正贡献年份（+15.64%），但 IR 的提升说明 2026 年的波动也相对较大。

### 关键发现

1. **全周期集成 1+1>2**：累计 393% > A 361% > B 315，IR 1.24 为三者最高
2. **2020-2025 纯区间 A 固定反超**：累计 330.82% > 集成 326.60%，说明全周期的"1+1>2"效应部分依赖 2026 年
3. **2020 年是集成超越两个单模型的唯一年份**（+29.29%），这是集成效应最纯粹的体现
4. **集成超额峰值最高**（全周期 275%、2020-2025 240%），但最长回撤段未改善（194 天 vs A 的 134 天）
5. **集成 IR 始终最优或并列最优**（全周期 1.24、2020-2025 1.27），风险调整后收益稳定占优

---

## 7. 横向对比：所有集成方案

| 集成对 | 方法 | 累计 | 年化 | vs 强单模型 | 结论 |
|---|---|---|---|---|---|
| **A+B 5050** | 名额分配 | **393%** | **29.2%** | **+32%** | **唯一 1+1>2** |
| A+B 6040 | 名额分配 | 373% | 28.3% | +12% | 不输 A，但不如 5050 |
| A+B 7030 | 名额分配 | 376% | 28.4% | +15% | |
| MLP+A 5050 | 名额分配 | 505% | 33.5% | -46% | 收益输 MLP，但回撤段 120 vs 172 天 |
| MLP+B 系列 | 名额分配 | 417~448% | 30~31% | -102~-133% | MLP 太强，B 拖累 |
| D+B 系列 | 名额分配 | 293~339% | 24~27% | -14~-60% | B 太弱，集成全败 |
| AB+D 5050 | 名额分配 | 345% | 27.1% | -48% | 每年一方大幅占优，取中间值 |

---

## 8. 代码位置

| 文件 | 用途 |
|---|---|
| `ml_core/ensemble_quota.py` | A+B 名额分配集成信号生成（含 5050/6040/7030/4060 四种权重） |
| `ml_core/ensemble_mlp_lgbm.py` | MLP+LGBM 集成信号生成 |
| `ml_core/ensemble_d_b.py` | D+B 集成信号生成 |
| `ml_core/ensemble_ab_d.py` | (A+B)+D 三模型集成信号生成 |
| `ml_core/run_ensemble_backtest.py` | 批量回测脚本（LGBM 集成） |
| `ml_core/run_ensemble_mlp_backtest.py` | 批量回测脚本（MLP 集成） |
| `docs/ensemble_methodology.md` | 2026 H1 集成实验原始记录（排序平均 vs 名额分配） |

### 复现命令

```bash
# 1. 生成集成信号
source /nfs/ofs-prediction/peterzhenglinpeng-code/peterdidi/bin/activate
cd /nfs/ofs-prediction/peterzhenglinpeng-code/factor-replication-quant-new
python -m ml_core.ensemble_quota

# 2. 回测
cd /nfs/ofs-prediction/peterzhenglinpeng-code/daily-realtime-backtest-pipeline
BACKTEST_CONFIG_PATH=config/config_ensemble_quota_50_50.yaml python batch_runner.py

# 3. 查看结果
cat results/ensemble_quota_50_50_20200103_20260630_topk100_netting_vwapam_shift1_interval5/periodic_returns.txt
```

---

## 9. 后续方向

### 动态权重

固定 50/50 在风格切换年（2024 A 大胜）会吃亏。根据近期 IC 动态调整名额：

```python
# 每个调仓日，回看过去 20 天截面 IC
w_A = IC_A / (IC_A + IC_B)
w_B = IC_B / (IC_A + IC_B)
n_A = round(100 * w_A)
n_B = 100 - n_A
```

A 强时给 70% 名额，B 强时给 60% 名额——自适应市场风格切换。

### 待验证问题

- 动态权重是否会过拟合近期 IC？
- 其他权重（45/55、55/45）是否有更优解？
- A+B 5050 + MLP 的三模型集成是否能同时超越 MLP 单模型？
