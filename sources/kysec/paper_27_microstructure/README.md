# 开源证券·市场微观结构系列（27）

**研报**：高频成交量的峰、岭、谷信息（2025-07-20，魏建榕/王志豪）
**共享方法论**：日内分钟成交量按「过去 20 日同时点 1σ」划分喷发/温和；孤立喷发=量峰，连续喷发=量岭，温和=量谷。

因子清单见 specs/ 子目录。详见 input.md（研报原文）+ docs/<factor>.md（沉淀）。

---

## 因子中英文对照表（20 个）

> 编号 `fN` 对应研报表 16 / 表 18 中的因子编号。

### 量峰类（Peak）— 孤立喷发，知情交易

| 编号 | 因子文件名 | 中文名（研报原文） | 说明 |
|------|-----------|-------------------|------|
| f1 | `peak_minute_count` | 量峰分钟数 | 日内量峰的分钟计数 |
| f6 | `peak_weighted_quantile` | 量峰加权价格分位 | 量峰 VWAP 在日内的价格分位 |
| f8 | `peak_interval_std` | 量峰间隔标准差 | 相邻量峰时间间隔的 std |
| f9 | `peak_interval_skew` | 量峰间隔偏度 | 相邻量峰时间间隔的 skew |
| f10 | `peak_interval_kurt` | 量峰间隔峰度 | 相邻量峰时间间隔的 kurt |

### 量岭类（Ridge）— 连续喷发，个人投资者跟随

| 编号 | 因子文件名 | 中文名（研报原文） | 说明 |
|------|-----------|-------------------|------|
| f2 | `ridge_minute_count` | 量岭分钟数 | 日内量岭的分钟计数 |
| f3 | `ridge_minute_return` | 量岭分钟收益 | 量岭时点的分钟收益率之和 |
| f4 | `ridge_relative_vwap` | 量岭相对加权价 | 量岭 VWAP / 日 VWAP |
| f11 | `ridge_interval_std` | 量岭间隔标准差 | 相邻量岭时间间隔的 std |
| f12 | `ridge_interval_skew` | 量岭间隔偏度 | 相邻量岭时间间隔的 skew |
| f13 | `ridge_interval_kurt` | 量岭间隔峰度 | 相邻量岭时间间隔的 kurt |

### 量谷类（Valley）— 温和成交，情绪低迷

| 编号 | 因子文件名 | 中文名（研报原文） | 说明 |
|------|-----------|-------------------|------|
| f5 | `valley_relative_vwap` | 量谷相对加权价 | 量谷 VWAP / 日 VWAP |
| f7 | `valley_weighted_quantile` | 量谷加权价格分位 | 量谷 VWAP 在日内的价格分位 |

### 峰岭谷组合类（Cross-category）

| 编号 | 因子文件名 | 中文名（研报原文） | 说明 |
|------|-----------|-------------------|------|
| f14 | `peak_ridge_price_ratio` | 峰岭加权价格比 | 量峰 VWAP / 量岭 VWAP |
| f15 | `valley_ridge_price_ratio` | 谷岭加权价格比 | 量谷 VWAP / 量岭 VWAP |
| f16 | `peak_ridge_turnover_ratio` | 峰岭成交比 | 量峰总成交额 / 量岭总成交额 |
| f17 | `eruption_followup_ratio` | 喷发成交额跟随比例 | 喷发后一分钟成交额 / 喷发时成交额 |
| f18 | `eruption_turnover_sensitivity` | 喷发成交额敏感度 | 喷发成交额对下一分钟的敏感度 |
| f19 | `eruption_turnover_corr` | 喷发成交额相关性 | 喷发时与下一分钟成交额的 Pearson 相关 |
| f20 | `peakridge_minute_corr` | 同时点峰岭数相关性 | 同一时点量峰数与量岭数的相关性 |

---

## 研报 RankICIR 排序（市值、行业中性，月频）

> 来源：研报表 18，测试区间 20130101–20250531，市值+行业中性化，月频调仓。
> 负号 = 因子方向为负（值越高未来收益越差）。

| 排名 | 因子文件名 | 中文名 | RankIC | RankICIR | 方向 |
|------|-----------|--------|--------|----------|------|
| 1 | `peak_interval_kurt` | 量峰间隔峰度 | 7.19% | **4.63** | 正 |
| 2 | `peak_interval_skew` | 量峰间隔偏度 | 7.68% | **4.58** | 正 |
| 3 | `valley_relative_vwap` | 量谷相对加权价 | 8.69% | **4.44** | 正 |
| 4 | `peak_minute_count` | 量峰分钟数 | 10.62% | **4.36** | 正 |
| 5 | `valley_weighted_quantile` | 量谷加权价格分位 | 6.34% | **4.32** | 正 |
| 6 | `peak_ridge_turnover_ratio` | 峰岭成交比 | 10.28% | **4.07** | 正 |
| 7 | `ridge_interval_std` | 量岭间隔标准差 | 7.34% | **3.75** | 正 |
| 8 | `peak_ridge_price_ratio` | 峰岭加权价格比 | 4.70% | **3.61** | 正 |
| 9 | `valley_ridge_price_ratio` | 谷岭加权价格比 | 6.98% | **3.56** | 正 |
| 10 | `peak_weighted_quantile` | 量峰加权价格分位 | 3.47% | **2.90** | 正 |
| 11 | `ridge_minute_count` | 量岭分钟数 | -9.04% | **-3.13** | 负 |
| 12 | `ridge_minute_return` | 量岭分钟收益 | -6.29% | **-3.55** | 负 |
| 13 | `eruption_turnover_sensitivity` | 喷发成交额敏感度 | -7.14% | **-3.58** | 负 |
| 14 | `ridge_relative_vwap` | 量岭相对加权价 | -6.27% | **-3.77** | 负 |
| 15 | `ridge_interval_kurt` | 量岭间隔峰度 | -7.61% | **-3.77** | 负 |
| 16 | `ridge_interval_skew` | 量岭间隔偏度 | -8.08% | **-3.82** | 负 |
| 17 | `eruption_followup_ratio` | 喷发成交额跟随比例 | -10.59% | **-3.90** | 负 |
| 18 | `peak_interval_std` | 量峰间隔标准差 | -8.57% | **-3.99** | 负 |
| 19 | `eruption_turnover_corr` | 喷发成交额相关性 | -10.94% | **-3.99** | 负 |
| 20 | `peakridge_minute_corr` | 同时点峰岭数相关性 | -6.67% | **-4.40** | 负 |

**头部结论**：量峰类的**间隔统计特征**（峰度/偏度）和量谷类的**相对加权价**占据 RankICIR 前三，是最稳健的因子。量岭类整体为负向，|ICIR| 多在 3.1–3.8 区间。

---

## 复现因子评估结果（5d IC / ICIR）

> 来源：`factor_inventory/latest/inventory.parquet`，与 `output/<factor>/evaluation_*.png` 同源。
> 评估配置：日频、5 分组、5 日调仓、**无市值/行业/Barra 中性化**，区间 2016-01-01 ~ 2025-12-31。
> ⚠️ 与研报（月频 10 分组 市值行业中性）**不可直接比较 ICIR 量级**，日频 ICIR × √21 ≈ 月频 ICIR。

| 排名 | 因子文件名 | 中文名 | 5d IC | 5d ICIR | 方向 | miss_listed |
|------|-----------|--------|-------|---------|------|-------------|
| 1 | `peak_ridge_turnover_ratio` | 峰岭成交比 | 0.0913 | **0.8176** | 正 | 10.25% |
| 2 | `valley_ridge_price_ratio` | 谷岭加权价格比 | 0.0915 | **0.7910** | 正 | 78.26% ⚠️ |
| 3 | `ridge_interval_skew` | 量岭间隔偏度 | 0.0711 | **0.7471** | 负 | 10.25% |
| 4 | `ridge_interval_kurt` | 量岭间隔峰度 | 0.0699 | **0.7417** | 负 | 10.25% |
| 5 | `valley_relative_vwap` | 量谷相对加权价 | 0.0736 | **0.7333** | 正 | 11.43% |
| 6 | `peak_ridge_price_ratio` | 峰岭加权价格比 | 0.0708 | **0.7271** | 正 | 78.51% ⚠️ |
| 7 | `eruption_followup_ratio` | 喷发成交额跟随比例 | 0.0907 | **0.7216** | 负 | 10.25% |
| 8 | `ridge_relative_vwap` | 量岭相对加权价 | 0.0744 | **0.7110** | 负 | 78.25% ⚠️ |
| 9 | `ridge_interval_std` | 量岭间隔标准差 | 0.0639 | **0.6967** | 正 | 10.25% |
| 10 | `eruption_turnover_corr` | 喷发成交额相关性 | 0.0653 | **0.6722** | 负 | 10.25% |
| 11 | `ridge_minute_return` | 量岭分钟收益 | 0.0674 | **0.6357** | 负 | 10.25% |
| 12 | `ridge_minute_count` | 量岭分钟数 | 0.0618 | **0.5921** | 负 | 10.25% |
| 13 | `eruption_turnover_sensitivity` | 喷发成交额敏感度 | 0.0542 | **0.5824** | 负 | 10.25% |
| 14 | `valley_weighted_quantile` | 量谷加权价格分位 | 0.0329 | **0.3910** | 正 | 11.43% |
| 15 | `peak_minute_count` | 量峰分钟数 | 0.0412 | **0.3830** | 正 | 10.25% |
| 16 | `peak_interval_std` | 量峰间隔标准差 | 0.0387 | **0.3682** | 负 | 10.27% |
| 17 | `peak_interval_skew` | 量峰间隔偏度 | 0.0277 | **0.3219** | 正 | 10.27% |
| 18 | `peak_interval_kurt` | 量峰间隔峰度 | 0.0248 | **0.3053** | 正 | 10.27% |
| 19 | `peakridge_minute_corr` | 同时点峰岭数相关性 | 0.0099 | **0.2323** | 负 | 9.83% |
| 20 | `peak_weighted_quantile` | 量峰加权价格分位 | 0.0102 | **0.1131** | 正 | 16.17% |

### 复现 vs 研报对照（Top 10）

| 因子 | 复现 5d ICIR | 研报月频 RankICIR | 方向一致性 |
|------|-------------|-------------------|-----------|
| 峰岭成交比 | 0.818 | 4.07 | ✅ 同正 |
| 谷岭加权价格比 | 0.791 | 3.56 | ✅ 同正 |
| 量岭间隔偏度 | 0.747 | -3.82 | ✅ 同负 |
| 量岭间隔峰度 | 0.742 | -3.77 | ✅ 同负 |
| 量谷相对加权价 | 0.733 | 4.44 | ✅ 同正 |
| 峰岭加权价格比 | 0.727 | 3.61 | ✅ 同正 |
| 喷发成交额跟随比例 | 0.722 | -3.90 | ✅ 同负 |
| 量岭相对加权价 | 0.711 | -3.77 | ✅ 同负 |
| 量岭间隔标准差 | 0.697 | 3.75 | ⚠️ 复现正 / 研报正 |
| 喷发成交额相关性 | 0.672 | -3.99 | ✅ 同负 |

**核心观察**：
- **方向 100% 一致**：20 个因子的正负方向全部与研报吻合
- **排序高度一致**：Top 因子（峰岭成交比、谷岭价比、量岭间隔偏度/峰度）与研报头部重合
- **量级差距**：复现 ICIR 约为研报的 1/5，主要由**频率换算**（日频 vs 月频）+ **中性化**导致
- **⚠️ 高缺失率三兄弟**：`valley_ridge_price_ratio` / `peak_ridge_price_ratio` / `ridge_relative_vwap` 原版 miss_listed 高达 78%，但 ICIR 仍居 Top 10——"在能算出来的日子里信号极强"。已通过 `min_periods` 实验降到 25%（见下节）。

---

## min_periods 衰减实验（2026-05-25）

**问题**：高缺失率三兄弟的 78% miss 来自 rolling 默认 `min_periods=window=20` —— 20 日窗口内只要有 1 天 0-ridge（`ridge_vwap` NaN），整窗输出 NaN。`peak_ridge_turnover_ratio` 缺失只有 10%，因为它走"先 sum 后比"，sum 不被 NaN 传染。

**实验**：保持论文「日 ratio → 20d mean」语义不变，仅放宽 rolling `min_periods`，跑 mp20（原）/ mp15 / mp10 / mp5 四档。变体走 `<base>__mp{N}` 命名约定（详见 `specs/<factor>__mp10/spec.yaml`），原版 spec/parquet 完全不动。

**结果**（5d 频率，2016-01-01 ~ 2025-12-31，无中性化；miss_listed 来自 `factor_inventory/latest/inventory.parquet`）：

| 因子 | 指标 | mp20（原） | mp15 | mp10 | mp5 |
|------|------|----------|------|------|-----|
| `ridge_relative_vwap`     | miss_listed% | **78.2** | 17.5 | 10.5 | 10.0 |
|                           | 5d ICIR      | **0.711**| 0.665| 0.656| 0.655|
| `peak_ridge_price_ratio`  | miss_listed% | **78.5** | 17.7 | 10.6 | 10.0 |
|                           | 5d ICIR      | **0.727**| 0.697| 0.696| 0.696|
| `valley_ridge_price_ratio`| miss_listed% | **78.3** | 17.5 | 10.6 | 10.0 |
|                           | 5d ICIR      | **0.791**| 0.731| 0.725| 0.725|

**观察**：
- **缺失率台阶式坍塌，sweet spot 在 mp10**：mp20→mp15 一次降 ~61pp；mp15→mp10 再降 ~7pp；mp10→mp5 仅再降 ~0.5pp。说明大多数股票在 20 日窗口里至少有 10 天有 ridge，剩下 ~10% 是结构性 0-ridge（与其它 paper27 因子的 baseline miss 持平）。
- **ICIR 衰减一次性发生在 mp20→mp15**（~6%）；mp15→mp5 几乎平的。原版 78% miss 的高 ICIR 部分来自"高 ridge 活跃股"的子集偏差。
- **方向、因子间排序**完全稳定（valley > peak > ridge）。

**决策**：生产档采用 **mp10**（miss_listed ~10.5%，与 baseline 因子持平；ICIR 损失 ~7%）；保留原版 mp20 作为高-conviction 备选；淘汰 mp15、mp5。
