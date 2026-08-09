# 改进动量因子族 wgt_return / exp_wgt_return（华泰证券 2016-12-20，多因子系列之四）

因子定义与研报原文引证见 `../paper.md`。本文记录实现与复现结论（8 因子共用一份）。

## 1. 因子原理

传统 N 月动量 return_Nm 在 A 股是反转因子。华泰的改进：**用日换手率给每日收益加权**。
直觉：换手率高的交易日，价格由更多"新进场投资者"决定，其定价错误（过度反应）更大，
后续反转也更强 → 给这些日子更高权重，能放大反转信号。

- `wgt_return_Nm` = Σ(turnover_i · ret_i) / Σ(turnover_i)，窗口 = 20N 交易日
- `exp_wgt_return_Nm` = 同上，但权重再乘 `exp(−x_i/(4N))`（x_i = 距截面日交易日数，
  当日=0）——越近的交易日权重越高，衰减尺度 4N 日。

## 2. 实现（新建件）

| 件 | 说明 |
|---|---|
| `data_fetching/turnover_rate_dquant.py` | dquant `get_turnover_rate.today` → `turnover_rate_panel`（T×N，%，2006-01 起） |
| `scripts/build_ret20_panel.py --window 1` | 后复权日收益面板 `ret1_panel`（原脚本泛化出 `--window`） |
| `core/operators/rolling_weighted_mean.py` | **新 L3 算子**：滚动归一化加权平均 Σ(d·w·v)/Σ(d·w)，`decay_scale` 可选指数衰减 |
| `core/operators/load_panel.py`（改） | 主表不存在时**用面板创建主表**（按 universe + fetch 区间裁剪），纯面板型因子起手式 |
| `specs/{wgt,exp_wgt}_return_{1,3,6,12}m` | 8 个 spec，各 3 步（load ret1 → load turnover → rolling_weighted_mean） |

算子实现：长表 → 宽表 → `for i in range(window)` 移位累加分子/分母/有效天数
（O(W·T·N) 纯 numpy，无逐股循环）；值或权重任一 NaN 的日子整体不进分子分母，
有效天数 < window//2 → NaN。全市场 240 日窗口约 25 秒。

## 3. 复现口径的两处拍板（研报表述含糊）

1. **"以换手率为权重求算术平均"** → 取**归一化加权平均** `Σ(w·r)/Σ(w)`（与摘要
   "换手率加权平均日收益率"一致），而非业界常见的 `mean(w·r)`。后者量纲含换手率
   水平，会与流动性因子强共线。
2. **N 月 = 20N 交易日**（1/3/6/12 → 20/60/120/240），`exp` 衰减尺度 = 4N 日。

## 4. 复现结果

**月频 IC 对照研报图表115**（neu=行业市值中性化，月末截面 vs 次月 vwap 收益）
`PYTHONPATH=. python scripts/htsec_momentum_verify.py`

| 因子 | 我 IC(2006-01~2016-11) | 我 IR | 论文 IC | 论文 IR | 我 IC(2016-01~2026-07) | 我 IR |
|---|---|---|---|---|---|---|
| wgt_return_1m | −7.98% | 0.98 | −7.24% | 0.92 | −5.88% | 0.74 |
| wgt_return_3m | −7.48% | 0.93 | −6.12% | 0.74 | −6.74% | 0.82 |
| wgt_return_6m | −7.01% | 0.86 | −5.16% | 0.60 | −6.59% | 0.77 |
| wgt_return_12m | −5.65% | 0.72 | −4.18% | 0.51 | −5.80% | 0.67 |
| exp_wgt_return_1m | −5.30% | 0.60 | −6.04% | 0.75 | −2.10% | 0.28 |
| exp_wgt_return_3m | −8.36% | 1.00 | −7.74% | 0.96 | −5.72% | 0.70 |
| exp_wgt_return_6m | **−9.26%** | **1.14** | −7.70% | 0.92 | −7.15% | 0.84 |
| exp_wgt_return_12m | −8.84% | 1.07 | −6.86% | 0.78 | −7.62% | 0.86 |

注：研报"IR 比率"是**月频** mean/std（非年化），由其"IC>0 占比"栏反推确认
（如 wgt_1m IR 0.92 ↔ 占比 18.18% ≈ Φ(−0.92)）。

日频评估（`run.py`，2016-2025，5 日 IC，dir=−1）：cleaned IC5d 排序
exp_6m 0.0793 > exp_12m 0.0767 > exp_3m 0.0737 > wgt_1m 0.0681 > wgt_3m 0.0663
> wgt_6m 0.0588 > exp_1m 0.0530 > wgt_12m 0.0472；单调性普遍 0.8~0.92。

**结论：论文区间量级与排序均对上**（研报 line 1008 称最强为 exp_3m/exp_6m，我们
exp_6m/exp_3m/exp_12m 居前）；8 个因子在 2016 后仍有效，仅 `exp_wgt_return_1m`
显著衰减（IR 0.60 → 0.28）。

## 5. 非平凡洞察

1. **短窗 + 强衰减 = 现代失效**：exp_wgt_return_1m（窗口 20 日、衰减尺度仅 4 日，
   有效权重集中在最近 ~8 个交易日）是唯一在 2016 后崩掉的成员（IR 0.28）。极短端
   反转已被 T+0 高频策略充分收割；而窗口越长、衰减越缓（exp_6m/12m）反而越稳。
   **改进动量的 alpha 在"中长端"，不在短端。**
2. **归一化 vs 非归一化不是小事**：若用 `mean(w·r)`，因子量纲 = 收益×换手率，
   截面上等价于"反转 × 流动性"复合，IC 会更高但与换手因子共线；归一化后是纯收益
   量纲，才对得上研报"加权平均日收益率"的语义。
3. **换手率数据起点 2006-01**（dquant `get_turnover_rate` 2005 年缺失）→ 12 月因子
   2007 年起才有值；论文区间 2005-05~2016-11 只能对照 2006-01 之后的部分。
4. **`load_panel` 的建表能力是通用基建**：改造后，"多个宽表面板 → 长表 → 算子链"
   成为纯面板型因子（无需 API fetch）的标准起手式，本族 8 个 + 开源长端动量 2 个
   spec 全部零 API 调用。

## 6. 与基准差距 / 改进路径

- 我方 wgt 系列 IC 系统性比论文强 0.7~1.9pct，exp_1m 弱 0.7pct。可能来源：换手率
  口径（dquant 流通股换手率 vs Wind）、股票池（我们含全 A 剔 ST/停牌/新股）。
- 未复现研报同篇的 `HAlpha` 与 `return_Nm`（4 个）——`return_Nm` 用 `rolling` 即可，
  `HAlpha` 需 60 月个股 vs 上证综指回归截距，可用 `rolling_ts_regress` 变体实现。
- 8 因子两两相关性高（研报 line 29 警告多重共线），接入 ML 前应先跑
  `scripts/factor_correlation.py --pattern '*wgt_return*'` 做去冗余。
