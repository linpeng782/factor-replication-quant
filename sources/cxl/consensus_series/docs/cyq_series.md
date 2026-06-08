# cyq_earningest / cnts_cyq_rpt — 业绩超预期

> 数据源：PIT 利润表首披净利润（实现）+ reports 明细（分析师预测）| 事件驱动（财报披露点 T）

## 1. 定义
财报首披点 T（最新报告期 Qk，k=季度号）：
- actual_q = 单季实现净利润 = 当年累计 cum_k − 同年上一季累计 prev_cum；r = 5−k
- 分析师预测：[T−90, T) 内对当年(fy1)做预测的首序分析师，每人最新全年预测 net_profit_t → min/max/count
- 单季预测 q_pred_i = (forecast_i − prev_cum)/r（拆全年预测到单季）
- **cyq_earningest** = (max_i q_pred_i − actual_q)/actual_q（actual_q>0、报告数≥3）
- **cnts_cyq_rpt** = 报告数（首序分析师数）

## 2. 复现结果（20160101–20251231, 全市场）
| 因子 | 方向 | neu IC 5/10/20d | neu ICIR 5/10/20d | 单调性 |
|---|---|---|---|---|
| cyq_earningest | −1 | 0.014/0.017/0.021 | 0.39/0.44/0.54 | 0.995 |
| cnts_cyq_rpt   | +1 | 0.018/0.023/0.030 | 0.31/0.38/0.48 | 0.871 |

## 3. 洞察 ⭐
**(a) 方向 −1 才是"超预期"的正确符号。** 因子是"分析师最乐观单季预测 − 实际单季"的相对差：
值大 = 实际远低于乐观预期（低于预期，利空）；值小甚至负 = 实际兑现超过最乐观预期（真超预期，
利好）。故"因子越小→收益越高"= direction −1，与 PEAD（盈利超预期后正漂移）完全一致。
ICIR 达 0.54、单调性 0.995——是本批最强的景气因子之一，且 IC 量级（0.02）无前视离群。

**(b) PIT 首披 = 利润表+快报的天然拼接。** statements='all' 取每 quarter 最早 info_date，
首披日往往是业绩快报日（早于正式年报），既实现研报"利润表与快报拼接"的时效，又避开
latest 版用追溯调整值的前视——一个字段选择同时解决覆盖度和 PIT 两个问题。

**(c) cnts_cyq_rpt 又见中性化悖论。** cleaned ICIR 0.04（报告数≈市值代理），neu 后 0.48；
关注度类因子必须看中性化版，与 cnts_ana_rpt90 同理。

## 4. 简化/改进
- 快报/预告未单独拉取（PIT 首披已含快报时效）；如需预告区间（forecast_np_floor/ceiling）
  进一步抢跑，可接 performance_forecast。
- "是否超预期"(85 分位≤实际) 辅助布尔未单独落因子，可作 cyq 的二值变体补充。

## 5. 引用
- 计算：`scripts/consensus_factors.py::build_cyq`
- 数据：`data_fetching/consensus.py::load_or_fetch_pit_first_netprofit`
- 评估图：`output/cxl/consensus_series/{cyq_earningest,cnts_cyq_rpt}/evaluation_*__{cleaned,neu}.png`
