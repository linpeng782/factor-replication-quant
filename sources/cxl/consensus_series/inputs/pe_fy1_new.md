# pe_fy1_new — 滚动一致预期 PE

**类别**：价值 | **方向**：-1（PE 越高越贵 → 预期收益越低）

## 研报释义（consensus.png）
分子为总市值，分母为 fy1 滚动一致预期值；每年的 12 月 31 日为基准日切换预测报告期，
每天计算当天距去年 12 月 31 日的时（天数），该时差除以 365 作为 fy1 分析师预测利润均值
的权重，剩余权重分配给 fy2 预测利润，将加权利润作为最终的滚动一致预期，作为分母。

## 工程实现
- 数据源：`get_consensus_comp_indicators(report_range=3)`（不考虑补录入，PIT 稳定）；
  取 `comp_con_net_profit_t1/t2/t3`（= 最近年报年度 report_year_t 的 +1/+2/+3 年一致预期净利润，元）。
- 滚动一致预期 = forward-12m 线性插值：当前日历年 Y，权重 w=当年剩余天数/365 落在 Y，
  (1-w) 落在 Y+1；NP(Y)/NP(Y+1) 按 (Y-report_year_t) 偏移从 t1/t2/t3 取。
- 分子市值用预计算 market_cap 面板（亿元 ×1e8 → 元）。
- 仅保留 forward_np > 0。
