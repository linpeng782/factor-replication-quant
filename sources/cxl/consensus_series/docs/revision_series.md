# 卖方分析师预期 · 调整/上调/覆盖系列

> up_ratio_fy1/fy2、adj_pct_fy1/fy2、cnts_ana_rpt90 共 5 因子 | 数据源
> `get_consensus_indicator(fiscal_year=Y, date_rule='rice_create_tm')` 报告级明细

## 1. 共享流水线（事件 → 日频截面）
1. **剔非个股报告**：report_main_id 数字主体 == order_book_id 主体。实测全市场只有 ~16%
   报告是真·个股盈利预测，其余 84% 是行业/策略研报误挂到成分股（report_main_id=None 或
   指向别的主体），必须剔除——否则"分析师预期"被行业噪声淹没。
2. **首序分析师**：lead = author 逗号分隔首位（同一篇报告归属第一作者）。
3. **1% 重复剔除**：同 (股, lead) 相邻预测相对偏离 < 1% 视为重复报告，不产出调整事件、不更新基准。
4. **调整事件**：每 lead 本次 net_profit 预测 vs 其上一篇 → adj=(new−prev)/|prev|、is_up=1{adj>0}。
5. **可回溯区间**：近似 lookback=180 天（约两个财报周期；分析师覆盖稀疏，半年比研报原文
   "上期财报发布前5天至今"更稳）。每股每日取活跃 lead 最新观点聚合，活跃数 < 3 置 NaN。
6. **fy1/fy2 滚动**：fiscal_year=Y 在 [Y-05-01, (Y+1)-04-30] 区间作 fy1（年报披露后切换，
   与 comp 的 report_year_t 滚动平均时点一致）。fy2 = 同区间同 fiscal_year 文件、value_col
   换成 net_profit_t1（对 Y+1 年的预测）。

各因子差异仅第 5 步聚合：up_ratio=上调数/总数；adj_pct=adj 均值；cnts_ana_rpt90=过去 90 天
不同 lead 数（月末快照+ffill，4s 全量）。

## 2. 复现结果（20160101–20251231, 全市场, n=5×g=5）
| 因子 | 方向 | neu IC 5/10/20d | neu ICIR20 | 单调性 | LS Sharpe |
|---|---|---|---|---|---|
| up_ratio_fy1 | +1 | 0.017/0.020/0.023 | 0.25 | 0.95 | 1.34 |
| up_ratio_fy2 | +1 | 0.016/0.021/0.026 | 0.27 | 0.96 | 1.57 |
| adj_pct_fy1  | +1 | 0.014/0.016/0.019 | 0.19 | 0.92 | 0.94 |
| adj_pct_fy2  | +1 | 0.014/0.017/0.021 | 0.20 | 0.99 | 1.23 |
| cnts_ana_rpt90 | +1 | 0.013/0.017/0.019 | 0.25 | 0.89 | 1.27 |

## 3. 非平凡洞察 ⭐
**(a) 漂移效应是真信号的指纹。** 5 个因子的 IC 一律 5d<10d<20d 单调递增——盈利预测调整的
信息缓慢扩散进价格（PEAD 的分析师版），这是教科书级的事件漂移特征，强烈佐证因子捕捉的是
真实的预期修正而非噪声/前视（前视污染往往表现为短期 IC 畸高后衰减，与此相反）。

**(b) cnts_ana_rpt90 的"中性化悖论"。** cleaned 版极弱（IC 0.008/ICIR 0.06/单调 0.68），
neu 版反而显著变强（IC 0.013/ICIR 0.17/单调 0.89）。原因：分析师覆盖数与市值高度共线（大票
覆盖多），cleaned 版信号几乎是市值代理；行业市值中性化剥离市值后，残差"同市值下关注度更高"
才显出正 alpha。**教训：覆盖/关注度类因子必须看中性化版，cleaned 版会被市值掩盖。**

**(c) fy1 ≈ fy2，fy2 甚至略强。** 远一年预测的信息没有衰减（up_ratio_fy2 LS Sharpe 1.57 >
fy1 1.34），说明分析师对 fy2 的修正同样被市场缓慢吸收，且 fy2 修正更"领先"。

## 4. 待确认/改进
- 可回溯区间用固定 180 天近似了研报的"上期财报发布前5天至今"；接入 `get_pit_financials_ex`
  的 info_date 可精确化（trds_from_issue_stmt_mrq 辅助指标即为此）。
- date_rule 用 rice_create_tm（入库时间）做 PIT，比 rpt_dt（研报落款）更防前视；漂移效应的
  存在反向印证了没有前视泄漏。

## 5. 引用
- 计算：`scripts/consensus_factors.py`（clean_reports/build_events/build_revision_panels/build_count_panel）
- 数据：`data_fetching/consensus.py::load_or_fetch_consensus_reports`
- 评估图：`output/cxl/consensus_series/<factor>/evaluation_20160101_20251231__{cleaned,neu}.png`
