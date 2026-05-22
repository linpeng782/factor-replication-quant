# 基本面因子复现进度追踪

> 本文件汇总已定义因子的释义、IC/IR 基准值及复现状态。
> 数据来源：`scripts/fundmental_factors.png` + `scripts/factors_ICIR.png`
> 后续新增因子直接追加到对应分类表格，复现完成后更新状态列。

---

## 一、单因子定义表

| 复现状态 | 因子名称 | 因子释义 | public_tag | private_tag | 备注 |
|---------|---------|---------|-----------|------------|------|
| ⬜ 待复现 | roic_ttm_ind_rnk8 | 以中证800+总市值前100个股为底池，计算过去8期ROIC_TTM行业内排名的最小值 | fundamental.quality.financial | 质量 | |
| ⬜ 待复现 | roic_ttm_dev_std8 | 1 ROIC_TTM / 过去8期ROIC_TTM的标准差 | fundamental.quality.financial | 质量 | |
| ⬜ 待复现 | roic_ttm_all_rnk8 | 以中证800+总市值前100个股为底池，计算过去8期ROIC_TTM排名的最小值 | fundamental.quality.financial | 质量 | |
| ✅ 已复现 | roe_pyoy_mrq | 单季度ROE同比（仅保留分母>0的因子值） | fundamental.growth.financial | 景气 | |
| ✅ 已复现 | roe_pqoq_mrq | 单季度ROE环比（仅保留分母>0的因子值） | fundamental.growth.financial | 景气 | |
| ✅ 已复现 | roe_mrq_new | 单季度ROE | fundamental.growth.financial | 景气 | |
| ✅ 已复现 | roe_ayoy_mrq | 单季度ROE同比（分母取绝对值） | fundamental.growth.financial | 景气 | |
| ✅ 已复现 | roe_apoq_mrq | 单季度ROE环比（分母取绝对值） | fundamental.growth.financial | 景气 | |
| ✅ 已复现 | reg_pe_hist | 经过历史增速变化调整的历史PE变化：取单季度净利润同比、净利润TTM、PE_TTM，分别取log后求60日的diff，先以delta(log(净利润同比))为y、delta(log(净利润ttm))为x回归残差，再以残差为x、delta(log(pe_ttm))为y回归残差 | fundamental.value.financial | 价值 | 详见docs/reg_pe_hist.md |
| ✅ 已复现 | reg_pb_gshe | 经过调整的PB估值因子：在中证全指范围内，取pb_ratio_lf、自行计算roe_mrq和ep，剔除pb和roe不为正的个股，求ep过去一年中位数后去极值截面分10组，组内对roe和log(pb)去极值，按log(pb)~roe回归取残差 | fundamental.value.financial | 价值 | 详见docs/reg_pb_gshe.md |
| ✅ 已复现 | pe_ttm_new | 总市值/TTM净利润 | fundamental.value.financial | 价值 | |
| ✅ 已复现 | pe_ttm_delta60 | PE_TTM的60日差值 | fundamental.value.financial | 价值 | diff已生效，结果与pe_ttm_new不同 |
| ✅ 已复现 | pe_mrq | 总市值/季度净利润 | fundamental.value.financial | 价值 | |
| ✅ 已复现 | npf_pyoy_mrq | 单季度净利润同比（仅保留分母>0的因子值） | fundamental.growth.financial | 景气 | |
| ✅ 已复现 | npf_pqoq_mrq | 单季度净利润环比（仅保留分母>0的因子值） | fundamental.growth.financial | 景气 | |
| ✅ 已复现 | npf_mrq_sue8 | SUE：取过去8个报告期单季度净利润，求净利润差分的均值和标准差，因子值为（去年同期净利润+差分均值）/差分标准差 | fundamental.growth.financial | 景气 | |
| ✅ 已复现 | npf_mrq_accs8 | 单季度净利润增加速度：以过去8期单季度净利润为y，以8个报告期间累计间隔天数为x，分别对x和y做zscore处理后，做二次回归ax²+bx+c，其中a即为因子值；计算前4期和后4期相关性，相关性过高置nan；不足8期置nan | fundamental.growth.financial | 景气 | |
| ✅ 已复现 | npf_ayoy_mrq | 单季度净利润同比（分母取绝对值） | fundamental.growth.financial | 景气 | |
| ✅ 已复现 | npf_apoq_mrq | 单季度净利润环比（分母取绝对值） | fundamental.growth.financial | 景气 | |
| ⬜ 待复现 | np_delta_rank | 最近一期单季度净利润在最近8期单季度净利润的排名 | fundamental.growth.financial | 景气 | |
| ⬜ 待复现 | np_delta_rank | 最近一期单季度净利润环比变化在最近8期单季度净利润环比变化的排名 | fundamental.growth.financial | 景气 | 与上行同名，注意区分 |
| ⬜ 待复现 | net_oper_cash_flow_ttm | 现金流TTM值 | fundamental.quality.financial | 质量 | |

---

## 二、辅助指标定义表

| 复现状态 | 因子名称 | 因子释义 | public_tag | private_tag | 备注 |
|---------|---------|---------|-----------|------------|------|
| ⬜ 待复现 | reverse_d3 | 短期反向久期=(x1×0.75+x2×0.5+x3×0.25)/(x1+x2+x3)，其中x1,x2,x3为最近3期的净利润TTM值 | fundamental.financial | 辅助 | |
| ⬜ 待复现 | reverse_d8 | 与短期反向久期计算方法一致，只是选择了8个报告期的数据进行计算 | fundamental.financial | 辅助 | |
| ⬜ 待复现 | trds_from_issue_stmt_mrq | 距上次最新财务报告发布时点的时间差 | fundamental.financial | 辅助 | |

---

## 三、IC/IR 基准排名表

> 下表为基准 IC/IR 值（来源：`scripts/factors_ICIR.png`），复现结果基于统一评估流程（2016-01-04 ~ 2025-12-31）。

| 排名 | 复现状态 | 因子名称 | 基准 IC | 基准 ICIR | 5d IC | 5d ICIR | 10d IC | 10d ICIR | 20d IC | 20d ICIR | 差异说明 |
|-----|---------|---------|--------|----------|-------|---------|--------|----------|--------|----------|---------|
| 1 | ✅ 已复现 | pe_ttm_delta60 | 0.066 | 0.925 | 0.021 | 0.276 | 0.026 | 0.350 | 0.031 | 0.419 | diff已生效，但与基准差距仍大 |
| 2 | ✅ 已复现 | roe_mrq_new | 0.043 | 0.867 | 0.018 | 0.145 | 0.020 | 0.150 | 0.022 | 0.156 | 低于基准 |
| 3 | ✅ 已复现 | reg_pe_hist | 0.063 | 0.849 | 0.032 | 0.236 | 0.041 | 0.307 | 0.051 | 0.386 | 低于基准 |
| 4 | ✅ 已复现 | npf_mrq_sue8 | 0.066 | 1.284 | 0.018 | 0.236 | 0.020 | 0.244 | 0.024 | 0.265 | 池子差异导致低于基准 |
| 5 | ⬜ 待复现 | roe_mrq | 0.039 | 0.818 | — | — | — | — | — | — | 与roe_mrq_new疑似同名 |
| 6 | ⬜ 待复现 | pb_reg_gshe | 0.032 | 0.586 | — | — | — | — | — | — | 与reg_pb_gshe疑似同名 |
| 7 | ⬜ 待复现 | pe_fy1_new | 0.031 | 0.453 | — | — | — | — | — | — | |
| 8 | ✅ 已复现 | roe_pyoy_mrq | 0.030 | 0.702 | 0.022 | 0.273 | 0.025 | 0.283 | 0.029 | 0.306 | 低于基准 |
| 9 | ✅ 已复现 | npf_pyoy_mrq | 0.028 | 0.621 | 0.020 | 0.221 | 0.022 | 0.226 | 0.025 | 0.235 | 池子差异导致低于基准 |
| 10 | ✅ 已复现 | roe_ayoy_mrq | 0.026 | 0.556 | 0.018 | 0.264 | 0.020 | 0.270 | 0.023 | 0.284 | 低于基准 |
| 11 | ✅ 已复现 | reg_pb_gshe | 0.025 | 0.531 | 0.039 | 0.366 | 0.048 | 0.428 | 0.057 | 0.490 | IC高于基准，ICIR低于基准 |
| 12 | ⬜ 待复现 | net_oper_cash_flow_ttm | 0.024 | 0.424 | — | — | — | — | — | — | |
| 13 | ✅ 已复现 | npf_ayoy_mrq | 0.022 | 0.450 | 0.016 | 0.222 | 0.018 | 0.224 | 0.020 | 0.228 | 池子差异导致低于基准 |
| 14 | ✅ 已复现 | pe_ttm_new | 0.008 | 0.239 | 0.014 | 0.142 | 0.018 | 0.181 | 0.023 | 0.214 | IC高于基准，ICIR低于基准 |
| 15 | ✅ 已复现 | pe_mrq | 0.009 | 0.319 | 0.012 | 0.151 | 0.016 | 0.189 | 0.020 | 0.221 | IC高于基准，ICIR低于基准 |
| 16 | ⬜ 待复现 | pe_ttm | 0.009 | 0.286 | — | — | — | — | — | — | |
| 17 | ✅ 已复现 | roe_apoq_mrq | 0.014 | 0.442 | 0.009 | 0.186 | 0.009 | 0.181 | 0.010 | 0.192 | 低于基准 |
| 18 | ✅ 已复现 | npf_apoq_mrq | 0.013 | 0.407 | 0.008 | 0.175 | 0.008 | 0.169 | 0.009 | 0.173 | 池子差异导致低于基准 |
| 19 | ✅ 已复现 | roe_pqoq_mrq | 0.012 | 0.322 | 0.014 | 0.218 | 0.015 | 0.227 | 0.017 | 0.251 | 低于基准 |
| 20 | ✅ 已复现 | npf_pqoq_mrq | 0.011 | 0.289 | 0.014 | 0.206 | 0.015 | 0.212 | 0.016 | 0.231 | 池子差异导致低于基准 |
| 21 | ✅ 已复现 | npf_apoq_mrq | 0.010 | 0.305 | 0.008 | 0.175 | 0.008 | 0.169 | 0.009 | 0.173 | 池子差异导致低于基准 |

---

## 四、复现统计

| 分类 | 总数 | 已复现 | 待复现 | 有bug |
|------|-----|--------|--------|-------|
| 单因子 | 22 | 19 | 3 | 0 |
| 辅助指标 | 3 | 0 | 3 | 0 |
| **合计** | **25** | **19** | **6** | **0** |

---

## 五、待办优先级

| 优先级 | 因子 | 理由 |
|--------|------|------|
| P1 | roic_ttm 三因子（ind_rnk8 / all_rnk8 / dev_std8） | 需要 merge + change-day rolling，留到所有简单因子验证完成后 |
| P2 | 适度冒险因子（moderate_risk） | 需要分钟级数据 + 截面去均值算子，需扩展算子库 |
| P3 | pe_fy1_new / pe_ttm / pb_reg_gshe | 基准表中待复现的剩余单因子 |
| P3 | 辅助指标 | 非核心因子，低优先级 |
