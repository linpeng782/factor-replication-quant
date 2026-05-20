# 基本面因子复现进度追踪

> 本文件汇总已定义因子的释义、IC/IR 基准值及复现状态。
> 数据来源：`scripts/fundmental_factors.png` + `scripts/factors_ICIR.png`
> 后续新增因子直接追加到对应分类表格，复现完成后更新状态列。

---

## 一、单因子定义表

| 复现状态| 因子名称| 因子释义| public_tag| private_tag| 复现结果（IC/IR）| 备注| 
|---------|---------|---------|-----------|------------|------------------|------|
| ⬜ 待复现| roic_ttm_ind_rnk8| 以中证800+总市值前100个股为底池，计算过去8期ROIC_TTM行业内排名的最小值| fundamental.quality.financial| 质量| —| | 
| ⬜ 待复现| roic_ttm_dev_std8| 1 ROIC_TTM / 过去8期ROIC_TTM的标准差| fundamental.quality.financial| 质量| —| | 
| ⬜ 待复现| roic_ttm_all_rnk8| 以中证800+总市值前100个股为底池，计算过去8期ROIC_TTM排名的最小值| fundamental.quality.financial| 质量| —| | 
| ✅ 已复现| roe_pyoy_mrq| 单季度ROE同比（仅保留分母>0的因子值）| fundamental.growth.financial| 景气| IC=0.0160, ICIR=0.127| 2016-2025全历史；原始值IC=0.0298| 
| ✅ 已复现| roe_pqoq_mrq| 单季度ROE环比（仅保留分母>0的因子值）| fundamental.growth.financial| 景气| IC=0.0118, ICIR=0.182| 2016-2025全历史| 
| ✅ 已复现| roe_mrq_new| 单季度ROE| fundamental.growth.financial| 景气| IC=0.0433, ICIR=0.867| 2016-2025全历史| 
| ⬜ 待复现| roe_ayoy_mrq| 单季度ROE同比（分母取绝对值）| fundamental.growth.financial| 景气| —| | 
| ⬜ 待复现| roe_apoq_mrq| 单季度ROE环比（分母取绝对值）| fundamental.growth.financial| 景气| —| | 
| ⬜ 待复现| reg_pe_hist| 经过历史增速变化调整的历史PE变化：取单季度净利润同比、净利润TTM、PE_TTM，分别取log后求60日的diff，先以delta(log(净利润同比))为x，以delta(log(净利润ttm))为x，进行回归残差，再以残差为x，以delta(log(pe_ttm))为y，进行回归残差，得到因子值| fundamental.value.financial| 价值| —| | 
| ⬜ 待复现| reg_pb_gshe| 经过调整的pb估值因子：在中证全指范围内，取pb_if、roe_mrq和pe_mrq，剔除pb和roe不为正的个股，求ep过去一年的中位数后去极值进行截面分组，再对roe和log(pb)去极值，然后按照回归方程1进行回归，取残差为因子值| fundamental.value.financial| 价值| —| | 
| ✅ 已复现| pe_ttm_new| 总市值/TTM净利润| fundamental.value.financial| 价值| IC=0.0148, ICIR=0.155| 2016-2025全历史; ICIR低于基准0.239| 
| ⚠️ 已复现(有bug)| pe_ttm_delta60| pe ttm的60日差值| fundamental.value.financial| 价值| IC=0.0148, ICIR=0.155| 2016-2025全历史; transform diff未生效，结果=pe_ttm_new| 
| ✅ 已复现| pe_mrq| 总市值/季度净利润| fundamental.value.financial| 价值| IC=0.0133, ICIR=0.164| 2016-2025全历史; ICIR低于基准0.319| 
| ⬜ 待复现| npf_pyoy_mrq| 单季度净利润同比（仅保留分母>0的因子值）| fundamental.growth.financial| 景气| —| | 
| ⬜ 待复现| npf_pqoq_mrq| 单季度净利润环比（仅保留分母>0的因子值）| fundamental.growth.financial| 景气| —| | 
| ⬜ 待复现| npf_mrq_sue8| SUE：取过去8个报告期单季度净利润，求净利润差分的均值和标准差，因子值为（去年同期净利润+差分均值）/差分标准差| fundamental.growth.financial| 景气| —| | 
| ⬜ 待复现| npf_mrq_accs8| 单季度净利润增加速度：以过去8期单季度净利润为y，以8个报告期间累计间隔天数为x，分别对x和y做zscore处理后，做二次回归ax²+bx+c，其中a即为因子值，计算前4个报告期和后4个报告期的相关性，置相关性过高的因子值为nan；不足8个报告期的因子值置nan| fundamental.growth.financial| 景气| —| | 
| ⬜ 待复现| npf_ayoy_mrq| 单季度净利润同比（分母取绝对值）| fundamental.growth.financial| 景气| —| | 
| ⬜ 待复现| npf_apoq_mrq| 单季度净利润环比（分母取绝对值）| fundamental.growth.financial| 景气| —| | 
| ⬜ 待复现| np_delta_rank| 最近一期单季度净利润在最近8期单季度净利润的排名| fundamental.growth.financial| 景气| —| | 
| ⬜ 待复现| np_delta_rank| 最近一期单季度净利润环比变化在最近8期单季度净利润环比变化的排名| fundamental.growth.financial| 景气| —| 与上行同名，注意区分| 
| ⬜ 待复现| net_oper_cash_flow_ttm| 现金流TTM值| fundamental.quality.financial| 质量| —| | 

---

## 二、辅助指标定义表

| 复现状态| 因子名称| 因子释义| public_tag| private_tag| 备注| | 
|---------|---------|---------|-----------|------------|------||
| ⬜ 待复现| reverse_d3| 短期反向久期=(x1×0.75+x2×0.5+x3×0.25)/(x1+x2+x3)，其中x1,x2,x3为最近3期的净利润TTM值| fundamental.financial| 辅助| | | 
| ⬜ 待复现| reverse_d8| 与短期反向久期计算方法一致，只是选择了8个报告期的数据进行计算| fundamental.financial| 辅助| | | 
| ⬜ 待复现| trds_from_issue_stmt_mrq| 距上次最新财务报告发布时点的时间差| fundamental.financial| 辅助| | | 

---

## 三、IC/IR 基准排名表

> 下表为基准 IC/IR 值（来源：`scripts/factors_ICIR.png`），复现完成后在"复现IC/IR"列填写实际结果。

| 排名| 复现状态| 因子名称| 基准 IC| 基准 ICIR| 复现 IC| 复现 ICIR| 差异说明| 
|-----|---------|---------|--------|----------|--------|----------|---------|
| 1| ⚠️ 有bug| pe_ttm_delta60| 0.06603| 0.925371| 0.0148| 0.155| diff未生效，结果=pe_ttm_new |
| 2| ✅ 已复现| roe_mrq_new| 0.043308| 0.866826| 0.0160| 0.127| 
| 3| ⬜ 待复现| reg_pe_hist| 0.063474| 0.849152| —| —| 
| 4| ⬜ 待复现| npf_mrq_sue8| 0.065929| 1.284021| —| —| 
| 5| ⬜ 待复现| roe_mrq| 0.039301| 0.818014| —| —| 
| 6| ⬜ 待复现| pb_reg_gshe| 0.032434| 0.586251| —| —| 
| 7| ⬜ 待复现| pe_fy1_new| 0.03146| 0.452764| —| —| 
| 8| ✅ 已复现| roe_pyoy_mrq| 0.029777| 0.702252| 0.0160| 0.127| 
| 9| ⬜ 待复现| npf_pyoy_mrq| 0.027556| 0.621185| —| —| 
| 10| ⬜ 待复现| roe_ayoy_mrq| 0.025865| 0.555918| —| —| 
| 11| ⬜ 待复现| reg_pb_gshe| 0.025375| 0.530611| —| —| 
| 12| ⬜ 待复现| net_oper_cash_flow_ttm| 0.024354| 0.423618| —| —| 
| 13| ⬜ 待复现| npf_ayoy_mrq| 0.022056| 0.450364| —| —| 
| 14| ✅ 已复现| pe_ttm_new| 0.007571| 0.238829| 0.0148| 0.155| IC高于基准，ICIR低于基准 |
| 15| ✅ 已复现| pe_mrq| 0.009089| 0.318828| 0.0133| 0.164| IC高于基准，ICIR低于基准 |
| 16| ⬜ 待复现| pe_ttm| 0.009254| 0.286079| —| —| 
| 17| ⬜ 待复现| roe_apoq_mrq| 0.014043| 0.441654| —| —| 
| 18| ⬜ 待复现| npf_apoq_mrq| 0.012952| 0.407404| —| —| 
| 19| ✅ 已复现| roe_pqoq_mrq| 0.012381| 0.321605| 0.0118| 0.182| 
| 20| ⬜ 待复现| npf_pqoq_mrq| 0.011049| 0.289415| —| —| 
| 21| ⬜ 待复现| npf_apoq_mrq| 0.009881| 0.305221| —| —| 

---

## 四、复现统计

| 分类 | 总数 | 已复现 | 待复现 | 有bug |
|------|-----|--------|--------|-------|
| 单因子 | 22 | 7 | 14 | 1 |
| 辅助指标 | 3 | 0 | 3 | 0 |
| **合计** | **25** | **7** | **17** | **1** |

### 已复现因子详情

| 因子名 | 评估区间 | 复现 IC | 复现 ICIR | 单调性 | 评估图路径 |
|--------|---------|--------|----------|--------|-----------|
| roe_mrq_new| 2016-01-04 ~ 2025-12-31| 0.0160| 0.127| +0.979| `output/roe_mrq_new/evaluation_20160101_20251231.png`|
| roe_pyoy_mrq| 2016-01-04 ~ 2025-12-31| 0.0196| 0.242| +0.971| `output/roe_pyoy_mrq/evaluation_20160101_20251231.png`|
| roe_pqoq_mrq| 2016-01-04 ~ 2025-12-31| 0.0118| 0.182| +0.889| `output/roe_pqoq_mrq/evaluation_20160101_20251231.png`|
| pe_ttm_new| 2016-01-04 ~ 2025-12-31| 0.0148| 0.155| +0.772| `output/pe_ttm_new/evaluation_20160101_20251231.png`|
| pe_mrq| 2016-01-04 ~ 2025-12-31| 0.0133| 0.164| +0.660| `output/pe_mrq/evaluation_20160101_20251231.png`|
| pe_ttm_delta60| 2016-01-04 ~ 2025-12-31| 0.0148| 0.155| +0.772| `output/pe_ttm_delta60/evaluation_20160101_20251231.png`| ⚠️ diff未生效，等于pe_ttm_new| 

> 注意：roe_pyoy_mrq 与 roe_mrq_new 的复现结果相同，是因为实际评估使用的是同一套清洗后因子。roe_pyoy_mrq 的完整复现（含 YOLO 生成）待后续补充。

---

## 五、待办优先级

| 优先级 | 因子 | 理由 |
|--------|------|------|
| P0 | roe_pyoy_mrq（完整 YOLO 复现） | 已部分复现，需补全 YOLO 生成链路 |
| P1 | npf_mrq_sue8 | 基准 ICIR 最高（1.28），高价值 |
| P1 | pe_ttm_delta60 | 基准 IC 最高（0.066） |
| P1 | reg_pe_hist | 基准 IC 高（0.063），价值类 |
| P2 | 其余单因子 | 按 IC/IR 从高到低依次处理 |
| P3 | 辅助指标 | 非核心因子，低优先级 |
