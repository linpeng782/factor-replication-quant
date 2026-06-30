# 基本面因子生产 —— 米筐 API / 字段需求清单（供 dquant 补齐）

> 目的：`factor_production/` 下 22 个基本面因子脚本目前直接用 **rqdatac** 取数。
> 本文列出它们依赖的米筐 API 与字段，并标注 dquant 当前是否已提供，
> 方便补齐 dquant 缺失字段后，把这条产线整体迁到 dquant。
>
> 结论速览：dquant `get_factor`（134 列估值/盈利比率）+ `get_fundamental`（仅市值）
> 只能覆盖 **5/22** 因子；其余 **17/22** 卡在**单季度 PIT 财报序列**（`*_mrq_n`）与
> **基础科目 TTM**（`*_ttm_0`），dquant 目前无任何 API 暴露这些字段。

---

## 1. 用到的 3 个米筐入口

| rqdatac 调用 | 用途 | dquant 现成对应 | 状态 |
|---|---|---|---|
| `all_instruments(type="CS")` | 全 A 股票池（order_book_id 列表） | `ddata.all_instruments(...)` 筛 `type=="CS"` | ✅ 已有 |
| `get_factor(stocks, fields, start_date, end_date)` | 取财务/估值字段，返回 (order_book_id, date) × fields | `ddata.get_factor(...)` | ⚠️ **部分字段缺**（见 §2） |
| `client.execute("__internal__zx2019_industry")` | 中信 2019 一级行业（仅 `roic_ttm_ind_rnk8` 用） | `ddata.get_industry_rq(..., industry_standard="citics_2019")` | ✅ 已有 |

> 即：**唯一缺口集中在 `get_factor` 的若干字段**，其余 API 已能对齐。

---

## 2. `get_factor` 字段需求（核心）

基本面因子去重后共依赖 **18 个 `get_factor` 字段**，分三类：

### 2.1 ❌ dquant 缺失 —— 需要补齐（共 14 个）

dquant `get_factor`（134 列）与 `get_fundamental`（只有市值）**全都没有**。

#### A. 单季度净利润 PIT 序列 `net_profit_mrq_{n}`（n=0..8，共 9 期）

| 字段 | 含义 | 被哪些因子依赖 |
|---|---|---|
| `net_profit_mrq_0` | 当期单季度净利润（最新已披露季） | 16 个因子（几乎全 npf/roe/pe_mrq/reg 系列） |
| `net_profit_mrq_1` | 上一单季度（环比 qoq 用） | npf_apoq/pqoq、roe_apoq/pqoq、np_rank、np_delta、accs8、sue8 |
| `net_profit_mrq_2` | 前 2 单季度 | np_rank、np_delta、accs8、sue8 |
| `net_profit_mrq_3` | 前 3 单季度 | 同上 |
| `net_profit_mrq_4` | 去年同期单季度（同比 yoy 用） | npf_ayoy/pyoy、roe_ayoy/pyoy、reg_pe_hist、np_rank、np_delta、accs8、sue8 |
| `net_profit_mrq_5` | 前 5 单季度 | np_rank、np_delta、accs8、sue8 |
| `net_profit_mrq_6` | 前 6 单季度 | 同上 |
| `net_profit_mrq_7` | 前 7 单季度 | 同上 |
| `net_profit_mrq_8` | 前 8 单季度 | np_delta_rank（需要 mrq_0..8 共 9 期算 8 个环比差） |

#### B. 单季度净资产 PIT 序列 `total_equity_mrq_{n}`

| 字段 | 含义 | 被哪些因子依赖 |
|---|---|---|
| `total_equity_mrq_0` | 当期单季度净资产（归母权益） | roe 全系列、reg_pb_gshe |
| `total_equity_mrq_1` | 上一单季度净资产 | roe_apoq_mrq、roe_pqoq_mrq |
| `total_equity_mrq_4` | 去年同期单季度净资产 | roe_ayoy_mrq、roe_pyoy_mrq |

#### C. 基础科目 TTM（带数字尾缀 `_0`）

| 字段 | 含义 | 被哪些因子依赖 |
|---|---|---|
| `net_profit_ttm_0` | 滚动 4 季净利润 TTM | reg_pe_hist |
| `cash_flow_from_operating_activities_ttm_0` | 经营活动现金流量净额 TTM | net_oper_cash_flow_ttm |

### 2.2 ✅ dquant 已有 —— 无需补（共 4 个，均在 `get_factor` 134 列中）

| 字段 | 含义 | 被哪些因子依赖 |
|---|---|---|
| `pe_ratio_ttm` | TTM 市盈率 | pe_ttm_new、pe_ttm_delta60、reg_pe_hist |
| `pb_ratio_lf` | 市净率（最新财报口径 last-financials） | reg_pb_gshe |
| `return_on_invested_capital_ttm` | 投入资本回报率 TTM | roic_ttm 全 3 个 |
| `market_cap_3` | 总市值（约定用 `_3` 口径） | pe_mrq、reg_pb_gshe |

---

## 3. 字段值语义（补 dquant 时务必对齐）

补齐时要让这些字段在 dquant ClickHouse 表里与 rqdatac 同口径，关键约束：

1. **面板形态**：`get_factor` 返回 (order_book_id, date) × field 的**日频**面板。
   财报字段在两次披露之间按**公告日 PIT 前向填充**（同一季度值持有到下季披露日切换），
   **不能用 trade_date 之后才公开的数据**（杜绝前视）。

2. **`_mrq_n` = 单季度值、按季滞后 n**：
   - `mrq_0` 是「截至当日**最新已披露**的单季度」值，`mrq_1` 是其上一季，依此类推。
   - 已是**单季度**口径（非累计），**不要再做季度 diff**。
   - n 是「整季滞后」，不是「日滞后」：同一只股票，跨过一次财报披露日时，
     `mrq_0` 会前移一季，`mrq_1..mrq_8` 整体顺移。

3. **`net_profit`/`total_equity` 口径**：净利润为**归属母公司**净利润；净资产为**归母股东权益**
   （与 rqdatac `get_factor` 同名字段一致；建议补齐后抽样 bit 级对账，见 §4）。

4. **`_ttm_0`**：滚动 4 个单季度求和（净利润）/ 期末值（现金流量表为 TTM 累加）。

5. **覆盖区间**：现有生产区间 `2010-01-01 ~ 至今`；ML 训练需要回溯到 **2005-01-01**
   （与 alpha158/paper27 对齐），建议尽量补到 2005。

---

## 4. 验收建议（补齐后）

对每个新增字段，抽 3~5 只股票 + 跨财报披露日的若干交易日，做 dquant vs rqdatac 逐格对账：

```python
import rqdatac; rqdatac.init()
from dquant import data as ddata
stocks = ["000001.XSHE", "600519.XSHG", "300750.XSHE"]
f = "net_profit_mrq_0"
rq = rqdatac.get_factor(stocks, f, start_date="20230101", end_date="20231231")
dq = ddata.get_factor(stocks, "20230101", "20231231")[["order_book_id","trade_date",f]]
# 对齐 (order_book_id, date) 后比对：要求 PIT 切换日一致、数值相对误差 < 1e-6
```

重点检查**财报披露日当天的值切换是否对齐**（PIT 正确性比数值精度更易出错）。

---

## 5. 缺失字段汇总（给同事的清单）

需在 dquant `get_factor`（或新 API）中补齐以下 **14 个字段**：

```
net_profit_mrq_0
net_profit_mrq_1
net_profit_mrq_2
net_profit_mrq_3
net_profit_mrq_4
net_profit_mrq_5
net_profit_mrq_6
net_profit_mrq_7
net_profit_mrq_8
total_equity_mrq_0
total_equity_mrq_1
total_equity_mrq_4
net_profit_ttm_0
cash_flow_from_operating_activities_ttm_0
```

补齐后 22 个因子可全部迁到 dquant（其余 4 个估值/盈利比率字段 + 行业 + 股池 + 市值已就绪）。
