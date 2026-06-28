# 米筐(RiceQuant/rqdatac)数据需求清单 — 供 CK 数据工程对接

> **用途**：本因子流水线目前直接调 `rqdatac` 拉数据。现改为从公司 ClickHouse(CK)取数，
> 本清单列出**所有需要的米筐数据集 + 精确字段 + 口径**，供数据工程团队在 CK 侧落地对应表。
> 全部标的池 = **A 股全市场普通股**：`rqdatac.all_instruments(type="CS")`（含已退市/暂停上市，全历史）。
> 代码格式 = 米筐 `order_book_id`（如 `000001.XSHE` / `600000.XSHG`）。

## 0. 汇总表

| # | 数据集 | 米筐 API | 频率 | 复权 | 历史起点 | 用途 |
|---|---|---|---|---|---|---|
| 1 | 股票主表/股票池 | `all_instruments(type="CS")` | — | — | 全量 | 所有数据的 universe |
| 2 | 日频行情 | `get_price(freq="1d")` | 日 | **不复权** | 2005-01-01 | alpha158 因子、收益标签 |
| 3 | 分钟行情 | `get_price(freq="1m")` | 1分钟 | **不复权** | 2005-01-01 | 全部分钟级微结构因子（**数据量最大**）|
| 4 | 复权因子 | `get_ex_factor` | 事件 | — | 2005-01-01 | 读时实时复权（日频/分钟共用）|
| 5 | 总市值 | `get_factor("market_cap_3")` | 日 | — | 2005-01-01 | 行业市值中性化 |
| 6 | 中信一级行业(PIT) | 内部表 `__internal__zx2019_industry` | 事件 | — | 2005-01-01 | 行业中性化/哑变量 |
| 7 | 中信一级行业指数 | `get_price(CI0050xx.INDX, "1d")` | 日 | — | 2010-04 | 行业联合动量因子 |
| 8 | 中证全指分钟 | `get_price("000985.XSHG","1m")` | 1分钟 | 指数不复权 | 2005-01-01 | APM 因子市场参照 |
| 9 | 基本面 PIT 财务 | `get_factor(财务字段)` | 日(快照) | — | 2010-01-01 | cxl 基本面因子 |
| 10 | 交易日历 | `get_trading_dates` 等 | — | — | 全量 | 增量调度/对齐（**必需**）|

> 附：**掩码数据**（ST/停牌/涨停/新股）当前由另一个项目产出，但底层也来自米筐——见文末附录。

---

## 1. 股票主表 / 股票池
- **API**：`all_instruments(type="CS")` → 取 `order_book_id` 列。
- **需要**：A 股全市场普通股清单（含**已退市/暂停上市**的全历史标的，不能只给在市的）。
- **建议 CK 字段**：`order_book_id`、`上市日期`、`退市日期`、`状态`、（可选）`symbol/简称`。
- 说明：上市日期用于"新股掩码"，退市/状态用于过滤僵尸标的。

## 2. 日频行情（不复权）
- **API**：`get_price(stocks, frequency="1d", adjust_type="none", skip_suspended=False, fields=[...])`
- **字段**：`open, high, low, close, volume, total_turnover, limit_up, limit_down`
  - `total_turnover`=成交额；`limit_up/limit_down`=当日涨/跌停价。
  - **VWAP 由 `total_turnover / volume` 派生**（收益标签用），不需单独字段。
- **复权口径**：`adjust_type="none"`（**原始未复权价**）。复权在读时用数据集④实时算。
- **停牌**：`skip_suspended=False`（停牌日也要返回，保留行）。
- **历史**：2005-01-01 至今，逐股全历史。
- **消费方**：alpha158 全部 158 个因子；ML 收益标签（forward_return = vwap[t+1+N]/vwap[t+1]-1）。

## 3. 分钟行情（不复权）— 数据量最大
- **API**：`get_price(stocks, frequency="1m", adjust_type="none", skip_suspended=False, fields=[...])`
- **字段**：`open, high, low, close, volume, total_turnover`
- **复权口径**：`adjust_type="none"`（原始未复权）；复权读时用④算。
- **历史**：2005-01-01 至今，**全市场 × 每个交易日 × 240 分钟条**（量级最大，CK 侧重点）。
- **消费方**：所有分钟级微结构因子的 superset（prv_v3/paper_27、apm、sm、tide、dazzle、pricejump 等）。

## 4. 复权因子（除权事件）
- **API**：`get_ex_factor(stocks)`
- **字段**：`ex_cum_factor, ex_factor, ex_end_date`（索引 `ex_date`=除权日）。
- **用途**：磁盘只存②③的不复权价 + 此稀疏复权因子；读时**价×ffill(ex_cum_factor)、量÷ex_cum_factor**得后复权。
- **历史**：2005-01-01 至今，仅除权事件日（稀疏）。

## 5. 总市值
- **API**：`get_factor(stocks, "market_cap_3")`，日频。
- **字段**：`market_cap_3` = A 股**总市值（含限售股）**。
- **历史**：2005-01-01 至今。
- **消费方**：行业市值中性化。

## 6. 中信(CITIC) 2019 一级行业 — PIT 事件表
- **API**：`rqdatac.client.get_client().execute("__internal__zx2019_industry")`（米筐内部表）
- **字段**：`order_book_id, start_date, first_industry_name`（中信 2019 **一级**行业名；按 `start_date` 记录行业变更事件）。
- **用途**：构建 (交易日 × 股票) 行业面板，做行业中性化/哑变量。一级行业共 30 个。
- **历史**：2005-01-01 至今。

## 7. 中信一级行业指数（日频收盘）
- **API**：`get_price(["CI005001.INDX",...,"CI005030.INDX"], frequency="1d", fields=["close"])`
  + `instruments(code).symbol` 取行业名（"中信XXX"）。
- **字段**：30 个指数的 `close`；指数代码 `CI005001.INDX ~ CI005030.INDX`。
- **历史**：2010-04 至今（中信一级行业指数最早 2010-04）。
- **消费方**：行业联合动量因子（co-momentum）。

## 8. 中证全指 000985 分钟
- **API**：`get_price("000985.XSHG", frequency="1m", fields=["open","close"])`（指数不复权）
- **字段**：`open, close`（分钟级）。
- **用途**：拆 隔夜/上午/下午 分段收益，作 APM 因子族的市场参照序列。
- **历史**：2005-01-01 至今。

## 9. 基本面 PIT 财务字段（get_factor）⚠️ 口径关键
- **API**：`get_factor(stocks, FIELDS, start_date, end_date)`，按交易日取**当日快照**。
- **PIT 语义（必须保留）**：`_mrq_N` = most-recent-quarter 往前第 N 季的**当时已披露**值；
  `_ttm` = trailing-twelve-months。某天拉到的值**当天冻结、历史不改写**（as-first-reported，免疫财报重述/前视）。
  CK 侧若只能给"最新口径"的财务，需额外提供**披露日期/快照日期**以便我方做 PIT 冻结。
- **字段清单（18 个）**：

  | 字段 | 含义 |
  |---|---|
  | `net_profit_mrq_0` … `net_profit_mrq_8` | 净利润，最近季~往前第8季（9 个）|
  | `net_profit_ttm_0` | 净利润 TTM |
  | `total_equity_mrq_0`, `_1`, `_4` | 股东权益(净资产) mrq，当季/前1季/前4季 |
  | `cash_flow_from_operating_activities_ttm_0` | 经营活动现金流 TTM |
  | `return_on_invested_capital_ttm` | 投入资本回报率 ROIC(TTM) |
  | `market_cap_3` | 总市值（同⑤，估值口径一并快照）|
  | `pb_ratio_lf` | 市净率（最新一期）|
  | `pe_ratio_ttm` | 市盈率（TTM）|

- **历史**：2010-01-01 至今。
- **消费方**：cxl 基本面因子族（ROE/ROIC/净利润增速/估值回归等约 22 个）。

## 10. 交易日历 / 工具（必需）
增量调度、起止日推断都依赖：`get_trading_dates(start,end)`、`get_next_trading_date`、
`get_previous_trading_date`、`get_latest_trading_date`。
- **需要**：标准 A 股**交易日历表**（交易日列表）。

---

## 附录：掩码数据（另一项目产出，底层同源米筐）

回测/信号过滤用的掩码当前由 `backtest_engine` 项目产出，但底层数据也来自米筐，CK 侧若一并提供更好：
- `is_st`（是否 ST/*ST）
- `is_suspended`（是否停牌）
- `is_limit_up`（是否涨停；也可由日频 `close == limit_up` 推）
- `is_new_stock`（是否新股；由上市日期推，见数据集①）
长表格式 `[order_book_id, datetime, <bool 列>]`。

---

## 关键口径备注（务必传达给数据工程）

1. **价格一律不复权存储**（`adjust_type="none"`），复权因子单独存（数据集④），复权在读时算 —— 避免复权基准变动导致历史漂移。
2. **停牌日保留**（`skip_suspended=False`），不要在数据层 drop。
3. **股票池含退市/暂停标的全历史**，不能只给当前在市。
4. **财务字段是 PIT（as-reported）**，`_mrq_N` 的 N 是季度回溯档位；最好带披露/快照日期。
5. **代码用米筐 order_book_id 体系**（`.XSHE`/`.XSHG` 后缀）；若 CK 用其它代码体系需提供映射。
