## Spec：pe_mrq（PE_MRQ）

### 1. 定义

MRQ市盈率（Price-to-Earnings Ratio, Most Recent Quarter），即总市值除以最近一个季度的净利润。基于最新单季度财报的估值指标，比TTM更及时反映当期盈利变化。

$$
\text{pe\_mrq} = \frac{\text{market\_cap}}{\text{net\_profit\_mrq\_0}}
$$

### 2. 变量说明

| 变量 | 米筐字段 | 含义 |
|------|---------|------|
| `market_cap` | `market_cap` | 总市值（日频） |
| `net_profit_mrq_0` | `net_profit_mrq_0` | 最近一期单季度净利润（PIT时点值） |

### 3. 数据对齐

- **接口**：`rq.get_factor()`
- **PIT 机制**：`_mrq_0` 后缀字段由米筐每日计算，自动根据 `trade_date` 返回当时已披露的最新财报数据。无需手动调用 `get_pit_financials_ex` 进行差分。
- **产出格式**：宽表 `date × order_book_id`，未上市股票为 NaN。

### 4. 计算步骤

1. **fetch**：`get_factor(fields=['market_cap', 'net_profit_mrq_0'])`
2. **compute**：`pe_mrq = market_cap / net_profit_mrq_0`

### 5. 股票池

全市场 A 股（`all_instruments(type='CS')`）。ST/停牌/上市不满60日等排除条件由回测引擎处理，因子计算阶段不做过滤。
