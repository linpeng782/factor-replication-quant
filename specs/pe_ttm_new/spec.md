## Spec：pe_ttm_new（PE_TTM）

### 1. 定义

TTM市盈率（Price-to-Earnings Ratio, Trailing Twelve Months），即总市值除以过去12个月的净利润。反映市场对公司最近一年盈利能力的估值水平。

$$
\text{pe\_ttm\_new} = \frac{\text{总市值}}{\text{净利润\_TTM}} = \text{pe\_ratio\_ttm}
$$

### 2. 变量说明

| 变量 | 米筐字段 | 含义 |
|------|---------|------|
| `pe_ratio_ttm` | `pe_ratio_ttm` | TTM市盈率（米筐预计算字段） |

### 3. 数据对齐

- **接口**：`rq.get_factor()`
- **PIT 机制**：`pe_ratio_ttm` 由米筐每日计算，自动根据 `trade_date` 返回当时有效的最新数据。无需手动处理财报期对齐。
- **产出格式**：宽表 `date × order_book_id`，未上市/退市股票为 NaN。

### 4. 计算步骤

1. **fetch**：`get_factor(fields=['pe_ratio_ttm'])`
2. **compute**：`pe_ttm_new = pe_ratio_ttm`

### 5. 股票池

全市场 A 股（`all_instruments(type='CS')`）。ST/停牌/上市不满60日等排除条件由回测引擎处理，因子计算阶段不做过滤。
