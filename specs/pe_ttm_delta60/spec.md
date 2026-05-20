## Spec：pe_ttm_delta60（PE_TTM 60日差值）

### 1. 定义

PE_TTM的60日差值，衡量估值短期变化趋势。核心逻辑：**PE下降越多（差值越负），说明估值收缩越明显，预期收益越高**。

$$
\text{pe\_ttm\_delta60}_t = \text{pe\_ratio\_ttm}_t - \text{pe\_ratio\_ttm}_{t-60}
$$

### 2. 变量说明

| 变量 | 米筐字段 | 含义 |
|------|---------|------|
| `pe_ratio_ttm` | `pe_ratio_ttm` | TTM市盈率（日频） |

### 3. 数据对齐

- **接口**：`rq.get_factor()`
- **PIT 机制**：`pe_ratio_ttm` 由米筐每日计算，自动对齐。
- **差分处理**：long格式下按 `order_book_id` 分组，逐股票计算60日差值（`diff(periods=60)`）。
- **产出格式**：宽表 `date × order_book_id`。每只股票前60个交易日前向缺失。

### 4. 计算步骤

1. **fetch**：`get_factor(fields=['pe_ratio_ttm'])`
2. **transform**：`diff(periods=60, group_column='order_book_id', columns=['pe_ratio_ttm'])`
3. **compute**：`pe_ttm_delta60 = pe_ratio_ttm`

### 5. 股票池

全市场 A 股（`all_instruments(type='CS')`）。ST/停牌/上市不满60日等排除条件由回测引擎处理，因子计算阶段不做过滤。
