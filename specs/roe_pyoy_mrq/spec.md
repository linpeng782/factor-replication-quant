## Spec：roe_pyoy_mrq（单季度ROE同比）

### 1. 定义

单季度ROE的同比增长率，仅保留去年同期单季度ROE大于0的样本。

$$
\text{roe\_pyoy\_mrq} = \frac{\text{roe\_mrq\_0} - \text{roe\_mrq\_4}}{|\text{roe\_mrq\_4}|}
$$

其中：

$$
\text{roe\_mrq\_0} = \frac{\text{net\_profit\_mrq\_0}}{\text{total\_equity\_mrq\_0}}, \quad
\text{roe\_mrq\_4} = \frac{\text{net\_profit\_mrq\_4}}{\text{total\_equity\_mrq\_4}}
$$

### 2. 变量说明

| 变量 | 米筐字段 | 含义 |
|------|---------|------|
| `net_profit_mrq_0` | `net_profit_mrq_0` | 最近一期单季度净利润（PIT） |
| `net_profit_mrq_4` | `net_profit_mrq_4` | 4个季度前单季度净利润（PIT） |
| `total_equity_mrq_0` | `total_equity_mrq_0` | 最近一期净资产（PIT） |
| `total_equity_mrq_4` | `total_equity_mrq_4` | 4个季度前净资产（PIT） |

### 3. 数据对齐

- **接口**：`rq.get_factor()`
- **PIT 机制**：`_mrq_n` 后缀字段由米筐每日计算，自动根据 `trade_date` 返回当时已披露的最新财报数据。无需手动差分或 lag。
- **产出格式**：宽表 `date × order_book_id`，未上市股票为 NaN。

### 4. 计算步骤

1. **fetch**：`get_factor(fields=['net_profit_mrq_0', 'net_profit_mrq_4', 'total_equity_mrq_0', 'total_equity_mrq_4'])`
2. **compute**：`roe_mrq_0 = net_profit_mrq_0 / total_equity_mrq_0`
3. **compute**：`roe_mrq_4 = net_profit_mrq_4 / total_equity_mrq_4`
4. **compute**：`roe_pyoy_mrq = (roe_mrq_0 - roe_mrq_4) / abs(roe_mrq_4)`
5. **filter**：`roe_mrq_4 > 0`（研报要求）

### 5. 股票池

全市场 A 股（`all_instruments(type='CS')`）。ST/停牌/上市不满60日等排除条件由回测引擎处理，因子计算阶段不做过滤。
