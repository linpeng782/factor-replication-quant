## Spec：roe_aqoq_mrq（单季度ROE环比_绝对值版）

### 1. 定义

单季度ROE的环比增长率，**分母取绝对值，不过滤负值样本**。

与 `roe_pqoq_mrq` 的区别：`pqoq` 仅保留分母>0的样本，`aqoq` 对所有样本计算，分母统一取 `abs()`。

$$
\text{roe\_aqoq\_mrq} = \frac{\text{roe\_mrq\_0} - \text{roe\_mrq\_1}}{|\text{roe\_mrq\_1}|}
$$

其中：

$$
\text{roe\_mrq\_0} = \frac{\text{net\_profit\_mrq\_0}}{\text{total\_equity\_mrq\_0}}, \quad
\text{roe\_mrq\_1} = \frac{\text{net\_profit\_mrq\_1}}{\text{total\_equity\_mrq\_1}}
$$

### 2. 变量说明

| 变量 | 米筐字段 | 含义 |
|------|---------|------|
| `net_profit_mrq_0` | `net_profit_mrq_0` | 最近一期单季度净利润（PIT） |
| `net_profit_mrq_1` | `net_profit_mrq_1` | 1个季度前单季度净利润（PIT） |
| `total_equity_mrq_0` | `total_equity_mrq_0` | 最近一期净资产（PIT） |
| `total_equity_mrq_1` | `total_equity_mrq_1` | 1个季度前净资产（PIT） |

### 3. 与 roe_pqoq_mrq 的区别

| 维度 | roe_pqoq_mrq | roe_aqoq_mrq |
|------|-------------|--------------|
| 计算方式 | `(roe_mrq_0 - roe_mrq_1) / abs(roe_mrq_1)` | 相同 |
| 负分母处理 | **过滤掉**（`roe_mrq_1 > 0`） | **保留**，分母取绝对值 |
| 样本覆盖 | 仅正值分母样本 | 全样本 |
| 适用场景 | 避免负ROE公司扭曲分布 | 保留全部信息，负值ROE的边际变化也纳入考量 |

### 4. 计算步骤

1. **fetch**：`get_factor(fields=['net_profit_mrq_0', 'net_profit_mrq_1', 'total_equity_mrq_0', 'total_equity_mrq_1'])`
2. **compute**：`roe_mrq_0 = net_profit_mrq_0 / total_equity_mrq_0`
3. **compute**：`roe_mrq_1 = net_profit_mrq_1 / total_equity_mrq_1`
4. **compute**：`roe_aqoq_mrq = (roe_mrq_0 - roe_mrq_1) / abs(roe_mrq_1)`

> 无 filter 步骤，与 `roe_pqoq_mrq` 的核心差异。

### 5. 股票池

全市场 A 股（`all_instruments(type='CS')`）。ST/停牌/上市不满60日等排除条件由回测引擎处理，因子计算阶段不做过滤。
