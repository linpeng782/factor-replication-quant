# pe_ttm_delta60 — PE_TTM 60 日差值

## 1. 基本信息

| 字段 | 值 |
|---|---|
| name | `pe_ttm_delta60` |
| name_cn | PE_TTM 60 日差值 |
| category | 价值 |
| direction | -1（负向） |
| 假设方向理由 | PE 下行（差值为负）→ 估值收缩 / 利润预期改善 → 预期未来收益更高 |
| primary_index | `ALL`（全市场 A 股） |

## 2. 因子定义

$$
\text{pe\_ttm\_delta60}_t \;=\; \text{pe\_ratio\_ttm}_t \;-\; \text{pe\_ratio\_ttm}_{t-60}
$$

按股票分组（`order_book_id`）做 60 个交易日的差分。

## 3. 数据来源

| 列名 | 米筐 API | 字段 | 频率 |
|---|---|---|---|
| `pe_ratio_ttm` | `get_factor` | `pe_ratio_ttm` | 日频 |

## 4. 计算步骤（与 spec.yaml 一一对应）

每个 step 的 **name / 输入 / 输出** 必须与 yaml 中对应 step 的 `name / source_column / output_column` 严格相等；如不一致由 spec_schema 静态校验拒绝。

1. **获取 PE_TTM**
   - 输入：无（首步 fetch）
   - 操作：从米筐 `get_factor` 拉取字段 `pe_ratio_ttm`
   - 输出主表新增列：`pe_ratio_ttm`

2. **计算 60 日差值**
   - 输入：`pe_ratio_ttm`
   - 操作：按 `order_book_id` 分组做 60 个交易日差分（`pandas.groupby.diff(periods=60)`）
   - 输出主表新增列：`pe_ttm_delta60`

最终：引擎按 `factor.column = pe_ttm_delta60` 将主表 pivot 为 (T, N) 宽表落盘。

## 5. 数据规则

| 场景 | 处理 | 来源 |
|---|---|---|
| 每只股票前 60 个交易日 | NaN（回看数据不足） | 算法必然结果 |

> 评估期 mask（ST / 停牌 / 涨停 / 新股）由评估管线统一处理，**不写入此处**。
