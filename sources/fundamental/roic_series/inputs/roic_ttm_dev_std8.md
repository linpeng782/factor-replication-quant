# 因子描述：roic_ttm_dev_std8（ROIC_TTM 稳定性因子）

## 研报字面

`roic_ttm_dev_std8 = 1 / 过去 8 期 ROIC_TTM 的标准差`

ROIC_TTM 越稳定（过去 8 期标准差越小），因子值越大。属于质量类因子。

## 原始变量

**ROIC_TTM** = 米筐字段 `return_on_invested_capital_ttm`（已预计算）
- 日频；值仅在公司财报发布日变化
- 银行类股票该字段始终为 NaN，自然产生因子 NaN

## 计算流程

1. **取 ROIC_TTM 日频时间序列**（fetch get_factor）
2. **过去 8 期变化日 rolling 标准差**：
   - 算子 `rolling`，`change_on = return_on_invested_capital_ttm`（仅在值变化的行采样）
   - `window = 8`（8 个财报期，约 2 年）
   - `min_periods = 4`（与新股 mask 252 天 ≈ 4 期财报对齐）
   - `agg = 'std'`
   - `fill_method = 'ffill'`
   - 输出列：`roic_ttm_std8`
3. **取倒数**（compute）：
   - `roic_ttm_dev_std8 = 1 / roic_ttm_std8`
   - 当 `roic_ttm_std8 == 0`（极少见，连续 8 期完全一样）时结果为 inf；评估清洗阶段会被处理
4. **输出**：因子列名 `roic_ttm_dev_std8`

## 因子方向

`direction = +1`（正向）：因子值越**大**（ROIC 越稳定）→ 预期未来收益越**好**。

## 底池

`universe.primary_index: ALL`（全市场 `all_instruments("CS")`，约 5500 只）。
研报原文是 "中证 800 + 总市值前 100"，本期忠于全市场底池先复现。
