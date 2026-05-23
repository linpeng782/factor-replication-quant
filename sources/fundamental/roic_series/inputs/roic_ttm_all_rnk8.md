# 因子描述：roic_ttm_all_rnk8（过去 8 期 ROIC_TTM 全市场排名最小值）

## 研报字面

以全市场个股为底池，计算过去 8 期 ROIC_TTM 排名的最小值（**不分行业**）。
ROIC_TTM 越高排名越靠前，取过去 8 期"最好排名"（即排名数值的最小值）。

属于质量类因子。与 `roic_ttm_ind_rnk8` 唯一区别：排名范围是全市场。

## 原始变量

**ROIC_TTM** = 米筐字段 `return_on_invested_capital_ttm`（已预计算）
- 日频；值仅在公司财报发布日变化
- 银行类股票该字段始终为 NaN，自然产生因子 NaN

## 计算流程

1. **取 ROIC_TTM 日频时间序列**（fetch get_factor）
2. **全市场截面排名**：
   - `group_by = [date]`（仅按交易日分组，不分行业）
   - `ascending = False`（ROIC_TTM 越高 → 排名数值越小 = 越靠前）
   - `rank_method = 'min'`（并列取最小排名）
   - `pct = False`（输出顺序整数排名）
   - 该行 ROIC_TTM 为 NaN 时不参与排名，输出 NaN
3. **过去 8 期变化日 rolling 取最小**：
   - 算子 `rolling`，`change_on = return_on_invested_capital_ttm`
   - `window = 8`（8 个财报期）
   - `min_periods = 4`
   - `agg = 'min'`
   - `fill_method = 'ffill'`
4. **输出**：因子列名 `roic_ttm_all_rnk8` = 上一步的 rolling min 列

## 因子方向

`direction = -1`（反向）：因子值越**小**意味着历史上全市场曾达到过更靠前的排名，
预期未来收益越**好**。

## 底池

`universe.primary_index: ALL`（全市场 `all_instruments("CS")`，约 5500 只）。
研报原文是 "中证 800 + 总市值前 100"，本期忠于全市场底池先复现。
