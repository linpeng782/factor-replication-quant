jin# 因子描述：roic_ttm_ind_rnk8（过去 8 期 ROIC_TTM 行业内排名最小值）

## 研报字面

以全市场个股为底池，计算过去 8 期 ROIC_TTM 行业内排名的最小值。
ROIC_TTM 越高排名越靠前，取过去 8 期"最好排名"（即排名数值的最小值）。

属于质量类因子。

## 原始变量

1. **ROIC_TTM** = 米筐字段 `return_on_invested_capital_ttm`（已预计算，无需手算 `operating_profitTTM / invested_capital_ttm`）
   - 日频；值仅在公司财报发布日变化（同一财报期内值保持不变）
   - 银行类股票该字段始终为 NaN（业务模式不适用 ROIC），自然产生因子 NaN
2. **行业分类**：中信一级行业（2019 版），通过 `fetch api=custom command=__internal__zx2019_industry` 取，
   算子内部已自动 pivot + ffill 转为日频 long 表 `(order_book_id, date, first_industry_name)`

## 计算流程

1. **取 ROIC_TTM 日频时间序列**（fetch get_factor）
2. **merge 行业列**到 ROIC_TTM 表（按 `[order_book_id, date]` 关联）
3. **行业内截面排名**：
   - `group_by = [date, first_industry_name]`
   - `ascending = False`（ROIC_TTM 越高 → 排名数值越小 = 越靠前）
   - `rank_method = 'min'`（并列取最小排名）
   - `pct = False`（输出顺序整数排名 1, 2, 3...）
   - 该行 ROIC_TTM 为 NaN 时不参与排名，输出 NaN
4. **过去 8 期变化日 rolling 取最小**：
   - 算子 `rolling`，`change_on = return_on_invested_capital_ttm`（仅在 ROIC_TTM 值变化的行采样）
   - `window = 8`（8 个财报期，约 2 年）
   - `min_periods = 4`（与新股 mask 252 天 ≈ 4 期财报对齐）
   - `agg = 'min'`
   - `fill_method = 'ffill'`（变化日之间沿用最近一次的 rolling 结果）
5. **输出**：因子列名 `roic_ttm_ind_rnk8` = 上一步的 rolling min 列

## 因子方向

`direction = -1`（反向）：因子值越**小**意味着历史上行业内曾达到过更靠前的排名，
代表"曾经辉煌"的龙头基因，预期未来收益越**好**。

## 底池

`universe.primary_index: ALL`（全市场 `all_instruments("CS")`，约 5500 只）。
研报原文是 "中证 800 + 总市值前 100"，本期忠于全市场底池先复现，后续 v2 再切池子。
