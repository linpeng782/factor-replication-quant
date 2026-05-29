"""
因子: roic_ttm_all_rnk8 — 过去 8 期 ROIC_TTM 全市场排名最小值

定义：
  1. 取 ROIC_TTM (return_on_invested_capital_ttm)
  2. 在每个交易日做全市场截面排名（值越大 → 排名越小，1 是最高）
  3. 按"财报变化日"事件驱动滚动取过去 8 期排名的最小值
       — 仅在 ROIC_TTM 发生变化的日子记一笔；用 ffill 持有
       — ffill 范围限定在米筐对该股票实际返回过的交易日内（含 NaN 行）

经济直觉：历史上"曾达到过的最好全市场排名"。值越小 → 历史最好排名越靠前。

方向：负向；分类：质量

实现：用 long format（跟米筐返回格式一致）做 rolling，输出转宽表。
"""

import numpy as np
import pandas as pd
import rqdatac


def main():
    start_date, end_date = "20100101", "20260527"
    rqdatac.init()
    stocks = rqdatac.all_instruments(type="CS")["order_book_id"].tolist()

    # 1) 拉 raw ROIC（long 格式，保留米筐返回的 NaN 行）
    print("拉 ROIC_TTM ...")
    raw = rqdatac.get_factor(stocks, "return_on_invested_capital_ttm",
                              start_date=start_date, end_date=end_date)
    if isinstance(raw, pd.Series):
        raw = raw.to_frame(name="return_on_invested_capital_ttm")
    raw = raw.reset_index()
    raw.columns = ["order_book_id", "date", "roic"]
    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw.sort_values(["order_book_id", "date"]).reset_index(drop=True)

    # 2) 每日全市场截面排名（roic 越大 → rank 越小）
    print("全市场截面排名 ...")
    raw["rank"] = raw.groupby("date")["roic"].rank(ascending=False, method="min")

    # 3) 变化日 mask: roic 跟上一行（同股票内）不同
    print("识别财报变化日 ...")
    raw["changed"] = raw.groupby("order_book_id")["roic"].transform(
        lambda x: x != x.shift(1)
    )

    # 4) 在变化日上 rolling 8 期取最小（仅在 roic 不为 NaN 的变化日采样）
    print("事件驱动滚动 8 期取最小 ...")
    change_rows = raw.loc[raw["changed"], ["order_book_id", "rank"]].dropna(subset=["rank"])
    rolled = change_rows.groupby("order_book_id")["rank"].transform(
        lambda x: x.rolling(window=8, min_periods=4).min()
    )

    # 5) 拼回全 long 表，ffill 持有（仅在米筐返回的行内 ffill）
    rolled_full = pd.Series(np.nan, index=raw.index, dtype="float64")
    rolled_full.loc[change_rows.index] = rolled.values
    rolled_full = rolled_full.groupby(raw["order_book_id"]).transform(lambda x: x.ffill())

    # 6) 转宽表
    raw["factor"] = rolled_full
    panel = raw.pivot(index="date", columns="order_book_id", values="factor")
    panel = panel.sort_index()

    print(f"因子 shape: {panel.shape}, 非空率 {panel.notna().mean().mean():.2%}")
    panel.to_parquet("roic_ttm_all_rnk8.parquet")


if __name__ == "__main__":
    main()
