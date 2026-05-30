"""
因子: roic_ttm_dev_std8 — ROIC_TTM 稳定性（倒数标准差）

定义：1 / std(ROIC_TTM 过去 8 期变化日值)
方向：正向（ROIC 越稳定因子值越大）；分类：质量
"""

import numpy as np
import pandas as pd
import rqdatac


def main():
    start_date, end_date = "20100101", "20260527"
    rqdatac.init()
    stocks = rqdatac.all_instruments(type="CS")["order_book_id"].tolist()

    print("拉 ROIC_TTM ...")
    raw = rqdatac.get_factor(stocks, "return_on_invested_capital_ttm",
                              start_date=start_date, end_date=end_date)
    if isinstance(raw, pd.Series):
        raw = raw.to_frame(name="return_on_invested_capital_ttm")
    raw = raw.reset_index()
    raw.columns = ["order_book_id", "date", "roic"]
    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw.sort_values(["order_book_id", "date"]).reset_index(drop=True)

    # 变化日 mask（含首行的 NaN 不等于自身比较，得 True）
    raw["changed"] = raw.groupby("order_book_id")["roic"].transform(
        lambda x: x != x.shift(1)
    )

    # 在变化日的 raw 值上滚动 8 期取 std
    print("事件驱动滚动 8 期取标准差 ...")
    change_rows = raw.loc[raw["changed"], ["order_book_id", "roic"]].dropna(subset=["roic"])
    rolled = change_rows.groupby("order_book_id")["roic"].transform(
        lambda x: x.rolling(window=8, min_periods=4).std(ddof=1)
    )

    rolled_full = pd.Series(np.nan, index=raw.index, dtype="float64")
    rolled_full.loc[change_rows.index] = rolled.values
    rolled_full = rolled_full.groupby(raw["order_book_id"]).transform(lambda x: x.ffill())

    # 因子 = 1 / std
    raw["factor"] = 1.0 / rolled_full
    panel = raw.pivot(index="date", columns="order_book_id", values="factor")
    panel = panel.sort_index()

    print(f"因子 shape: {panel.shape}, 非空率 {panel.notna().mean().mean():.2%}")
    panel.to_parquet("roic_ttm_dev_std8.parquet")


if __name__ == "__main__":
    main()
