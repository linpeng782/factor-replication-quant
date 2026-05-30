"""
因子: roic_ttm_ind_rnk8 — 过去 8 期 ROIC_TTM 行业内排名最小值

定义：在中信一级行业内做截面排名，按变化日滚动 8 期取最小排名
方向：负向；分类：质量
"""

import numpy as np
import pandas as pd
import rqdatac


def fetch_zx2019_industry_long(trading_days, stocks):
    """
    取中信 2019 一级行业分类 → long 表 [order_book_id, date, first_industry_name]
    每股票按 (start_date, industry) 段持有，到下个段开始日切换。
    """
    raw = rqdatac.client.get_client().execute("__internal__zx2019_industry")
    df = pd.DataFrame(raw)
    df["start_date"] = pd.to_datetime(df["start_date"])

    value_col = "first_industry_name"
    if value_col not in df.columns:
        cands = [c for c in df.columns if "industry" in c.lower()]
        value_col = cands[0]

    # pivot 到 (start_date × stock)，ffill 段内持有
    wide = (
        df.sort_values(["order_book_id", "start_date"])
        .pivot(index="start_date", columns="order_book_id", values=value_col)
        .ffill()
    )
    wide = wide.reindex(index=trading_days).ffill()

    # 转回 long
    long = wide.reset_index().melt(id_vars=["start_date"], var_name="order_book_id",
                                     value_name="industry").rename(columns={"start_date": "date"})
    long = long.dropna(subset=["industry"])
    long["date"] = pd.to_datetime(long["date"])
    return long


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

    print("拉中信一级行业 ...")
    trading_days = sorted(raw["date"].unique())
    industry = fetch_zx2019_industry_long(trading_days, stocks)

    # merge 行业到 raw
    raw = raw.merge(industry, on=["order_book_id", "date"], how="left")

    # 行业内截面排名（每天 + 每个行业）
    print("行业内截面排名 ...")
    raw["rank"] = raw.groupby(["date", "industry"])["roic"].rank(ascending=False, method="min")

    # 变化日 mask
    print("事件驱动滚动 8 期取最小 ...")
    raw["changed"] = raw.groupby("order_book_id")["roic"].transform(
        lambda x: x != x.shift(1)
    )

    change_rows = raw.loc[raw["changed"], ["order_book_id", "rank"]].dropna(subset=["rank"])
    rolled = change_rows.groupby("order_book_id")["rank"].transform(
        lambda x: x.rolling(window=8, min_periods=4).min()
    )

    rolled_full = pd.Series(np.nan, index=raw.index, dtype="float64")
    rolled_full.loc[change_rows.index] = rolled.values
    rolled_full = rolled_full.groupby(raw["order_book_id"]).transform(lambda x: x.ffill())

    raw["factor"] = rolled_full
    panel = raw.pivot(index="date", columns="order_book_id", values="factor")
    panel = panel.sort_index()

    print(f"因子 shape: {panel.shape}, 非空率 {panel.notna().mean().mean():.2%}")
    panel.to_parquet("roic_ttm_ind_rnk8.parquet")


if __name__ == "__main__":
    main()
