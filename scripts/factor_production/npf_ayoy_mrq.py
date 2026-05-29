"""
因子: npf_ayoy_mrq — 单季度净利润同比（绝对值版，不过滤）

定义：(net_profit_mrq_0 - net_profit_mrq_4) / abs(net_profit_mrq_4)
方向：正向；分类：景气
"""

import pandas as pd
import rqdatac


def fetch_factor(stocks, field, start_date, end_date):
    df = rqdatac.get_factor(stocks, field, start_date=start_date, end_date=end_date)
    if isinstance(df, pd.Series):
        df = df.to_frame(name=field)
    df.index.names = ["order_book_id", "date"]
    panel = df[field].unstack(level="order_book_id")
    panel.index = pd.to_datetime(panel.index)
    return panel.sort_index()


def main():
    start_date, end_date = "20100101", "20260527"
    rqdatac.init()
    stocks = rqdatac.all_instruments(type="CS")["order_book_id"].tolist()

    np_0 = fetch_factor(stocks, "net_profit_mrq_0", start_date, end_date)
    np_4 = fetch_factor(stocks, "net_profit_mrq_4", start_date, end_date)

    factor = (np_0 - np_4) / np_4.abs()

    print(f"因子 shape: {factor.shape}, 非空率 {factor.notna().mean().mean():.2%}")
    factor.to_parquet("npf_ayoy_mrq.parquet")


if __name__ == "__main__":
    main()
