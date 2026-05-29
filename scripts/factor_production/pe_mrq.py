"""
因子: pe_mrq — 单季度市盈率（总市值 / 单季度净利润）

定义：market_cap_3 / net_profit_mrq_0
方向：负向（PE 越低估值越便宜）
分类：价值

数据来源：rqdatac.get_factor()
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

    market_cap = fetch_factor(stocks, "market_cap_3", start_date, end_date)
    net_profit = fetch_factor(stocks, "net_profit_mrq_0", start_date, end_date)

    factor = market_cap / net_profit

    print(f"因子 shape: {factor.shape}")
    print(f"日期范围: {factor.index.min().date()} ~ {factor.index.max().date()}")
    print(f"非空率: {factor.notna().mean().mean():.2%}")
    factor.to_parquet("pe_mrq.parquet")
    print("已保存: pe_mrq.parquet")


if __name__ == "__main__":
    main()
