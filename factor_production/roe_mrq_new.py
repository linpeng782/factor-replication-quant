"""
因子: roe_mrq_new — 单季度 ROE

定义：net_profit_mrq_0 / total_equity_mrq_0
方向：正向（ROE 越高质量越好）
分类：景气

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

    np_0 = fetch_factor(stocks, "net_profit_mrq_0", start_date, end_date)
    eq_0 = fetch_factor(stocks, "total_equity_mrq_0", start_date, end_date)

    factor = np_0 / eq_0

    print(f"因子 shape: {factor.shape}")
    print(f"日期范围: {factor.index.min().date()} ~ {factor.index.max().date()}")
    print(f"非空率: {factor.notna().mean().mean():.2%}")
    factor.to_parquet("roe_mrq_new.parquet")
    print("已保存: roe_mrq_new.parquet")


if __name__ == "__main__":
    main()
