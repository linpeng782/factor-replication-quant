"""
因子: roe_pqoq_mrq — 单季度 ROE 环比（分母 > 0 版）

定义：roe_mrq_0/1 见 roe_apoq_mrq；因子 = (roe_mrq_0 - roe_mrq_1) / abs(roe_mrq_1)
过滤：roe_mrq_1 > 0
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

    np_0 = fetch_factor(stocks, "net_profit_mrq_0",   start_date, end_date)
    eq_0 = fetch_factor(stocks, "total_equity_mrq_0", start_date, end_date)
    np_1 = fetch_factor(stocks, "net_profit_mrq_1",   start_date, end_date)
    eq_1 = fetch_factor(stocks, "total_equity_mrq_1", start_date, end_date)

    roe_0 = np_0 / eq_0
    roe_1 = np_1 / eq_1
    factor = (roe_0 - roe_1) / roe_1.abs()
    factor = factor.where(roe_1 > 0)   # 过滤分母 > 0

    print(f"因子 shape: {factor.shape}, 非空率 {factor.notna().mean().mean():.2%}")
    factor.to_parquet("roe_pqoq_mrq.parquet")


if __name__ == "__main__":
    main()
