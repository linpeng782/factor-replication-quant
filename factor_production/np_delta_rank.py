"""
因子: np_delta_rank — 当期净利润环比变化 d_0 在 8 期 {d_0..d_7} 中的 min-rank

定义：
  d_i = net_profit_mrq_i - net_profit_mrq_{i+1}, i = 0..7   (8 个环比变化)
  factor = d_0 在 [d_0..d_7] 中的 min-rank (升序, 最小=1, 最大=8)
         = 1 + sum(d_0 > d_i for i in 1..7)
  需要 mrq_0..mrq_8 共 9 期数据。

方向：正向（环比变化排名越高景气改善越显著）；分类：景气
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

    # 拉 9 期单季度净利润 (mrq_0..mrq_8)
    np_panels = {
        i: fetch_factor(stocks, f"net_profit_mrq_{i}", start_date, end_date)
        for i in range(9)
    }

    # 8 个环比变化 d_0..d_7
    d = {i: np_panels[i] - np_panels[i + 1] for i in range(8)}

    # d_0 在 8 期中的 min-rank: 1 + sum(d_0 > d_i for i in 1..7)
    factor = (d[0] - d[0]) + 1   # NaN 守门
    for i in range(1, 8):
        factor = factor + (d[0] > d[i]).astype(float)

    print(f"因子 shape: {factor.shape}, 非空率 {factor.notna().mean().mean():.2%}")
    factor.to_parquet("np_delta_rank.parquet")


if __name__ == "__main__":
    main()
