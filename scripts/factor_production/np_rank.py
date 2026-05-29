"""
因子: np_rank — 最近一期单季度净利润在 8 期中的 min-rank

定义：每个 (股票, 日期)，对 [net_profit_mrq_0, ..., net_profit_mrq_7]
      求 mrq_0 在这 8 个值里的 min-rank（升序，最小=1，最大=8）。
      实现：1 + sum(mrq_0 > mrq_i for i in 1..7)
      mrq_0 缺失时结果为 NaN。

方向：正向（排名越高景气越好）；分类：景气
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

    # 拉 8 期单季度净利润
    panels = {
        i: fetch_factor(stocks, f"net_profit_mrq_{i}", start_date, end_date)
        for i in range(8)
    }

    # mrq_0 在 8 期中的 min-rank：1 + sum(mrq_0 > mrq_i, i=1..7)
    factor = (panels[0] - panels[0]) + 1  # 把不变常数 1 嵌入：mrq_0 缺失则 NaN
    for i in range(1, 8):
        factor = factor + (panels[0] > panels[i]).astype(float)

    # 上一步 (mrq_0 > mrq_i) 在两边都是 NaN 时返回 False（=0），需要修正
    # 实际正确语义：mrq_0 是 NaN 时整体应为 NaN；mrq_i 是 NaN 时不计入比较
    # 这里"减自身 + 1"的 trick 已保证 mrq_0 NaN → NaN 传播
    # 所以不需要额外处理 mrq_i NaN

    print(f"因子 shape: {factor.shape}, 非空率 {factor.notna().mean().mean():.2%}")
    factor.to_parquet("np_rank.parquet")


if __name__ == "__main__":
    main()
