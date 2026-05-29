"""
因子: pe_ttm_delta60 — PE_TTM 60 日差值

定义：pe_ratio_ttm.diff(60)，逐股做 60 个交易日的差值
方向：负向（PE 下降越多预期收益越高）
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

    pe = fetch_factor(stocks, "pe_ratio_ttm", start_date, end_date)
    factor = pe.diff(60)   # 逐股按交易日索引 diff 60 期

    print(f"因子 shape: {factor.shape}")
    print(f"日期范围: {factor.index.min().date()} ~ {factor.index.max().date()}")
    print(f"非空率: {factor.notna().mean().mean():.2%}")
    factor.to_parquet("pe_ttm_delta60.parquet")
    print("已保存: pe_ttm_delta60.parquet")


if __name__ == "__main__":
    main()
