"""
因子: roe_pyoy_mrq — 单季度 ROE 同比增长率（仅保留分母 > 0 样本）

定义：(roe_mrq_0 - roe_mrq_4) / abs(roe_mrq_4)
  其中 roe_mrq_0 = 当期单季度 ROE = net_profit_mrq_0 / total_equity_mrq_0
       roe_mrq_4 = 去年同期单季度 ROE = net_profit_mrq_4 / total_equity_mrq_4

过滤：保留 roe_mrq_4 > 0 的样本（剔除负基数样本，避免同比方向误判）
方向：正向（ROE 改善预期收益更高）
分类：景气

数据来源：rqdatac.get_factor()  —  PIT 字段 net_profit_mrq_X / total_equity_mrq_X
"""

import numpy as np
import pandas as pd
import rqdatac


def fetch_pit_field(stocks: list, field: str, start_date: str, end_date: str) -> pd.DataFrame:
    """拉单个 PIT 字段，返回 (date × stock) 宽表。"""
    df = rqdatac.get_factor(
        order_book_ids=stocks,
        factor=field,
        start_date=start_date,
        end_date=end_date,
    )
    if isinstance(df, pd.Series):
        df = df.to_frame(name=field)
    df.index.names = ["order_book_id", "date"]
    panel = df[field].unstack(level="order_book_id")
    panel.index = pd.to_datetime(panel.index)
    return panel.sort_index()


def compute_roe_pyoy_mrq(start_date: str, end_date: str) -> pd.DataFrame:
    """
    单季度 ROE 同比因子计算主流程。
    """
    rqdatac.init()
    universe = rqdatac.all_instruments(type="CS")
    stocks = universe["order_book_id"].tolist()

    # 拉 4 个 PIT 字段
    np_0 = fetch_pit_field(stocks, "net_profit_mrq_0", start_date, end_date)
    eq_0 = fetch_pit_field(stocks, "total_equity_mrq_0", start_date, end_date)
    np_4 = fetch_pit_field(stocks, "net_profit_mrq_4", start_date, end_date)
    eq_4 = fetch_pit_field(stocks, "total_equity_mrq_4", start_date, end_date)

    # 当期与去年同期单季度 ROE
    roe_0 = np_0 / eq_0
    roe_4 = np_4 / eq_4

    # 同比增长率
    factor = (roe_0 - roe_4) / roe_4.abs()

    # 过滤分母 > 0
    factor = factor.where(roe_4 > 0)

    return factor


def main():
    start_date = "20100101"
    end_date = "20260527"

    factor = compute_roe_pyoy_mrq(start_date, end_date)

    print(f"因子 shape: {factor.shape}")
    print(f"日期范围: {factor.index.min().date()} ~ {factor.index.max().date()}")
    print(f"非空率: {factor.notna().mean().mean():.2%}")

    factor.to_parquet("roe_pyoy_mrq.parquet")
    print(f"已保存: roe_pyoy_mrq.parquet")


if __name__ == "__main__":
    main()
