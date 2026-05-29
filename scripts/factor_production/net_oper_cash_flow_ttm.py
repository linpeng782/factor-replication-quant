"""
因子: net_oper_cash_flow_ttm — 经营活动现金流量净额 TTM

定义：直接取米筐字段 cash_flow_from_operating_activities_ttm_0
（最近报告期为终点的 4 季度滚动）作为因子值。

方向：正向（现金流越大质量越高）
分类：质量

注：原始 TTM 量纲与市值高度相关；如需中性化版本可后续做衍生因子。
本因子按"现金流 TTM 值"原义实现，不做缩放。

数据来源：rqdatac.get_factor()
"""

import pandas as pd
import rqdatac


def fetch_factor(stocks: list, field: str, start_date: str, end_date: str) -> pd.DataFrame:
    """拉单个字段，返回 (date × stock) 宽表。"""
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


def main():
    start_date = "20100101"
    end_date = "20260527"

    rqdatac.init()
    universe = rqdatac.all_instruments(type="CS")
    stocks = universe["order_book_id"].tolist()

    factor = fetch_factor(stocks, "cash_flow_from_operating_activities_ttm_0", start_date, end_date)

    print(f"因子 shape: {factor.shape}")
    print(f"日期范围: {factor.index.min().date()} ~ {factor.index.max().date()}")
    print(f"非空率: {factor.notna().mean().mean():.2%}")

    factor.to_parquet("net_oper_cash_flow_ttm.parquet")
    print(f"已保存: net_oper_cash_flow_ttm.parquet")


if __name__ == "__main__":
    main()
