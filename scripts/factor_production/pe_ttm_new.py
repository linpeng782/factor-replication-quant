"""
因子: pe_ttm_new — TTM 市盈率（总市值 / TTM 净利润）

定义：直接取米筐字段 pe_ratio_ttm 作为因子值。
方向：负向（PE 越低估值越便宜，预期收益越高）。
分类：价值

数据来源：rqdatac.get_factor()
"""

import pandas as pd
import rqdatac


def fetch_pe_ttm(start_date: str, end_date: str) -> pd.DataFrame:
    """
    全市场拉取 PE_TTM。

    参数:
        start_date: 起始日，'YYYYMMDD' 格式
        end_date: 结束日，'YYYYMMDD' 格式

    返回:
        宽表 (date × order_book_id) 的 PE_TTM 矩阵
    """
    rqdatac.init()

    # 全 A 股票池
    universe = rqdatac.all_instruments(type="CS")
    stocks = universe["order_book_id"].tolist()

    # 拉因子
    df = rqdatac.get_factor(
        order_book_ids=stocks,
        factor="pe_ratio_ttm",
        start_date=start_date,
        end_date=end_date,
    )

    # rqdatac 返回 (order_book_id, date) MultiIndex；转成宽表
    if isinstance(df, pd.Series):
        df = df.to_frame(name="pe_ratio_ttm")
    df.index.names = ["order_book_id", "date"]
    panel = df["pe_ratio_ttm"].unstack(level="order_book_id")
    panel.index = pd.to_datetime(panel.index)
    panel = panel.sort_index()

    return panel


def main():
    start_date = "20100101"
    end_date = "20260527"

    factor = fetch_pe_ttm(start_date, end_date)

    print(f"因子 shape: {factor.shape}")
    print(f"日期范围: {factor.index.min().date()} ~ {factor.index.max().date()}")
    print(f"非空率: {factor.notna().mean().mean():.2%}")

    factor.to_parquet(f"pe_ttm_new.parquet")
    print(f"已保存: pe_ttm_new.parquet")


if __name__ == "__main__":
    main()
