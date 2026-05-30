"""
因子: npf_mrq_sue8 — 单季度净利润 SUE（基于过去 8 期环比差分）

定义：
    SUE = (net_profit_mrq_0 - net_profit_mrq_4 - diff_mean) / diff_std

  其中：
    diff_i = net_profit_mrq_i - net_profit_mrq_{i+1}   (i = 0..6)
    diff_mean = mean(diff_0, diff_1, ..., diff_6)
    diff_std  = std (diff_0, diff_1, ..., diff_6)，ddof=1

经济直觉：
  当期相对去年同期的"超预期净利润"（用过去 7 期环比差分的均值与标准差归一化）。

方向：正向（SUE 越高预期收益越高）
分类：景气

数据来源：rqdatac.get_factor()  —  PIT 字段 net_profit_mrq_0 ... net_profit_mrq_7
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


def compute_npf_mrq_sue8(start_date: str, end_date: str) -> pd.DataFrame:
    """
    SUE8 因子计算主流程。
    """
    rqdatac.init()
    universe = rqdatac.all_instruments(type="CS")
    stocks = universe["order_book_id"].tolist()

    # 一次性拉取 8 期单季度净利润：mrq_0 (当期) ... mrq_7 (7 季前)
    fields = [f"net_profit_mrq_{i}" for i in range(8)]
    panels = {f: fetch_pit_field(stocks, f, start_date, end_date) for f in fields}

    # 每个 panel 都是 (date × stock)，对齐 index 后做 element-wise 运算
    np_mrq = {i: panels[f"net_profit_mrq_{i}"] for i in range(8)}

    # 7 个环比差分：d_01, d_12, ..., d_67
    diffs = [np_mrq[i] - np_mrq[i + 1] for i in range(7)]   # 长度 = 7

    # 沿"周期维"求均值/标准差：把 7 个 (date × stock) panel 堆成 (date × stock × 7) 三维数组
    stacked = np.stack([d.values for d in diffs], axis=-1)   # shape (T, N, 7)

    diff_mean = np.nanmean(stacked, axis=-1)
    diff_std  = np.nanstd (stacked, axis=-1, ddof=1)

    diff_mean = pd.DataFrame(diff_mean, index=diffs[0].index, columns=diffs[0].columns)
    diff_std  = pd.DataFrame(diff_std,  index=diffs[0].index, columns=diffs[0].columns)

    # SUE8 = (mrq_0 - mrq_4 - diff_mean) / diff_std
    factor = (np_mrq[0] - np_mrq[4] - diff_mean) / diff_std

    # diff_std=0 会产 inf，统一置 NaN
    factor = factor.replace([np.inf, -np.inf], np.nan)

    return factor


def main():
    start_date = "20100101"
    end_date = "20260527"

    factor = compute_npf_mrq_sue8(start_date, end_date)

    print(f"因子 shape: {factor.shape}")
    print(f"日期范围: {factor.index.min().date()} ~ {factor.index.max().date()}")
    print(f"非空率: {factor.notna().mean().mean():.2%}")

    factor.to_parquet("npf_mrq_sue8.parquet")
    print(f"已保存: npf_mrq_sue8.parquet")


if __name__ == "__main__":
    main()
