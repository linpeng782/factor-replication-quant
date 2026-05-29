"""
因子: npf_mrq_accs8 — 单季度净利润增加速度（8 期二次回归二次项系数）

定义：
  对每个 (股票, 日期)：
    1. y = [net_profit_mrq_0, ..., net_profit_mrq_7]
    2. x = [0, 1, 2, ..., 7]，做 z-score 归一化（ddof=1）
    3. 对 (x_z, y_z) 做二次回归 a2*x^2 + a1*x + a0，取 a2
       — y 也做 z-score 归一化（让不同体量公司的 a 系数可比）
       — 任一 y_i NaN 或 std(y)=0 → 该行结果 NaN
    4. 计算 (y[0..3], y[4..7]) 之间的 Pearson 相关性
    5. |corr| >= 0.9 时输出 NaN（前后两半"过相似"则不可信）

方向：正向；分类：景气

实现：
  固定 x（8 期等距 z-score）+ 任意系数对应的权重向量 W[i]，整体退化为
    a2 = (Y_zscored @ weights)，单次矩阵乘法搞定全市场。
"""

import numpy as np
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


def row_polyfit_a2(y_stack: np.ndarray) -> np.ndarray:
    """
    向量化二次回归取 a2 系数。
    y_stack: shape (T, N, 8)
    返回 a2: shape (T, N)

    思路：x = z-score([0..7]) 是固定的，所以 Vandermonde V 和 pinv(V) 都只算一次。
    取 a2 对应的权重向量 weights = pinv(V)[0]（高次在前），shape (8,)。
    然后 a2 = (y_zscored @ weights)，单次矩阵乘法。
    """
    n = y_stack.shape[-1]
    x_raw = np.arange(n, dtype=np.float64)
    x_z = (x_raw - x_raw.mean()) / x_raw.std(ddof=1)

    # Vandermonde (n, 3) = [x^2, x^1, x^0]，degree=2，高次在前
    V = np.column_stack([x_z ** 2, x_z, np.ones(n)])
    W = np.linalg.pinv(V)         # (3, 8)
    weights = W[0]                # 取 a2 → 第 0 行

    # 行向 z-score y（ddof=1）
    y_mean = np.nanmean(y_stack, axis=-1, keepdims=True)
    y_std  = np.nanstd(y_stack, axis=-1, ddof=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        y_z = np.where(y_std > 0, (y_stack - y_mean) / y_std, np.nan)

    # 任一 y_i NaN 或 std==0 → 整行 NaN
    nan_mask = np.isnan(y_stack).any(axis=-1) | (y_std.squeeze(-1) == 0)

    # a2 = y_z @ weights
    a2 = y_z @ weights            # NaN 位会传播
    a2 = np.where(nan_mask, np.nan, a2)
    return a2


def row_pearson(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    行向 Pearson 相关系数。A, B shape (..., n)，返回 (...)。
    任一侧含 NaN → 该行 NaN（用 .mean() 而非 nanmean，对齐 spec 行为）。
    """
    nan_mask = np.isnan(A).any(axis=-1) | np.isnan(B).any(axis=-1)

    mean_a = A.mean(axis=-1, keepdims=True)
    mean_b = B.mean(axis=-1, keepdims=True)
    A_dev = A - mean_a
    B_dev = B - mean_b

    cov_ab = (A_dev * B_dev).mean(axis=-1)
    var_a  = (A_dev ** 2).mean(axis=-1)
    var_b  = (B_dev ** 2).mean(axis=-1)

    denom = np.sqrt(var_a * var_b)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.where(denom > 0, cov_ab / denom, np.nan)
    corr = np.where(nan_mask, np.nan, corr)
    return corr


def main():
    start_date, end_date = "20100101", "20260527"
    rqdatac.init()
    stocks = rqdatac.all_instruments(type="CS")["order_book_id"].tolist()

    # 8 期单季度净利润
    panels = [
        fetch_factor(stocks, f"net_profit_mrq_{i}", start_date, end_date)
        for i in range(8)
    ]

    # 堆成 (T, N, 8) 三维数组
    y_stack = np.stack([p.values for p in panels], axis=-1)

    print("行向二次回归取 a2 ...")
    a2 = row_polyfit_a2(y_stack)

    print("前 4 期 vs 后 4 期 Pearson 相关 ...")
    corr = row_pearson(y_stack[..., :4], y_stack[..., 4:])

    # |corr| >= 0.9 → NaN
    factor_arr = np.where(np.abs(corr) < 0.9, a2, np.nan)
    factor = pd.DataFrame(factor_arr, index=panels[0].index, columns=panels[0].columns)

    print(f"因子 shape: {factor.shape}, 非空率 {factor.notna().mean().mean():.2%}")
    factor.to_parquet("npf_mrq_accs8.parquet")


if __name__ == "__main__":
    main()
