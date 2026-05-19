"""
分层回测
------------------------------------------------------------
每 n 个交易日按因子值横截面分 g 组：
  - 组内等权
  - 持有期内不再平衡（T 日建仓 → 持有到下一个调仓日）
  - 停牌日贡献 0（nanmean 自动忽略该股票）

输出：
  - group_nav     : (T, g+1) 各组 + 多空(G_g - G_1) 累积净值
  - group_returns : (T, g+1) 每日收益
  - turnover      : (n_rebal-1, g) 单边换手率（新入组股票占新组规模比例）
  - summary       : 各组年化收益、年化波动、Sharpe、平均换手率
  - monotonicity  : 分组序 vs 年化收益 的 Pearson 相关（>0 = 正向单调）
"""

import warnings

import numpy as np
import pandas as pd
from loguru import logger


def layered_backtest(
    factor: pd.DataFrame,
    return_1d: pd.DataFrame,
    n: int = 5,
    g: int = 5,
) -> dict:
    """
    参数:
        factor   : (T, N) 清洗后因子（NaN = 不可交易）
        return_1d: (T, N) 未来 1 日 vwap-to-vwap 收益
        n        : 调仓周期（每 n 个交易日重新分组）
        g        : 分组数

    返回:
        dict: group_nav / group_returns / turnover / summary / monotonicity
    """
    # 对齐
    factor = factor.reindex(index=return_1d.index, columns=return_1d.columns)
    T, N = factor.shape

    # ==================== 1. 调仓日 & 分组 ====================
    rebal_idx = np.arange(0, T, n)  # 调仓日在 factor.index 中的整数位置
    n_rebal = len(rebal_idx)
    factor_rebal = factor.iloc[rebal_idx]  # (n_rebal, N)

    # 横截面 pct rank → 分组 [0, g-1]，NaN 自动归到 -1（不持有）
    pct = factor_rebal.rank(axis=1, pct=True, method="first").values
    pct_filled = np.where(np.isnan(pct), 0.0, pct)  # NaN → 0 → ceil(0)-1 = -1
    group_at_rebal = (np.ceil(pct_filled * g) - 1).astype(np.int32)
    group_at_rebal = np.clip(group_at_rebal, -1, g - 1)  # (n_rebal, N)

    # ==================== 2. 广播到每日持仓 ====================
    group_panel = np.full((T, N), -1, dtype=np.int32)
    for i, start in enumerate(rebal_idx):
        end = rebal_idx[i + 1] if i + 1 < n_rebal else T
        group_panel[start:end] = group_at_rebal[i]

    # ==================== 3. 每日每组收益 ====================
    ret_arr = return_1d.values
    group_returns = np.full((T, g + 1), np.nan)
    # 抑制 nanmean 遇到全 NaN 行的 RuntimeWarning（该情况结果本身就是 NaN，无害）
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for gi in range(g):
            in_group = group_panel == gi  # (T, N) bool
            ret_in = np.where(in_group, ret_arr, np.nan)
            group_returns[:, gi] = np.nanmean(ret_in, axis=1)
    # 多空: G_g (top) - G_1 (bottom)
    group_returns[:, g] = group_returns[:, g - 1] - group_returns[:, 0]

    cols = [f"G{i+1}" for i in range(g)] + ["LongShort"]
    group_ret_df = pd.DataFrame(group_returns, index=factor.index, columns=cols)
    group_nav = (1 + group_ret_df.fillna(0)).cumprod()

    # ==================== 4. 换手率（向量化） ====================
    # one_hot: (n_rebal, N, g) bool
    gi_range = np.arange(g)[None, None, :]
    one_hot = group_at_rebal[..., None] == gi_range
    added = (~one_hot[:-1] & one_hot[1:]).sum(axis=1)  # (n_rebal-1, g)
    curr_size = one_hot[1:].sum(axis=1)  # (n_rebal-1, g)
    turnover = added / np.maximum(curr_size, 1)
    turnover_df = pd.DataFrame(
        turnover,
        index=factor.index[rebal_idx[1:]],
        columns=[f"G{i+1}" for i in range(g)],
    )

    # ==================== 5. 汇总指标 ====================
    n_years = T / 252.0
    summary_rows = {}
    for col in cols:
        total_ret = group_nav[col].iloc[-1] - 1
        ann_ret = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else np.nan
        ann_vol = group_ret_df[col].std(ddof=0) * np.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
        if col in turnover_df.columns:
            mean_to = float(turnover_df[col].mean())
        elif col == "LongShort":
            # 多空组合的单边换手 = 多头组换手 + 空头组换手
            mean_to = float(turnover_df[f"G{g}"].mean() + turnover_df["G1"].mean())
        else:
            mean_to = np.nan
        summary_rows[col] = dict(
            ann_return=float(ann_ret),
            ann_vol=float(ann_vol),
            sharpe=float(sharpe) if np.isfinite(sharpe) else np.nan,
            mean_turnover=mean_to,
        )
    summary_df = pd.DataFrame(summary_rows).T

    # 单调性: 分组序 (1..g) vs 年化收益 的 Pearson
    group_ann = np.array([summary_rows[f"G{i+1}"]["ann_return"] for i in range(g)])
    if np.all(np.isfinite(group_ann)):
        monotonicity = float(np.corrcoef(np.arange(g), group_ann)[0, 1])
    else:
        monotonicity = float("nan")

    # ==================== 日志 ====================
    logger.info(f"[layered] n={n}日调仓 × g={g}组，调仓次数={n_rebal}")
    for col in cols:
        row = summary_rows[col]
        logger.info(
            f"  {col}: ann_return={row['ann_return']:+.2%}, "
            f"ann_vol={row['ann_vol']:.2%}, "
            f"sharpe={row['sharpe']:+.3f}, "
            f"avg_turnover={row['mean_turnover']:.2%}"
        )
    logger.info(
        f"[layered] monotonicity (Pearson G_id vs ann_return) = {monotonicity:+.3f}"
    )

    return dict(
        group_nav=group_nav,
        group_returns=group_ret_df,
        turnover=turnover_df,
        summary=summary_df,
        monotonicity=monotonicity,
    )
