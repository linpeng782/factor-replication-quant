"""
IC（Information Coefficient）计算
------------------------------------------------------------
- compute_ic_series : 每日横截面 IC 时序
- ic_summary        : 从 IC 时序生成汇总指标
- compute_ic_report : 多持有期批量 IC 报告

IC 定义：
    IC[T] = corr(factor[T], forward_return[T])  （横截面）
    method = 'spearman'（Rank IC，默认）或 'pearson'

向量化实现：
    Spearman = Pearson on rank（对原面板先做行 rank）
    用 numpy 批量算每行的 Pearson，避免 pandas per-row apply
"""

import numpy as np
import pandas as pd
from loguru import logger


def _cross_sectional_pearson(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    (T, N) 两矩阵每行 Pearson（nan-safe，向量化）

    返回: (T,) ndarray
    """
    mask = ~(np.isnan(a) | np.isnan(b))
    n = mask.sum(axis=1)  # (T,) 每日有效点数

    # 带 mask 的均值
    a_m = np.where(mask, a, 0.0)
    b_m = np.where(mask, b, 0.0)
    n_safe = np.maximum(n, 1)
    mean_a = a_m.sum(axis=1) / n_safe
    mean_b = b_m.sum(axis=1) / n_safe

    # 去均值（无效位置置 0，对分子分母都无贡献）
    a_c = np.where(mask, a - mean_a[:, None], 0.0)
    b_c = np.where(mask, b - mean_b[:, None], 0.0)

    num = (a_c * b_c).sum(axis=1)
    den = np.sqrt((a_c**2).sum(axis=1) * (b_c**2).sum(axis=1))
    # 用 np.divide 的 out/where 参数避免除零警告
    ic = np.full_like(den, np.nan)
    np.divide(num, den, out=ic, where=den > 0)
    # 有效点 < 2 的天 IC 无意义
    ic = np.where(n >= 2, ic, np.nan)
    return ic


def compute_ic_series(
    factor: pd.DataFrame,
    forward_return: pd.DataFrame,
    method: str = "spearman",
) -> pd.Series:
    """
    每日横截面 IC 时序

    参数:
        factor        : (T, N) 清洗后因子
        forward_return: (T, N) 未来收益
        method        : 'spearman' 或 'pearson'

    返回:
        (T,) Series，NaN 位置表示当日有效点不足
    """
    forward_return = forward_return.reindex(index=factor.index, columns=factor.columns)

    if method == "spearman":
        # Spearman = Pearson on rank；rank 对 NaN 保持 NaN
        f = factor.rank(axis=1, method="average").values
        r = forward_return.rank(axis=1, method="average").values
    elif method == "pearson":
        f = factor.values
        r = forward_return.values
    else:
        raise ValueError(f"不支持的 method: {method}")

    ic = _cross_sectional_pearson(f, r)
    return pd.Series(ic, index=factor.index, name=f"IC_{method}")


def ic_summary(ic: pd.Series) -> dict:
    """从 IC 时序生成汇总指标"""
    s = ic.dropna()
    n = len(s)
    if n == 0:
        return dict(n_days=0, ic_mean=np.nan, ic_std=np.nan)

    ic_mean = float(s.mean())
    ic_std = float(s.std(ddof=0))
    icir = ic_mean / ic_std if ic_std > 0 else np.nan
    t_stat = ic_mean / (ic_std / np.sqrt(n)) if ic_std > 0 else np.nan

    return dict(
        n_days=n,
        ic_mean=ic_mean,
        ic_std=ic_std,
        icir=icir,
        ic_t=t_stat,
        pct_positive=float((s > 0).mean()),
        pct_abs_gt_2pct=float((s.abs() > 0.02).mean()),
        pct_abs_gt_5pct=float((s.abs() > 0.05).mean()),
        ic_skew=float(s.skew()),
        ic_kurt=float(s.kurt()),
    )


def compute_ic_report(
    factor: pd.DataFrame,
    forward_returns: dict,
    method: str = "spearman",
):
    """
    对多个持有期批量计算 IC 汇总

    参数:
        factor         : (T, N) 清洗后因子
        forward_returns: dict[N -> DataFrame]，build_forward_returns 的输出
        method         : IC 类型

    返回:
        (summary_df, ic_series_dict)
            summary_df     : index=horizon (如 '2d')，columns=指标
            ic_series_dict : {horizon (int) -> pd.Series}
    """
    summary_rows = {}
    series_dict = {}
    for n, r in forward_returns.items():
        ic = compute_ic_series(factor, r, method=method)
        series_dict[n] = ic
        summary_rows[f"{n}d"] = ic_summary(ic)

    summary_df = pd.DataFrame(summary_rows).T
    summary_df.index.name = "horizon"

    logger.info(f"[ic] {method.upper()} IC report:")
    for h, row in summary_df.iterrows():
        logger.info(
            f"  {h}: ic_mean={row['ic_mean']:+.4f}, "
            f"ic_std={row['ic_std']:.4f}, "
            f"icir={row['icir']:+.3f}, "
            f"t_stat={row['ic_t']:+.2f}, "
            f"positive_pct={row['pct_positive']:.2%}, "
            f"n_days={int(row['n_days'])}"
        )
    return summary_df, series_dict
