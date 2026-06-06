"""
行业 + 市值中性化（横截面 OLS 残差，FWL 向量化）
============================================================
严格等价于逐日横截面 OLS：
    factor ~ [行业 one-hot | log(size)],  hasconst=False,  取残差

用 Frisch–Waugh–Lovell 定理向量化（与上式逐元素相同，非近似）：
    1. 行业组内去均值 y、log(size)        —— 等价于投影掉满行业哑变量
    2. 去均值后 y 对去均值后 log(size) 过原点回归 → β
    3. 残差 = y* − β·log(size)*
等价性由 tests/test_neu.py 对拍 statsmodels.OLS 验证（容差 1e-10）。

有效截面（复刻 factor_analysis_fund.py 的三重 dropna）：
    每日只在「factor & industry & size 同时非空」的股票上回归，其余置 NaN。

单股独占某行业当天：组内去均值后 y*=s*=0 → 残差=0，与满哑变量 OLS 一致。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from alpha_shared.cleaning.preprocess import standardize_zscore


def neutralize(
    factor: pd.DataFrame,
    industry: pd.DataFrame,
    size: pd.DataFrame,
    *,
    min_samples: int = 30,
    restandardize: bool = True,
) -> pd.DataFrame:
    """
    行业 + 市值中性化，返回与 factor 同 (T, N) 网格的残差因子。

    参数：
        factor        : (T, N) 标准化后因子（一般为 zscore 之后）
        industry       : (T, N) 行业名（如中信一级行业名），将 reindex 到 factor 网格
        size           : (T, N) 市值面板（正值，单位任意；内部取 log）
        min_samples    : 每日有效股票数下限；当日 < max(行业数+2, min_samples) 则该日全 NaN
        restandardize  : 残差是否再做横截面 zscore（对齐原脚本外层 standardize）

    返回：
        (T, N) 中性化残差因子（无效截面/被剔除处为 NaN）
    """
    # 1. 对齐到 factor 网格 + log(size)
    industry = industry.reindex(index=factor.index, columns=factor.columns)
    size = size.reindex(index=factor.index, columns=factor.columns)
    log_size = np.log(size.where(size > 0))

    # 2. 三重非空交集 → 长表（复刻原版逐层 dropna）
    mask = factor.notna() & industry.notna() & log_size.notna()
    y_long = factor.where(mask).stack()
    if y_long.empty:
        logger.warning("[neutralize] 无有效截面，返回全 NaN")
        return pd.DataFrame(np.nan, index=factor.index, columns=factor.columns)
    long = pd.DataFrame(
        {
            "y": y_long,
            "ind": industry.stack().reindex(y_long.index),
            "s": log_size.stack().reindex(y_long.index),
        }
    )
    long["date"] = long.index.get_level_values(0)

    # 3. FWL 第①步：行业组内去均值
    g_di = long.groupby(["date", "ind"], sort=False)
    long["y_star"] = long["y"] - g_di["y"].transform("mean")
    long["s_star"] = long["s"] - g_di["s"].transform("mean")

    # 4. FWL 第②③步：逐日单变量过原点回归 β = Σ(y*·s*) / Σ(s*²)，残差 = y* − β·s*
    long["ys"] = long["y_star"] * long["s_star"]
    long["ss"] = long["s_star"] * long["s_star"]
    g_d = long.groupby("date", sort=False)
    num = g_d["ys"].transform("sum")
    den = g_d["ss"].transform("sum")
    # den==0（行业内市值无变异）→ β=0，残差退化为「仅行业中性」的组内去均值
    beta = np.where(den.to_numpy() > 0, num.to_numpy() / np.where(den.to_numpy() > 0, den.to_numpy(), 1.0), 0.0)
    long["resid"] = long["y_star"] - beta * long["s_star"]

    # 5. 最小样本守门：当日有效股票数 < max(行业数+2, min_samples) → 该日全 NaN
    cnt = g_d["y"].transform("size").to_numpy()
    n_ind = g_d["ind"].transform("nunique").to_numpy()
    thresh = np.maximum(n_ind + 2, min_samples)
    long.loc[cnt < thresh, "resid"] = np.nan

    # 6. 回写 (T, N) 网格
    out = long["resid"].unstack().reindex(index=factor.index, columns=factor.columns)

    n_days_kept = out.notna().any(axis=1).sum()
    logger.info(
        f"[neutralize] 有效截面 {int(mask.values.sum()):,} 单元 → "
        f"残差非空 {int(out.notna().values.sum()):,}，覆盖 {n_days_kept}/{len(factor)} 天"
        f"{'（残差已再标准化）' if restandardize else ''}"
    )

    # 7. 残差再标准化（对齐原脚本 standardize(neutralization(...))）
    if restandardize:
        out = standardize_zscore(out)
    return out
