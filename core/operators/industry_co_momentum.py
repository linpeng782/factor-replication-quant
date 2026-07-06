"""
行业/市场联合动量算子（国信 2024-01「个股与行业的共振」）
============================================================
创建主表的特殊聚合算子（同类：minute_intraday_aggregate）。一个算子服务 5 因子，
由 spec params 区分：
  sub_factor ∈ {ICM, VICM, VICR, CMC}，signal_source ∈ {industry, market}
  · ICM/VICM/VICR/CMC + industry → 行业联合动量族
  · CMC + market → MCMC（市场联合动量，signal 换中证全指 000985）

算法（个股过去 window 日，半衰期权重 wᵢ = 2^(−i/(n−1))）：
  ICM  = 纯涨幅 top n_mom 日 → 加权所属行业收益
  VICM = (涨幅×量) top n_mom 日；VICR = (涨幅×量) bottom n_rev 日；CMC = VICM − VICR
自包含加载：后复权 OHLCV(load_adjusted_panels) + 行业指数收益面板 + 行业映射 / 中证全指。
产出 long 主表 (order_book_id, date, <output_column>)，yolo_engine 后续 pivot 落 raw。

⚠️ 算法唯一真相源 = 本模块。验证脚本(scripts/comomentum_*.py)从这里 import，勿重复实现。
复现/口径结论见 sources/guosen/co_momentum/docs/co_momentum.md（生产用 neu 版）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

import config
from core.data.adjusted_panels import load_adjusted_panels

from . import Context, OpRegistry

SUB_FACTORS = ("ICM", "VICM", "VICR", "CMC")
SIGNAL_SOURCES = ("industry", "market")


def _w(n: int) -> np.ndarray:
    """半衰期权重列向量 (n,1)：w[0]=1（排名1），w[n-1]=0.5（排名n）。"""
    return (2.0 ** (-np.arange(n) / (n - 1)))[:, None]


def build_ind_ret_panel(dates, stocks) -> pd.DataFrame:
    """每股每日【所属行业】指数收益面板 (T×N)，向量化 broadcast（含旧名别名，零特判）。"""
    ind_idx = pd.read_parquet(config.INDUSTRY_INDEX_RETURN_PATH)
    ind_idx.index = pd.to_datetime(ind_idx.index)
    panel = pd.read_parquet(config.INDUSTRY_PANEL_ZX_PATH)
    panel.index = pd.to_datetime(panel.index)
    panel = panel.reindex(index=dates, columns=stocks)
    ind_idx = ind_idx.reindex(dates)
    out = pd.DataFrame(np.nan, index=dates, columns=stocks)
    for name in ind_idx.columns:
        mask = (panel == name)
        if mask.values.any():
            bc = np.broadcast_to(ind_idx[name].values[:, None], mask.shape)
            out = out.mask(mask, pd.DataFrame(bc, index=dates, columns=stocks))
    return out


def build_mkt_ret_panel(dates, stocks, fetcher) -> pd.DataFrame:
    """中证全指(000985)收益 broadcast 到 (T×N)（MCMC 用）。"""
    fetcher._init_rq()
    mkt = fetcher._rq.get_price(
        "000985.XSHG", start_date=str(dates.min().date()),
        end_date=str(dates.max().date()), frequency="1d", fields=["close"])
    mc = mkt["close"]
    if mc.index.nlevels > 1:
        mc = mc.reset_index(level=0, drop=True)
    mc.index = pd.to_datetime(mc.index)
    r = mc.pct_change().reindex(dates)
    return pd.DataFrame(np.broadcast_to(r.values[:, None], (len(dates), len(stocks))),
                        index=dates, columns=stocks)


def compute_factor(ret_np, vol_np, sig_np, t, window, n_mom, n_rev, sub) -> np.ndarray:
    """下标 t：全市场每股 sub 因子值 (N,)。窗口内含 NaN 的股置 NaN。"""
    sl = slice(t - window + 1, t + 1)
    r, v, sig = ret_np[sl], vol_np[sl], sig_np[sl]
    valid = ~(np.isnan(r).any(0) | np.isnan(v).any(0) | np.isnan(sig).any(0))
    if sub == "ICM":
        o = np.argsort(np.where(np.isnan(r), -np.inf, r), axis=0, kind="stable")
        f = (np.take_along_axis(sig, o[-n_mom:][::-1], 0) * _w(n_mom)).sum(0)
    else:
        s = r * v
        o = np.argsort(np.where(np.isnan(s), -np.inf, s), axis=0, kind="stable")
        vicm = (np.take_along_axis(sig, o[-n_mom:][::-1], 0) * _w(n_mom)).sum(0)
        vicr = (np.take_along_axis(sig, o[:n_rev], 0) * _w(n_rev)).sum(0)
        f = {"VICM": vicm, "VICR": vicr, "CMC": vicm - vicr}[sub]
    f[~valid] = np.nan
    return f


@OpRegistry.register("industry_co_momentum")
def op_industry_co_momentum(ctx: Context, step: dict, fetcher) -> None:
    """唯一入口算子：自包含加载 → 逐日算因子 → long 主表写 ctx。"""
    target = step.get("output_dataframe", "data")
    if ctx.has_df(target):
        raise ValueError(f"industry_co_momentum: 主表 {target!r} 已存在；本算子负责创建主表")
    if not ctx.universe:
        raise ValueError("industry_co_momentum: ctx.universe 为空（先确定股票池）")

    sub = step["sub_factor"]
    sig_src = step["signal_source"]
    out_col = step["output_column"]
    window = int(step.get("window", 20))
    n_mom = int(step.get("n_mom", 5))
    n_rev = int(step.get("n_rev", 15))
    if sub not in SUB_FACTORS:
        raise ValueError(f"sub_factor 须 ∈ {SUB_FACTORS}, 得到 {sub!r}")
    if sig_src not in SIGNAL_SOURCES:
        raise ValueError(f"signal_source 须 ∈ {SIGNAL_SOURCES}, 得到 {sig_src!r}")

    start = ctx.start_date or "2010-01-01"
    logger.info(
        f"[industry_co_momentum] sub={sub} signal={sig_src} "
        f"window={window} n_mom={n_mom} n_rev={n_rev} | load {start}~{ctx.end_date} "
        f"({len(ctx.universe)} 股候选)")
    panels = load_adjusted_panels(
        stocks=list(ctx.universe), start=start, end=ctx.end_date, fields=("close", "volume"))
    close, volume = panels["close"], panels["volume"]
    ret = close.pct_change(fill_method=None)
    dates, stocks = ret.index, list(close.columns)
    sig = (build_ind_ret_panel(dates, stocks) if sig_src == "industry"
           else build_mkt_ret_panel(dates, stocks, fetcher))

    ret_np, vol_np, sig_np = ret.values, volume.values, sig.values
    arr = np.full((len(dates), len(stocks)), np.nan)
    for t in range(window - 1, len(dates)):
        arr[t] = compute_factor(ret_np, vol_np, sig_np, t, window, n_mom, n_rev, sub)
    wide = pd.DataFrame(arr, index=dates, columns=stocks)
    wide.index.name = "date"

    long = (wide.reset_index()
            .melt(id_vars="date", var_name="order_book_id", value_name=out_col)
            .dropna(subset=[out_col]).reset_index(drop=True))
    logger.info(f"[industry_co_momentum] {sub} → 主表 {long.shape}（非空 {len(long):,}）")
    ctx.set_df(target, long)
