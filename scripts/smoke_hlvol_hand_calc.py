"""
高/低位放量 冒烟验证：纯 pandas 手算 对照 HLVolReducer + rolling_group_ratio
============================================================================
3 只股票端到端：分钟 → 日内波动率/日收盘价 → 4 个占比因子（20日/5组）。
手算路径刻意与算子实现不同（groupby+pct_change / 逐日循环排序），逐值 assert。
"""
from __future__ import annotations
import numpy as np, pandas as pd
from loguru import logger
from core.minute_data import load_adjusted_minute_window
from core.operators.minute_hlvol import HLVolReducer
from core.operators.rolling_group_ratio import _one_group

STOCKS = ["300868.XSHE", "000001.XSHE", "600519.XSHG"]
START, END = "2020-09-01", "2020-12-31"
W, G = 20, 5


def hand_daily(df_one: pd.DataFrame) -> pd.DataFrame:
    """手算：逐日 1 分钟收益 std（ddof=1）+ 日收盘价。与 reducer 的 pivot 路径独立。"""
    df_one = df_one.sort_values("datetime")
    day = df_one["datetime"].dt.normalize()
    rows = []
    for d, g in df_one.groupby(day):
        close = g["close"].astype(float)
        ret = close.pct_change().dropna()
        vol = ret.std(ddof=1) if len(ret) >= 30 else np.nan
        rows.append((d, vol, close.iloc[-1]))
    return pd.DataFrame(rows, columns=["date", "vol", "close"]).set_index("date")


def hand_factor(vals: np.ndarray, sorts: np.ndarray, hi: bool) -> float:
    """手算单窗口占比：排序分组（逐日循环版）。"""
    if np.isnan(vals).any() or np.isnan(sorts).any():
        return np.nan
    order = np.argsort(sorts, kind="stable")
    k = W // G
    idx = order[-k:] if hi else order[:k]
    return vals[idx].mean() / vals.mean()


def main():
    allm = load_adjusted_minute_window(START, END, stocks=STOCKS)
    red = HLVolReducer(cache_key="smoke")
    n_checked = 0
    for ob, g in allm.groupby("order_book_id"):
        # ① 日频归约对照
        got = red.reduce(ob, g).set_index("date")
        exp = hand_daily(g)
        j = got.join(exp, how="inner")
        assert len(j) > 50, f"{ob}: 交集日数过少 {len(j)}"
        np.testing.assert_allclose(j["intraday_vol"], j["vol"], rtol=1e-10, equal_nan=True)
        np.testing.assert_allclose(j["close_d"], j["close"], rtol=1e-12)
        # ② 4 个占比因子对照（算子向量化 vs 手算逐窗）
        v = j["intraday_vol"].to_numpy()
        c = j["close_d"].to_numpy()
        for val, srt, hi, name in [
            (v, c, True, "d_highpos_vol_ratio"), (v, c, False, "d_lowpos_vol_ratio"),
            (c, v, True, "d_highvol_price_ratio"), (c, v, False, "d_lowvol_price_ratio"),
        ]:
            got_f = _one_group(val, srt, W, W // G, hi)
            exp_f = np.array([np.nan] * (W - 1) + [
                hand_factor(val[i - W + 1:i + 1], srt[i - W + 1:i + 1], hi)
                for i in range(W - 1, len(val))
            ])
            np.testing.assert_allclose(got_f, exp_f, rtol=1e-12, equal_nan=True)
            n_checked += np.isfinite(exp_f).sum()
        logger.info(f"{ob}: {len(j)} 日归约 + 4 因子逐值一致 ✅")
    logger.info(f"冒烟通过：3 股 × 4 因子，共 {n_checked} 个有效因子值逐值一致 ✅")


if __name__ == "__main__":
    main()
