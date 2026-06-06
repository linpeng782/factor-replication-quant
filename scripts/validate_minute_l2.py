"""
阶段2 V2.1 验证：新算子(读 minute/raw + 读时复权) superset == 旧 v4 缓存(bit级)
============================================================
逻辑：load_adjusted_minute_window(窗口, stocks) → 按股切表 → _compute_one_stock
      → 与 intermediate-cache/prv_v3__hc09d46528f/<ob>.parquet 逐列比对。
从全史起点 2005-01-04 起算 → warmup(前 std_window 日)与旧缓存自动对齐。

用法: python scripts/validate_minute_l2.py [--end 2008-12-31] [--stocks 000001.XSHE,...]
"""
import argparse
import numpy as np
import pandas as pd

from core import config
from core.minute_data import load_adjusted_minute_window
from core.operators.minute_intraday_aggregate import (
    _compute_one_stock, _SUPERSET_COLUMNS,
)

CACHE = config.INTERMEDIATE_CACHE_DIR / "prv_v3__hc09d46528f"
START = "2005-01-04"
STD_WINDOW, STD_THRESHOLD = 20, 1.0
# 不受 close 复权影响 → 应 bit 精确相等的列（raw 量 / 计数 / 分钟位置派生）
EXACT_COLS = [c for c in _SUPERSET_COLUMNS
              if not any(k in c for k in ("vwap", "return", "daily_high",
                                          "daily_low", "daily_close"))]
FLOAT_COLS = [c for c in _SUPERSET_COLUMNS if c not in EXACT_COLS]


def _cmp_col(a: np.ndarray, b: np.ndarray):
    """返回 (n_mismatch, max_rel, max_abs)。NaN==NaN 视为相等。
    带符号量(如 ridge_return_sum)会净到≈0 → rel 失真，故同时报 abs。"""
    both_nan = np.isnan(a) & np.isnan(b)
    one_nan = np.isnan(a) ^ np.isnan(b)
    absd = np.where(both_nan, 0.0, np.abs(a - b))
    rel = np.where(both_nan, 0.0, absd / np.maximum(np.abs(b), 1e-12))
    rel = np.where(one_nan, np.inf, rel)
    absd = np.where(one_nan, np.inf, absd)
    return (int((rel > 1e-9).sum()),
            float(np.nanmax(rel) if len(rel) else 0.0),
            float(np.nanmax(absd) if len(absd) else 0.0))


def _col_pass(c: str, max_rel: float, max_abs: float, n_mismatch: int) -> bool:
    """exact 列须 bit 精确；float 列 rel 或 abs 任一 ≤1e-6 即过(带符号量靠 abs)。"""
    if c in EXACT_COLS:
        return n_mismatch == 0
    return max_rel <= 1e-6 or max_abs <= 1e-6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--end", default="2008-12-31")
    ap.add_argument("--stocks", default="000001.XSHE,600000.XSHG,000002.XSHE,"
                                        "600036.XSHG,000651.XSHE,600519.XSHG")
    a = ap.parse_args()
    stocks = a.stocks.split(",")
    print(f"窗口 {START}~{a.end} | {len(stocks)} 股 | std_window={STD_WINDOW}")

    print("读 minute/raw + 复权 ...")
    allm = load_adjusted_minute_window(START, a.end, stocks=stocks)
    if allm.empty:
        print("!! 窗口内无 raw 数据"); return
    print(f"  loaded {len(allm):,} 行, 实际股票 {allm['order_book_id'].nunique()}")

    worst = {}  # col -> (max_rel, max_abs, n_mismatch)
    n_ok = 0
    for ob, sub in allm.groupby("order_book_id", sort=True):
        cache_p = CACHE / f"{ob}.parquet"
        if not cache_p.exists():
            print(f"  {ob}: 无缓存, skip"); continue
        new = _compute_one_stock(ob, sub, STD_WINDOW, STD_THRESHOLD)
        old = pd.read_parquet(cache_p)
        m = new.merge(old, on="date", suffixes=("_new", "_old"))
        m = m[(m["date"] >= START) & (m["date"] <= a.end)]
        bad = []
        for c in _SUPERSET_COLUMNS:
            nm, mr, ma = _cmp_col(m[f"{c}_new"].to_numpy(float),
                                  m[f"{c}_old"].to_numpy(float))
            if not _col_pass(c, mr, ma, nm):
                bad.append(c)
            prev = worst.get(c, (0.0, 0.0, 0))
            worst[c] = (max(prev[0], mr), max(prev[1], ma), prev[2] + nm)
        flag = "✅" if not bad else f"❌ {bad}"
        print(f"  {ob}: 重叠 {len(m)} 行 {flag}")
        if not bad:
            n_ok += 1

    print(f"\n=== 逐列汇总 (跨全部股票) ===")
    for c in _SUPERSET_COLUMNS:
        mr, ma, nm = worst[c]
        kind = "exact" if c in EXACT_COLS else "float"
        flag = "✅" if _col_pass(c, mr, ma, nm) else "❌"
        print(f"  {flag} {c:32s}[{kind}] max_rel={mr:.2e} max_abs={ma:.2e} 不等={nm}")
    print(f"\n结果: {n_ok}/{len(stocks)} 股 bit 级通过")


if __name__ == "__main__":
    main()
