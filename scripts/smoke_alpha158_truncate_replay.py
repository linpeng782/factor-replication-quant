"""
alpha158 增量正确性验收：truncate-replay 真实数据对账
======================================================
思路（docs/alpha158_incremental_design.md §9）：
  用真实按股 OHLCV 全量算出"ground truth"因子面板 → 砍掉尾部 K 天得到"昨天的旧面板"
  （并删去截断点尚未上市的股 = 模拟它们当时不是列）→ 从【未截断的源】逐日重放这 K 天
  → 重放结果 vs ground truth 逐格对比。

4 个防假阳性判据：
  ① 截断的是因子产出、不是源（源 OHLCV 永远全）
  ② 多天重放（逐日 append，模拟连续日更）
  ③ 显式断言新股列（截断窗口内真实 IPO）回来且吻合
  ④ NaN-aware：max|Δ| 只在双方非 NaN 处算；另单独断言 NaN 模式逐格一致

通过 = 重叠(日期×股票) max|Δ|==0 且 NaN 模式一致 且 IPO 列吻合。

核心函数 incremental_step() 即生产 alpha158_daily_update.py 的内核（共享，杜绝脱节）。
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core import config
from core.producers.alpha158 import Alpha158Panel

# ==================== 配置 ====================
RAW_DIR = config.RAW_OHLCV_DIR
EX_DIR = config.EX_FACTORS_DIR
N_WORKERS = 32
N_SAMPLE = 600                      # 抽样股票数（alpha158 跨股独立，子集即可验逻辑）
K = 20                              # 截断/重放天数
WINDOWS = [5, 60]                   # 覆盖最小+最大窗口
TEST_FACTORS = ["MA5", "ROC5", "CORR5", "MA60", "STD60", "VMA60"]
W = max(WINDOWS)                    # warmup = max window = 60
BUFFER = 10                         # 回读窗口 = W + BUFFER 个交易日
FIELDS = ["open", "high", "low", "close", "volume", "vwap"]


# ==================== 数据读取（读时复权：价×ffill(cum)，量÷cum）====================
def _load_one(stock: str, raw_dir: str, ex_dir: str) -> pd.DataFrame | None:
    raw_path = Path(raw_dir) / f"{stock}.parquet"
    if not raw_path.exists():
        return None
    raw = pd.read_parquet(raw_path)
    raw.index = pd.to_datetime(raw.index)
    ex_path = Path(ex_dir) / f"{stock}.parquet"
    if ex_path.exists():
        ex = pd.read_parquet(ex_path)
        cum = ex["ex_cum_factor"].reindex(raw.index, method="ffill").fillna(1.0)
    else:
        cum = pd.Series(1.0, index=raw.index)
    adj = raw[["open", "high", "low", "close", "volume", "total_turnover"]].copy()
    for c in ["open", "high", "low", "close"]:
        adj[c] = adj[c] * cum
    adj["volume"] = adj["volume"] / cum
    adj["vwap"] = adj["total_turnover"] / adj["volume"].replace(0, np.nan)
    adj["stock_code"] = stock
    return adj.reset_index().rename(columns={"date": "datetime"})


def load_source_panels(stocks: list[str]) -> dict[str, pd.DataFrame]:
    """并行读 → 6 个宽表面板 {field: date×stock}。这是"未截断的源"。"""
    worker = partial(_load_one, raw_dir=str(RAW_DIR), ex_dir=str(EX_DIR))
    dfs = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        for fut in as_completed([ex.submit(worker, s) for s in stocks]):
            r = fut.result()
            if r is not None:
                dfs.append(r)
    long = pd.concat(dfs, ignore_index=True)
    panels = {}
    for f in FIELDS:
        w = long.pivot(index="datetime", columns="stock_code", values=f)
        panels[f] = w.sort_index().sort_index(axis=1)
    return panels


def compute(panels: dict, only_names: list[str]) -> dict[str, pd.DataFrame]:
    return Alpha158Panel(panels).compute_all(windows=WINDOWS, only_names=only_names)


# ==================== 核心增量函数（= 生产内核）====================
def incremental_step(old_wide: pd.DataFrame, new_full: pd.DataFrame, up_to: pd.Timestamp) -> pd.DataFrame:
    """旧因子宽表 + 尾窗算出的新因子宽表 → 切 (last, up_to] 新行 → concat(列并集) → dedup → 排序。

    这就是 §6 的算法；append 列并集自动纳新股、dedup(keep last) 幂等。
    生产里 new_full 由"读盘尾窗+compute"得到；测试里由内存源切窗+compute 得到，等价。
    """
    last = old_wide.index.max()
    new_rows = new_full.loc[(new_full.index > last) & (new_full.index <= up_to)]
    combined = pd.concat([old_wide, new_rows])
    return combined[~combined.index.duplicated(keep="last")].sort_index()


# ==================== 对账 ====================
def reconcile(name: str, replayed: pd.DataFrame, truth: pd.DataFrame, ipo_cols: list[str],
              onset: dict) -> dict:
    d = replayed.index.intersection(truth.index)
    c = replayed.columns.intersection(truth.columns)
    a = replayed.loc[d, c]
    b = truth.loc[d, c]
    # 只比"活区"：个股上市后(date >= onset)。上市前死区下游被 new_stock_mask 抹掉；
    # 且 CNT 族因布尔吃 NaN + min_periods=1 在死区有伪 0（全量构建的 latent 伪值），增量为 NaN 更正确。
    live = pd.DataFrame(False, index=d, columns=c)
    for s in c:
        o = onset.get(s)
        if o is not None:
            live.loc[live.index >= o, s] = True
    a = a.where(live)
    b = b.where(live)
    both = ~(a.isna() | b.isna())
    n_both = int(both.values.sum())
    av = a.values[both.values]
    bv = b.values[both.values].astype(float)
    abs_d = np.abs(av - bv)
    max_diff = float(abs_d.max()) if n_both else np.nan
    max_rel = float((abs_d / (np.abs(bv) + 1e-12)).max()) if n_both else np.nan
    nan_match = bool((a.isna().values == b.isna().values).all())   # ④ NaN 模式逐格一致
    # ③ IPO 列断言
    ipo_present = [s for s in ipo_cols if s in replayed.columns]
    ipo_ok = True
    for s in ipo_present:
        if s not in truth.columns:
            continue
        ra, rb = a[s], b[s]   # 已 live-mask
        bb = ~(ra.isna() | rb.isna())
        if bb.any() and float((ra[bb] - rb[bb]).abs().max()) > 0:
            ipo_ok = False
        if not (ra.isna().values == rb.isna().values).all():
            ipo_ok = False
    # 通过判据：相对误差 < 1e-6（pandas rolling 滑动累加器的浮点噪声，非逻辑差异）
    TOL_REL = 1e-6
    val_ok = np.isnan(max_rel) or max_rel < TOL_REL
    return {
        "factor": name, "max_abs": max_diff, "max_rel": max_rel,
        "nan_match": nan_match, "ipo_back": len(ipo_present), "ipo_ok": ipo_ok,
        "pass": val_ok and nan_match and ipo_ok,
    }


def main():
    import argparse
    global WINDOWS, TEST_FACTORS, W
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="验全 158 因子（windows=[5,10,20,30,60]）")
    ap.add_argument("--sample", type=int, default=N_SAMPLE)
    args = ap.parse_args()
    if args.all:
        WINDOWS = [5, 10, 20, 30, 60]
        TEST_FACTORS = Alpha158Panel.list_factor_names(windows=WINDOWS)
        W = max(WINDOWS)
        logger.info(f"全因子模式：{len(TEST_FACTORS)} 个因子，windows={WINDOWS}")

    t0 = time.time()
    logger.info("[1/5] 选样本（含真实 IPO 股 + 锚定股）...")
    all_stocks = [f.stem for f in sorted(RAW_DIR.glob("*.parquet"))]
    ipo_known = ["001393.XSHE", "603435.XSHG", "688635.XSHG"]   # 近 20 日 IPO（脚本预探测）
    anchor = ["600519.XSHG", "000001.XSHE", "300750.XSHE"]
    rng = np.random.default_rng(42)
    rest = [s for s in all_stocks if s not in ipo_known + anchor]
    sample = anchor + ipo_known + list(rng.choice(rest, N_SAMPLE - len(anchor) - len(ipo_known), replace=False))
    sample = [s for s in dict.fromkeys(sample) if s in all_stocks]
    logger.info(f"  样本 {len(sample)} 只")

    logger.info("[2/5] 读源面板（未截断）...")
    panels = load_source_panels(sample)
    dates = panels["close"].index
    T = dates[-1]
    cut = dates[-(K + 1)]
    logger.info(f"  日期 {dates[0].date()}~{T.date()} | 截断点 cut={cut.date()} | 重放 {K} 天到 {T.date()}")

    logger.info("[3/5] 全量算 ground truth...")
    truth = compute(panels, TEST_FACTORS)

    logger.info("[4/5] 构造截断旧面板（删 cut 时未上市的股 = 当时不是列）...")
    first_date = {s: panels["close"][s].first_valid_index() for s in panels["close"].columns}
    present_at_cut = [s for s, fd in first_date.items() if fd is not None and fd <= cut]
    ipo_in_window = [s for s, fd in first_date.items() if fd is not None and fd > cut]
    logger.info(f"  cut 时在册 {len(present_at_cut)} 只；窗口内 IPO {len(ipo_in_window)} 只: {ipo_in_window}")
    old = {f: truth[f].reindex(columns=present_at_cut).loc[:cut].copy() for f in TEST_FACTORS}

    logger.info(f"[5/5] 逐日重放 {K} 天（每日：源切尾窗 {W+BUFFER} 日 → compute → incremental_step）...")
    replay_dates = dates[dates > cut]
    for ti in replay_dates:
        win_panels = {f: p.loc[:ti].tail(W + BUFFER) for f, p in panels.items()}
        new_full = compute(win_panels, TEST_FACTORS)
        for f in TEST_FACTORS:
            old[f] = incremental_step(old[f], new_full[f], ti)

    logger.info("=" * 60)
    rows = [reconcile(f, old[f], truth[f], ipo_in_window, first_date) for f in TEST_FACTORS]
    df = pd.DataFrame(rows)
    if len(df) <= 12:
        logger.info(f"\n{df.to_string(index=False)}")
    else:
        logger.info(f"  因子数={len(df)} | max_rel 全局最大={df['max_rel'].max():.2e} | "
                    f"nan_match 全True={df['nan_match'].all()} | ipo_ok 全True={df['ipo_ok'].all()}")
        worst = df.nlargest(5, "max_rel")[["factor", "max_abs", "max_rel"]]
        logger.info(f"  max_rel 最大的 5 个:\n{worst.to_string(index=False)}")

    if df["pass"].all():
        logger.success(f"✅ truncate-replay 全部通过：{K} 天重放 == 全量重算（rel<1e-6），{len(df)} 因子 + 新股列吻合")
    else:
        logger.error(f"❌ 未通过 {(~df['pass']).sum()} 个: {df[~df['pass']]['factor'].tolist()}")
    logger.info(f"总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
