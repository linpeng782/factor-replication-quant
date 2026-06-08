"""
指数日频分段收益构建 / 日更
============================================================
产出 market-data/index/000985_segments.parquet（APM 因子回归的市场参照序列）：
  index   : DatetimeIndex, name='date'
  columns : ret_overnight_idx  前收→当日开盘（隔夜）
            ret_am_idx          当日开盘→午收（9:31 open ~ 11:30 close）
            ret_pm_idx          午后开盘→收盘（13:01 open ~ 15:00 close）
            ret_pm_late_idx     14:01 open ~ 15:00 close（APM_1 用）

数据来源：rqdatac.get_price('000985.XSHG', frequency='1m') 原始价格（指数不复权）。
段边界（bar mod = hour*60+minute）：
  开盘 bar = 最早 mod（典型 571 = 09:31）
  午收 bar = mod 690（11:30）
  下午开 bar = 最早 mod >= 780
  尾盘开 bar = 最早 mod >= 840
  收盘 bar = 最晚 bar（典型 mod 900 = 15:00）

用法：
  python data_fetching/index_segments.py         # 增量（补缺日）
  python data_fetching/index_segments.py --full  # 全量重建（2005~今）
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import rqdatac
from loguru import logger

SYMBOL = "000985.XSHG"
FULL_START = "2005-01-01"
ACCOUNTS = [("13522652015", "123456"), ("18555079000", "123456")]


def _data_root() -> Path:
    return Path(os.environ.get("FACTOR_REPL_DATA_ROOT", "/nfs/ofs-prediction/peterzhenglinpeng"))


def _out_path() -> Path:
    return _data_root() / "market-data" / "index" / "000985_segments.parquet"


def _extract_segments(df1m: pd.DataFrame) -> pd.DataFrame:
    """单日级 long 表（datetime, open, close）→ 日频四段收益宽表。"""
    df = df1m.copy()
    dt = pd.to_datetime(df["datetime"])
    df["date"] = dt.dt.normalize()
    df["mod"] = dt.dt.hour * 60 + dt.dt.minute

    # 段边界 mod 范围
    AM_LO, AM_HI = 570, 690    # 09:30~11:30
    PM_LO, PM_HI = 780, 900    # 13:00~15:00
    PML_LO = 840               # 14:00~

    def first_open(sub, lo, hi):
        m = sub[(sub["mod"] >= lo) & (sub["mod"] <= hi)]
        return m.sort_values("mod")["open"].iloc[0] if len(m) else np.nan

    def last_close(sub, lo, hi):
        m = sub[(sub["mod"] >= lo) & (sub["mod"] <= hi)]
        return m.sort_values("mod")["close"].iloc[-1] if len(m) else np.nan

    rows = []
    for day, g in df.groupby("date"):
        am_open  = first_open(g, AM_LO, AM_HI)
        am_close = last_close(g, AM_LO, AM_HI)
        pm_open  = first_open(g, PM_LO, PM_HI)
        pm_close = last_close(g, PM_LO, PM_HI)    # = day close
        pml_open = first_open(g, PML_LO, PM_HI)

        rows.append({
            "date": day,
            "_open": am_open,
            "_am_close": am_close,
            "_pm_open": pm_open,
            "_pm_close": pm_close,
            "_pml_open": pml_open,
        })

    seg = pd.DataFrame(rows).set_index("date").sort_index()
    # overnight: today open / prev close - 1
    prev_close = seg["_pm_close"].shift(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        seg["ret_overnight_idx"] = seg["_open"] / prev_close - 1
        seg["ret_am_idx"]        = seg["_am_close"] / seg["_open"] - 1
        seg["ret_pm_idx"]        = seg["_pm_close"] / seg["_pm_open"] - 1
        seg["ret_pm_late_idx"]   = seg["_pm_close"] / seg["_pml_open"] - 1

    return seg[["ret_overnight_idx", "ret_am_idx", "ret_pm_idx", "ret_pm_late_idx"]]


def build(out_path: Path, full: bool = False) -> None:
    rqdatac.init(*ACCOUNTS[0])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    end = pd.Timestamp(rqdatac.get_latest_trading_date())

    if full or not out_path.exists():
        start = pd.Timestamp(FULL_START)
        logger.info(f"[全量] {SYMBOL} 分段收益 {start.date()} ~ {end.date()}")
    else:
        existing = pd.read_parquet(out_path)
        existing.index = pd.to_datetime(existing.index)
        last = existing.index.max()
        nxt = pd.Timestamp(rqdatac.get_next_trading_date(last))
        if nxt > end:
            logger.success(f"[日更] {SYMBOL} 已最新（到 {last.date()}），无需更新")
            return
        start = nxt
        logger.info(f"[日更] {SYMBOL} 补 {start.date()} ~ {end.date()}")

    df1m = rqdatac.get_price(
        SYMBOL,
        start_date=str(start.date()),
        end_date=str(end.date()),
        frequency="1m",
        fields=["open", "close"],
        expect_df=True,
    )
    if df1m is None or len(df1m) == 0:
        logger.warning("get_price 返回空")
        return

    df1m = df1m.reset_index()
    df1m = df1m.rename(columns={"trading_date": "date"}) if "trading_date" in df1m.columns else df1m
    logger.info(f"  获取 {len(df1m):,} 行分钟数据")

    new_seg = _extract_segments(df1m)

    if not full and out_path.exists():
        old = pd.read_parquet(out_path)
        old.index = pd.to_datetime(old.index)
        combined = pd.concat([old, new_seg])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    else:
        combined = new_seg

    combined.to_parquet(out_path)
    logger.success(
        f"已保存 {out_path} | shape={combined.shape} | "
        f"{combined.index.min().date()} ~ {combined.index.max().date()} | "
        f"NaN率 overnight={combined['ret_overnight_idx'].isna().mean():.1%}"
    )


def main():
    ap = argparse.ArgumentParser(description="000985 指数日频分段收益 构建/日更")
    ap.add_argument("--full", action="store_true", help="全量重建")
    ap.add_argument("--output", type=Path, default=None)
    a = ap.parse_args()
    out = a.output or _out_path()
    logger.info(f"输出: {out} | 模式: {'全量' if a.full else '增量'}")
    build(out, full=a.full)


if __name__ == "__main__":
    main()
