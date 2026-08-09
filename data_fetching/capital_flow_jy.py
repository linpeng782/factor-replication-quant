# -*- coding: utf-8 -*-
"""jy 分单资金流全量回填 + 增量更新（开源系列12 大单/小单资金流因子原料）
================================================================
三阶段：
  Stage A: 按月拉全市场 → per-month/<YYYY-MM>.parquet（原始长表，增量基线）
  Stage B: per-month → per-stock/<股>.parquet（逐股宽表，源真相，16 列）
  Stage C: per-month → panels/{lb,sb}_{net,gross}_value.parquet（T×N 消费面板）

数据口径（scripts/verify_jy_capital_flow_tiers.py 已验证 + 聚源官方表说明证实）：
  - 来源: dquant get_capital_flow(source="jy") ← 聚源 MySQL DZ_TradingCapitalFlow（信息来源恒生电子）
  - 官方语义: ValueRange=单笔成交金额区间；buy=主动买成交(外盘)，sell=主动卖成交(内盘)
    ——单侧计数不配平，与 Wind"按挂单金额双边归档"口径不同（见 paper_12_moneyflow/paper.md）
  - value_range 实测映射（与 dquant docstring 相反！）:
      1=小单(<4万) 2=中单(4-20万) 3=大单(20-100万) 4=特大单(>100万)
  - 配平: 四档 buy+sell 金额/数量之和 = 当日成交额/量（bit 级，浮点误差 e-16）
  - 起点: 实际 2009 年起有数（2005-2008 无）；官方数据范围 2012-07-09 至今，
    2009~2012.7 为口径外回填段，使用需谨慎。覆盖率: 缺行≈停牌（当日无成交），
    有成交但缺数的仅 ~4 只/日 → 面板在个股存续期内缺口填 0（无成交=零资金流）
  - dquant 端 cache_on_fetch=False（不落本地缓存），每次调用逐日查远端 MySQL，
    ~19s/月，全量约 1 小时 → 必须本地落盘冻结

运行: 直接 python data_fetching/capital_flow_jy.py（参数在下方参数区手动修改）
"""
from __future__ import annotations

import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import dquant  # noqa: F401  (确保 fork 子进程能继承)
from dquant.data import get_capital_flow

from config import CAPITAL_FLOW_JY_DIR, CAPITAL_FLOW_JY_PANEL_DIR
from data_fetching.dquant_source import latest_trading_date

# ==================== 参数区（手动修改） ====================
FULL_START = "2009-01-01"   # jy 数据起点（2005-2008 无数据）
STAGES = ("A", "B", "C")    # 要执行的阶段
FETCH_WORKERS = 4           # Stage A 并行进程数（远端 MySQL，克制并行）
WRITE_WORKERS = 64          # Stage B 写盘并行进程数
# ==========================================================

PER_MONTH_DIR = CAPITAL_FLOW_JY_DIR / "per-month"
PER_STOCK_DIR = CAPITAL_FLOW_JY_DIR / "per-stock"

# value_range → 档位前缀（实测映射，见模块 docstring）
TIER_MAP = {1: "sb", 2: "md", 3: "lb", 4: "xl"}
RAW_FIELDS = ["buy_value", "sell_value", "buy_volume", "sell_volume"]
# 逐股宽表列序: sb_buy_value, sb_sell_value, ..., xl_sell_volume（4档×4字段=16列）
STOCK_COLS = [f"{t}_{f}" for t in ("sb", "md", "lb", "xl") for f in RAW_FIELDS]


# ==================== Stage A: 按月拉取 → per-month ====================

def _month_list(start: str, end: str) -> list[str]:
    """[start, end] 覆盖的月份列表（YYYY-MM）。"""
    return [str(p) for p in pd.period_range(start[:7], end[:7], freq="M")]


def _fetch_one_month(month: str) -> dict:
    """fork 子进程入口: 拉单月全市场 jy 资金流 → per-month parquet（原子写）。"""
    t0 = time.time()
    out_path = PER_MONTH_DIR / f"{month}.parquet"
    r = {"month": month, "status": "error", "rows": 0, "elapsed": 0.0, "error": ""}
    try:
        m_start = f"{month}-01"
        m_end = str(pd.Period(month, freq="M").end_time.date())
        df = get_capital_flow(None, m_start, m_end, source="jy")  # None=全市场，防子集缺数
        if df is None or len(df) == 0:
            r["status"] = "empty"
        else:
            tmp = out_path.with_suffix(".parquet.tmp")
            df.to_parquet(tmp, index=False)
            os.replace(tmp, out_path)
            r["status"], r["rows"] = "ok", len(df)
    except Exception as e:
        r["error"] = f"{type(e).__name__}: {e}"
    r["elapsed"] = time.time() - t0
    return r


def run_stage_a(start: str, end: str, workers: int = FETCH_WORKERS) -> None:
    """Stage A: 按月并行拉取。已有月份跳过；最后一个已有月份重拉（可能不完整）。"""
    PER_MONTH_DIR.mkdir(parents=True, exist_ok=True)
    months = _month_list(start, end)

    existing = sorted(f.stem for f in PER_MONTH_DIR.glob("*.parquet"))
    refetch_last = existing[-1] if existing else None  # 增量: 尾月重拉覆盖
    todo = [m for m in months if m not in existing or m == refetch_last]
    logger.info(f"[Stage A] 月份 {months[0]}~{months[-1]} 共 {len(months)}，待拉 {len(todo)}（{workers} 进程）")
    if not todo:
        logger.success("[Stage A] 已最新，无需拉取")
        return

    t0 = time.time()
    ok = empty = err = 0
    done = 0
    mp_ctx = multiprocessing.get_context("fork")
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp_ctx) as ex:
        futs = {ex.submit(_fetch_one_month, m): m for m in todo}
        for fut in as_completed(futs):
            r = fut.result()
            if r["status"] == "ok":
                ok += 1
            elif r["status"] == "empty":
                empty += 1
            else:
                err += 1
                logger.warning(f"{r['month']} 失败: {r['error']}")
            done += 1
            if done % 10 == 0 or done == len(todo):
                el = time.time() - t0
                eta = (len(todo) - done) / (done / el) / 60 if done else 0
                logger.info(f"[A] {done}/{len(todo)} ok={ok} empty={empty} err={err} ETA={eta:.1f}min")
    logger.success(f"[Stage A] 完成: ok={ok} empty={empty} err={err} 耗时={time.time()-t0:.0f}s")


# ==================== 公共: 读全部 per-month 长表 ====================

def _load_all_months() -> pd.DataFrame:
    files = sorted(PER_MONTH_DIR.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"{PER_MONTH_DIR} 无 per-month 文件，先跑 Stage A")
    logger.info(f"读 {len(files)} 个 per-month parquet...")
    t0 = time.time()
    long_df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    long_df["trade_date"] = pd.to_datetime(long_df["trade_date"])
    logger.info(f"合并长表 {len(long_df):,} 行, {long_df['order_book_id'].nunique():,} 股, "
                f"耗时 {time.time()-t0:.0f}s")
    return long_df


# ==================== Stage B: per-month → per-stock 宽表 ====================

def _write_one_stock(args) -> tuple[str, str]:
    """fork 子进程入口: 单股长表 → 16 列宽表（value_range 展开），原子写。"""
    code, g = args
    try:
        wide = g.pivot(index="trade_date", columns="value_range", values=RAW_FIELDS)
        wide.columns = [f"{TIER_MAP[vr]}_{f}" for f, vr in wide.columns]
        wide = wide.reindex(columns=STOCK_COLS).sort_index().astype(np.float64)
        wide.index.name = "date"
        out_path = PER_STOCK_DIR / f"{code}.parquet"
        tmp = out_path.with_suffix(".parquet.tmp")
        wide.to_parquet(tmp)
        os.replace(tmp, out_path)
        return code, "ok"
    except Exception as e:
        return code, f"error: {type(e).__name__}: {e}"


def run_stage_b(long_df: pd.DataFrame, workers: int = WRITE_WORKERS) -> None:
    """Stage B: 全量重建逐股宽表（源真相在 per-month，重建语义最干净）。"""
    PER_STOCK_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("[Stage B] 按股切分...")
    t0 = time.time()
    groups = [(code, g) for code, g in long_df.groupby("order_book_id")]

    ok = err = 0
    done = 0
    mp_ctx = multiprocessing.get_context("fork")
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp_ctx) as ex:
        futs = {ex.submit(_write_one_stock, g): g[0] for g in groups}
        for fut in as_completed(futs):
            code, status = fut.result()
            if status == "ok":
                ok += 1
            else:
                err += 1
                logger.warning(f"{code} 写盘失败: {status}")
            done += 1
            if done % 1000 == 0 or done == len(groups):
                logger.info(f"[B] {done}/{len(groups)} ok={ok} err={err}")
    logger.success(f"[Stage B] 完成: ok={ok} err={err} 耗时={time.time()-t0:.0f}s")


# ==================== Stage C: per-month → 消费面板 ====================

def _lifespan_fill_zero(panel: pd.DataFrame) -> pd.DataFrame:
    """个股存续期内（首末有效值之间）缺口填 0：无行=停牌=零资金流；存续期外保持 NaN。"""
    alive = panel.ffill().notna() & panel.bfill().notna()
    return panel.where(~(alive & panel.isna()), 0.0)


def run_stage_c(long_df: pd.DataFrame) -> None:
    """Stage C: 大单(lb=range3)/小单(sb=range1) × 净流入/买卖总额 4 张 T×N 面板。"""
    CAPITAL_FLOW_JY_PANEL_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    # 对齐全交易日历（个别日期个别股缺行 → pivot 自然 NaN → 存续期内填 0）
    for tier, vr in (("lb", 3), ("sb", 1)):
        sub = long_df.loc[long_df["value_range"] == vr, ["trade_date", "order_book_id", "buy_value", "sell_value"]]
        for kind, series in (
            ("net", sub["buy_value"] - sub["sell_value"]),
            ("gross", sub["buy_value"] + sub["sell_value"]),
        ):
            panel = (
                sub[["trade_date", "order_book_id"]]
                .assign(v=series)
                .pivot(index="trade_date", columns="order_book_id", values="v")
                .sort_index()
                .pipe(_lifespan_fill_zero)
                .astype(np.float64)
            )
            panel.index.name = "date"
            out = CAPITAL_FLOW_JY_PANEL_DIR / f"{tier}_{kind}_value.parquet"
            tmp = out.with_suffix(".parquet.tmp")
            panel.to_parquet(tmp)
            os.replace(tmp, out)
            logger.info(f"[C] {out.name}: shape={panel.shape}, 非空={panel.notna().values.sum():,}")
    logger.success(f"[Stage C] 完成 耗时={time.time()-t0:.0f}s")


# ==================== 主入口 ====================

def main():
    end = latest_trading_date()
    logger.info(f"输出: {CAPITAL_FLOW_JY_DIR} | 区间 {FULL_START} ~ {end} | 阶段 {STAGES}")

    if "A" in STAGES:
        run_stage_a(FULL_START, end)

    if "B" in STAGES or "C" in STAGES:
        long_df = _load_all_months()
        if "B" in STAGES:
            run_stage_b(long_df)
        if "C" in STAGES:
            run_stage_c(long_df)

    logger.success("✅ capital_flow_jy 完成")


if __name__ == "__main__":
    main()
