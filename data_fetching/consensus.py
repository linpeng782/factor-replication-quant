"""
卖方分析师一致预期数据获取 + 缓存
============================================================
两类原始数据（均来自米筐 rqdatac）：

1. comp_indicators —— 每交易日一行的「现成一致预期」面板
   get_consensus_comp_indicators(report_range=3)：不考虑补录入(无未来补填) + PIT 稳定，
   返回 comp_con_net_profit_t1/t2/t3（= report_year_t 的 +1/+2/+3 年一致预期净利润，元）。
   用于：pe_fy1_new / reg_pe_fy1 等「滚动一致预期」类因子。

2. consensus_reports —— 每份研报一行的「分析师明细」
   get_consensus_indicator(fiscal_year=Y, date_rule)：返回 institute/author/net_profit_t,_t1,_t2 等。
   用于：adj_pct / up_ratio / cnts / cyq_earningest 等需要逐报告状态的因子。

缓存落 <DATA_ROOT>/market-data/consensus/ ；再生成本高(全市场分钟级)，故 parquet 持久化。
"""
from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

import pandas as pd
from loguru import logger

import core.config as config

CONSENSUS_DIR = config._MKT / "consensus"

COMP_NP_FIELDS = [
    "comp_con_net_profit_t1",
    "comp_con_net_profit_t2",
    "comp_con_net_profit_t3",
]


def _init_rq():
    import rqdatac as rq
    try:
        rq.init()
    except Exception as e:
        logger.warning(f"rqdatac.init(): {e}")
    return rq


def all_cs_ids(rq) -> List[str]:
    return rq.all_instruments(type="CS")["order_book_id"].tolist()


# ───────────────────────── comp_indicators ─────────────────────────

def fetch_comp_indicators(
    order_book_ids: List[str],
    start_date: str,
    end_date: str,
    report_range: int = 3,
    batch_size: int = 800,
    workers: int = 6,
    rq=None,
) -> pd.DataFrame:
    """全市场 comp_indicators 拉取（批量+多线程）。返回 long:
    [order_book_id, date, report_year_t, comp_con_net_profit_t1/t2/t3]。
    """
    rq = rq or _init_rq()
    batches = [order_book_ids[i:i + batch_size] for i in range(0, len(order_book_ids), batch_size)]

    def _one(batch):
        df = rq.get_consensus_comp_indicators(
            batch, start_date=start_date, end_date=end_date, report_range=report_range,
        )
        if df is None or len(df) == 0:
            return None
        d = df.reset_index()
        keep = ["order_book_id", "date", "report_year_t"] + COMP_NP_FIELDS
        return d[[c for c in keep if c in d.columns]].copy()

    out = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=min(workers, len(batches))) as ex:
        futs = {ex.submit(_one, b): i for i, b in enumerate(batches)}
        for f in as_completed(futs):
            r = f.result()
            if r is not None:
                out.append(r)
            logger.info(f"[comp] batch {futs[f]+1}/{len(batches)} done ({time.time()-t0:.0f}s)")
    if not out:
        raise RuntimeError("comp_indicators 全空")
    res = pd.concat(out, ignore_index=True)
    res["date"] = pd.to_datetime(res["date"])
    # report_year_t 可能为空（无年报基准）→ nullable Int，下游计算时 dropna
    res["report_year_t"] = pd.to_numeric(res["report_year_t"], errors="coerce").astype("Int64")
    return res.sort_values(["order_book_id", "date"]).reset_index(drop=True)


def load_or_fetch_comp_indicators(
    start_date: str = "20100101",
    end_date: str = "20260527",
    report_range: int = 3,
    refresh: bool = False,
) -> pd.DataFrame:
    CONSENSUS_DIR.mkdir(parents=True, exist_ok=True)
    cache = CONSENSUS_DIR / f"comp_indicators_r{report_range}.parquet"
    if cache.exists() and not refresh:
        logger.info(f"📂 comp_indicators 缓存命中: {cache}")
        return pd.read_parquet(cache)
    rq = _init_rq()
    ids = all_cs_ids(rq)
    logger.info(f"拉取 comp_indicators: {len(ids)} 股 {start_date}~{end_date} r={report_range}")
    df = fetch_comp_indicators(ids, start_date, end_date, report_range=report_range, rq=rq)
    df.to_parquet(cache)
    logger.info(f"💾 落缓存 {cache} shape={df.shape}")
    return df


# ───────────────────────── consensus_reports (明细) ─────────────────────────

REPORT_KEEP = [
    "order_book_id", "date", "institute", "author", "report_title",
    "report_main_id", "net_profit_t", "net_profit_t1", "net_profit_t2",
    "grade_coef", "targ_price",
]


def fetch_consensus_reports_year(
    order_book_ids: List[str],
    fiscal_year: int,
    start_date: str,
    end_date: str,
    date_rule: str = "rice_create_tm",
    batch_size: int = 400,
    workers: int = 6,
    rq=None,
) -> pd.DataFrame:
    """单 fiscal_year 的分析师明细报告（批量+多线程）。
    date_rule='rice_create_tm' → 用米筐入库时间做 PIT（避免研报发布日的前视）。
    """
    rq = rq or _init_rq()
    batches = [order_book_ids[i:i + batch_size] for i in range(0, len(order_book_ids), batch_size)]

    def _one(batch):
        try:
            df = rq.get_consensus_indicator(
                batch, fiscal_year=str(fiscal_year),
                start_date=start_date, end_date=end_date, date_rule=date_rule,
            )
        except Exception as e:
            logger.warning(f"[rpt fy{fiscal_year}] batch err: {e}")
            return None
        if df is None or len(df) == 0:
            return None
        d = df.reset_index()
        d["fiscal_year"] = fiscal_year
        return d[[c for c in REPORT_KEEP if c in d.columns] + ["fiscal_year"]].copy()

    out = []
    with ThreadPoolExecutor(max_workers=min(workers, len(batches))) as ex:
        for r in ex.map(_one, batches):
            if r is not None:
                out.append(r)
    if not out:
        return pd.DataFrame(columns=REPORT_KEEP + ["fiscal_year"])
    res = pd.concat(out, ignore_index=True)
    res["date"] = pd.to_datetime(res["date"])
    return res


def load_or_fetch_consensus_reports(
    fiscal_years: List[int],
    date_rule: str = "rice_create_tm",
    refresh: bool = False,
) -> pd.DataFrame:
    """多 fiscal_year 明细：每年研报在 [Y-2, Y] 期间发布，逐年拉取并缓存合并。"""
    CONSENSUS_DIR.mkdir(parents=True, exist_ok=True)
    rq = _init_rq()
    ids = all_cs_ids(rq)
    frames = []
    for fy in fiscal_years:
        cache = CONSENSUS_DIR / f"reports_fy{fy}_{date_rule}.parquet"
        if cache.exists() and not refresh:
            logger.info(f"📂 reports fy{fy} 缓存命中")
            frames.append(pd.read_parquet(cache))
            continue
        sd, ed = f"{fy-2}0101", f"{fy+1}0630"
        logger.info(f"拉取 reports fy{fy} {sd}~{ed} ({len(ids)} 股)")
        df = fetch_consensus_reports_year(ids, fy, sd, ed, date_rule=date_rule, rq=rq)
        df.to_parquet(cache)
        logger.info(f"💾 reports fy{fy} shape={df.shape}")
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--what", choices=["comp", "reports"], default="comp")
    p.add_argument("--start", default="20100101")
    p.add_argument("--end", default="20260527")
    p.add_argument("--fy", default="")
    p.add_argument("--refresh", action="store_true")
    a = p.parse_args()
    if a.what == "comp":
        df = load_or_fetch_comp_indicators(a.start, a.end, refresh=a.refresh)
    else:
        fys = [int(x) for x in a.fy.split(",")] if a.fy else list(range(2014, 2027))
        df = load_or_fetch_consensus_reports(fys, refresh=a.refresh)
    print(df.shape)
    print(df.head())
