"""
cxl 基本面 L3 因子增量正确性验收：truncate-replay 真实数据对账
============================================================
三胞胎之一（孪生 smoke_minute_l3_truncate_replay / alpha158/smoke_replay.py）。
docs/cxl_fundamental_incremental_design.md §9。

思路：以**本地基本面 PIT 基础层**（market-data/fundamentals/<field>.parquet，冻结快照）为源——
全量跑 spec 的 L3 步骤（fetch 之后的 compute/rank/filter/regress/rolling）算 ground-truth 因子面板
→ 砍尾 K 天得旧面板（删 cut 时未上市的股）→ 从未截断 L1 逐日重放 → 逐格对账。

杜绝脱节：L3 计算走**真实算子**（经最小 Context + OpRegistry）；写盘走**生产内核**
core.yolo_engine.incremental_append（与 run.py 同一函数）。仅砍尾/重放是测试编排。

4 判据：①截因子产出非源 ②多天重放 ③显式断言窗口内 IPO 列 ④NaN-aware 活区比较。

用法:
  python scripts/smoke_cxl_l3_truncate_replay.py                               # 默认 roe_apoq_mrq
  python scripts/smoke_cxl_l3_truncate_replay.py --factor cxl/cross_section_regress/reg_pe_hist --k 25
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml as _yaml
from core.config import FUNDAMENTALS_DIR
from core.operators import Context, OpRegistry
from core.spec_resolver import incremental_safe, max_warmup_window, resolve_spec_path
from core.yolo_engine import incremental_append

DEFAULT_FACTOR = "cxl/roe_series/roe_apoq_mrq"
N_SAMPLE = 800
K = 20
BUFFER = 10
TOL_REL = 1e-6


def parse_spec(factor: str):
    """返回 (fetch_fields, l3_steps, factor_column, w2)。l3_steps = fetch 之后的步骤。"""
    spec = yaml.safe_load(open(resolve_spec_path(factor), encoding="utf-8"))
    steps = spec.get("calculation_steps") or []
    fetch_step = next((s for s in steps if s.get("action") == "fetch"), None)
    if fetch_step is None or fetch_step.get("api") != "get_factor":
        raise ValueError(f"{factor}: 非 get_factor 基本面因子")
    fields = fetch_step.get("fields") or list((fetch_step.get("output_columns") or {}).keys())
    # 重命名（L3/factor.column 引用重命名后的列）：支持 output_columns(dict) 与 output_column(单字段)
    rename = {k: v for k, v in (fetch_step.get("output_columns") or {}).items() if k != v}
    if fetch_step.get("output_column") and len(fields) == 1:
        rename[fields[0]] = fetch_step["output_column"]
    l3_steps = [s for s in steps if s.get("action") != "fetch"]
    return fields, rename, l3_steps, spec["factor"]["column"], max_warmup_window(spec)


def load_local_source(fields, rename, stocks: list[str]) -> pd.DataFrame:
    """读本地 fundamentals 面板（sample 股，全史）→ long [order_book_id, date, *fields(重命名后)]。"""
    universe = set(stocks)
    series = []
    for f in fields:
        p = FUNDAMENTALS_DIR / f"{f}.parquet"
        w = pd.read_parquet(p)
        w.index = pd.to_datetime(w.index)
        cols = [c for c in w.columns if c in universe]
        s = w[cols].stack(dropna=False)
        s.name = rename.get(f, f)
        series.append(s)
    long = pd.concat(series, axis=1).reset_index()
    long.columns = ["date", "order_book_id"] + [rename.get(f, f) for f in fields]
    long = long.dropna(subset=[rename.get(f, f) for f in fields], how="all")
    long["date"] = pd.to_datetime(long["date"])
    return long


def compute_l3(long: pd.DataFrame, l3_steps: list[dict], factor_column: str) -> pd.DataFrame:
    """真实算子跑 L3 步骤 → pivot 因子宽表 date×stock。"""
    ctx = Context(factor_name="__smoke__")
    ctx.set_df("data", long.sort_values(["order_book_id", "date"]).reset_index(drop=True))
    for step in l3_steps:
        OpRegistry.get(step["action"])(ctx, step, None)
    data = ctx.get_df("data")
    wide = data.pivot(index="date", columns="order_book_id", values=factor_column)
    return wide.sort_index().sort_index(axis=1)


def reconcile(replayed, truth, ipo_cols, onset) -> dict:
    d = replayed.index.intersection(truth.index)
    c = replayed.columns.intersection(truth.columns)
    a, b = replayed.loc[d, c], truth.loc[d, c]
    live = pd.DataFrame(False, index=d, columns=c)
    for s in c:
        o = onset.get(s)
        if o is not None:
            live.loc[live.index >= o, s] = True
    a, b = a.where(live), b.where(live)
    av = a.values.astype(float); bv = b.values.astype(float)
    # inf 感知：基本面因子常含 inf（如 /abs(基数=0)）。NaN+inf 模式分别逐格断言；
    # 数值 max_rel 只在双方有限处算（否则 inf 会污染 max → nan）。
    nan_match = bool((np.isnan(av) == np.isnan(bv)).all())
    inf_match = bool((np.isposinf(av) == np.isposinf(bv)).all()
                     and (np.isneginf(av) == np.isneginf(bv)).all())
    fin = np.isfinite(av) & np.isfinite(bv)
    abs_d = np.abs(av[fin] - bv[fin])
    max_abs = float(abs_d.max()) if fin.any() else np.nan
    max_rel = float((abs_d / (np.abs(bv[fin]) + 1e-12)).max()) if fin.any() else np.nan
    ipo_present = [s for s in ipo_cols if s in replayed.columns]
    ipo_ok = True
    for s in ipo_present:
        if s not in truth.columns:
            continue
        ra, rb = a[s], b[s]
        bb = ~(ra.isna() | rb.isna())
        if bb.any() and float((ra[bb] - rb[bb]).abs().max()) > 0:
            ipo_ok = False
        if not (ra.isna().values == rb.isna().values).all():
            ipo_ok = False
    val_ok = np.isnan(max_rel) or max_rel < TOL_REL
    return {"max_abs": max_abs, "max_rel": max_rel, "nan_match": nan_match,
            "inf_match": inf_match, "ipo_back": len(ipo_present), "ipo_ok": ipo_ok,
            "pass": val_ok and nan_match and inf_match and ipo_ok}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--factor", default=DEFAULT_FACTOR)
    ap.add_argument("--sample", type=int, default=N_SAMPLE)
    ap.add_argument("--k", type=int, default=K)
    args = ap.parse_args()
    t0 = time.time()

    if not incremental_safe(_yaml.safe_load(open(resolve_spec_path(args.factor), encoding="utf-8"))):
        logger.warning(
            f"⏭️  {args.factor} 含 filter→时序算子（warmup 无界）→ 走全量重算、非有界尾窗增量；"
            "truncate-replay 不适用（生产 run.py 会全量重算，确定性、历史不漂移）。跳过。"
        )
        return

    fields, rename, l3_steps, factor_column, w2 = parse_spec(args.factor)
    logger.info(f"[1/5] {args.factor}: fetch字段={fields} | L3步骤={[s['action'] for s in l3_steps]} | "
                f"factor_column={factor_column} | W2={w2}")

    logger.info("[2/5] 选样本 + 读本地 L1 源...")
    any_panel = FUNDAMENTALS_DIR / f"{fields[0]}.parquet"
    all_stocks = sorted(pd.read_parquet(any_panel).columns.tolist())
    rng = np.random.default_rng(42)
    n = min(args.sample, len(all_stocks))
    sample = sorted(rng.choice(all_stocks, n, replace=False).tolist())
    long = load_local_source(fields, rename, sample)
    logger.info(f"  样本 {len(sample)} 只 | long {len(long):,} 行")

    logger.info("[3/5] 全量算 ground truth（真实算子）...")
    truth = compute_l3(long, l3_steps, factor_column)
    dates = truth.index
    T = dates[-1]; cut = dates[-(args.k + 1)]
    logger.info(f"  日期 {dates[0].date()}~{T.date()} | cut={cut.date()} | 重放 {args.k} 天")

    logger.info("[4/5] 构造截断旧面板（删 cut 时未上市的股）...")
    first_valid = {s: truth[s].first_valid_index() for s in truth.columns}
    present = [s for s, fv in first_valid.items() if fv is not None and fv <= cut]
    ipo_in_window = [s for s, fv in first_valid.items() if fv is not None and fv > cut]
    logger.info(f"  cut 时在册 {len(present)} 只；窗口内 IPO {len(ipo_in_window)} 只: {ipo_in_window[:8]}")

    logger.info(f"[5/5] 逐日重放 {args.k} 天（源切尾窗 {w2+BUFFER} 交易日 → 真实算子 → 生产内核 append）...")
    with tempfile.TemporaryDirectory() as td:
        panel_path = Path(td) / "factor.parquet"
        truth.reindex(columns=present).loc[:cut].to_parquet(panel_path)
        cur_last = cut
        for ti in dates[dates > cut]:
            win_dates = dates[dates <= ti][-(w2 + BUFFER):]
            win_long = long[long["date"].isin(win_dates)]
            wide = compute_l3(win_long, l3_steps, factor_column)
            incremental_append(panel_path, wide, cur_last)
            cur_last = ti
        replayed = pd.read_parquet(panel_path)
        replayed.index = pd.to_datetime(replayed.index)

    r = reconcile(replayed, truth, ipo_in_window, first_valid)
    logger.info("=" * 60)
    logger.info(f"  max_abs={r['max_abs']:.2e} | max_rel={r['max_rel']:.2e} | "
                f"NaN模式一致={r['nan_match']} | inf模式一致={r['inf_match']} | "
                f"IPO列回来={r['ipo_back']}/{len(ipo_in_window)} ok={r['ipo_ok']}")
    if r["pass"]:
        logger.success(f"✅ cxl L3 truncate-replay 通过：{args.k} 天重放 == 全量重算（活区 rel<{TOL_REL:g}）"
                       f"、NaN 模式一致、新股列吻合 | 耗时 {time.time()-t0:.0f}s")
    else:
        logger.error(f"❌ 未通过：{r} | 耗时 {time.time()-t0:.0f}s")
        sys.exit(1)


if __name__ == "__main__":
    main()
