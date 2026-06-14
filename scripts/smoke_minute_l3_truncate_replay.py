"""
分钟 L3 因子增量正确性验收：truncate-replay 真实数据对账
============================================================
孪生于 scripts/smoke_alpha158_truncate_replay.py（同一套黄金标准），但验的是
**分钟侧 L3**（superset → spec rolling → 因子宽表 append）。

思路（docs/minute_incremental_design.md §13.1）：
  以**真实 per-stock superset 缓存**为源（append-only 冻结，是 L3 的稳定输入；
  pooled corr 等跨日耦合已在 L2 算完，L3 是纯单股 rolling → 与 alpha158 同构）。
  全量跑 spec 的 L3 步骤算出 ground-truth 因子面板 → 砍掉尾部 K 天得"昨天的旧面板"
  （并删去截断点尚未上市的股 = 当时不是列）→ 从【未截断的 superset】逐日重放 K 天
  → 重放结果 vs ground-truth 逐格对比。

杜绝脱节（对齐 alpha158 §9 红利）：
  · L3 计算口径 = **真实 rolling 算子**（core.operators.rolling.op_rolling，经最小 Context）
  · 写盘 = **生产内核** core.yolo_engine.incremental_append（与 run.py 同一函数）
  只有"砍尾/逐日重放"的编排是测试代码。

4 个防假阳性判据（同 alpha158）：
  ① 截断的是因子产出、不是源（superset 永远全）
  ② 多天重放（逐日 append，模拟连续日更）
  ③ 显式断言新股列（截断窗口内真实 IPO）回来且吻合
  ④ NaN-aware：max|Δ| 只在双方非 NaN 处算；另单独断言 NaN 模式逐格一致

通过 = 重叠(日期×股) 活区 max_rel<1e-6 且 NaN 模式一致 且 IPO 列吻合。

用法:
  python scripts/smoke_minute_l3_truncate_replay.py                       # 默认 pj_peak_minute_count, 600 股
  python scripts/smoke_minute_l3_truncate_replay.py --factor xxx --sample 400 --k 20
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

import core.yolo_engine  # noqa: F401  触发所有 reducer 注册
from core.operators import Context, OpRegistry
from core.operators.minute_engine import REDUCER_BY_ACTION
from core.spec_resolver import max_rolling_window, resolve_spec_path
from core.yolo_engine import incremental_append

DEFAULT_FACTOR = "pj_peak_minute_count"
N_SAMPLE = 600
K = 20                       # 截断/重放天数
BUFFER = 10                  # 回读窗口 = W2 + BUFFER 个交易日
TOL_REL = 1e-6


# ==================== spec 解析：定位 superset 源 + L3 步骤 ====================
def parse_spec(factor: str):
    """返回 (superset_dir, l3_steps, features, factor_column, w2)。

    superset_dir = spec 里 minute 聚合步骤（action ∈ REDUCER_BY_ACTION）对应的缓存目录；
    l3_steps     = 聚合步骤之后的所有 calculation_steps（= L3 计算，如 rolling）。
    """
    spec = yaml.safe_load(open(resolve_spec_path(factor), encoding="utf-8"))
    steps = spec.get("calculation_steps") or []
    agg_step = next((s for s in steps if s.get("action") in REDUCER_BY_ACTION), None)
    if agg_step is None:
        raise ValueError(f"{factor}: spec 无 minute 聚合步骤，非分钟 L3 因子")
    reducer = REDUCER_BY_ACTION[agg_step["action"]].from_step(agg_step)
    superset_dir = reducer.cache_dir()
    features = list(agg_step["features"])
    l3_steps = [s for s in steps if s.get("action") not in REDUCER_BY_ACTION]
    factor_column = spec["factor"]["column"]
    return superset_dir, l3_steps, features, factor_column, max_rolling_window(spec)


# ==================== 源读取：per-stock superset 长表 ====================
def load_superset_long(superset_dir: Path, stocks: list[str], features: list[str]) -> pd.DataFrame:
    cols = ["order_book_id", "date"] + features
    dfs = []
    for s in stocks:
        p = superset_dir / f"{s}.parquet"
        if not p.exists():
            continue
        d = pd.read_parquet(p, columns=cols)
        if len(d):
            dfs.append(d)
    long = pd.concat(dfs, ignore_index=True)
    long["date"] = pd.to_datetime(long["date"])
    return long


# ==================== L3 计算：真实 rolling 算子（= 生产口径）====================
def compute_l3(long: pd.DataFrame, l3_steps: list[dict], factor_column: str) -> pd.DataFrame:
    """把 superset 长表过一遍 spec 的 L3 步骤（真实算子）→ pivot 因子宽表 date×stock。"""
    ctx = Context(factor_name="__smoke__")
    ctx.set_df("data", long.sort_values(["order_book_id", "date"]).reset_index(drop=True))
    for step in l3_steps:
        OpRegistry.get(step["action"])(ctx, step, None)
    data = ctx.get_df("data")
    wide = data.pivot(index="date", columns="order_book_id", values=factor_column)
    return wide.sort_index().sort_index(axis=1)


# ==================== 对账（NaN-aware + 活区 + IPO 断言）====================
def reconcile(replayed: pd.DataFrame, truth: pd.DataFrame, ipo_cols: list[str], onset: dict) -> dict:
    d = replayed.index.intersection(truth.index)
    c = replayed.columns.intersection(truth.columns)
    a, b = replayed.loc[d, c], truth.loc[d, c]
    # 只比"活区"（个股上市后 date >= onset），上市前死区下游被 new_stock_mask 抹掉
    live = pd.DataFrame(False, index=d, columns=c)
    for s in c:
        o = onset.get(s)
        if o is not None:
            live.loc[live.index >= o, s] = True
    a, b = a.where(live), b.where(live)
    both = ~(a.isna() | b.isna())
    av = a.values[both.values]
    bv = b.values[both.values].astype(float)
    abs_d = np.abs(av - bv)
    max_abs = float(abs_d.max()) if both.values.any() else np.nan
    max_rel = float((abs_d / (np.abs(bv) + 1e-12)).max()) if both.values.any() else np.nan
    nan_match = bool((a.isna().values == b.isna().values).all())
    # ③ IPO 列断言
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
    return {
        "max_abs": max_abs, "max_rel": max_rel, "nan_match": nan_match,
        "ipo_back": len(ipo_present), "ipo_ok": ipo_ok,
        "pass": val_ok and nan_match and ipo_ok,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--factor", default=DEFAULT_FACTOR)
    ap.add_argument("--sample", type=int, default=N_SAMPLE)
    ap.add_argument("--k", type=int, default=K)
    args = ap.parse_args()
    t0 = time.time()

    superset_dir, l3_steps, features, factor_column, w2 = parse_spec(args.factor)
    logger.info(
        f"[1/5] {args.factor}: superset={superset_dir.name} | features={features} | "
        f"L3 步骤={[s['action'] for s in l3_steps]} | factor_column={factor_column} | W2={w2}"
    )
    if not superset_dir.exists():
        logger.error(f"superset 缓存不存在: {superset_dir}（先 refresh_supersets.py 建库）")
        sys.exit(1)

    logger.info("[2/5] 选样本股（superset 现存 + 随机）...")
    all_stocks = sorted(p.stem for p in superset_dir.glob("*.parquet"))
    rng = np.random.default_rng(42)
    n = min(args.sample, len(all_stocks))
    sample = sorted(rng.choice(all_stocks, n, replace=False).tolist())
    logger.info(f"  样本 {len(sample)} / {len(all_stocks)} 只")

    logger.info("[3/5] 读 superset 长表 → 全量算 ground truth（真实 rolling 算子）...")
    long = load_superset_long(superset_dir, sample, features)
    truth = compute_l3(long, l3_steps, factor_column)
    dates = truth.index
    T = dates[-1]
    cut = dates[-(args.k + 1)]
    logger.info(f"  日期 {dates[0].date()}~{T.date()} | 截断点 cut={cut.date()} | 重放 {args.k} 天")

    logger.info("[4/5] 构造截断旧面板（删 cut 时未上市的股）...")
    first_valid = {s: truth[s].first_valid_index() for s in truth.columns}
    present = [s for s, fv in first_valid.items() if fv is not None and fv <= cut]
    ipo_in_window = [s for s, fv in first_valid.items() if fv is not None and fv > cut]
    logger.info(f"  cut 时在册 {len(present)} 只；窗口内 IPO {len(ipo_in_window)} 只: {ipo_in_window[:10]}")

    logger.info(f"[5/5] 逐日重放 {args.k} 天（源切尾窗 {w2 + BUFFER} 交易日 → 真实算子 → 生产内核 append）...")
    with tempfile.TemporaryDirectory() as td:
        panel_path = Path(td) / "factor.parquet"
        truth.reindex(columns=present).loc[:cut].to_parquet(panel_path)
        cur_last = cut
        replay_dates = dates[dates > cut]
        for ti in replay_dates:
            win_dates = dates[dates <= ti][-(w2 + BUFFER):]
            win_long = long[long["date"].isin(win_dates)]
            wide = compute_l3(win_long, l3_steps, factor_column)
            incremental_append(panel_path, wide, cur_last)   # ★ 生产内核
            cur_last = ti
        replayed = pd.read_parquet(panel_path)
        replayed.index = pd.to_datetime(replayed.index)

    logger.info("=" * 60)
    r = reconcile(replayed, truth, ipo_in_window, first_valid)
    logger.info(
        f"  max_abs={r['max_abs']:.2e} | max_rel={r['max_rel']:.2e} | "
        f"NaN模式一致={r['nan_match']} | IPO列回来={r['ipo_back']}/{len(ipo_in_window)} ok={r['ipo_ok']}"
    )
    if r["pass"]:
        logger.success(
            f"✅ 分钟 L3 truncate-replay 通过：{args.k} 天重放 == 全量重算（活区 rel<{TOL_REL:g}）"
            f"、NaN 模式逐格一致、新股列吻合 | 耗时 {time.time()-t0:.0f}s"
        )
    else:
        logger.error(f"❌ 未通过：{r} | 耗时 {time.time()-t0:.0f}s")
        sys.exit(1)


if __name__ == "__main__":
    main()
