"""
批量 L3：同一 superset「读一次、算多个因子」——消除 N× 冗余 superset 读
============================================================
背景：逐个 run.py 时，同一 superset(如 prv_v3 的 5505 个 per-stock 文件)会被**每个因子各读一遍**
（23 个 paper_27 因子 → ~23×11万次文件读），慢在冗余 I/O 而非计算。

本脚本：扫 spec 按 superset 分组 → **每组并行读一次** superset 进内存全表 → 串行对每个因子
切增量窗口→跑 L3 算子→pivot→incremental_append。计算口径**逐字节复刻** run.py 路径
（op_minute_aggregate consume 切片 + 同样的增量窗口 + 同一个 incremental_append 内核），
故产出与逐个 run.py **bit 一致**，只是 I/O 从 N× 降到 1×。

适用：spec 的 L3 是「minute 聚合 + 纯算子(rolling/compute/...)」且 incremental_safe（有界尾窗）。
不安全因子(filter→rolling / change_on，见 spec_resolver.incremental_safe)本脚本跳过，仍走 run.py --rebuild。

用法：
  python pipeline/refresh_factors_batch.py --cache-key prv_v3 --end-date 20260612
  python pipeline/refresh_factors_batch.py --factor-glob 'sources/kysec/paper_27_microstructure/specs/*'
"""
from __future__ import annotations

import argparse
import glob
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import yaml
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import core.yolo_engine  # noqa: F401  触发 reducer 注册
from core.config import RAW_FACTOR_BASE
from core.operators import Context, OpRegistry
from core.operators.minute_engine import REDUCER_BY_ACTION, MinuteAggregateEngine
from core.spec_resolver import (
    factor_name_from_arg,
    incremental_safe,
    max_warmup_window,
    resolve_namespace_safe,
)
from core.yolo_engine import incremental_append

BUFFER = 10              # 与 run.py INCREMENTAL_BUFFER 一致
N_READ_THREADS = 32


def _parse_spec(path: str) -> dict | None:
    """解析一个 spec → 批量所需元信息；非 minute-聚合 或 不可安全增量 → 返回 None(交回 run.py)。"""
    spec = yaml.safe_load(open(path, encoding="utf-8"))
    steps = spec.get("calculation_steps") or []
    agg = next((s for s in steps if s.get("action") in REDUCER_BY_ACTION), None)
    if agg is None:
        return None
    if not incremental_safe(spec):
        return None
    reducer = REDUCER_BY_ACTION[agg["action"]].from_step(agg)
    p = Path(path)
    qp = "/".join(p.parts[p.parts.index("sources") + 1:p.parts.index("specs")] + (p.parent.name,))
    return {
        "qp": qp,
        "factor_name": p.parent.name,
        "factor_column": spec["factor"]["column"],
        "features": list(agg["features"]),
        "l3_steps": [s for s in steps if s is not agg and s.get("action") != "fetch"],
        "cache_dir": str(reducer.cache_dir()),
        "w2": max_warmup_window(spec),
        "panel": RAW_FACTOR_BASE / resolve_namespace_safe(qp) / f"{factor_name_from_arg(qp)}.parquet",
    }


def _read_superset_once(cache_dir: Path, win_start: pd.Timestamp) -> pd.DataFrame:
    """并行读该 superset 全部 per-stock 文件(切到 win_start 之后) → 一张长表(含所有 superset 列)。"""
    files = sorted(cache_dir.glob("*.parquet"))

    def _one(p: Path):
        d = pd.read_parquet(p)
        d["date"] = pd.to_datetime(d["date"])
        return d[d["date"] >= win_start]

    parts = []
    with ThreadPoolExecutor(max_workers=N_READ_THREADS) as ex:
        for d in ex.map(_one, files):
            if len(d):
                parts.append(d)
    long = pd.concat(parts, ignore_index=True)
    return long


def _compute_one(long_all: pd.DataFrame, meta: dict, end: pd.Timestamp) -> str:
    """复刻 run.py 增量路径：切 [last-(W2+buf), end] 窗口 → 投影 features → 跑 L3 → pivot → append。"""
    panel = meta["panel"]
    last = None
    if panel.exists():
        idx = pd.to_datetime(pd.read_parquet(panel, columns=[]).index)
        last = idx.max() if len(idx) else None
    if last is not None:
        lookback = math.ceil((meta["w2"] + BUFFER) * 1.6)          # 与 run.py 一致(日历日)
        start = last - pd.Timedelta(days=lookback)
    else:
        start = long_all["date"].min()                            # 首建：全量
    # op_minute_aggregate consume 复刻：切窗口 + 投影 [obid,date]+features + 排序
    cols = ["order_book_id", "date"] + meta["features"]
    data = long_all.loc[
        (long_all["date"] >= start) & (long_all["date"] <= end), cols
    ].sort_values(["order_book_id", "date"]).reset_index(drop=True)
    if data.empty:
        return "empty"
    ctx = Context(factor_name=meta["factor_name"])
    ctx.set_df("data", data)
    for step in meta["l3_steps"]:
        OpRegistry.get(step["action"])(ctx, step, None)
    out = ctx.get_df("data")
    wide = out.pivot(index="date", columns="order_book_id", values=meta["factor_column"])
    wide.index = pd.to_datetime(wide.index)
    incremental_append(panel, wide, last)                          # 同一生产内核(>last append, dedup, 原子写)
    n_new = int((wide.index > last).sum()) if last is not None else len(wide)
    return f"ok(+{n_new}日)"


def main():
    ap = argparse.ArgumentParser(description="批量 L3：同一 superset 读一次算多个因子")
    ap.add_argument("--cache-key", default=None, help="只跑用该 cache_key 的因子(如 prv_v3)")
    ap.add_argument("--factor-glob", default="sources/*/*/specs/*", help="spec 范围 glob")
    ap.add_argument("--end-date", default=None, help="因子面板结束日 YYYYMMDD；默认=superset 末日")
    a = ap.parse_args()
    t0 = time.time()

    # 1. 扫 + 分组（按 superset cache_dir）
    groups: dict[str, list] = {}
    skipped = []
    for path in sorted(glob.glob(a.factor_glob + "/spec.yaml")):
        m = _parse_spec(path)
        if m is None:
            skipped.append(path)
            continue
        if a.cache_key and Path(m["cache_dir"]).name.split("__")[0] != a.cache_key:
            continue
        groups.setdefault(m["cache_dir"], []).append(m)
    if not groups:
        logger.warning("无匹配的可批量因子（检查 --cache-key / --factor-glob；不安全因子走 run.py --rebuild）")
        return
    logger.info(f"分组: {[(Path(k).name, len(v)) for k,v in groups.items()]} | 跳过(非批量){len(skipped)}")

    total_ok = 0
    for cache_dir, metas in groups.items():
        cdir = Path(cache_dir)
        # 前置：superset 应已最新（daily_update step6 / 先跑 refresh_supersets.py）；本脚本只做 L3。
        # 读窗口起点 = 各因子 max(last-(W2+buf)) 的最小值（保守覆盖最早新日的 warmup）
        starts = []
        for m in metas:
            if m["panel"].exists():
                idx = pd.to_datetime(pd.read_parquet(m["panel"], columns=[]).index)
                if len(idx):
                    starts.append(idx.max() - pd.Timedelta(days=math.ceil((m["w2"] + BUFFER) * 1.6)))
        win_start = min(starts) if starts else pd.Timestamp("2005-01-01")
        logger.info(f"▶ {cdir.name}: {len(metas)} 因子 | 读窗口起点 {win_start.date()} | 并行读 superset…")
        tr = time.time()
        long_all = _read_superset_once(cdir, win_start)
        end = pd.Timestamp(a.end_date) if a.end_date else long_all["date"].max()
        logger.info(f"  superset 读毕 {time.time()-tr:.0f}s | 长表 {len(long_all):,} 行 × {long_all.shape[1]} 列 | end={end.date()}")
        # 4. 串行对每个因子算+append（计算很快；瓶颈已在读那一步消除）
        tc = time.time()
        for m in metas:
            r = _compute_one(long_all, m, end)
            total_ok += r.startswith("ok")
        logger.info(f"  {len(metas)} 因子算+append 完成 {time.time()-tc:.0f}s")
        del long_all

    logger.success(f"✅ 批量 L3 完成：{total_ok} 因子更新 | 跳过(非批量){len(skipped)} | 总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
