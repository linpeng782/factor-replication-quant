"""
Alpha158 全量生产：读后复权面板 → 算 158 因子 → 落 factors/raw/alpha158/<group>/
============================================================
内存安全：alpha158 所有算子逐列(逐股)独立(无横截面算子)，故按股票分块计算 = 与整体等价。
  外层遍历因子、内层按股票块算再拼列、直接写最终面板 → 峰值内存 ~5GB（16GB Mac 安全），无 temp。

group ∈ {kline, price, rolling, volume}（见 core.producers.alpha158.groups）。落盘 float32。

用法：python alpha158/build.py            # 增量：只算缺的
      python alpha158/build.py --full     # 全量重算覆盖
      python alpha158/build.py --only KMID MA20
      python alpha158/build.py --chunk 1200   # 调股票分块大小（内存↔速度）
"""
from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import pandas as pd
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core import config
from core.producers.alpha158 import Alpha158Panel, adjusted_panels
from core.producers.alpha158.groups import factor_group

WINDOWS = [5, 10, 20, 30, 60]
CHUNK_COLS = 1200                                          # 每块股票数（控内存）
ALPHA158_RAW_BASE = config.ALPHA158_RAW_BASE                # factors/raw/alpha158[-dquant]/<group>/


def _existing_names() -> set[str]:
    return {p.stem for p in ALPHA158_RAW_BASE.glob("*/*.parquet")}


def build(full: bool = False, only: list[str] | None = None,
          start: str | None = None, end: str | None = None, chunk: int = CHUNK_COLS) -> None:
    all_names = Alpha158Panel.list_factor_names(WINDOWS)          # 158 个
    target = only if only else all_names
    if not full and not only:
        have = _existing_names()
        target = [n for n in all_names if n not in have]
    if not target:
        logger.success("factors/raw/alpha158 下 158 因子已全（--full 强制重算）"); return

    logger.info(f"Alpha158 生产 | {'全量' if full else ('指定' if only else '增量')} "
                f"| 目标 {len(target)} 个 | 区间 {start or '全史'}~{end or '今'} | 股票块={chunk}")

    t0 = time.time()
    logger.info("[1/3] 加载后复权面板（open/high/low/close/volume/vwap）...")
    panels = adjusted_panels.load_adjusted_panels(
        start=start, end=end, fields=("open", "high", "low", "close", "volume", "vwap"),
    )
    cols = list(panels["close"].columns)
    idx = panels["close"].index
    logger.info(f"  shape={panels['close'].shape} {idx.min().date()}~{idx.max().date()} 耗时 {time.time()-t0:.1f}s")

    # 预切分成股票块（各持独立副本，释放整体引用）
    col_chunks = [cols[i:i + chunk] for i in range(0, len(cols), chunk)]
    chunk_panels = [{f: panels[f][c] for f in panels} for c in col_chunks]
    del panels; gc.collect()
    logger.info(f"[2/3] 按 {len(col_chunks)} 块股票逐因子计算 + 直写 float32 → {ALPHA158_RAW_BASE}/<group>/")

    t1 = time.time()
    for i, name in enumerate(target, 1):
        group = factor_group(name)
        parts = [Alpha158Panel(cp).compute_all(windows=WINDOWS, only_names=[name])[name]
                 for cp in chunk_panels]
        full_df = pd.concat(parts, axis=1).reindex(columns=cols).astype("float32")
        out_dir = ALPHA158_RAW_BASE / group
        out_dir.mkdir(parents=True, exist_ok=True)
        full_df.to_parquet(out_dir / f"{name}.parquet")
        del parts, full_df; gc.collect()
        if i % 20 == 0 or i == len(target):
            logger.info(f"  [{i}/{len(target)}] 已写 {name} ({group}) 累计 {time.time()-t1:.0f}s")
    logger.success(f"[3/3] 完成：{len(target)} 个因子 → {ALPHA158_RAW_BASE}/  总耗时 {time.time()-t0:.0f}s")


def main():
    ap = argparse.ArgumentParser(description="Alpha158 全量生产 → factors/raw/alpha158/<group>/")
    ap.add_argument("--full", action="store_true", help="全量重算覆盖（默认增量补缺）")
    ap.add_argument("--only", nargs="+", default=None, help="只算指定因子名")
    ap.add_argument("--start", default=None, help="加载起点 YYYY-MM-DD（默认全史）")
    ap.add_argument("--end", default=None, help="加载终点 YYYY-MM-DD（默认今）")
    ap.add_argument("--chunk", type=int, default=CHUNK_COLS, help=f"股票分块大小（默认{CHUNK_COLS}）")
    args = ap.parse_args()
    build(full=args.full, only=args.only, start=args.start, end=args.end, chunk=args.chunk)


if __name__ == "__main__":
    main()
