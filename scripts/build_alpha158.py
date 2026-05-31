"""
Alpha158 全量生产：读后复权面板 → 算 158 因子 → 落 factors/raw/alpha158/<group>/
============================================================
与 spec/yolo 路径并行；产出后用本仓标准 clean→neu→eval 评估（见 build_alpha158_eval 或
直接 evaluate_single_factor(name, df, namespace=f"alpha158/{group}")）。

group ∈ {kline, price, rolling, volume}（见 core.producers.alpha158.groups）。
落盘 float32，与现有 market-data/alpha158 口径一致。

用法：python scripts/build_alpha158.py            # 增量：只算 factors/raw/alpha158 下缺的
      python scripts/build_alpha158.py --full     # 全量重算覆盖
      python scripts/build_alpha158.py --only KMID MA20 ...   # 指定子集
"""
from __future__ import annotations

import argparse
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
ALPHA158_RAW_BASE = config.RAW_FACTOR_BASE / "alpha158"   # factors/raw/alpha158/<group>/


def _existing_names() -> set[str]:
    return {p.stem for p in ALPHA158_RAW_BASE.glob("*/*.parquet")}


def build(full: bool = False, only: list[str] | None = None,
          start: str | None = None, end: str | None = None) -> None:
    all_names = Alpha158Panel.list_factor_names(WINDOWS)          # 158 个
    target = only if only else all_names
    if not full and not only:
        have = _existing_names()
        target = [n for n in all_names if n not in have]
    if not target:
        logger.success("factors/raw/alpha158 下 158 因子已全，无需计算（--full 强制重算）")
        return

    logger.info(f"Alpha158 生产 | 模式={'全量' if full else ('指定' if only else '增量')} "
                f"| 目标 {len(target)} 个 | 区间 {start or '全史'}~{end or '今'}")

    t0 = time.time()
    logger.info("[1/3] 加载后复权面板（open/high/low/close/volume/vwap）...")
    panels = adjusted_panels.load_adjusted_panels(
        start=start, end=end, fields=("open", "high", "low", "close", "volume", "vwap"),
    )
    logger.info(f"  close shape={panels['close'].shape} 耗时 {time.time()-t0:.1f}s")

    logger.info("[2/3] 批量计算 ...")
    t1 = time.time()
    computed = Alpha158Panel(panels).compute_all(windows=WINDOWS, only_names=target)
    logger.info(f"  算完 {len(computed)} 个 耗时 {time.time()-t1:.1f}s")

    logger.info(f"[3/3] 写盘 → {ALPHA158_RAW_BASE}/<group>/ (float32)")
    t2 = time.time()
    for name in target:
        if name not in computed:
            logger.warning(f"  {name} 未算出，跳过"); continue
        group = factor_group(name)
        out_dir = ALPHA158_RAW_BASE / group
        out_dir.mkdir(parents=True, exist_ok=True)
        computed[name].astype("float32").to_parquet(out_dir / f"{name}.parquet")
    logger.success(f"完成：{len(target)} 个因子 写盘 {time.time()-t2:.1f}s | 总 {time.time()-t0:.1f}s")


def main():
    ap = argparse.ArgumentParser(description="Alpha158 全量生产 → factors/raw/alpha158/<group>/")
    ap.add_argument("--full", action="store_true", help="全量重算覆盖（默认增量补缺）")
    ap.add_argument("--only", nargs="+", default=None, help="只算指定因子名")
    ap.add_argument("--start", default=None, help="加载起点 YYYY-MM-DD（默认全史）")
    ap.add_argument("--end", default=None, help="加载终点 YYYY-MM-DD（默认今）")
    args = ap.parse_args()
    build(full=args.full, only=args.only, start=args.start, end=args.end)


if __name__ == "__main__":
    main()
