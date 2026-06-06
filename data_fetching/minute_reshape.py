"""
一次性迁移：旧 1m_post（单股大文件，烤死后复权）→ 新 minute/raw（全股按日分片，不复权）
============================================================
机制（已 V1 验证 vs rqdatac 真原始 maxRel~6e-8）：
  原始价 = 后复权价 ÷ ffill(ex_cum_factor)   # 仅 4 个价格列；量/额原样搬运
输出：<raw_dir>/<YYYY-MM-DD>.parquet（含 order_book_id 列，按日一文件）

内存有界（流式两段）：
  ① 按股票分组（每组 chunk 只）→ ÷cf 反推 → pyarrow 按 date 分区【追加写】到临时 parts/（不重写已有）
  ② compaction：逐日把该日多个 part 合并成单个 <date>.parquet（内存只占一日）
峰值内存 ≈ 一组（chunk≈60 只 → ~4GB）。

用法：
  python minute_reshape.py --limit 10 --chunk 5 --raw-dir /tmp/raw_test   # 测试
  python minute_reshape.py --chunk 60                                      # 全量(本地/服务器)
"""
from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger

PRICE = ["open", "high", "low", "close"]
COLS = ["order_book_id", "datetime"] + PRICE + ["volume", "total_turnover"]


def _root() -> Path:
    return Path(os.environ.get("FACTOR_REPL_DATA_ROOT", "/nfs/ofs-prediction/peterzhenglinpeng"))


def _to_raw(post: pd.DataFrame, stock: str, ex_dir: Path) -> pd.DataFrame:
    """旧 post 单股 → 原始：价格 ÷ ffill(cum_factor)，量/额不动；补 order_book_id。"""
    post = post.copy()
    post["datetime"] = pd.to_datetime(post["datetime"])
    days = post["datetime"].dt.normalize()
    exp = ex_dir / f"{stock}.parquet"
    if exp.exists():
        ex = pd.read_parquet(exp, columns=["ex_cum_factor"]); ex.index = pd.to_datetime(ex.index)
        cf = ex["ex_cum_factor"].sort_index().reindex(days.values, method="ffill").fillna(1.0).to_numpy()
    else:
        cf = np.ones(len(post))
    for c in PRICE:
        post[c] = (post[c].to_numpy() / cf).astype("float32")
    post["order_book_id"] = stock
    return post[COLS]


def reshape(post_dir: Path, ex_dir: Path, raw_dir: Path,
            limit: int | None = None, chunk: int = 60) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    parts_root = raw_dir.parent / (raw_dir.name + "_parts")
    if parts_root.exists():
        shutil.rmtree(parts_root)
    files = sorted(post_dir.glob("*.parquet"))
    if limit:
        files = files[:limit]
    logger.info(f"重塑 {len(files)} 只 | chunk={chunk} | 临时parts={parts_root} | 终输出={raw_dir}")

    # ① 流式分组 → 分区追加写
    for gi in range(0, len(files), chunk):
        group = files[gi:gi + chunk]
        frames = [_to_raw(pd.read_parquet(f), f.stem, ex_dir) for f in group]
        g = pd.concat(frames, ignore_index=True); del frames
        g["date"] = g["datetime"].dt.strftime("%Y-%m-%d")
        pq.write_to_dataset(pa.Table.from_pandas(g, preserve_index=False),
                            root_path=str(parts_root), partition_cols=["date"])
        logger.info(f"  组 {gi//chunk+1}/{(len(files)+chunk-1)//chunk}: {len(group)} 只 → 分区写 {len(g):,} 行")
        del g

    # ② compaction：逐日合并 parts → 单文件
    date_dirs = sorted(parts_root.glob("date=*"))
    logger.info(f"compaction: {len(date_dirs)} 个交易日")
    for i, dd in enumerate(date_dirs, 1):
        d = dd.name.split("=", 1)[1]
        m = pd.read_parquet(dd)[COLS]
        m = m.drop_duplicates(["order_book_id", "datetime"], keep="last") \
             .sort_values(["order_book_id", "datetime"]).reset_index(drop=True)
        tmp = tempfile.mktemp(suffix=".parquet", dir=str(raw_dir))
        m.to_parquet(tmp); os.replace(tmp, raw_dir / f"{d}.parquet")
        if i % 500 == 0:
            logger.info(f"  compaction {i}/{len(date_dirs)}")
    shutil.rmtree(parts_root)
    logger.success(f"重塑完成：{len(date_dirs)} 个日文件 → {raw_dir}")


def main():
    ap = argparse.ArgumentParser(description="旧1m_post → 新minute/raw（按日分片，内存有界流式）")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=60)
    ap.add_argument("--raw-dir", type=Path, default=None, help="输出目录(默认数据根 minute/raw)")
    a = ap.parse_args()
    root = _root()
    reshape(post_dir=root / "market-data" / "minute" / "stock_data_1m_post",
            ex_dir=root / "market-data" / "daily" / "stock-ex-factors",
            raw_dir=a.raw_dir or (root / "market-data" / "minute" / "raw"),
            limit=a.limit, chunk=a.chunk)


if __name__ == "__main__":
    main()
