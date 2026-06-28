"""
alpha158 长表构建 + can_train 过滤
===================================

将 158 个 alpha158 宽表拼装为长表，并用 can_train 三条件过滤。

can_train(T, X) = has_factor(T, X) ∧ can_buy(T, X) ∧ has_label(T, X)

  has_factor : alpha158 全部 158 因子在 (T, X) 均非 NaN → MLP 输入完整
  can_buy    : T+1 日 X 非 ST / 停牌 / 新股 → 能买入（涨停不过滤：因子和标签均可观测）
  has_label  : forward_return_20d[T, X] 非 NaN → 远期收益可计算

输出:
  alpha158_long.parquet      长表 (date, stock) × [158 因子 + can_train 等]
  build_stats.txt            过滤漏斗统计
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core import config

# ==================== 路径 ====================

ALPHA158_BASE = config.RAW_FACTOR_BASE / "alpha158"
GROUPS = ["kline", "price", "rolling", "volume"]

COMBO_MASK_PATH = config.COMBO_MASK_PATH
NEW_STOCK_MASK_PATH = config.NEW_STOCK_MASK_PATH
LABEL_PATH = config.LABELS_DIR / "forward_return_20d.parquet"

OUT_DIR = Path("/nfs/ofs-prediction/peterzhenglinpeng/ml/ht")
OUT_LONG = OUT_DIR / "alpha158_long.parquet"
OUT_STATS = OUT_DIR / "build_stats.txt"


# ==================== 工具函数 ====================


def _to_wide(df: pd.DataFrame, dates: pd.DatetimeIndex, stocks: pd.Index) -> np.ndarray:
    """长表 pivot 到宽表，对齐到目标 dates/stocks。bool 输出，缺失填 False。"""
    w = df.pivot(index="datetime", columns="order_book_id")
    w = w.reindex(index=dates, columns=stocks)
    return w.to_numpy(dtype=bool, na_value=False)


def _next_day_map(dates: pd.DatetimeIndex) -> dict:
    """每个交易日 → 下一个交易日。最后一天返回 None。"""
    return {dates[i]: dates[i + 1] for i in range(len(dates) - 1)}


def _print_funnel(
    total: int,
    n_factor: int,
    n_buy: int,
    n_label: int,
    n_train: int,
    dates: pd.DatetimeIndex,
    flat_can_train: np.ndarray,
):
    """打印过滤漏斗 + 逐年样本数。"""

    def pct(n):
        return f"{n / total * 100:.1f}%"

    lines = [
        "=" * 72,
        "can_train 过滤漏斗",
        "=" * 72,
        f"  总单元格       {total:>12,}",
        f"  has_factor     {n_factor:>12,}  ({pct(n_factor)})",
        f"  can_buy        {n_buy:>12,}  ({pct(n_buy)})",
        f"  has_label      {n_label:>12,}  ({pct(n_label)})",
        f"  ────────────────────────────",
        f"  can_train 交集 {n_train:>12,}  ({pct(n_train)})",
        "",
    ]

    ct_2d = flat_can_train.reshape(len(dates), -1)
    years = dates.year.to_numpy()
    lines += [
        f"{'年份':>6} | {'交易日':>6} | {'can_train':>12} | {'日均股票':>8}",
        "─" * 42,
    ]
    for y in sorted(set(years)):
        mask_y = years == y
        n_days = int(mask_y.sum())
        n_cells = int(ct_2d[mask_y].sum())
        avg = n_cells // n_days if n_days else 0
        lines.append(f"{y:>6} | {n_days:>6} | {n_cells:>12,} | {avg:>8,}")
    lines.append("=" * 72)

    text = "\n".join(lines)
    print(text)
    return text


# ==================== 主流程 ====================


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ── Step 1: 基准 index ──
    logger.info("Step 1/6: 读取基准 index (KMID)...")
    import pyarrow.parquet as pq
    kmid_path = ALPHA158_BASE / "kline" / "KMID.parquet"
    schema = pq.read_schema(kmid_path)
    stocks = pd.Index([n for n in schema.names if not n.startswith("__")])
    ref = pd.read_parquet(kmid_path, columns=[stocks[0]])
    dates = ref.index
    T, N = len(dates), len(stocks)
    total = T * N
    logger.info(f"  {T} dates × {N} stocks = {total:,} cells")
    del ref

    # ── Step 2: 读 158 因子 + has_factor ──
    logger.info("Step 2/6: 读取 158 因子...")
    factor_names: list[str] = []
    matrices: list[np.ndarray] = []

    for g in GROUPS:
        gdir = ALPHA158_BASE / g
        for f in sorted(gdir.glob("*.parquet")):
            name = f.stem
            df = pd.read_parquet(f).reindex(index=dates, columns=stocks)
            factor_names.append(name)
            matrices.append(df.to_numpy(dtype=np.float32))
            if len(factor_names) % 20 == 0:
                logger.info(f"  已读 {len(factor_names)} 个因子...")

    logger.info(f"  共 {len(factor_names)} 个因子")

    logger.info("  has_factor: 全部 158 因子非 NaN...")
    factor_stack = np.stack(matrices)  # (158, T, N)
    has_factor = ~np.isnan(factor_stack).any(axis=0)  # (T, N) — 全部因子非 NaN
    n_factor = int(has_factor.sum())
    logger.info(f"  has_factor = True: {n_factor:,} ({n_factor / total:.1%})")

    # ── Step 3: can_buy（原始 mask，无 shift，查 T+1）──
    logger.info("Step 3/6: 构建 can_buy（T+1 可买入）...")

    logger.info("  读 combo_mask...")
    combo = pd.read_parquet(
        COMBO_MASK_PATH, columns=["order_book_id", "datetime", "is_st", "is_suspended"]
    )
    logger.info("  读 new_stock_mask...")
    ns = pd.read_parquet(NEW_STOCK_MASK_PATH, columns=["order_book_id", "datetime", "is_new_stock"])

    logger.info("  pivot 到宽表（涨停不过滤：因子和标签均可观测）...")
    cannot_buy = np.zeros((T, N), dtype=bool)
    for col in ["is_st", "is_suspended"]:
        w = combo.pivot(index="datetime", columns="order_book_id", values=col)
        cannot_buy |= w.reindex(index=dates, columns=stocks).to_numpy(dtype=bool, na_value=False)

    w_ns = ns.pivot(index="datetime", columns="order_book_id", values="is_new_stock")
    cannot_buy |= w_ns.reindex(index=dates, columns=stocks).to_numpy(dtype=bool, na_value=False)

    del combo, ns, w_ns

    nmap = _next_day_map(dates)
    can_buy = np.zeros((T, N), dtype=bool)
    for i in range(T - 1):
        t_next = nmap.get(dates[i])
        if t_next is not None:
            j = dates.get_loc(t_next)
            can_buy[i] = ~cannot_buy[j]
        # 最后一天 can_buy 保持 False

    n_buy = int(can_buy.sum())
    logger.info(f"  can_buy = True: {n_buy:,} ({n_buy / total:.1%})")
    del cannot_buy

    # ── Step 4: has_label ──
    logger.info("Step 4/6: 构建 has_label...")
    label_wide = pd.read_parquet(LABEL_PATH).reindex(index=dates, columns=stocks)
    has_label = ~np.isnan(label_wide.to_numpy(dtype=np.float32))
    n_label = int(has_label.sum())
    logger.info(f"  has_label = True: {n_label:,} ({n_label / total:.1%})")
    del label_wide

    # ── Step 5: can_train 交集 ──
    logger.info("Step 5/6: 计算 can_train 交集...")
    can_train = has_factor & can_buy & has_label
    n_train = int(can_train.sum())

    flat_factor = has_factor.ravel()
    flat_buy = can_buy.ravel()
    flat_label = has_label.ravel()
    flat_ct = can_train.ravel()

    stats_text = _print_funnel(total, n_factor, n_buy, n_label, n_train, dates, flat_ct)

    # ── Step 6: 构建长表 ──
    logger.info("Step 6/6: 构建长表...")
    any_mask = flat_factor | flat_buy | flat_label  # 至少满足一个条件才保留
    n_long = int(any_mask.sum())
    logger.info(f"  长表行数 (any_mask=True): {n_long:,}")

    logger.info("  提取因子值...")
    data = np.empty((n_long, len(factor_names)), dtype=np.float32)
    for k, mat in enumerate(matrices):
        data[:, k] = mat.reshape(-1)[any_mask]
    del matrices, factor_stack

    logger.info("  构建 DataFrame...")
    long_df = pd.DataFrame(data, columns=factor_names)
    del data

    long_df["has_factor"] = flat_factor[any_mask]
    long_df["can_buy"] = flat_buy[any_mask]
    long_df["has_label"] = flat_label[any_mask]
    long_df["can_train"] = flat_ct[any_mask]

    # 构建 MultiIndex
    date_arr = np.repeat(dates.to_numpy(), N)
    stock_arr = np.tile(stocks.to_numpy(), T)
    long_df.index = pd.MultiIndex.from_arrays(
        [date_arr[any_mask], stock_arr[any_mask]], names=["date", "stock_code"]
    )
    del date_arr, stock_arr, any_mask

    logger.info(f"  长表: {long_df.shape}")

    # ── NaN 残留验证 ──
    ct_rows = long_df["can_train"].to_numpy(dtype=bool)
    n_ct = int(ct_rows.sum())
    factor_cols = factor_names
    if n_ct > 0:
        nan_in_ct = long_df.loc[ct_rows, factor_cols].isna().sum()
        nan_factors = nan_in_ct[nan_in_ct > 0]
        if len(nan_factors) > 0:
            print(f"\n  ⚠️  can_train=True 中仍有 NaN 的因子 ({len(nan_factors)} 个):")
            for name, count in nan_factors.items():
                print(f"    {name}: {count:,} NaN ({count / n_ct:.2%})")
        else:
            print(f"\n  ✅ can_train=True 的 {n_ct:,} 行中，158 因子列零 NaN")

    # 非 can_train 行的 NaN 概况
    non_ct = ~ct_rows
    n_non_ct = int(non_ct.sum())
    if n_non_ct > 0:
        nan_per_factor = long_df.loc[non_ct, factor_cols].isna().sum()
        nan_factors_non = nan_per_factor[nan_per_factor > 0]
        print(f"\n  非 can_train 行 ({n_non_ct:,} 行) 中 NaN 因子数: {len(nan_factors_non)}")
        if len(nan_factors_non) > 0:
            top10 = nan_factors_non.nlargest(10)
            for name, count in top10.items():
                print(f"    {name}: {count:,} NaN")

    # ── 落盘 ──
    logger.info(f"  写 {OUT_LONG}...")
    long_df.to_parquet(OUT_LONG, engine="pyarrow")
    logger.info(f"  文件大小: {OUT_LONG.stat().st_size / 1024**3:.2f} GB")

    OUT_STATS.write_text(stats_text)
    logger.success(f"完成！耗时 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
