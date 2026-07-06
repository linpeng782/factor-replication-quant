"""
Alpha158 抽取复现校验
============================================================
目的：证明从 my-alpha-engine 抽到 alpha158/engine/ 的计算逻辑，
      用本地 raw meta 重算后与现有 market-data/alpha158/*.parquet 数值一致。

做法（不写盘）：
  1. core.data.adjusted_panels 加载后复权宽表面板（含 vwap）
  2. Alpha158Panel.compute_all 重算指定因子（默认一组覆盖各家族的代表）
  3. 对每个因子 reindex 到现有面板的 (date×stock) 网格，比对：
     max/mean 绝对差、相关系数、float32 容差内匹配率

为快与省内存：默认只加载 LOAD_START 起的数据（给 60d 窗口足够 warmup），
              只在 COMPARE_START~COMPARE_END 区间比对。改 FACTORS=None 比全 158。

用法：python alpha158/verify_repro.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from alpha158.engine import Alpha158Panel
from core.data import adjusted_panels

# ── 参数区 ──
EXISTING_DIR = config._MKT / "alpha158"          # 现有成品面板目录
WINDOWS = [5, 10, 20, 30, 60]
LOAD_START = "2022-01-01"                         # 加载起点（留 warmup）
COMPARE_START, COMPARE_END = "2024-01-01", "2026-04-30"  # 比对区间（对齐 152 主流面板末日）
TOL = 1e-4                                        # float32 量级容差
# 代表性子集（覆盖 K线/价格/各 rolling 家族/各量家族）；设 None 比全部 158
FACTORS = [
    "KMID", "KLEN", "KSFT2",                      # kline
    "OPEN0", "VWAP0",                             # price
    "ROC20", "MA20", "STD30", "BETA20", "RSQR20", "RESI20",
    "MAX10", "QTLU20", "RANK20", "RSV10", "IMAX10", "IMXD20",
    "CORR20", "CORD20", "CNTP10", "SUMP20",       # rolling
    "VMA20", "VSTD20", "WVMA20", "VSUMP20",       # volume
]


def main():
    if not EXISTING_DIR.exists():
        logger.error(f"现有面板目录不存在: {EXISTING_DIR}"); sys.exit(1)

    names = FACTORS if FACTORS is not None else Alpha158Panel.list_factor_names(WINDOWS)
    logger.info(f"校验 {len(names)} 个因子 | 加载起点 {LOAD_START} | 比对 {COMPARE_START}~{COMPARE_END}")

    logger.info("[1/3] 加载后复权面板（open/high/low/close/volume/vwap）...")
    panels = adjusted_panels.load_adjusted_panels(
        start=LOAD_START, end=None,
        fields=("open", "high", "low", "close", "volume", "vwap"),
    )
    logger.info(f"  close shape={panels['close'].shape}, "
                f"{panels['close'].index.min().date()}~{panels['close'].index.max().date()}")

    logger.info("[2/3] 重算因子...")
    alpha = Alpha158Panel(panels)
    only = list(names)
    computed = alpha.compute_all(windows=WINDOWS, only_names=only)

    logger.info("[3/3] 逐因子比对现有面板 ...")
    cs, ce = pd.Timestamp(COMPARE_START), pd.Timestamp(COMPARE_END)
    rows = []
    for name in names:
        exist_p = EXISTING_DIR / f"{name}.parquet"
        if name not in computed:
            rows.append((name, "重算缺失", np.nan, np.nan, np.nan)); continue
        if not exist_p.exists():
            rows.append((name, "成品缺失", np.nan, np.nan, np.nan)); continue
        old = pd.read_parquet(exist_p); old.index = pd.to_datetime(old.index)
        new = computed[name]
        old = old.loc[(old.index >= cs) & (old.index <= ce)]
        new = new.reindex(index=old.index, columns=old.columns)
        a, b = new.to_numpy(np.float64), old.to_numpy(np.float64)
        both = ~np.isnan(a) & ~np.isnan(b)
        n = int(both.sum())
        if n == 0:
            rows.append((name, "无重叠", np.nan, np.nan, np.nan)); continue
        diff = np.abs(a[both] - b[both])
        # 相对差：量类因子(分母+1e-12，volume→0 时值飙到 1e12)绝对差无意义，统一用相对差判定
        rel = diff / np.maximum(np.abs(b[both]), 1e-6)
        maxd, maxrel = float(diff.max()), float(rel.max())
        match = float((rel <= TOL).mean())          # 相对差在容差内的比例
        nan_mismatch = int((np.isnan(a) != np.isnan(b)).sum())
        verdict = "✅" if (maxrel <= TOL and nan_mismatch == 0) else (
                  "≈ " if match > 0.999 else "❌")
        rows.append((name, verdict, maxd, maxrel, match))
        logger.info(f"  {verdict} {name:8s} maxΔ={maxd:.2e} max相对差={maxrel:.2e} "
                    f"相对匹配率={match:.4%} nan不一致={nan_mismatch} n={n}")

    df = pd.DataFrame(rows, columns=["factor", "判定", "max_abs_diff", "max_rel_diff", "rel_match_rate"])
    ok = (df["判定"] == "✅").sum()
    near = (df["判定"] == "≈ ").sum()
    bad = len(df) - ok - near
    logger.info("=" * 60)
    logger.info(f"复现校验汇总：✅完全一致 {ok} | ≈高度一致 {near} | ❌/异常 {bad} / 共 {len(df)}")
    print(df.to_string(index=False))
    if bad > 0:
        logger.warning("存在 ❌/异常因子，需排查（注意 6 个 odd 面板 CORR5/10·MA5/10·ROC10·STD20 口径不同）")


if __name__ == "__main__":
    main()
