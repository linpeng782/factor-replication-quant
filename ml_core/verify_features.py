"""
对齐验证：ml_core.features 的 has_factor 策略 + 组装口径。
============================================================
运行（dquant 后端）：
    ALPHA158_DATA_BACKEND=dquant ML_HT_BACKEND=dquant \
        python -m ml_core.verify_features

检查：
  1. discover：alpha158-dquant 应 158 个；kysec-dquant（软链）应能发现 paper_27。
  2. ALL 策略（MLP）：base=can_buy&has_label 上组装 alpha158-dquant → 保留行数
     应 ≈ ml_ht 长表 can_train（13,071,992，mask 源时间差内）。
  3. NONE 策略（LGBM）：保留行数 == base_mask.sum()（不卡完整性）。
  4. strict：抽几格因子值 vs 直接 load_factor_grid 逐元素相等。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from core import config
from ml_core.universe import build_universe
from ml_core.features import (
    HasFactorPolicy,
    build_feature_matrix,
    discover_features,
    load_factor_grid,
)


def main() -> None:
    # ── 1) discover ──
    logger.info("=== 1) discover ===")
    a158 = discover_features(["alpha158-dquant"])
    kysec = discover_features(["kysec-dquant"])
    logger.info(f"  alpha158-dquant={len(a158)} | kysec-dquant(软链)={len(kysec)}")
    assert len(a158) == 158, f"❌ alpha158-dquant 应 158 个，实得 {len(a158)}"
    assert len(kysec) > 0, "❌ kysec-dquant 软链未发现因子"
    logger.success("  ✅ discover 正常（含软链 kysec-dquant）")

    # ── 底座 + 训练候选池 base = can_buy & has_label ──
    u = build_universe()
    base = u.can_buy & u.has_label
    logger.info(f"  训练候选池 base(can_buy&has_label)={base.sum():,}")

    # ── 2) ALL 策略（MLP）：保留行数 ≈ ml_ht can_train ──
    logger.info("=== 2) ALL 策略（alpha158-dquant，全史）===")
    fm_all = build_feature_matrix(u, base, sources=["alpha158-dquant"],
                                  has_factor_policy=HasFactorPolicy.ALL)
    n_all = len(fm_all.X)
    lt_path = config.ML_HT_BASE / "alpha158_long.parquet"
    if lt_path.exists():
        ct = int(pd.read_parquet(lt_path, columns=["can_train"])["can_train"].sum())
        logger.info(f"  ALL 保留={n_all:,} | ml_ht can_train={ct:,} | 差={n_all - ct:+,} "
                    f"({abs(n_all - ct) / ct:.3%})")
    else:
        logger.warning(f"  长表不存在，跳过 can_train 比对：{lt_path}")
    assert np.isfinite(fm_all.X).all(), "❌ ALL 策略输出仍含 NaN"
    logger.success("  ✅ ALL 策略输出零 NaN")

    # ── 4) strict：抽样因子值逐元素对照 ──
    logger.info("=== 4) strict 抽样对照 ===")
    name0 = fm_all.feature_names[0]
    grid = load_factor_grid(a158[name0], u.dates, u.stocks)
    didx = {d: i for i, d in enumerate(u.dates)}
    sidx = {s: i for i, s in enumerate(u.stocks)}
    ok = True
    for k in range(0, len(fm_all.X), max(1, len(fm_all.X) // 5))[:5]:
        di, si = didx[pd.Timestamp(fm_all.dates[k])], sidx[fm_all.stocks[k]]
        a, b = fm_all.X[k, 0], grid[di, si]
        if not (np.isclose(a, b) or (np.isnan(a) and np.isnan(b))):
            ok = False
            logger.error(f"  ❌ 不一致 {fm_all.dates[k]} {fm_all.stocks[k]}: {a} != {b}")
    assert ok, "❌ 抽样因子值与直接读取不一致"
    logger.success(f"  ✅ 因子 {name0} 抽样值与 load_factor_grid 逐元素一致")

    # ── 3) NONE 策略（LGBM）：保留 == base.sum()（max_features=1 省 IO，结论同全量）──
    logger.info("=== 3) NONE 策略（不卡完整性，只读 1 因子省 IO）===")
    fm_none = build_feature_matrix(u, base, sources=["alpha158-dquant"],
                                   has_factor_policy=HasFactorPolicy.NONE, max_features=1)
    assert len(fm_none.X) == int(base.sum()), "❌ NONE 策略丢了候选行"
    logger.success(f"  ✅ NONE 保留={len(fm_none.X):,} == base={int(base.sum()):,}（不卡完整性）")


if __name__ == "__main__":
    main()
