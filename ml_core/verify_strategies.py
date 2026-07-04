"""
对齐验证：labels / scaling / splits 策略 vs ml + ml_ht 原实现，逐元素一致。
============================================================
运行：python -m ml_core.verify_strategies
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger


def _eq(a, b, tol=1e-6):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return a.shape == b.shape and np.all((np.abs(a - b) <= tol) | (np.isnan(a) & np.isnan(b)))


def main() -> None:
    # ── 1) ExcessReturn vs ml.labels.build_excess_label ──
    logger.info("=== 1) ExcessReturn vs ml ===")
    from ml.labels import build_excess_label, load_forward_return
    from ml.dataset import load_can_buy_mask
    from ml_core.labels import ExcessReturn

    ret = load_forward_return(20)
    pm = load_can_buy_mask().reindex(index=ret.index, columns=ret.columns)
    ml_excess = build_excess_label(ret, pm).to_numpy(dtype=np.float32)
    can_buy = pm.astype("boolean").fillna(False).to_numpy(dtype=bool)
    mine = ExcessReturn().build_panel(ret.to_numpy(np.float32), can_buy, can_buy)
    assert _eq(ml_excess, mine), "❌ ExcessReturn 与 ml 不一致"
    logger.success("  ✅ ExcessReturn 与 ml.build_excess_label 逐元素一致")

    # ── 2) BinaryMedian vs ml_ht.dataset.binarize_labels（在 pool 位置）──
    logger.info("=== 2) BinaryMedian vs ml_ht ===")
    from ml_ht.dataset import binarize_labels
    from ml_core.labels import BinaryMedian

    # 取一小段网格构造长表数组
    sub = ret.iloc[-200:, :800]
    T, N = sub.shape
    pool = np.isfinite(sub.to_numpy())                       # pool=有标签处
    dates_long = np.repeat(sub.index.to_numpy(), N)
    labels_long = sub.to_numpy(np.float32).reshape(-1)
    pool_long = pool.reshape(-1)
    ht_bin = binarize_labels(labels_long, dates_long, pool_long).reshape(T, N)
    mine_bin = BinaryMedian().build_panel(sub.to_numpy(np.float32),
                                          np.zeros_like(pool), pool)
    assert _eq(ht_bin[pool], mine_bin[pool]), "❌ BinaryMedian 与 ml_ht 在 pool 位置不一致"
    logger.success("  ✅ BinaryMedian 与 ml_ht.binarize_labels 在 pool 位置逐元素一致")

    # ── 3) WholeSetRobustZ vs ml.preprocess.RobustZScoreScaler ──
    logger.info("=== 3) WholeSetRobustZ vs ml ===")
    from ml.preprocess import RobustZScoreScaler
    from ml_core.scaling import WholeSetRobustZ

    rng = np.random.default_rng(0)
    M = rng.standard_normal((5000, 12)).astype(np.float32)
    M[rng.random(M.shape) < 0.05] = np.nan                   # 撒点 NaN
    cols = [f"f{i}" for i in range(12)]
    dfM = pd.DataFrame(M, columns=cols)
    ml_sc = RobustZScoreScaler().fit(dfM)
    ml_z = ml_sc.transform(dfM).to_numpy()
    my = WholeSetRobustZ().fit(M); my.feature_names_ = cols
    my_z = my.transform(M)
    assert _eq(ml_sc.median_.to_numpy(), my.median_) and _eq(ml_sc.scale_.to_numpy(), my.scale_), "❌ median/scale 不一致"
    assert _eq(ml_z, my_z), "❌ transform 不一致"
    logger.success("  ✅ WholeSetRobustZ 与 ml.RobustZScoreScaler 逐元素一致（median/scale/transform）")

    # ── 4) DailyCrossSectionMAD vs ml_ht.dataset.build_predict_set ──
    logger.info("=== 4) DailyCrossSectionMAD vs ml_ht ===")
    from ml_ht.dataset import build_predict_set
    from ml_core.scaling import DailyCrossSectionMAD

    nd, ns, nf = 5, 600, 10
    feats = rng.standard_normal((nd * ns, nf)).astype(np.float32)
    dts = np.repeat(np.arange(nd).astype("datetime64[D]"), ns)
    stks = np.tile(np.arange(ns).astype(str), nd)
    has_f = np.ones(nd * ns, dtype=bool)
    can_b = np.ones(nd * ns, dtype=bool)
    ht_z, _, _ = build_predict_set(feats, has_f, can_b, dts, stks)
    my_z2 = DailyCrossSectionMAD().transform(feats.copy(), dts)
    assert _eq(ht_z, my_z2), "❌ DailyCrossSectionMAD 与 ml_ht 不一致"
    logger.success("  ✅ DailyCrossSectionMAD 与 ml_ht.build_predict_set 逐元素一致")

    # ── 5) splits vs ml SPLIT ──
    logger.info("=== 5) splits vs ml SPLIT ===")
    from ml.dataset import SPLIT as ML_SPLIT
    from ml_core.splits import assign_segments, SplitConfig

    dates = ret.index
    seg = assign_segments(dates, SplitConfig.default())
    for name, (lo, hi) in ML_SPLIT.items():
        ml_mask = np.asarray((dates >= pd.Timestamp(lo)) & (dates <= pd.Timestamp(hi)))
        # ml_core default test 末日开口；只比 train/valid 完全相等，test 比 ml 子集关系
        if name == "test":
            assert (seg["test"] >= ml_mask).all(), "❌ test 段未覆盖 ml test"
        else:
            assert np.array_equal(seg[name], ml_mask), f"❌ {name} 段与 ml 不一致"
    logger.success("  ✅ splits train/valid 与 ml 逐日一致；test 覆盖 ml test（开口到末日）")


if __name__ == "__main__":
    main()
