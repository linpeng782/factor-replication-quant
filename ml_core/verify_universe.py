"""
对齐验证：ml_core.universe 复现 ml/ + ml_ht/ 的底座谓词，零漂移。
============================================================
运行（dquant 后端，全集网格）：
    ALPHA158_DATA_BACKEND=dquant ML_HT_BACKEND=dquant \
        python -m ml_core.verify_universe

三步：
  1. 全集网格底座 build_universe() → 打印 can_buy / has_label 计数。
  2. 【strict】对齐 ml：在 ml 旧网格（标签 index/columns）上，
     can_buy 应逐 bit == ml.dataset.load_can_buy_mask，has_label 应逐 bit == isfinite(标签)。
  3. 【descriptive】交叉核对 ml_ht 长表的 can_buy/has_label 计数（mask 源若不同则仅近似）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from core import config
from ml_core.universe import build_universe, dquant_grid


def main() -> None:
    # ── 1) 全集网格底座 ──
    logger.info("=== 1) 全集网格底座 ===")
    u = build_universe()
    logger.info(f"  形状 {u.shape} | can_buy={u.can_buy.sum():,} | has_label={int(u.has_label.sum()):,}")

    # ── 2) strict：对齐 ml（同 masks 源、ml 旧网格=标签 index/columns）──
    logger.info("=== 2) 对齐 ml（逐 bit）===")
    from ml.dataset import load_can_buy_mask
    from ml.labels import load_forward_return

    ret = load_forward_return(20)
    u_ml = build_universe(dates=ret.index, stocks=ret.columns, horizon=20)

    pm = load_can_buy_mask().reindex(index=ret.index, columns=ret.columns).fillna(False).to_numpy(dtype=bool)
    hl = np.isfinite(ret.to_numpy(dtype=np.float32))
    assert np.array_equal(u_ml.can_buy, pm), "❌ can_buy 与 ml can_buy_mask 不一致"
    assert np.array_equal(u_ml.has_label, hl), "❌ has_label 与 ml 不一致"
    logger.success(f"  ✅ ml 网格 {u_ml.shape}：can_buy / has_label 与 ml 逐 bit 一致")

    # ── 3) descriptive：交叉核对 ml_ht 长表 ──
    logger.info("=== 3) 交叉核对 ml_ht 长表（描述性）===")
    lt_path = config.ML_HT_BASE / "alpha158_long.parquet"
    if not lt_path.exists():
        logger.warning(f"  长表不存在，跳过：{lt_path}")
        return
    lt = pd.read_parquet(lt_path, columns=["can_buy", "has_label", "can_train", "has_factor"])
    logger.info(f"  ml_ht 长表 {lt_path}")
    logger.info(f"    long can_buy={int(lt['can_buy'].sum()):,} | has_label={int(lt['has_label'].sum()):,} "
                f"| has_factor={int(lt['has_factor'].sum()):,} | can_train={int(lt['can_train'].sum()):,}")
    # 把全集底座限制到长表的日期区间再比 can_buy（长表只存 any_mask 行，故只比总量级）
    g_dates, g_stocks = dquant_grid()
    lt_dates = pd.DatetimeIndex(sorted(lt.index.get_level_values(0).unique()))
    in_range = (g_dates >= lt_dates.min()) & (g_dates <= lt_dates.max())
    u_rng = build_universe(dates=g_dates[in_range], stocks=g_stocks, horizon=20)
    logger.info(f"    universe(同区间全网格) can_buy={u_rng.can_buy.sum():,} | "
                f"has_label={int(u_rng.has_label.sum()):,}")
    logger.info("  注：长表仅存 any_mask=True 行且 mask 源可能早于底座更新，计数仅作量级 sanity，"
                "非逐 bit；strict 对齐以第 2 步 ml 比对为准。")


if __name__ == "__main__":
    main()
