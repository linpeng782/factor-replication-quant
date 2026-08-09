"""
头部可买性诊断 —— 解释「诊断脚本的 topN 超额」与「回测年化」为何背离
================================================================================
背景：scripts/diagnose_topn_tail_profile.py 显示 E1（纯 2d 标签）的 top5 每期超额
      高达 +85.6bp，但 topk 网格回测（scripts/summarize_topk_grid.py）显示
      E1 的年化随 top_k 缩小【单调下降】（100→24.0% / 50→23.9% / 30→16.1% / 20→12.6%），
      与诊断完全反向。

排除项：forward_return_2d[T] = vwap(T+3)/vwap(T+1)-1，已是 T+1 执行口径，
        所以背离【不是】执行延迟造成的。

本脚本检验剩下的头号嫌疑：**头部票在 T+1 根本买不到**（涨停/停牌/ST/新股）。
对每个模型、每个 N 输出：
  可买占比      = topN 中 (can_buy & not_limit_up) 的比例
  原始超额      = 全部 topN 的平均超额（= tail_profile 的口径，理想化）
  可买后超额    = 只保留可买票后的平均超额（≈ 回测真实能吃到的）
  锐度蒸发      = 原始超额 - 可买后超额

参数写在下方 PARAMS，不走命令行。
用法：PYTHONPATH=. python scripts/diagnose_topn_tradability.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from alpha_shared.cleaning.mask_loader import load_filter_masks  # noqa: E402

# ============================== PARAMS ==============================
HORIZON = 2
START, END = "2020-01-02", "2026-05-26"
TOPNS = (5, 10, 20, 30, 50, 100)

# 宽表 (date × stock) parquet 直接给路径；长表 (date, code, score) 给 dict(path=..., long=True)
MODELS: dict[str, object] = {
    "E0 基线 20d": str(config.ML_PREDICTIONS_DIR / "lgbm_shap128_csrank5_dq" / "pred_panel_live.parquet"),
    "E1 纯 2d":    str(config.ML_PREDICTIONS_DIR / "lgbm_shap128_h2_csrank5_dq" / "pred_panel_live.parquet"),
    "joey base49": dict(path="/nfs/ofs-prediction/peterzhenglinpeng-code/joey-project/run_rebuild49/scores.parquet",
                        long=True),
}
# ====================================================================


def _load_panel(spec: object, ref_cols: pd.Index) -> pd.DataFrame:
    """宽表/长表统一成 (date × stock) 宽表；长表无后缀 code 按前缀映射到参照列名。
    与 diagnose_topn_tail_profile.py 的同名函数保持一致口径。"""
    if isinstance(spec, dict) and spec.get("long"):
        df = pd.read_parquet(spec["path"])
        pre = {c.split(".")[0]: c for c in ref_cols}
        df["date"] = pd.to_datetime(df["date"])
        df["col"] = df["code"].astype(str).str.zfill(6).map(pre)
        df = df.dropna(subset=["col"])
        panel = df.pivot_table(index="date", columns="col", values="score")
    else:
        panel = pd.read_parquet(spec if isinstance(spec, str) else spec["path"])
        panel.index = pd.to_datetime(panel.index)
    return panel.sort_index()


def main() -> None:
    ret = pd.read_parquet(config.LABELS_DIR / f"forward_return_{HORIZON}d.parquet")
    ret.index = pd.to_datetime(ret.index)

    can_buy, not_lu = load_filter_masks(
        combo_mask_path=config.COMBO_MASK_PATH,
        new_stock_mask_path=config.NEW_STOCK_MASK_PATH,
        reindex_columns=ret.columns,
    )
    tradable = (can_buy & not_lu).reindex(index=ret.index, columns=ret.columns, fill_value=False)
    logger.info(f"标签 {ret.shape} | 可买 mask {tradable.shape} | 区间 {START}~{END}")

    rows = []
    for name, spec in MODELS.items():
        panel = _load_panel(spec, ret.columns)

        d = panel.index.intersection(ret.index)
        d = d[(d >= pd.Timestamp(START)) & (d <= pd.Timestamp(END))]
        c = panel.columns.intersection(ret.columns)
        a = panel.loc[d, c].to_numpy(dtype=np.float64)
        b = ret.loc[d, c].to_numpy(dtype=np.float64)
        tb = tradable.loc[d, c].to_numpy()
        a = np.where(np.isfinite(b), a, np.nan)          # 与 tail_profile 同口径

        rk = pd.DataFrame(-a).rank(axis=1, method="first").to_numpy()
        mkt = float(np.nanmean(np.nanmean(b, axis=1)))

        for k in TOPNS:
            sel = (rk <= k) & np.isfinite(b)
            sel_t = sel & tb
            # 逐日均值后再跨日平均：与 tail_profile 一致，避免大截面日主导
            raw = np.nansum(np.where(sel, b, 0.0), axis=1) / np.maximum(sel.sum(axis=1), 1)
            trd = np.nansum(np.where(sel_t, b, 0.0), axis=1) / np.maximum(sel_t.sum(axis=1), 1)
            valid = sel_t.sum(axis=1) > 0                # 全不可买的日子不计入
            raw_ex = float(np.nanmean(raw)) - mkt
            trd_ex = float(np.nanmean(trd[valid])) - mkt
            rows.append({
                "模型": name, "N": k,
                "可买占比": f"{sel_t.sum() / max(sel.sum(), 1) * 100:.1f}%",
                "原始超额": f"{raw_ex*1e4:+.1f}bp",
                "可买后超额": f"{trd_ex*1e4:+.1f}bp",
                "锐度蒸发": f"{(raw_ex - trd_ex)*1e4:+.1f}bp",
            })
        logger.info(f"  完成 {name}")

    print(f"\n全市场每期均值 = {mkt*1e4:.1f}bp（超额均以此为基准）\n")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
