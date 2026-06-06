"""
实盘推理支路：用训好的 GBDT + 训练段 scaler_x，对「所有 pre_mask 位置」打分（不过 label）
============================================================
评估口径（ml.run 的 test IC，卡在 label 能兑现的最后一天 2026-03-31）完全不动。
本支路只新增一条推理路径：raw 特征 → 同一把 scaler_x(train段拟合) → model.predict，
把预测补到「最新因子日」（默认自动取所有入选因子共同覆盖的最末日），供实盘信号导出。

与评估路径的唯一区别：样本只过 pre_mask（剔 ST/停牌/新股），**不过 label 非 NaN**，
故候选票池是评估面板的超集；两面板在共有 (date, stock) 格子上的 ŷ 必然逐元素相等。

用法：
    python -m ml.predict_live                                   # run-id=full_gbdt, 2022-01-01→自动末日
    python -m ml.predict_live --run-id full_gbdt --start 2022-01-01 --end 2026-05-27
产物：
    FACTOR_REPL_DATA_ROOT/ml/predictions/<run_id>/pred_panel_live.parquet   (date × stock)
"""
from __future__ import annotations

import argparse
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from loguru import logger

from core import config
from ml.dataset import discover_features, load_pre_mask
from ml.preprocess import RobustZScoreScaler


def _load_scaler(model_dir, features: list[str]) -> RobustZScoreScaler:
    """读回 train 段拟合的 median/scale，子集对齐到入选因子顺序，重建 scaler。"""
    df = pd.read_parquet(model_dir / "scaler_x.parquet").reindex(features)
    sc = RobustZScoreScaler()
    sc.median_, sc.scale_, sc.columns_ = df["median"], df["scale"], list(features)
    return sc


def predict_live(run_id: str = "full_gbdt", start: str = "2022-01-01", end: str | None = None) -> pd.DataFrame:
    model_dir = config.ML_MODELS_DIR / run_id
    scaler_path = model_dir / "scaler_x.parquet"
    if not scaler_path.exists():
        raise FileNotFoundError(f"缺 scaler：{scaler_path}（先用新版 ml.run 重训 {run_id} 以物化尺子）")
    model = lgb.Booster(model_file=str(model_dir / "model.txt"))
    sel: list[str] = json.loads((model_dir / "selected_features.json").read_text())["features"]
    sx = _load_scaler(model_dir, sel)
    pathmap = discover_features()
    missing = [f for f in sel if f not in pathmap]
    if missing:
        raise KeyError(f"入选因子在 factors/raw 找不到：{missing[:5]}…（{len(missing)} 个）")

    # 末日：未指定则自动取所有入选因子「共同覆盖」的最末交易日（读 index 不读数据，廉价）
    if end is None:
        ends = [pd.to_datetime(pd.read_parquet(pathmap[f], columns=[]).index).max() for f in sel]
        end = min(ends)
        logger.info(f"[live] 未指定 --end，自动取入选因子共同覆盖末日 = {pd.Timestamp(end).date()}")

    # 样本网格：pre_mask 位置（剔 ST/停牌/新股），不过 label
    pre_mask = load_pre_mask()
    dates = pre_mask.index[(pre_mask.index >= pd.Timestamp(start)) & (pre_mask.index <= pd.Timestamp(end))]
    pre_mask = pre_mask.loc[dates]
    stocks = pre_mask.columns
    pm = pre_mask.fillna(False).to_numpy(dtype=bool)
    rr, cc = np.where(pm)
    logger.info(f"[live] {run_id}: {len(dates)} 天 × pre_mask → {len(rr):,} 样本 | 因子={len(sel)} | "
                f"区间 {dates.min().date()}~{dates.max().date()}")

    # 逐因子读 raw → inf→NaN → 在样本位置取值填列（只读入选 64 个，内存友好）
    mat = np.full((len(rr), len(sel)), np.nan, dtype=np.float32)
    for j, name in enumerate(sel):
        df = pd.read_parquet(pathmap[name]); df.index = pd.to_datetime(df.index)
        arr = df.reindex(index=dates, columns=stocks).to_numpy(dtype=np.float32, copy=True)  # 可写副本：避免 pyarrow 只读视图
        arr[~np.isfinite(arr)] = np.nan
        mat[:, j] = arr[rr, cc]
        if (j + 1) % 20 == 0:
            logger.info(f"[live]   填列 {j+1}/{len(sel)}")

    midx = pd.MultiIndex.from_arrays([dates[rr], stocks[cc]], names=["date", "stock"])
    Xz = sx.transform(pd.DataFrame(mat, index=midx, columns=sel))
    yhat = model.predict(Xz)
    panel = pd.Series(yhat, index=midx, name="yhat").unstack("stock")   # (T, N)

    out_dir = config.ML_PREDICTIONS_DIR / run_id; out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "pred_panel_live.parquet"
    panel.to_parquet(out_path)
    logger.success(f"[live] {run_id}: 面板 {panel.shape} "
                   f"({panel.index.min().date()}~{panel.index.max().date()}) → {out_path}")
    return panel


def main() -> None:
    ap = argparse.ArgumentParser(description="实盘推理：补全到最新因子日的 ŷ 面板（评估口径不动）")
    ap.add_argument("--run-id", default="full_gbdt")
    ap.add_argument("--start", default="2022-01-01")
    ap.add_argument("--end", default=None, help="留空=自动取入选因子共同覆盖末日")
    args = ap.parse_args()
    predict_live(args.run_id, args.start, args.end)


if __name__ == "__main__":
    main()
