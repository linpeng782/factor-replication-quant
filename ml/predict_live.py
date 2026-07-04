"""
实盘推理支路：用训好的 GBDT + 训练段 scaler_x，对「所有 can_buy_mask 位置」打分（不过 label）
============================================================
评估口径（ml.run 的 test IC，卡在 label 能兑现的最后一天 2026-03-31）完全不动。
本支路只新增一条推理路径：raw 特征 → 同一把 scaler_x(train段拟合) → model.predict，
把预测补到「最新因子日」（默认自动取所有入选因子共同覆盖的最末日），供实盘信号导出。

与评估路径的唯一区别：样本只过 can_buy_mask（剔 ST/停牌/新股），**不过 label 非 NaN**，
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
from ml.dataset import discover_features, load_factor_grid, load_can_buy_mask
from ml.preprocess import RobustZScoreScaler

# ── 覆盖校验（C）：防「面板末日新鲜但近日大面积 NaN」静默污染信号（见 superset 前沿锁死类 bug）──
COVERAGE_RECENT = 10        # 检查最近 N 个交易日
COVERAGE_DROP = 0.6         # 近期非空股票数 < 0.6×历史基线 → 判骤降


def _load_scaler(model_dir, features: list[str]) -> RobustZScoreScaler:
    """读回 train 段拟合的 median/scale，子集对齐到入选因子顺序，重建 scaler。"""
    df = pd.read_parquet(model_dir / "scaler_x.parquet").reindex(features)
    sc = RobustZScoreScaler()
    sc.median_, sc.scale_, sc.columns_ = df["median"], df["scale"], list(features)
    return sc


def predict_live(run_id: str = "full_gbdt", start: str = "2022-01-01", end: str | None = None,
                 strict_coverage: bool = False) -> pd.DataFrame:
    model_dir = config.ML_MODELS_DIR / run_id
    scaler_path = model_dir / "scaler_x.parquet"
    if not scaler_path.exists():
        raise FileNotFoundError(f"缺 scaler：{scaler_path}（先用新版 ml.run 重训 {run_id} 以物化尺子）")
    model = lgb.Booster(model_file=str(model_dir / "model.txt"))
    meta = json.loads((model_dir / "selected_features.json").read_text())
    sel: list[str] = meta["features"]
    sx = _load_scaler(model_dir, sel)
    # 按训练时持久化的 sources/neu_sources 解析特征路径：rq/dquant 同名因子
    # （alpha158 vs alpha158-dquant）不串源；老模型 json 无此字段 → None → 回退全量(旧行为)。
    raw_map = discover_features(meta.get("sources"))
    neu_map = discover_features(meta.get("neu_sources"), stage="neu") if meta.get("neu_sources") else {}
    pathmap = {**neu_map, **raw_map}
    missing = [f for f in sel if f not in pathmap]
    if missing:
        raise KeyError(f"入选因子在 factors/raw 找不到：{missing[:5]}…（{len(missing)} 个）")

    # 末日：未指定则自动取所有入选因子「共同覆盖」的最末交易日（读 index 不读数据，廉价）
    if end is None:
        ends = [pd.to_datetime(pd.read_parquet(pathmap[f], columns=[]).index).max() for f in sel]
        end = min(ends)
        logger.info(f"[live] 未指定 --end，自动取入选因子共同覆盖末日 = {pd.Timestamp(end).date()}")

    # 样本网格：can_buy_mask 位置（剔 ST/停牌/新股），不过 label
    can_buy_mask = load_can_buy_mask()
    dates = can_buy_mask.index[(can_buy_mask.index >= pd.Timestamp(start)) & (can_buy_mask.index <= pd.Timestamp(end))]
    can_buy_mask = can_buy_mask.loc[dates]
    stocks = can_buy_mask.columns
    pm = can_buy_mask.fillna(False).to_numpy(dtype=bool)
    rr, cc = np.where(pm)
    logger.info(f"[live] {run_id}: {len(dates)} 天 × can_buy_mask → {len(rr):,} 样本 | 因子={len(sel)} | "
                f"区间 {dates.min().date()}~{dates.max().date()}")

    # 逐因子读 raw（共享组装 load_factor_grid，与训练 build_dataset 同口径，杜绝 train/serve skew）
    # → 在样本位置取值填列（只读入选 64 个，内存友好）+ 逐日覆盖校验（C）
    mat = np.full((len(rr), len(sel)), np.nan, dtype=np.float32)
    cov_warn: list[tuple[str, int, list]] = []   # (因子, 基线, [(日期, 近期非空数), ...])
    for j, name in enumerate(sel):
        arr = load_factor_grid(pathmap[name], dates, stocks)   # (T, N) float32, inf→NaN
        # C: 逐日非空股票数，近 COVERAGE_RECENT 日 vs 历史基线，骤降则记（防面板假最新 → 信号退化）
        if len(dates) > COVERAGE_RECENT * 3:
            nn = np.isfinite(arr).sum(axis=1)
            baseline = float(np.median(nn[:-COVERAGE_RECENT]))
            tail = range(len(dates) - COVERAGE_RECENT, len(dates))
            bad = [(dates[i].date(), int(nn[i])) for i in tail
                   if baseline > 0 and nn[i] < COVERAGE_DROP * baseline]
            if bad:
                cov_warn.append((name, int(baseline), bad))
        mat[:, j] = arr[rr, cc]
        if (j + 1) % 20 == 0:
            logger.info(f"[live]   填列 {j+1}/{len(sel)}")

    # C: 汇总覆盖校验
    if cov_warn:
        logger.warning(f"[live][coverage] ⚠️ {len(cov_warn)}/{len(sel)} 因子近 {COVERAGE_RECENT} 日非空覆盖骤降"
                       f"（疑面板陈旧/superset 前沿锁死类 → 候选池缩水、信号静默退化）：")
        for name, base, bad in cov_warn:
            logger.warning(f"    {name}: 基线~{base}/日 → 近期 {bad}")
        if strict_coverage:
            raise RuntimeError(f"[live][coverage] {len(cov_warn)} 因子覆盖骤降且 --strict-coverage，中止以防污染信号")
    else:
        logger.info(f"[live][coverage] ✅ {len(sel)} 因子近 {COVERAGE_RECENT} 日覆盖正常")

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
    ap.add_argument("--strict-coverage", action="store_true",
                    help="任一入选因子近 N 日非空覆盖骤降则中止（默认仅 WARNING 不中止）")
    args = ap.parse_args()
    predict_live(args.run_id, args.start, args.end, strict_coverage=args.strict_coverage)


if __name__ == "__main__":
    main()
