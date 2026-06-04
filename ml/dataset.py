"""
数据集组装 + 时间划分（含 embargo）+ mask 对齐
============================================================
流程（内存友好：先定样本索引，再逐因子填列，避免巨型中间体）：
  1. 读 label → 超额(demean) → 确定 train/valid/test 各段「样本索引」
       样本 = 段内日期 × pre_mask(剔ST/停牌/新股) × label 非 NaN
  2. 预分配 (N样本, F因子) float32 矩阵；逐因子读 raw → inf→NaN → 在样本位置取值填列
  3. RobustZScaler 在【train 段】fit（特征一套 + 标签一套），三段 transform
  4. post_mask(涨停) 记录于样本元信息，供预测/可买集合使用
返回 Split（X/y train/valid/test + 两个 scaler + 元信息）。

冒烟测试旋钮：date_sample（每 k 个交易日取 1）、max_features（只取前 m 个因子）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from core import config
from ml.labels import build_excess_label, load_forward_return
from ml.preprocess import RobustZScoreScaler

# 方案 A 划分（含 embargo 月：2019-12 / 2021-12 不落入任何段）
SPLIT = {
    "train": ("2012-01-01", "2019-11-30"),
    "valid": ("2020-01-01", "2021-11-30"),
    "test":  ("2022-01-01", "2026-03-31"),
}
HORIZON = 20


@dataclass
class Split:
    X_train: pd.DataFrame; y_train: pd.Series
    X_valid: pd.DataFrame; y_valid: pd.Series
    X_test: pd.DataFrame;  y_test: pd.Series
    feature_cols: list[str]
    scaler_x: RobustZScoreScaler
    scaler_y: RobustZScoreScaler
    # 元信息：每段样本的 (date, stock) 索引 + 涨停 post_mask（True=可买）
    meta: dict = field(default_factory=dict)


def discover_features(sources: list[str] | None = None) -> dict[str, Path]:
    """从 factors/raw 收集 {因子名: parquet 路径}（默认全部；sources 按 <source> 前缀过滤）。"""
    pathmap = {p.stem: p for p in sorted(config.RAW_FACTOR_BASE.glob("*/*/*.parquet"))}
    if sources:
        pathmap = {n: p for n, p in pathmap.items()
                   if any(str(p.relative_to(config.RAW_FACTOR_BASE)).startswith(s) for s in sources)}
    return pathmap


def _load_mask(value_col: str) -> pd.DataFrame:
    """combo_mask 长表 → 宽表布尔。value_col: 'tradable'(pre) 或 'is_limit_up'(post)。"""
    m = pd.read_parquet(config.COMBO_MASK_PATH)
    w = m.pivot(index="datetime", columns="order_book_id", values=value_col)
    w.index = pd.to_datetime(w.index)
    return w


def load_pre_mask() -> pd.DataFrame:
    """(T,N) 布尔：True=参与（tradable=NOT st/suspended/new）。涨停保留。"""
    return _load_mask("tradable").astype("boolean")


def _segment_dates(all_dates: pd.DatetimeIndex, date_sample: int | None) -> dict[str, pd.DatetimeIndex]:
    out = {}
    for seg, (lo, hi) in SPLIT.items():
        d = all_dates[(all_dates >= pd.Timestamp(lo)) & (all_dates <= pd.Timestamp(hi))]
        if date_sample and date_sample > 1:
            d = d[::date_sample]
        out[seg] = d
    return out


def build_dataset(
    sources: list[str] | None = None,
    date_sample: int | None = None,
    max_features: int | None = None,
) -> Split:
    """组装长表 → 划分 → train 段 fit RobustZScore(特征+标签) → transform 三段。"""
    pathmap = discover_features(sources)
    feat_names = sorted(pathmap)
    if max_features:
        feat_names = feat_names[:max_features]
    logger.info(f"[dataset] 因子数={len(feat_names)} sources={sources or 'ALL'}")

    # --- 1) 标签(超额) + 网格 ---
    ret = load_forward_return(HORIZON)
    pre_mask = load_pre_mask().reindex(index=ret.index, columns=ret.columns)
    excess = build_excess_label(ret, pre_mask)
    all_dates = ret.index
    all_stocks = ret.columns
    seg_dates = _segment_dates(all_dates, date_sample)

    # --- 2) 各段样本索引：段内日期 × pre_mask × label非NaN ---
    pm = pre_mask.fillna(False).to_numpy(dtype=bool)
    lab = excess.to_numpy(dtype=np.float32)
    date_pos = {d: i for i, d in enumerate(all_dates)}
    seg_idx = {}  # seg -> (row_pos, col_pos)
    for seg, dts in seg_dates.items():
        rows = np.array([date_pos[d] for d in dts], dtype=np.int64)
        sub_pm = pm[rows]; sub_lab = lab[rows]
        keep = sub_pm & np.isfinite(sub_lab)
        rr, cc = np.where(keep)
        seg_idx[seg] = (rows[rr], cc)
        logger.info(f"[dataset] {seg}: {len(dts)} 天 → {len(rr):,} 样本")

    # --- 3) 预分配特征矩阵, 逐因子填列 ---
    def alloc(seg):
        return np.full((len(seg_idx[seg][0]), len(feat_names)), np.nan, dtype=np.float32)
    mats = {seg: alloc(seg) for seg in SPLIT}
    for j, name in enumerate(feat_names):
        df = pd.read_parquet(pathmap[name]); df.index = pd.to_datetime(df.index)
        df = df.reindex(index=all_dates, columns=all_stocks)
        arr = df.to_numpy(dtype=np.float32, copy=True)  # 强制可写副本：避免 pyarrow zero-copy 只读视图
        arr[~np.isfinite(arr)] = np.nan      # inf→NaN
        for seg in SPLIT:
            rp, cp = seg_idx[seg]
            mats[seg][:, j] = arr[rp, cp]
        if (j + 1) % 50 == 0:
            logger.info(f"[dataset]   填列 {j+1}/{len(feat_names)}")

    # --- 4) 组装 DataFrame + MultiIndex(date,stock) ---
    def to_df(seg):
        rp, cp = seg_idx[seg]
        midx = pd.MultiIndex.from_arrays([all_dates[rp], all_stocks[cp]], names=["date", "stock"])
        X = pd.DataFrame(mats[seg], index=midx, columns=feat_names)
        y = pd.Series(lab[rp, cp], index=midx, name="excess")
        return X, y
    Xtr, ytr = to_df("train"); Xva, yva = to_df("valid"); Xte, yte = to_df("test")

    # --- 5) RobustZScore: train 段 fit(特征 + 标签), 三段 transform ---
    sx = RobustZScoreScaler().fit(Xtr)
    Xtr_z, Xva_z, Xte_z = sx.transform(Xtr), sx.transform(Xva), sx.transform(Xte)
    sy = RobustZScoreScaler().fit(ytr.to_frame())
    ytr_z = sy.transform(ytr.to_frame())["excess"]
    yva_z = sy.transform(yva.to_frame())["excess"]
    yte_z = sy.transform(yte.to_frame())["excess"]

    return Split(
        X_train=Xtr_z, y_train=ytr_z, X_valid=Xva_z, y_valid=yva_z,
        X_test=Xte_z, y_test=yte_z, feature_cols=feat_names,
        scaler_x=sx, scaler_y=sy,
        meta={"excess_raw": excess, "ret": ret, "pre_mask": pre_mask,
              "seg_idx": seg_idx, "all_dates": all_dates, "all_stocks": all_stocks},
    )
