"""
ml_core.features —— 统一因子读取 + has_factor 模型策略（动态层）
============================================================
职责：给定底座（universe）+ 候选样本掩码，把多源宽表因子组装成 (样本, 因子) 矩阵，
并按【模型策略】施加 has_factor 完整性过滤。

与底座（universe）的分工：
  - universe 给出公共网格 + can_buy + has_label（模型/因子无关、稳定）
  - features 现算 has_factor（随因子集 + 模型策略变化，**不物化**）

has_factor 是模型驱动的策略（重构换词汇时最易踩的串味坑，显式化）：
  ALL  —— MLP：158 因子全非 NaN 才入选（稠密输入，缺一即排除）
  NONE —— LGBM：原生吃 NaN，不卡完整性（trivially-true，保留全部候选行）
  ⚠️ LGBM 误用 ALL 会悄悄丢样本、改训练分布、动 IC 且不报错 → 默认 NONE，并在入口显式声明。

discover_features / load_factor_grid 直接沿用 ml.dataset 的已验证实现（逐字节同口径），
确保 train 组装与 live 推理共用一处、杜绝 train/serve skew；待 ml 接入后改为从此处 import。
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from core import config
from ml_core.universe import Universe


class HasFactorPolicy(str, Enum):
    """has_factor 完整性策略（模型驱动）。"""

    ALL = "all"     # MLP：全部因子非 NaN
    NONE = "none"   # LGBM：不卡（原生吃 NaN）


def discover_features(sources: list[str] | None = None, stage: str = "raw") -> dict[str, Path]:
    """从 factors/<stage> 收集 {因子名: parquet 路径}（沿用 ml.dataset.discover_features）。

    两条防串味硬约束：
      1. 先按 source 过滤、再按 stem 去重（避免同名因子被另一源路径抢占）。
      2. source 用「路径分量」匹配（rel==s 或 rel 以 s+"/" 开头），
         避免 "alpha158" 误配 "alpha158-dquant"。
    glob 跟随软链 → factors/raw/kysec-dquant（→ raw-dquant/kysec）可一并发现。
    """
    base = config.RAW_FACTOR_BASE if stage == "raw" else config.NEU_FACTOR_BASE
    paths = sorted(base.glob("*/*/*.parquet"))
    if sources:
        def _match(rel: str) -> bool:
            return any(rel == s or rel.startswith(s + "/") for s in sources)
        paths = [p for p in paths if _match(str(p.relative_to(base)))]
    return {p.stem: p for p in paths}


def load_factor_grid(path: Path, dates: pd.DatetimeIndex, stocks: pd.Index) -> np.ndarray:
    """读单因子宽表 → reindex 到 (dates × stocks) → float32 → inf→NaN（沿用 ml.dataset）。

    train 组装与 live 推理共用此函数：组装口径（reindex/dtype/inf-NaN）只此一处定义。
    """
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    arr = df.reindex(index=dates, columns=stocks).to_numpy(dtype=np.float32, copy=True)
    arr[~np.isfinite(arr)] = np.nan
    return arr


@dataclass
class FeatureMatrix:
    """组装好的样本矩阵 + 行索引 (date, stock) + 因子名。"""

    X: np.ndarray              # (n_samples, n_factors) float32（NONE 策略下可含 NaN）
    dates: np.ndarray          # (n_samples,) datetime64 —— 每行对应日期
    stocks: np.ndarray         # (n_samples,) —— 每行对应股票
    feature_names: list[str]

    @property
    def index(self) -> pd.MultiIndex:
        return pd.MultiIndex.from_arrays([self.dates, self.stocks], names=["date", "stock"])

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.X, index=self.index, columns=self.feature_names)


def build_feature_matrix(
    universe: Universe,
    base_mask: np.ndarray,
    sources: list[str] | None = None,
    neu_sources: list[str] | None = None,
    has_factor_policy: HasFactorPolicy = HasFactorPolicy.NONE,
    max_features: int | None = None,
    feature_order: list[str] | None = None,
) -> FeatureMatrix:
    """组装 (样本, 因子) 矩阵并按 has_factor 策略过滤行。

    base_mask     : (T, N) bool 候选样本掩码（调用方组合，如训练 can_buy&has_label、推理 can_buy）。
    feature_order : 显式因子顺序（如 LGBM 选出的 top-k，须与模型/scaler 列序一致）；
                    留空则用 sorted(discover) 全集。
    返回 FeatureMatrix：ALL 策略下 X 行已保证全非 NaN；NONE 策略下保留全部候选行（可含 NaN）。
    """
    raw_map = discover_features(sources, stage="raw")
    neu_map = discover_features(neu_sources, stage="neu") if neu_sources else {}
    conflicts = set(raw_map) & set(neu_map)
    if conflicts:
        raise ValueError(f"因子名冲突（raw 与 neu 同名）: {sorted(conflicts)}")
    pathmap = {**raw_map, **neu_map}
    if feature_order is not None:
        missing = [f for f in feature_order if f not in pathmap]
        if missing:
            raise KeyError(f"指定因子未找到：{missing[:5]}…（{len(missing)} 个）")
        feat = list(feature_order)
    else:
        feat = sorted(pathmap)
        if max_features:
            feat = feat[:max_features]
    if not feat:
        raise ValueError(f"未发现任何因子：sources={sources} neu_sources={neu_sources}")

    dates, stocks = universe.dates, universe.stocks
    if base_mask.shape != (len(dates), len(stocks)):
        raise ValueError(f"base_mask 形状 {base_mask.shape} != 网格 {(len(dates), len(stocks))}")

    rr, cc = np.where(base_mask)   # 候选样本位置：行=日期 idx，列=股票 idx
    logger.info(f"[features] 因子={len(feat)} | 候选样本={len(rr):,} | "
                f"has_factor策略={has_factor_policy.value} | 来源 raw={sources or 'ALL'} neu={neu_sources or 'NONE'}")

    mat = np.full((len(rr), len(feat)), np.nan, dtype=np.float32)
    for j, name in enumerate(feat):
        arr = load_factor_grid(pathmap[name], dates, stocks)   # (T, N) float32, inf→NaN
        mat[:, j] = arr[rr, cc]
        if (j + 1) % 40 == 0:
            logger.info(f"[features]   填列 {j+1}/{len(feat)}")

    # has_factor 策略：ALL 保留全非 NaN 行（MLP）；NONE 保留全部（LGBM）
    if has_factor_policy == HasFactorPolicy.ALL:
        keep = np.isfinite(mat).all(axis=1)
        logger.info(f"[features] has_factor=ALL：{len(rr):,} → 保留全非NaN {int(keep.sum()):,} 行 "
                    f"（剔 {len(rr) - int(keep.sum()):,}）")
    else:
        keep = np.ones(len(rr), dtype=bool)

    mat, rr, cc = mat[keep], rr[keep], cc[keep]
    return FeatureMatrix(
        X=mat,
        dates=dates.to_numpy()[rr],
        stocks=stocks.to_numpy()[cc],
        feature_names=feat,
    )
