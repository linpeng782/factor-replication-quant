"""
ml_core.pipeline —— 串联底座→特征→标签→切分→标准化→模型（两条线共享）
============================================================
两个编排入口，模型/标签/标准化/has_factor 策略全部由参数注入：

  predict_live(...)  实盘推理：预测池=can_buy（不要 label），加载既有模型+尺子 → ŷ 面板。
                     模型无关（LGBM 给 WholeSetRobustZ+NONE；MLP 给 DailyCrossSectionMAD+ALL）。
  run_train(...)     训练编排：底座→特征(has_factor策略)→标签→切分→train段fit尺子→模型.fit。

predict_live 与训练共用 build_universe / build_feature_matrix（组装口径唯一）→ 杜绝 train/serve skew。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from loguru import logger

from core import config
from ml_core.features import HasFactorPolicy, build_feature_matrix
from ml_core.labels import LabelTransform, load_forward_return
from ml_core.model import ModelAdapter
from ml_core.scaling import Standardizer
from ml_core.splits import SplitConfig, assign_segments
from ml_core.universe import Universe, build_universe


@dataclass
class PipelineConfig:
    """一次实验的全部口径（模型外的管线配置）。"""

    sources: list[str] | None
    neu_sources: list[str] | None = None
    has_factor_policy: HasFactorPolicy = HasFactorPolicy.NONE
    horizon: int = 20
    feature_order: list[str] | None = None        # 显式因子子集顺序（LGBM 选出的 top-k）
    exclude_features: list[str] | None = None     # 从全集剔除的因子（如天然稀疏因子，避免 has_factor=ALL 大量丢样本）
    split_cfg: SplitConfig = field(default_factory=SplitConfig.default)


def _ret_on_grid(u: Universe) -> np.ndarray:
    """远期收益对齐到底座网格 (T,N)。"""
    ret = load_forward_return(u.horizon).reindex(index=u.dates, columns=u.stocks)
    return ret.to_numpy(dtype=np.float32)


def predict_live(
    model_dir: str | Path,
    adapter: ModelAdapter,
    standardizer: Standardizer,
    cfg: PipelineConfig,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """实盘推理：预测池=can_buy（不过 label）→ ŷ 面板 (date×stock)。

    adapter 须已 load 好；standardizer 须已 load（有状态）或实例化（无状态）。
    cfg.feature_order 决定喂入模型的因子顺序（与尺子/模型列序一致）。
    """
    u = build_universe(horizon=cfg.horizon, start=start, end=end)
    base = u.can_buy                                  # 预测池：只过 can_buy，不要 label
    rr = int(base.sum())
    logger.info(f"[pipeline.live] 网格 {u.shape} | 预测池 can_buy={rr:,} | "
                f"区间 {u.dates.min().date()}~{u.dates.max().date()}")

    fm = build_feature_matrix(
        u, base, sources=cfg.sources, neu_sources=cfg.neu_sources,
        has_factor_policy=cfg.has_factor_policy, feature_order=cfg.feature_order,
        exclude_features=cfg.exclude_features,
    )
    Xz = standardizer.transform(fm.X, dates=fm.dates)
    yhat = adapter.predict(Xz)
    panel = pd.Series(yhat, index=fm.index, name="yhat").unstack("stock")
    logger.success(f"[pipeline.live] 面板 {panel.shape} "
                   f"({panel.index.min().date()}~{panel.index.max().date()})")
    return panel


def run_train(
    cfg: PipelineConfig,
    label: LabelTransform,
    standardizer: Standardizer,
    adapter: ModelAdapter,
    scale_label: bool = None,
    date_sample: int | None = None,
    selector: Callable | None = None,
) -> dict:
    """训练编排（模型无关）。返回 {adapter, standardizer, feature_names, selected, splits 元信息}。

    scale_label 留空时按 label.is_regression 决定（回归才对 y 再套同款标准化）。
    date_sample 仅冒烟用（每 k 个交易日取 1）。
    selector 给定时走【两阶段】：先在 train+valid 全特征上 selector 选 top-k（如 ml_core.select.select_by_shap），
      再仅用入选因子重训（对齐 ml.run 的「筛选→合成」）。selector 签名：
      (X_train, y_train, X_valid, y_valid, feature_names) → (selected: list[str], scores: pd.Series)。
      尺子(standardizer)仍按【全特征】train 段拟合并返回（与 ml.run 一致：实盘按 feature_order 子集对齐）。
    """
    scale_label = label.is_regression if scale_label is None else scale_label
    u = build_universe(horizon=cfg.horizon)
    ret = _ret_on_grid(u)

    # 训练候选池 = can_buy & has_label；features 施加 has_factor 策略
    base = u.can_buy & u.has_label
    if date_sample and date_sample > 1:
        keep_day = np.zeros(len(u.dates), dtype=bool)
        keep_day[::date_sample] = True
        base = base & keep_day[:, None]

    fm = build_feature_matrix(
        u, base, sources=cfg.sources, neu_sources=cfg.neu_sources,
        has_factor_policy=cfg.has_factor_policy, feature_order=cfg.feature_order,
        exclude_features=cfg.exclude_features,
    )
    # 标签：在 has_factor 过滤后的样本池上算（二分类中位数池口径）
    sample_mask = np.zeros(u.shape, dtype=bool)
    didx = {d: i for i, d in enumerate(u.dates)}
    sidx = {s: i for i, s in enumerate(u.stocks)}
    ri = np.array([didx[pd.Timestamp(d)] for d in fm.dates])
    ci = np.array([sidx[s] for s in fm.stocks])
    sample_mask[ri, ci] = True
    target_panel = label.build_panel(ret, u.can_buy, sample_mask)
    y = target_panel[ri, ci]

    # 时间切分
    seg = assign_segments(u.dates, cfg.split_cfg)
    seg_row = {s: seg[s][ri] for s in seg}            # 每个样本属于哪段
    tr, va, te = seg_row["train"], seg_row["valid"], seg_row["test"]
    logger.info(f"[pipeline.train] 样本 {len(fm.X):,} | train={tr.sum():,} valid={va.sum():,} test={te.sum():,}")

    # train 段 fit 尺子（特征）；可选 y 标准化
    standardizer.feature_names_ = fm.feature_names if standardizer.stateful else None
    standardizer.fit(fm.X[tr], dates=fm.dates[tr])
    Xz = standardizer.transform(fm.X, dates=fm.dates)
    if scale_label:
        from ml_core.scaling import WholeSetRobustZ
        ysc = WholeSetRobustZ().fit(y[tr].reshape(-1, 1))
        yz = ysc.transform(y.reshape(-1, 1)).ravel()
    else:
        yz = y

    # Stage-1（可选）：selector 在全特征 train+valid 上选 top-k；尺子仍保持全特征
    selected, scores = None, None
    feature_names, Xz_model = fm.feature_names, Xz
    if selector is not None:
        selected, scores = selector(Xz[tr], yz[tr], Xz[va], yz[va], fm.feature_names)
        sel_idx = [fm.feature_names.index(f) for f in selected]
        feature_names, Xz_model = selected, Xz[:, sel_idx]
        logger.info(f"[pipeline.train] 两阶段：{len(fm.feature_names)} 因子 → 选 top-{len(selected)} 重训")

    # Stage-2：用入选因子（或全特征）重训
    adapter.fit(Xz_model[tr], yz[tr], Xz_model[va], yz[va])
    return {"adapter": adapter, "standardizer": standardizer,
            "feature_names": feature_names, "selected": selected, "scores": scores,
            "seg_row": seg_row, "fm": fm, "y": y, "Xz": Xz_model,
            "u": u, "target_panel": target_panel}
