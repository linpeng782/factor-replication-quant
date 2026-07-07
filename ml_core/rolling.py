"""
ml_core.rolling —— 滚动训练编排（按股票切分，对齐 dnn-gru-v2 方案）
============================================================
与 run_train（时间切分）并列，但切分逻辑完全不同：

  run_train   ：train/valid/test 按时间段切，embargo 月隔离
  rolling     ：train/valid 按股票随机切（80/20），同一时间段不同股票
                标签 & 标准化用全市场算（截面统计量不受切分影响）
                无 test 段——训练完直接 predict_live 预测 year+1

适用场景：模型适应当下市场风格，每年滚动重训。
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

import config
from ml_core.features import HasFactorPolicy, build_feature_matrix
from ml_core.labels import ExcessReturn, load_forward_return
from ml_core.model import LGBMAdapter
from ml_core.scaling import WholeSetRobustZ
from ml_core.select import select_by_shap
from ml_core.splits import SplitConfig, assign_segments
from ml_core.universe import Universe, build_universe


@dataclass
class RollingConfig:
    """滚动训练配置。"""

    year: int                          # 训练截止年，预测 year+1
    train_years_back: int = 10         # 往前推 N 年
    train_sample_ratio: float = 0.8    # 股票随机切分训练比例（split_mode=stock 用）
    seed: int = 42                     # 随机种子
    sources: list[str] = None
    neu_sources: list[str] = None
    horizon: int = 20
    exclude_features: list[str] = None
    select_method: str = "shap"        # shap | null
    top_k: int = 64
    lgbm_params: dict = None
    # ── split_mode=time（标准时间滚动 + 选因子解耦）专用 ──
    split_mode: str = "stock"          # stock（旧，按股票切）| time（新，按时间切）
    valid_months: int = 12             # 最近 N 个月做时间 valid（早停用）
    embargo_months: int = 1            # valid 与预测年之间的 embargo 月（>horizon 天，防标签重叠）


def _train_window(year: int, years_back: int) -> tuple[str, str]:
    """训练区间：year-years_back 年 0101 ~ year 年 1130（留 12 月给 label 兑现）。"""
    return f"{year - years_back}-01-01", f"{year}-11-30"


def _split_by_stock(
    u: Universe, base: np.ndarray, ratio: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """按股票随机切分 → (train_mask, valid_mask)，均为 (T,N) bool。

    对齐 dnn-gru-v2 的 split_datasets：
      1. 取训练区间中间日的股票列表作为采样基准
      2. 随机选 ratio 比例的股票 → 训练集（这些股票所有日期都进训练）
      3. 剩余股票 → 验证集
    """
    mid_idx = len(u.dates) // 2
    mid_date = u.dates[mid_idx]
    # 中间日当天 base=True 的股票列表
    mid_stocks = u.stocks[base[mid_idx]]
    logger.info(f"[rolling] 股票切分基准日={mid_date.date()} | 候选股票={len(mid_stocks)}")

    rng = random.Random(seed)
    n_train = int(len(mid_stocks) * ratio)
    train_codes = set(rng.sample(list(mid_stocks), n_train))

    # 构建 (T,N) 掩码：train 股票所有日期进 train，其余进 valid
    is_train_stock = np.isin(u.stocks, list(train_codes))
    train_mask = base & is_train_stock[None, :]
    valid_mask = base & ~is_train_stock[None, :]
    logger.info(
        f"[rolling] 切分完成: train={int(train_mask.sum()):,} valid={int(valid_mask.sum()):,} "
        f"({ratio:.0%}/{1-ratio:.0%})"
    )
    return train_mask, valid_mask


def run_rolling_train_time(cfg: RollingConfig) -> dict:
    """时间滚动训练（完全仿照 pipeline.run_train）：数据→特征→标签→标准化→选因子→训练。

    与 run_train 唯一区别：训练区间由 _train_window(year) 给出（预测年 year+1 是真正的样本外，
    交给 predict_live 单独出信号，故这里只切 train/valid 两段）。切分同样走 assign_segments +
    SplitConfig：valid = 训练窗最近 valid_months 个月（最贴近预测年），train 与 valid 之间空出
    embargo_months 个月（>horizon 天，防 20 日标签重叠泄露）。
    标签/标准化/选因子/训练与 run_train 的 LGBM 路径逐行一致（train 段 fit 尺子、选因子与正式训练
    共用同一时间切分、valid 早停后直接用该模型）。
    """
    start, end = _train_window(cfg.year, cfg.train_years_back)
    logger.info(f"[rolling-time] year={cfg.year} | 训练区间 {start} ~ {end} | 预测 {cfg.year + 1}")

    # 1. 底座：universe（现算 eligible_today / can_buy / has_label）
    u = build_universe(horizon=cfg.horizon, start=start, end=end)

    # 2. 特征矩阵（全特征，选因子后再定；LGBM 原生吃 NaN）
    #    训练池 = eligible_today(T日因子有效) & can_buy(label可实现) & has_label，同 run_train
    fm = build_feature_matrix(
        u, u.eligible_today & u.can_buy & u.has_label,
        sources=cfg.sources, neu_sources=cfg.neu_sources,
        has_factor_policy=HasFactorPolicy.NONE, feature_order=None,
        exclude_features=cfg.exclude_features,
    )

    # 3. 长表映射（网格 → 样本行）+ 标签（在 has_factor 过滤后的样本池上算，同 run_train）
    ret = load_forward_return(cfg.horizon).reindex(
        index=u.dates, columns=u.stocks).to_numpy(dtype=np.float32)
    didx = {d: i for i, d in enumerate(u.dates)}
    sidx = {s: i for i, s in enumerate(u.stocks)}
    ri = np.array([didx[pd.Timestamp(d)] for d in fm.dates])
    ci = np.array([sidx[s] for s in fm.stocks])
    sample_mask = np.zeros(u.shape, dtype=bool)
    sample_mask[ri, ci] = True
    # demean 市场基准池 = eligible_today & can_buy（与训练投资域同口径，同 run_train）
    y = ExcessReturn().build_panel(ret, u.eligible_today & u.can_buy, sample_mask)[ri, ci]

    # 4. 时间切分（仿照 run_train：assign_segments + SplitConfig，段间留 embargo 空档）
    #    valid = 训练窗最近 valid_months 个月（最贴近预测年）；train 与 valid 之间空出
    #    embargo_months 个月（>horizon 天，防 20 日标签重叠泄露）；预测年为真正 test，交给
    #    predict_live 单独出信号，故这里不设 test 段。选因子与正式训练共用同一时间切分。
    end_ts = pd.Timestamp(end)
    valid_lo = end_ts - pd.DateOffset(months=cfg.valid_months) + pd.Timedelta(days=1)
    train_hi = valid_lo - pd.DateOffset(months=cfg.embargo_months) - pd.Timedelta(days=1)
    split_cfg = SplitConfig(segments={
        "train": (start, train_hi.strftime("%Y-%m-%d")),
        "valid": (valid_lo.strftime("%Y-%m-%d"), end),
    })
    seg = assign_segments(u.dates, split_cfg)
    sample_train = seg["train"][ri]
    sample_valid = seg["valid"][ri]
    logger.info(f"[rolling-time] 因子矩阵 {fm.X.shape} | "
                f"train(<= {train_hi.date()})={sample_train.sum():,} "
                f"valid([{valid_lo.date()} ~ {end}])={sample_valid.sum():,} "
                f"| 预测年 {cfg.year + 1} 为 test（交给 predict_live）")

    # 5. 标准化（train 段 fit）
    standardizer = WholeSetRobustZ()
    standardizer.feature_names_ = fm.feature_names
    standardizer.fit(fm.X[sample_train])
    Xz = standardizer.transform(fm.X)

    # 6. y 标准化（回归标签 robust z-score，仅 train 段 fit）
    y_scaler = WholeSetRobustZ().fit(y[sample_train].reshape(-1, 1))
    yz = y_scaler.transform(y.reshape(-1, 1)).ravel()

    # 7. 选因子（train fit + valid 早停 → SHAP top-k；与正式训练共用同一时间切分，不做随机抽样）
    selected, scores = None, None
    feature_names, Xz_model = fm.feature_names, Xz
    if cfg.select_method == "shap":
        selected, scores = select_by_shap(
            Xz[sample_train], yz[sample_train],
            Xz[sample_valid], yz[sample_valid],
            fm.feature_names, top_k=cfg.top_k,
        )
        sel_idx = [fm.feature_names.index(f) for f in selected]
        feature_names, Xz_model = selected, Xz[:, sel_idx]
        logger.info(f"[rolling-time] SHAP 选因子: {len(fm.feature_names)} → top-{len(selected)}")

    # 8. 训练（train fit + valid 早停，早停后直接用该模型）
    adapter = LGBMAdapter(params=cfg.lgbm_params or {})
    adapter.fit(
        Xz_model[sample_train], yz[sample_train],
        Xz_model[sample_valid], yz[sample_valid],
        tag=f"rolling-time-{cfg.year}",
    )

    # 9. 产物
    return {
        "adapter": adapter,
        "standardizer": standardizer,
        "feature_names": feature_names,
        "selected": selected,
        "scores": scores,
        "u": u,
        "all_features": fm.feature_names,
        "best_iter": adapter.best_iteration,
    }


def run_rolling_train(cfg: RollingConfig) -> dict:
    """滚动训练编排：按股票切分 → 全市场标签/标准化 → SHAP 选因子 → LGBM 训练。

    返回 {adapter, standardizer, feature_names, selected, scores, u, run_meta}。
    """
    start, end = _train_window(cfg.year, cfg.train_years_back)
    logger.info(f"[rolling] year={cfg.year} | 训练区间 {start} ~ {end} | 预测 {cfg.year + 1}")

    # 1. 底座：只取训练区间（不需要 test 段）
    u = build_universe(horizon=cfg.horizon, start=start, end=end)
    ret = load_forward_return(cfg.horizon).reindex(index=u.dates, columns=u.stocks).to_numpy(dtype=np.float32)

    # 2. 候选池 = eligible_today(T日因子有效) & can_buy(label可实现) & has_label
    base = u.eligible_today & u.can_buy & u.has_label

    # 3. 按股票随机切分
    train_mask, valid_mask = _split_by_stock(u, base, cfg.train_sample_ratio, cfg.seed)

    # 4. 组装因子矩阵（全候选样本，后续按 train/valid mask 取子集）
    fm = build_feature_matrix(
        u, base, sources=cfg.sources, neu_sources=cfg.neu_sources,
        has_factor_policy=HasFactorPolicy.NONE,  # LGBM 原生吃 NaN
        feature_order=None,  # 全特征，SHAP 选完再定
        exclude_features=cfg.exclude_features,
    )

    # 5. 标签：demean 池 = eligible_today & can_buy（与训练投资域同口径，不受股票切分影响）
    target_panel = ExcessReturn().build_panel(ret, u.eligible_today & u.can_buy, base)
    didx = {d: i for i, d in enumerate(u.dates)}
    sidx = {s: i for i, s in enumerate(u.stocks)}
    ri = np.array([didx[pd.Timestamp(d)] for d in fm.dates])
    ci = np.array([sidx[s] for s in fm.stocks])
    y = target_panel[ri, ci]

    # 6. train/valid 样本索引（从 fm 长表映射回网格掩码）
    fm_train = np.zeros(u.shape, dtype=bool)
    fm_valid = np.zeros(u.shape, dtype=bool)
    fm_train[ri, ci] = False
    fm_valid[ri, ci] = False
    # fm 的样本顺序对应 ri, ci；train_mask/valid_mask 是网格上的掩码
    sample_train = train_mask[ri, ci]
    sample_valid = valid_mask[ri, ci]
    logger.info(f"[rolling] 因子矩阵 {fm.X.shape} | train={sample_train.sum():,} valid={sample_valid.sum():,}")

    # 7. 标准化：全市场 train 段 fit（不按股票切，用全市场统计量）
    standardizer = WholeSetRobustZ()
    standardizer.feature_names_ = fm.feature_names
    standardizer.fit(fm.X[sample_train])
    Xz = standardizer.transform(fm.X)

    # 8. y 标准化（回归标签 robust z-score）
    ysc = WholeSetRobustZ().fit(y[sample_train].reshape(-1, 1))
    yz = ysc.transform(y.reshape(-1, 1)).ravel()

    # 9. Stage-1：SHAP 选因子（在 train 股票上）
    selected, scores = None, None
    feature_names = fm.feature_names
    Xz_model = Xz
    if cfg.select_method == "shap":
        selected, scores = select_by_shap(
            Xz[sample_train], yz[sample_train],
            Xz[sample_valid], yz[sample_valid],
            fm.feature_names, top_k=cfg.top_k,
        )
        sel_idx = [fm.feature_names.index(f) for f in selected]
        feature_names = selected
        Xz_model = Xz[:, sel_idx]
        logger.info(f"[rolling] SHAP 选因子: {len(fm.feature_names)} → top-{len(selected)}")

    # 10. Stage-2：LGBM 训练（train 股票 fit，valid 股票 early stopping）
    adapter = LGBMAdapter(params=cfg.lgbm_params or {})
    adapter.fit(
        Xz_model[sample_train], yz[sample_train],
        Xz_model[sample_valid], yz[sample_valid],
        tag=f"rolling-{cfg.year}",
    )

    return {
        "adapter": adapter,
        "standardizer": standardizer,
        "feature_names": feature_names,
        "selected": selected,
        "scores": scores,
        "u": u,
        "all_features": fm.feature_names,
    }


def save_rolling_model(result: dict, cfg: RollingConfig, run_id: str) -> Path:
    """保存滚动训练产物（模型 + scaler + run_meta），兼容 predict_live 直读。"""
    model_dir = config.ML_MODELS_DIR / run_id
    model_dir.mkdir(parents=True, exist_ok=True)

    # 模型
    result["adapter"].save(model_dir)

    # 标准化参数
    result["standardizer"].save(model_dir / "scaler_x.parquet")

    # 因子列表
    feature_names = result["feature_names"]
    (model_dir / "feature_names.json").write_text(
        json.dumps({"features": feature_names}, ensure_ascii=False, indent=2)
    )

    # run_meta（predict_live 直读 → 零漂移）
    meta = {
        "model": "lgbm",
        "sources": cfg.sources,
        "neu_sources": cfg.neu_sources,
        "has_factor_policy": HasFactorPolicy.NONE.value,
        "horizon": cfg.horizon,
        "select_method": cfg.select_method,
        "feature_order": feature_names,
        "exclude_features": cfg.exclude_features,
        "rolling": True,
        "rolling_year": cfg.year,
        "rolling_train_years_back": cfg.train_years_back,
        "split_mode": cfg.split_mode,
    }
    if cfg.split_mode == "time":
        meta.update({
            "rolling_valid_months": cfg.valid_months,
            "rolling_embargo_months": cfg.embargo_months,
            "rolling_best_iter": result.get("best_iter"),
        })
    else:
        meta["rolling_train_sample_ratio"] = cfg.train_sample_ratio
    (model_dir / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))

    # SHAP 选因子明细
    if result["selected"] is not None:
        (model_dir / "selected_features.json").write_text(json.dumps({
            "method": f"{cfg.select_method}_gain",
            "top_k": len(result["selected"]),
            "features": result["selected"],
            "sources": cfg.sources,
            "neu_sources": cfg.neu_sources,
            "scores": {k: float(v) for k, v in result["scores"].items()},
        }, ensure_ascii=False, indent=2))

    logger.success(f"[rolling] 模型已保存: {model_dir}")
    return model_dir
