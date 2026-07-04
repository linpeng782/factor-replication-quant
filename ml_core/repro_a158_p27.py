"""
通过 ml_core 复现 ml.run 训练的 LGBM 模型 a158_p27_shap_dquant_htsplit
============================================================
目的：验证「ml_core 管线」能等价复现「ml/ 老管线」训出的两阶段 LGBM 模型。

复现对象（原始日志 ml/logs/a158_p27_shap_dquant_htsplit_20260629_200101.log）：
  sources = alpha158-dquant + kysec-dquant/paper_27_microstructure（共 181 因子）
  两阶段：181 因子 → SHAP 选 top-64 → 重训 → 样本外 test IC
  原始结果：64 因子，test IC=+0.1242 ICIR=+1.239 天数=1511（split=train 2005/valid 2018/test 2020）

等价级别（方案 A，与用户敲定）：**数值高度接近、非逐 bit**。
  唯一已知漂移源：标签截面 demean —— ml.run 用 pandas.mean（float64 累加器），
  ml_core.ExcessReturn 用 np.nanmean（float32 累加器），实测每日市场均值差 ~5e-7。
  该差异经 1000 轮 GBDT 放大后破坏逐 bit，但不改变管线行为（IC / 选因子 / 预测高度一致）。

实现形态（**薄壳**）：训练全程交给日常入口 ml_core.pipeline.run_train ——
  与「随便起一个新实验」走完全同一条编排（组装→标签→切分→标准化→两阶段→重训），
  两阶段只是给 run_train 注入 selector=select_by_shap。本脚本自己不再手写任何训练逻辑，
  只负责①给定复现口径②把产物落到隔离 run_id③与原始模型逐项对照。
  → 它既是「ml_core 能复现生产模型」的一次性证明，也是改动 ml_core 后的回归基准。

管线对齐点（均由 run_train 内部保证，与 ml.run 一致、不引入差异）：
  - 因子组装 build_feature_matrix（discover/load 原语与 ml.dataset 同一真相源）
  - 标准化 WholeSetRobustZ ≡ ml.preprocess.RobustZScoreScaler；回归目标 y 同样 train 段 fit
  - 时间切分 train≤2017-11-30 / valid 2018-01~2019-11-30 / test 2020-01~2026-03-31
  - LGBM 参数 seed=42 deterministic num_threads=64；SHAP 采样 random_state=0 top-64

产物隔离：写到独立 run_id，绝不覆盖原始 a158_p27_shap_dquant_htsplit。
运行：python -m ml_core.repro_a158_p27
"""
from __future__ import annotations

import json
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

import config
from ml_core.features import HasFactorPolicy
from ml_core.labels import ExcessReturn
from ml_core.metrics import model_ic_panel
from ml_core.model import LGBMAdapter
from ml_core.pipeline import PipelineConfig, run_train
from ml_core.scaling import WholeSetRobustZ
from ml_core.select import select_by_shap                  # 选因子已上提 ml_core（不再依赖 ml.select）
from ml_core.splits import SplitConfig

# ── 复现参数（与原始 run 对齐；脚本内固定，不走命令行）──
ORIG_RUN_ID = "a158_p27_shap_dquant_htsplit"
REPRO_RUN_ID = "a158_p27_shap_dquant_htsplit_repro_mlcore"
SOURCES = ["alpha158-dquant", "kysec-dquant/paper_27_microstructure"]
TOP_K = 64
HORIZON = 20
# split 对齐当前 ml.dataset（即训练 htsplit 模型时所用，commit 68298c0「用dquant数据更新完毕」）：
# train 2005~2017-11 / valid 2018~2019-11 / test 2020~2026-03-31（embargo 月 2017-12 / 2019-12 落空）。
# 三段天数与 htsplit 日志吻合：train 3138 / valid 465 / test 1511。
SPLIT = SplitConfig(segments={
    "train": (None, "2017-11-30"),
    "valid": ("2018-01-01", "2019-11-30"),
    "test": ("2020-01-01", "2026-03-31"),
})
# 日志落盘目录：与 ml/logs 同惯例，放 ml_core/logs/（运行时产物，应受 .gitignore 排除）
LOG_DIR = Path(__file__).parent / "logs"


def _run() -> None:
    logger.info(f"=== ml_core 复现 {ORIG_RUN_ID} → 隔离产物 {REPRO_RUN_ID} ===")

    # ── 训练：全程交给日常入口 run_train（两阶段=注入 select_by_shap）──
    cfg = PipelineConfig(
        sources=SOURCES,
        has_factor_policy=HasFactorPolicy.NONE,   # LGBM 原生吃 NaN，不卡 has_factor
        horizon=HORIZON,
        split_cfg=SPLIT,
    )
    res = run_train(
        cfg,
        label=ExcessReturn(),                     # 截面 demean 超额收益（回归目标）
        standardizer=WholeSetRobustZ(),           # ≡ ml.preprocess.RobustZScoreScaler
        adapter=LGBMAdapter(),                    # seed=42 deterministic
        scale_label=True,                         # 回归：y 再套同款 RobustZ（与 ml.run 一致）
        selector=partial(select_by_shap, top_k=TOP_K),  # Stage-1：SHAP 选 top-64
    )
    selected, scores = res["selected"], res["scores"]
    fm, u, te = res["fm"], res["u"], res["seg_row"]["test"]
    adapter, sx = res["adapter"], res["standardizer"]
    logger.info(f"[repro] SHAP 入选 top-{len(selected)}（前5）：{selected[:5]}")

    # ── 落产物（隔离 run_id；sx 已由 run_train 按全特征 train 段 fit 好）──
    model_dir = config.ML_MODELS_DIR / REPRO_RUN_ID
    adapter.save(model_dir)
    sx.save(model_dir / "scaler_x.parquet")
    (model_dir / "selected_features.json").write_text(json.dumps({
        "method": "shap_gain", "top_k": len(selected), "features": selected,
        "sources": SOURCES, "neu_sources": None,
        "scores": {k: float(v) for k, v in scores.items()},
    }, ensure_ascii=False, indent=2))

    # ── 样本外 test IC（真值=未标准化超额收益面板，run_train 已返回 target_panel）──
    pred_te = adapter.predict(res["Xz"][te])                 # Xz 已是入选 64 列
    pred_panel = pd.Series(pred_te, index=fm.index[te], name="yhat").unstack("stock")
    excess_raw = pd.DataFrame(res["target_panel"], index=u.dates, columns=u.stocks)
    ic = model_ic_panel(pred_panel, excess_raw)
    logger.success(f"[repro] test IC 均值={ic.mean():+.4f} ICIR={ic.mean()/ic.std():+.3f} "
                   f"t={ic.mean()/ic.std()*len(ic)**0.5:+.2f} 天数={len(ic)}")
    pred_dir = config.ML_PREDICTIONS_DIR / REPRO_RUN_ID
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_panel.to_parquet(pred_dir / "pred_panel.parquet")

    # ════════════════ 与原始模型对照 ════════════════
    logger.info(f"=== 与原始 {ORIG_RUN_ID} 对照 ===")
    orig_dir = config.ML_MODELS_DIR / ORIG_RUN_ID
    orig_meta = json.loads((orig_dir / "selected_features.json").read_text())
    orig_feats = orig_meta["features"]

    # 1) 选因子对照：集合重叠 + 同位次
    inter = set(orig_feats) & set(selected)
    same_pos = sum(1 for a, b in zip(orig_feats, selected) if a == b)
    logger.success(f"  选因子: 交集 {len(inter)}/{TOP_K} | 同位次 {same_pos}/{TOP_K}")
    only_orig = [f for f in orig_feats if f not in set(selected)]
    only_new = [f for f in selected if f not in set(orig_feats)]
    if only_orig or only_new:
        logger.warning(f"    仅原始={only_orig}")
        logger.warning(f"    仅复现={only_new}")

    # 2) IC 对照
    orig_ic = pd.read_parquet(config.ML_PREDICTIONS_DIR / ORIG_RUN_ID / "ic_series.parquet")["ic"]
    logger.success(f"  IC 均值: 原始={orig_ic.mean():+.4f} 复现={ic.mean():+.4f} "
                   f"差={ic.mean()-orig_ic.mean():+.5f} | 天数 原始={len(orig_ic)} 复现={len(ic)}")

    # 3) 预测面板逐格对照（最有力：两条管线的最终输出有多接近）
    orig_panel = pd.read_parquet(config.ML_PREDICTIONS_DIR / ORIG_RUN_ID / "pred_panel.parquet")
    di = orig_panel.index.intersection(pred_panel.index)
    si = orig_panel.columns.intersection(pred_panel.columns)
    a = orig_panel.loc[di, si].to_numpy(np.float64)
    b = pred_panel.loc[di, si].to_numpy(np.float64)
    both = np.isfinite(a) & np.isfinite(b)
    av, bv = a[both], b[both]
    max_abs = float(np.max(np.abs(av - bv)))
    pear = float(np.corrcoef(av, bv)[0, 1])
    logger.success(f"  pred_panel 逐格: 共有 {len(di)}日×{len(si)}股 非空 {both.sum():,} | "
                   f"max_abs={max_abs:.3e} | Pearson={pear:.6f}")
    logger.success(f"=== 复现完成（方案A：数值接近）。产物：{model_dir} ===")


def main() -> None:
    """接 loguru 文件 sink（控制台 + 落盘），保证训练全过程可复看。"""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"{REPRO_RUN_ID}_{ts}.log"
    sink = logger.add(log_path, level="INFO",
                      format="{time:YYYY-MM-DD HH:mm:ss} | {level: <7} | {message}")
    logger.info(f"日志落盘：{log_path}")
    try:
        _run()
    except Exception:
        logger.exception("复现失败")    # 异常栈也写进日志
        raise
    finally:
        logger.remove(sink)


if __name__ == "__main__":
    main()
