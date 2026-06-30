"""
ml_core 日常实验入口 —— 配模型 + 选策略，跑一次完整实验
============================================================
这是「用整套 ml_core 跑一个新实验」的主入口：顶部改配置 → python -m ml_core.run。
重活全在 ml_core.pipeline.run_train（组装→标签→切分→标准化→可选两阶段→训练），
本文件只负责①给定这次实验的口径②把产物落到 run_id③报告样本外 IC。

支持两条线（换模型=换三件套，其余编排不变）：
  LGBM ：ExcessReturn(回归) + WholeSetRobustZ + has_factor=NONE，可两阶段(SHAP/GBDT 选 top-k)
  MLP  ：BinaryMedian(二分类) + DailyCrossSectionMAD + has_factor=ALL（一般不选因子）

产物（与 ml.run 同惯例，便于下游/回测直读）：
  ML_MODELS_DIR/<run_id>/        模型 + scaler_x.parquet + feature_names.json(+ selected_features.json)
  ML_PREDICTIONS_DIR/<run_id>/   pred_panel.parquet + ic_series.parquet
  ml_core/logs/<run_id>_<时间戳>.log   训练全程
"""
from __future__ import annotations

import json
from datetime import datetime
from functools import partial
from pathlib import Path

import pandas as pd
from loguru import logger

from core import config
from ml_core.features import HasFactorPolicy
from ml_core.labels import BinaryMedian, ExcessReturn
from ml_core.metrics import model_ic_panel
from ml_core.model import LGBMAdapter, MLPAdapter
from ml_core.pipeline import PipelineConfig, run_train
from ml_core.scaling import DailyCrossSectionMAD, WholeSetRobustZ
from ml_core.select import select_by_gbdt_importance, select_by_shap
from ml_core.splits import SplitConfig

# ════════════════ 实验配置（改这里，不走命令行）════════════════
RUN_ID = None                       # None → 用启动时间戳；复用名字也不会覆盖日志（带时间戳）
MODEL = "lgbm"                      # "lgbm" | "mlp"
SOURCES = ["alpha158-dquant", "kysec-dquant/paper_27_microstructure"]
NEU_SOURCES = None                  # 从 factors/neu 读的因子源（与 SOURCES 互补）
SELECT_METHOD = "shap"              # None=单阶段全特征 | "shap" | "gbdt"（两阶段选 top-k）
TOP_K = 64                          # 两阶段入选因子数（SELECT_METHOD 为 None 时忽略）
HORIZON = 20                        # 远期收益天数
DATE_SAMPLE = None                  # 冒烟用：每 k 个交易日取 1（None=全量）
N_FEATURES = 158                    # 仅 MLP：输入维度（= 因子数；MLP 一般不选因子）
SPLIT = SplitConfig.default()       # 时间切分；要复现 htsplit 等自定义可改 segments
LOG_DIR = Path(__file__).parent / "logs"


def _strategy():
    """按 MODEL 选「标签 / 标准化 / 模型适配器 / has_factor 策略」四件套。"""
    if MODEL == "lgbm":
        return (ExcessReturn(), WholeSetRobustZ(), LGBMAdapter(), HasFactorPolicy.NONE)
    if MODEL == "mlp":
        return (BinaryMedian(), DailyCrossSectionMAD(),
                MLPAdapter(n_features=N_FEATURES), HasFactorPolicy.ALL)
    raise ValueError(f"未知 MODEL={MODEL!r}（仅 lgbm/mlp）")


def _selector():
    """按 SELECT_METHOD 注入 Stage-1 选因子；None 则单阶段。"""
    if SELECT_METHOD is None:
        return None
    fn = {"shap": select_by_shap, "gbdt": select_by_gbdt_importance}[SELECT_METHOD]
    return partial(fn, top_k=TOP_K)


def _run() -> None:
    run_id = RUN_ID or datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(f"=== ml_core 实验 run_id={run_id} | 模型={MODEL} | 选因子={SELECT_METHOD} ===")

    label, standardizer, adapter, policy = _strategy()
    cfg = PipelineConfig(sources=SOURCES, neu_sources=NEU_SOURCES,
                         has_factor_policy=policy, horizon=HORIZON, split_cfg=SPLIT)
    res = run_train(cfg, label, standardizer, adapter,
                    date_sample=DATE_SAMPLE, selector=_selector())

    selected = res["selected"]
    fm, u, te = res["fm"], res["u"], res["seg_row"]["test"]
    adapter, sx = res["adapter"], res["standardizer"]

    # ── 落模型产物（scaler 无状态时 save 为空操作，无害）──
    model_dir = config.ML_MODELS_DIR / run_id
    adapter.save(model_dir)
    sx.save(model_dir / "scaler_x.parquet")
    (model_dir / "feature_names.json").write_text(
        json.dumps({"features": res["feature_names"]}, ensure_ascii=False, indent=2))
    if selected is not None:                      # 两阶段额外存入选明细（含重要性，兼容 ml.run）
        (model_dir / "selected_features.json").write_text(json.dumps({
            "method": f"{SELECT_METHOD}_gain", "top_k": len(selected), "features": selected,
            "sources": SOURCES, "neu_sources": NEU_SOURCES,
            "scores": {k: float(v) for k, v in res["scores"].items()},
        }, ensure_ascii=False, indent=2))

    # ── 样本外评估：模型 IC（pred vs 标签面板，对齐 ml.evaluate.model_ic）──
    pred_te = adapter.predict(res["Xz"][te])
    pred_panel = pd.Series(pred_te, index=fm.index[te], name="yhat").unstack("stock")
    target_panel = pd.DataFrame(res["target_panel"], index=u.dates, columns=u.stocks)
    ic = model_ic_panel(pred_panel, target_panel)
    logger.success(f"[run {run_id}] 选 {len(res['feature_names'])} 因子 | test IC 均值={ic.mean():+.4f} "
                   f"ICIR={ic.mean()/ic.std():+.3f} t={ic.mean()/ic.std()*len(ic)**0.5:+.2f} 天数={len(ic)}")

    pred_dir = config.ML_PREDICTIONS_DIR / run_id
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_panel.to_parquet(pred_dir / "pred_panel.parquet")
    ic.to_frame("ic").to_parquet(pred_dir / "ic_series.parquet")
    logger.success(f"=== 完成。模型={model_dir} | 预测={pred_dir} ===")


def main() -> None:
    """接 loguru 文件 sink（控制台 + 落盘），保证训练全过程可复看。"""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"{RUN_ID or ts}_{ts}.log"
    sink = logger.add(log_path, level="INFO",
                      format="{time:YYYY-MM-DD HH:mm:ss} | {level: <7} | {message}")
    logger.info(f"日志落盘：{log_path}")
    try:
        _run()
    except Exception:
        logger.exception("实验失败")
        raise
    finally:
        logger.remove(sink)


if __name__ == "__main__":
    main()
