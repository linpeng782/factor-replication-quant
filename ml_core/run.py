"""
ml_core 训练入口 —— 读 train_config.yaml，跑一次完整实验
============================================================
用法：改 ml_core/train_config.yaml → python -m ml_core.run。本文件不含任何实验参数。
重活全在 ml_core.pipeline.run_train（组装→标签→切分→标准化→可选两阶段→训练），
本文件只负责①把 train_config.yaml 翻译成模型/策略②落产物到 run_id③报告样本外 IC。
（推理 / 导信号是独立入口 ml_core.predict + predict_config.yaml，本文件只管训练评估。）

支持两条线（换模型=train_config.yaml 的 model 改一行，超参各自在 lgbm:/mlp: 段）：
  LGBM ：ExcessReturn(回归) + WholeSetRobustZ + has_factor=NONE
  MLP  ：BinaryMedian(二分类) + DailyCrossSectionMAD + has_factor=ALL（输入维度按数据自动定）
两阶段 select_method=shap/gbdt 对两条线都适用（GBDT 先选 top_k 降维，再用本模型重训）。

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
import yaml
from loguru import logger

import config
from ml_core.features import HasFactorPolicy
from ml_core.labels import BinaryMedian, ExcessReturn
from ml_core.metrics import model_ic_panel
from ml_core.model import LGBMAdapter, MLPAdapter
from ml_core.pipeline import PipelineConfig, run_train
from ml_core.scaling import DailyCrossSectionMAD, WholeSetRobustZ
from ml_core.select import select_by_gbdt_importance, select_by_shap
from ml_core.splits import SplitConfig

CONFIG_PATH = Path(__file__).parent / "train_config.yaml"
LOG_DIR = Path(__file__).parent / "logs"
_SELECTORS = {"shap": select_by_shap, "gbdt": select_by_gbdt_importance}


def load_config(path: Path = CONFIG_PATH) -> dict:
    """读 train_config.yaml（唯一实验参数来源）。"""
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _strategy(c: dict):
    """按 model 选「标签 / 标准化 / 模型适配器 / has_factor 策略」四件套，超参取对应段。"""
    model = c["model"]
    if model == "lgbm":
        return (
            ExcessReturn(),
            WholeSetRobustZ(),
            LGBMAdapter(params=c.get("lgbm") or {}),
            HasFactorPolicy.NONE,
        )
    if model == "mlp":
        return (
            BinaryMedian(),
            DailyCrossSectionMAD(),
            MLPAdapter(**(c.get("mlp") or {})),
            HasFactorPolicy.ALL,
        )  # 维度按数据自动定
    raise ValueError(f"未知 model={model!r}（仅 lgbm/mlp）")


def _selector(c: dict):
    """按 select_method 注入 Stage-1 选因子；null 则单阶段。"""
    sm = c.get("select_method")
    if sm is None:
        return None
    if sm not in _SELECTORS:
        raise ValueError(f"未知 select_method={sm!r}（仅 shap/gbdt 或 null）")
    return partial(_SELECTORS[sm], top_k=c["top_k"])


def _split(c: dict) -> SplitConfig:
    """yaml 的 split 段 [起,止] → SplitConfig（list→tuple，null→None 自动）。"""
    seg = c.get("split")
    if not seg:
        return SplitConfig.default()
    return SplitConfig(segments={k: tuple(v) for k, v in seg.items()})


def _run(c: dict) -> None:
    run_id = c.get("run_id") or datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(
        f"=== ml_core 实验 run_id={run_id} | 模型={c['model']} | 选因子={c.get('select_method')} ==="
    )

    label, standardizer, adapter, policy = _strategy(c)
    cfg = PipelineConfig(
        sources=c["sources"],
        neu_sources=c.get("neu_sources"),
        has_factor_policy=policy,
        horizon=c["horizon"],
        split_cfg=_split(c),
        exclude_features=c.get("exclude_features"),
    )
    res = run_train(
        cfg,
        label,
        standardizer,
        adapter,
        date_sample=c.get("date_sample"),
        selector=_selector(c),
    )

    selected = res["selected"]
    fm, u, te = res["fm"], res["u"], res["seg_row"]["test"]
    adapter, sx = res["adapter"], res["standardizer"]

    # ── 落模型产物（scaler 无状态时 save 为空操作，无害）──
    model_dir = config.ML_MODELS_DIR / run_id
    adapter.save(model_dir)
    sx.save(model_dir / "scaler_x.parquet")
    (model_dir / "feature_names.json").write_text(
        json.dumps({"features": res["feature_names"]}, ensure_ascii=False, indent=2)
    )
    # run_meta：推理自包含的唯一真相源（ml_core.predict 直读 → 零 train/serve 漂移）
    (model_dir / "run_meta.json").write_text(
        json.dumps(
            {
                "model": c["model"],
                "sources": c["sources"],
                "neu_sources": c.get("neu_sources"),
                "has_factor_policy": policy.value,
                "horizon": c["horizon"],
                "select_method": c.get("select_method"),
                "feature_order": res["feature_names"],
                "exclude_features": c.get("exclude_features"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if selected is not None:  # 两阶段额外存入选明细（含重要性，兼容 ml.run）
        (model_dir / "selected_features.json").write_text(
            json.dumps(
                {
                    "method": f"{c['select_method']}_gain",
                    "top_k": len(selected),
                    "features": selected,
                    "sources": c["sources"],
                    "neu_sources": c.get("neu_sources"),
                    "scores": {k: float(v) for k, v in res["scores"].items()},
                },
                ensure_ascii=False,
                indent=2,
            )
        )

    # ── 样本外评估：模型 IC（pred vs 标签面板，对齐 ml.evaluate.model_ic）──
    pred_te = adapter.predict(res["Xz"][te])
    pred_panel = pd.Series(pred_te, index=fm.index[te], name="yhat").unstack("stock")
    target_panel = pd.DataFrame(res["target_panel"], index=u.dates, columns=u.stocks)
    ic = model_ic_panel(pred_panel, target_panel)
    logger.success(
        f"[run {run_id}] 选 {len(res['feature_names'])} 因子 | test IC 均值={ic.mean():+.4f} "
        f"ICIR={ic.mean()/ic.std():+.3f} t={ic.mean()/ic.std()*len(ic)**0.5:+.2f} 天数={len(ic)}"
    )

    pred_dir = config.ML_PREDICTIONS_DIR / run_id
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_panel.to_parquet(pred_dir / "pred_panel.parquet")
    ic.to_frame("ic").to_parquet(pred_dir / "ic_series.parquet")
    logger.success(f"=== 训练完成。模型={model_dir} | 预测={pred_dir} ===")

    # ── 训练后自动推理 + 导信号（对齐 run_rolling.py 的一步到位）──
    pc = c.get("predict") or {}
    if pc.get("after_train"):
        _predict_after_train(c, res, model_dir, pred_dir, run_id)


def _predict_after_train(
    c: dict, res: dict, model_dir: Path, pred_dir: Path, run_id: str
) -> None:
    """训完 reload 磁盘模型 → predict_live → export_panel（与 run_rolling.py 同口径）。"""
    from ml_core.pipeline import predict_live as _predict_live
    from ml_core.signals import export_panel

    pc = c["predict"]
    start = pc.get("start")
    end = pc.get("end")
    top_n = pc.get("top_n", 500)
    rebuild = pc.get("rebuild", True)

    logger.info(f"[run {run_id}] === 训练后推理：reload 模型 → predict_live → 导信号 ===")

    # reload 磁盘模型（确保和实盘一致，不直接用内存里的 adapter）
    model = c["model"]
    if model == "lgbm":
        from ml_core.model import LGBMAdapter
        from ml_core.scaling import WholeSetRobustZ
        adapter = LGBMAdapter().load(model_dir)
        sx = WholeSetRobustZ.load(model_dir / "scaler_x.parquet", res["feature_names"])
    else:
        from ml_core.model import MLPAdapter
        from ml_core.scaling import DailyCrossSectionMAD
        adapter = MLPAdapter().load(model_dir)
        sx = DailyCrossSectionMAD()

    cfg = PipelineConfig(
        sources=c["sources"],
        neu_sources=c.get("neu_sources"),
        has_factor_policy=HasFactorPolicy.NONE if c["model"] == "lgbm" else HasFactorPolicy.ALL,
        horizon=c["horizon"],
        feature_order=res["feature_names"],
        exclude_features=c.get("exclude_features"),
    )
    panel = _predict_live(model_dir, adapter, sx, cfg, start=start, end=end)
    panel.to_parquet(pred_dir / "pred_panel_live.parquet")
    logger.info(f"[run {run_id}] 推理面板 → {pred_dir / 'pred_panel_live.parquet'}")

    export_panel(panel, pred_dir / "signals", top_n=top_n, rebuild=rebuild)
    logger.success(f"[run {run_id}] === 信号导出完成 → {pred_dir / 'signals'} ===")


def main() -> None:
    """读配置 + 接 loguru 文件 sink（控制台 + 落盘），保证训练全过程可复看。"""
    c = load_config()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"{c.get('run_id') or ts}_{ts}.log"
    sink = logger.add(
        log_path,
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <7} | {message}",
    )
    logger.info(f"日志落盘：{log_path}")
    try:
        _run(c)
    except Exception:
        logger.exception("实验失败")
        raise
    finally:
        logger.remove(sink)


if __name__ == "__main__":
    main()
