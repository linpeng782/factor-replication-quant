"""
ml_core.run_rolling —— 滚动训练入口
============================================================
读 rolling_config.yaml → run_rolling_train → 保存模型 → predict_live 预测 year+1。

用法：python -m ml_core.run_rolling
"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import yaml
from loguru import logger

from core import config
from ml_core.features import HasFactorPolicy
from ml_core.model import LGBMAdapter
from ml_core.pipeline import PipelineConfig, predict_live
from ml_core.rolling import (
    RollingConfig,
    run_rolling_train,
    run_rolling_train_time,
    save_rolling_model,
)

# 支持环境变量 ML_CORE_ROLLING_CONFIG 指定独立 config（批量并行训练用）
CONFIG_PATH = Path(os.environ.get("ML_CORE_ROLLING_CONFIG",
                                  str(Path(__file__).parent / "rolling_config.yaml")))
LOG_DIR = Path(__file__).parent / "logs"


def load_config(path: Path = CONFIG_PATH) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _rolling_cfg(c: dict) -> RollingConfig:
    return RollingConfig(
        year=c["year"],
        train_years_back=c.get("train_years_back", 10),
        train_sample_ratio=c.get("train_sample_ratio", 0.8),
        seed=c.get("seed", 42),
        sources=c["sources"],
        neu_sources=c.get("neu_sources"),
        horizon=c.get("horizon", 20),
        exclude_features=c.get("exclude_features"),
        select_method=c.get("select_method"),
        top_k=c.get("top_k", 64),
        lgbm_params=c.get("lgbm") or {},
        split_mode=c.get("split_mode", "stock"),
        valid_months=c.get("valid_months", 12),
        embargo_months=c.get("embargo_months", 1),
    )


def _run(c: dict) -> None:
    run_id = c.get("run_id") or f"lgbm_rolling_{c['year']}"
    rcfg = _rolling_cfg(c)
    logger.info(f"=== ml_core 滚动训练 run_id={run_id} | year={rcfg.year} → 预测 {rcfg.year + 1} "
                f"| split_mode={rcfg.split_mode} ===")

    # 1. 训练（按 split_mode 分流：stock=老按股票切 / time=方案1+2 时间滚动）
    if rcfg.split_mode == "time":
        result = run_rolling_train_time(rcfg)
    elif rcfg.split_mode == "stock":
        result = run_rolling_train(rcfg)
    else:
        raise ValueError(f"未知 split_mode={rcfg.split_mode!r}（仅 stock/time）")

    # 2. 保存模型
    model_dir = save_rolling_model(result, rcfg, run_id)

    # 3. 推理预测 year+1（predict_live 复用现有管线）
    pred_year = rcfg.year + 1
    start = f"{pred_year}-01-01"
    end = f"{pred_year}-12-31"

    # 重新加载模型（确保 predict_live 用的是磁盘上的产物，和实盘一致）
    adapter = LGBMAdapter().load(model_dir)
    from ml_core.scaling import WholeSetRobustZ
    sx = WholeSetRobustZ.load(model_dir / "scaler_x.parquet", result["feature_names"])

    pcfg = PipelineConfig(
        sources=rcfg.sources,
        neu_sources=rcfg.neu_sources,
        has_factor_policy=HasFactorPolicy.NONE,
        horizon=rcfg.horizon,
        feature_order=result["feature_names"],
        exclude_features=rcfg.exclude_features,
    )
    panel = predict_live(model_dir, adapter, sx, pcfg, start=start, end=end)

    # 4. 存面板 + 导信号
    pred_dir = config.ML_PREDICTIONS_DIR / run_id
    pred_dir.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(pred_dir / "pred_panel_live.parquet")
    logger.info(f"[rolling] 预测面板 → {pred_dir / 'pred_panel_live.parquet'}")

    from ml_core.signals import export_panel
    export_panel(panel, pred_dir / "signals", top_n=500, rebuild=True)
    logger.success(f"=== 滚动训练完成。模型={model_dir} | 预测={pred_dir} ===")


def main() -> None:
    c = load_config()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"{c.get('run_id') or ts}_{ts}.log"
    sink = logger.add(log_path, level="INFO",
                      format="{time:YYYY-MM-DD HH:mm:ss} | {level: <7} | {message}")
    logger.info(f"日志落盘：{log_path}")
    try:
        _run(c)
    except Exception:
        logger.exception("滚动训练失败")
        raise
    finally:
        logger.remove(sink)


if __name__ == "__main__":
    main()
