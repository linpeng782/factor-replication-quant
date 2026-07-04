"""
ml_core 推理 / 导信号入口 —— 读 predict_config.yaml，对既有模型出回测信号
============================================================
独立于训练（run.py）：训练低频、人手动、产模型；推理高频、可 cron 日更、消费模型。
两个维度是同一条路径的配置差异（不是两个脚本）：
  全量重生成（新模型）：latest_n=null + 全区间（+ rebuild=true 覆盖旧信号）
  日更增量（老模型）  ：latest_n=N + append-only（自动补缺、冻结历史）

自包含：模型口径全读训练落盘的 run_meta.json（model/sources/策略/feature_order）→ 零 train/serve 漂移。
（老模型无 run_meta.json 时，从 selected_features.json/feature_names.json + 模型文件类型回退拼装。）
覆盖守门：近 N 日面板非空骤降则告警/中止，防因子面板陈旧 → 信号静默退化。
运行：python -m ml_core.predict
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from loguru import logger

from core import config
from ml_core.features import HasFactorPolicy, discover_features
from ml_core.model import LGBMAdapter, MLPAdapter
from ml_core.pipeline import PipelineConfig, predict_live
from ml_core.scaling import DailyCrossSectionMAD, WholeSetRobustZ
from ml_core.signals import export_panel

CONFIG_PATH = Path(__file__).parent / "predict_config.yaml"
LOG_DIR = Path(__file__).parent / "logs"


def load_config(path: Path = CONFIG_PATH) -> dict:
    """读 predict_config.yaml（推理参数来源）。"""
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _load_meta(model_dir: Path) -> dict:
    """推理口径真相源：优先 run_meta.json；老模型回退拼装（兼容历史产物）。"""
    mp = model_dir / "run_meta.json"
    if mp.exists():
        return json.loads(mp.read_text())
    # 回退：探测模型类型 + 从 selected/feature_names 拼 sources/feature_order
    model = "lgbm" if (model_dir / "model.txt").exists() else "mlp"
    feats = sources = neu = None
    sf = model_dir / "selected_features.json"
    if sf.exists():
        m = json.loads(sf.read_text())
        feats, sources, neu = m["features"], m.get("sources"), m.get("neu_sources")
    fn = model_dir / "feature_names.json"
    if feats is None and fn.exists():
        feats = json.loads(fn.read_text())["features"]
    if feats is None or sources is None:
        raise FileNotFoundError(
            f"{model_dir} 缺 run_meta.json 且无法回退（需 selected_features.json 提供 sources/features）"
        )
    logger.warning(
        f"[predict] {model_dir.name} 无 run_meta.json，按老产物回退拼装（model={model}）"
    )
    return {
        "model": model,
        "sources": sources,
        "neu_sources": neu,
        "has_factor_policy": "none" if model == "lgbm" else "all",
        "horizon": 20,
        "select_method": None,
        "feature_order": feats,
    }


def _build_adapter_scaler(model_dir: Path, meta: dict):
    """按 run_meta 恢复 (adapter, standardizer, policy)；尺子用训练段 fit 的 scaler_x。"""
    feats = meta["feature_order"]
    if meta["model"] == "lgbm":
        adapter = LGBMAdapter().load(model_dir)
        sx = WholeSetRobustZ.load(model_dir / "scaler_x.parquet", feats)
    else:
        adapter = MLPAdapter().load(model_dir)  # 输入维度按权重 shape 自适应
        sx = DailyCrossSectionMAD()  # 无状态，无需 load
    return adapter, sx, HasFactorPolicy(meta["has_factor_policy"])


def _pathmap(meta: dict) -> dict:
    """入选因子名 → parquet 路径（raw + neu，与训练同口径）。"""
    raw = discover_features(meta["sources"])
    neu = (
        discover_features(meta["neu_sources"], stage="neu")
        if meta.get("neu_sources")
        else {}
    )
    return {**neu, **raw}


def _factor_end(pathmap: dict, feats: list[str]) -> pd.Timestamp:
    """入选因子「共同覆盖」的最末交易日（读 index 不读数据，廉价）。"""
    ends = [
        pd.to_datetime(pd.read_parquet(pathmap[f], columns=[]).index).max()
        for f in feats
    ]
    return pd.Timestamp(min(ends))


def _resolve_window(pc: dict, meta: dict) -> tuple[str | None, str]:
    """解析推理区间：latest_n（日更）优先；否则 start/end（全量）。end=null → 因子共同覆盖末日。"""
    pathmap = _pathmap(meta)
    feats = meta["feature_order"]
    end = pd.Timestamp(pc["end"]) if pc.get("end") else _factor_end(pathmap, feats)
    if pc.get("latest_n"):
        cal = pd.to_datetime(
            pd.read_parquet(pathmap[feats[0]], columns=[]).index
        ).sort_values()
        cal = cal[cal <= end]
        start = cal[-int(pc["latest_n"])]
        logger.info(
            f"[predict] latest_n={pc['latest_n']} → 区间 {start.date()}~{end.date()}（日更增量）"
        )
        return str(start.date()), str(end.date())
    logger.info(f"[predict] 全量区间 {pc.get('start') or '*'}~{end.date()}")
    return pc.get("start"), str(end.date())


def _coverage_guard(
    panel: pd.DataFrame, recent: int, drop: float, strict: bool
) -> None:
    """面板新鲜度守门：近 recent 日每日非空股票数 vs 历史基线，骤降则告警/中止。"""
    nn = panel.notna().sum(axis=1).to_numpy()
    if len(nn) <= recent * 3:
        logger.info(f"[predict][coverage] 区间过短（{len(nn)}日），跳过覆盖守门")
        return
    baseline = float(np.median(nn[:-recent]))
    bad = [
        (d.date(), int(n))
        for d, n in zip(panel.index[-recent:], nn[-recent:])
        if baseline > 0 and n < drop * baseline
    ]
    if bad:
        logger.warning(
            f"[predict][coverage] ⚠️ 近 {recent} 日非空骤降（基线~{int(baseline)}/日）：{bad}"
        )
        if strict:
            raise RuntimeError(
                "[predict][coverage] 覆盖骤降且 strict_coverage=true，中止以防污染信号"
            )
    else:
        logger.info(
            f"[predict][coverage] ✅ 近 {recent} 日覆盖正常（基线~{int(baseline)}/日）"
        )


def _run(pc: dict) -> None:
    run_id = pc["model_run_id"]
    model_dir = config.ML_MODELS_DIR / run_id
    if not model_dir.exists():
        raise FileNotFoundError(f"模型目录不存在：{model_dir}")
    meta = _load_meta(model_dir)
    logger.info(
        f"=== ml_core 推理 model_run_id={run_id} | 模型={meta['model']} "
        f"| 因子={len(meta['feature_order'])} ==="
    )

    adapter, sx, policy = _build_adapter_scaler(model_dir, meta)
    start, end = _resolve_window(pc, meta)
    cfg = PipelineConfig(
        sources=meta["sources"],
        neu_sources=meta.get("neu_sources"),
        has_factor_policy=policy,
        horizon=meta["horizon"],
        feature_order=meta["feature_order"],
    )
    panel = predict_live(model_dir, adapter, sx, cfg, start=start, end=end)

    _coverage_guard(
        panel,
        pc.get("coverage_recent", 10),
        pc.get("coverage_drop", 0.6),
        pc.get("strict_coverage", False),
    )

    # 存预测面板（供集成等下游消费；信号 txt 只有排名丢了分数，面板保留连续分）
    pred_dir = config.ML_PREDICTIONS_DIR / run_id
    pred_dir.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(pred_dir / "pred_panel_live.parquet")
    logger.info(f"[predict] 面板已存 → {pred_dir / 'pred_panel_live.parquet'}")

    signal_dir = pc.get("signal_dir") or (
        config.ML_PREDICTIONS_DIR / run_id / "signals"
    )
    export_panel(
        panel, signal_dir, top_n=pc.get("top_n", 500), rebuild=pc.get("rebuild", False)
    )
    logger.success(f"=== 推理完成。信号 → {signal_dir} ===")


def main() -> None:
    """读配置 + 接 loguru 文件 sink（控制台 + 落盘）。"""
    pc = load_config()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"predict_{pc['model_run_id']}_{ts}.log"
    sink = logger.add(
        log_path,
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <7} | {message}",
    )
    logger.info(f"日志落盘：{log_path}")
    try:
        _run(pc)
    except Exception:
        logger.exception("推理失败")
        raise
    finally:
        logger.remove(sink)


if __name__ == "__main__":
    main()
