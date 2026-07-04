"""
CLI 入口：串起 dataset → select(Stage1) → train(Stage2) → evaluate 全流程
============================================================
用法：
    python -m ml.run                          # 全因子 → GBDT重要性选64 → 重训 → test模型IC
    python -m ml.run --top-k 64 --run-id exp001
    python -m ml.run --date-sample 5 --max-features 60   # 冒烟
    python -m ml.run --num-threads 32         # 并行跑多个实验时降低线程数（默认64）
                                              # 经验值：单机128核，N个并行 → --num-threads 128//N
产物：
    FACTOR_REPL_DATA_ROOT/ml/models/<run_id>/       model.txt + selected_features.json
    FACTOR_REPL_DATA_ROOT/ml/predictions/<run_id>/  pred_panel + ic_series
    <repo>/ml/logs/<run_id>_<时间戳>.log            训练全记录 + 入选因子重要性（方便查看）
"""
from __future__ import annotations

import argparse
from datetime import datetime

import pandas as pd
from loguru import logger

import config
from ml.dataset import build_dataset
from ml.evaluate import model_ic, predict_panel
from ml.select import save_selection, select_by_gbdt_importance, select_by_shap
from ml.train import train_gbdt


def main() -> None:
    ap = argparse.ArgumentParser(description="LightGBM 因子合成（两阶段：筛选→合成）")
    ap.add_argument("--sources", nargs="*", default=None)
    ap.add_argument("--neu-sources", nargs="*", default=None,
                    help="从 factors/neu 读取的因子源（与 --sources 互补，如 guosen/co_momentum）")
    ap.add_argument("--select-method", default="gbdt", choices=["gbdt", "shap"])
    ap.add_argument("--top-k", type=int, default=64)
    ap.add_argument("--date-sample", type=int, default=None)
    ap.add_argument("--max-features", type=int, default=None)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--num-threads", type=int, default=None,
                    help="LightGBM 线程数（默认用 train.py 的 DEFAULT_PARAMS=64）。"
                         "并行跑 N 个实验时建议传 128//N，避免 CPU 超额订阅。")
    ap.add_argument("--seed", type=int, default=None,
                    help="随机种子（默认用 train.py 的 DEFAULT_PARAMS=42）。"
                         "多种子实验时传不同值，如 1/42/123/2024/7。")
    args = ap.parse_args()
    launch_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = args.run_id or launch_ts

    # 日志：控制台 + 直接落盘 logs/<run_id>_<时间戳>.log（扁平，无 run_id 子目录）
    # 文件名带启动时间戳 → 即使复用 run_id，每次训练也是独立文件，绝不覆盖/追加
    config.ML_LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = config.ML_LOGS_DIR / f"{run_id}_{launch_ts}.log"
    log_sink = logger.add(log_path, level="INFO",
                          format="{time:YYYY-MM-DD HH:mm:ss} | {level: <7} | {message}")
    # num_threads / seed 覆盖：只在显式传参时生效，否则沿用 DEFAULT_PARAMS
    thread_params = {}
    if args.num_threads:
        thread_params["num_threads"] = args.num_threads
    if args.seed is not None:
        thread_params["seed"] = args.seed
    if thread_params:
        logger.info(f"[run {run_id}] 参数覆盖: {thread_params}")

    logger.info(f"[run {run_id}] 启动 @ {launch_ts} | 日志: {log_path} | "
                f"参数: sources={args.sources} neu_sources={args.neu_sources}")

    sp = build_dataset(args.sources, neu_sources=args.neu_sources, date_sample=args.date_sample, max_features=args.max_features)

    # Stage 1: 筛选（只用 train+valid）
    selector = select_by_shap if args.select_method == "shap" else select_by_gbdt_importance
    selected, scores, _ = selector(sp.X_train, sp.y_train, sp.X_valid, sp.y_valid,
                                   top_k=args.top_k, params=thread_params or None)
    model_dir = config.ML_MODELS_DIR / run_id
    save_selection(selected, scores, f"{args.select_method}_gain", model_dir,
                   sources=args.sources, neu_sources=args.neu_sources)
    # 存 RobustZScore 尺子(median/scale, train段拟合)：实盘推理(predict_live)复用同一把尺，口径一致
    model_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"median": sp.scaler_x.median_, "scale": sp.scaler_x.scale_}).to_parquet(
        model_dir / "scaler_x.parquet")
    # 特征选择结果写入 run.log（入选 top-k + 重要性），不另产 csv
    logger.info(f"[select-{args.select_method}] 入选 top-{len(selected)}（按重要性降序）：")
    for i, f in enumerate(selected, 1):
        logger.info(f"    {i:>3}. {f:<40} importance={scores[f]:.2f}")

    # Stage 2: 仅用选出的因子重训
    model, best_it, _ = train_gbdt(sp.X_train[selected], sp.y_train, sp.X_valid[selected], sp.y_valid,
                                   params=thread_params or None, tag="final")
    model.save_model(str(model_dir / "model.txt"), num_iteration=best_it)

    # 评估：样本外模型 IC
    pred = predict_panel(model, sp.X_test[selected], best_it)
    ic = model_ic(pred, sp.meta["excess_raw"])
    logger.success(f"[run {run_id}] 选 {len(selected)} 因子 | 样本外模型 IC 均值={ic.mean():+.4f} "
                   f"ICIR={ic.mean()/ic.std():+.3f} t={ic.mean()/ic.std()*len(ic)**0.5:+.2f} 天数={len(ic)}")
    pred_dir = config.ML_PREDICTIONS_DIR / run_id; pred_dir.mkdir(parents=True, exist_ok=True)
    pred.to_parquet(pred_dir / "pred_panel.parquet")
    ic.to_frame("ic").to_parquet(pred_dir / "ic_series.parquet")
    logger.remove(log_sink)
    return ic


if __name__ == "__main__":
    main()
