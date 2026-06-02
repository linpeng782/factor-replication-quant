"""
CLI 入口：串起 dataset → select(Stage1) → train(Stage2) → evaluate 全流程
============================================================
用法：
    python -m ml.run                          # 全203因子 → GBDT重要性选64 → 重训 → test模型IC
    python -m ml.run --top-k 64 --run-id exp001
    python -m ml.run --date-sample 5 --max-features 60   # 冒烟
产物落 FACTOR_REPL_DATA_ROOT/ml/{models,predictions}/<run_id>/
"""
from __future__ import annotations

import argparse
from datetime import datetime

from loguru import logger

from core import config
from ml.dataset import build_dataset
from ml.evaluate import model_ic, predict_panel
from ml.select import save_selection, select_by_gbdt_importance, select_by_shap
from ml.train import train_gbdt


def main() -> None:
    ap = argparse.ArgumentParser(description="LightGBM 因子合成（两阶段：筛选→合成）")
    ap.add_argument("--sources", nargs="*", default=None)
    ap.add_argument("--select-method", default="gbdt", choices=["gbdt", "shap"])
    ap.add_argument("--top-k", type=int, default=64)
    ap.add_argument("--date-sample", type=int, default=None)
    ap.add_argument("--max-features", type=int, default=None)
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")

    sp = build_dataset(args.sources, date_sample=args.date_sample, max_features=args.max_features)

    # Stage 1: 筛选（只用 train+valid）
    selector = select_by_shap if args.select_method == "shap" else select_by_gbdt_importance
    selected, scores, _ = selector(sp.X_train, sp.y_train, sp.X_valid, sp.y_valid, top_k=args.top_k)
    model_dir = config.ML_MODELS_DIR / run_id
    save_selection(selected, scores, f"{args.select_method}_gain", model_dir)

    # Stage 2: 仅用选出的因子重训
    model, best_it, _ = train_gbdt(sp.X_train[selected], sp.y_train, sp.X_valid[selected], sp.y_valid)
    model.save_model(str(model_dir / "model.txt"), num_iteration=best_it)

    # 评估：样本外模型 IC
    pred = predict_panel(model, sp.X_test[selected], best_it)
    ic = model_ic(pred, sp.meta["excess_raw"])
    logger.success(f"[run {run_id}] 选 {len(selected)} 因子 | 样本外模型 IC 均值={ic.mean():+.4f} "
                   f"ICIR={ic.mean()/ic.std():+.3f} t={ic.mean()/ic.std()*len(ic)**0.5:+.2f} 天数={len(ic)}")
    pred_dir = config.ML_PREDICTIONS_DIR / run_id; pred_dir.mkdir(parents=True, exist_ok=True)
    pred.to_parquet(pred_dir / "pred_panel.parquet")
    ic.to_frame("ic").to_parquet(pred_dir / "ic_series.parquet")
    return ic


if __name__ == "__main__":
    main()
