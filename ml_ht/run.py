"""CLI 入口：串联 dataset → train → predict → export。

用法:
    python ml_ht/run.py --train              # 训练模型（落 run 记录）
    python ml_ht/run.py --predict            # 预测 + 导出信号
    python ml_ht/run.py --train --predict    # 训练后立即预测
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
from loguru import logger

from core import config
from .dataset import load_long_table, preprocess, build_loaders
from .model import StockMLP
from .train import train
from .predict import predict
from .export_signal import export_signals
from .metrics import daily_rank_ic, daily_long_short, yearly_report

MODEL_DIR = config.ML_ROOT / "ht" / "models"
SIGNAL_DIR = config.ML_ROOT / "ht" / "signals"
RUNS_DIR = config.ML_ROOT / "ht" / "runs"


def _test_report(model, data, device: str, run_dir: Path) -> dict:
    """best 模型在 test 上逐年评估，落 test_report.json。"""
    import numpy as np
    model.eval()
    X = torch.from_numpy(data["X_test"])
    probs = np.empty(len(X), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(X), 8192):
            probs[i : i + 8192] = torch.sigmoid(
                model(X[i : i + 8192].to(device))
            ).cpu().numpy().ravel()

    ret, dates = data["ret_test"], data["test_dates"]
    overall_ic = daily_rank_ic(probs, ret, dates)
    overall_ls = daily_long_short(probs, ret, dates)
    by_year = yearly_report(probs, ret, dates)

    report = {"overall": {**overall_ic, **overall_ls}, "by_year": by_year}
    (run_dir / "test_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))

    logger.info("=== Test 逐年报告 ===")
    logger.info(f"  overall: IC={overall_ic['ic_mean']:.4f} ICIR={overall_ic['icir']:.2f} "
                f"L-S={overall_ls['long_short']:.4f}")
    for y, m in by_year.items():
        logger.info(f"  {y}: IC={m['ic_mean']:.4f} ICIR={m['icir']:.2f} "
                    f"L-S={m['long_short']:.4f} pos={m['ic_pos_ratio']:.2f} ({m['n_days']}d)")
    return report


def cmd_train(device: str, lr: float, patience: int, max_epochs: int, batch_size: int):
    """训练流程。"""
    run_dir = RUNS_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = {"device": device, "lr": lr, "patience": patience,
           "max_epochs": max_epochs, "batch_size": batch_size,
           "started": datetime.now().isoformat()}
    (run_dir / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2))
    logger.info(f"Run 目录: {run_dir}")

    logger.info("=== 加载数据 ===")
    features, labels, can_train, factor_names, dates, stocks = load_long_table()
    logger.info(f"长表: {len(features):,} 行, {len(factor_names)} 因子")

    logger.info("=== 预处理 ===")
    data = preprocess(features, labels, can_train, dates, stocks)
    logger.info(
        f"train={data['n_train']:,} | valid={data['n_valid']:,} | test={data['n_test']:,}"
    )
    del features, labels, can_train

    logger.info("=== 构建 DataLoader ===")
    loaders = build_loaders(data, batch_size=batch_size)

    logger.info("=== 训练 ===")
    model = StockMLP(n_features=len(factor_names))
    logger.info(f"模型参数: {sum(p.numel() for p in model.parameters()):,}")
    model = train(
        model, loaders["train"], loaders["valid"],
        val_ret=data["ret_valid"], val_dates=data["valid_dates"],
        device=device, lr=lr, patience=patience, max_epochs=max_epochs,
        run_dir=run_dir,
    )

    # 保存（run_dir + 最新指针）
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), run_dir / "model.pt")
    torch.save(model.state_dict(), MODEL_DIR / "stock_mlp.pt")
    logger.success(f"模型保存: {run_dir / 'model.pt'}")

    # 训练后 test 逐年评估
    _test_report(model, data, device, run_dir)

    return model, data


def cmd_predict(model, data, device: str, top_k: int | None):
    """预测 + 导出信号。"""
    logger.info("=== 预测 ===")
    date_strs, stocks_per_day, probs_per_day = predict(
        model, data["X_test"], data["test_dates"], data["test_stocks"],
        device=device,
    )

    logger.info("=== 导出信号 ===")
    export_signals(date_strs, stocks_per_day, probs_per_day, SIGNAL_DIR, top_k=top_k)


def main():
    ap = argparse.ArgumentParser(description="ml_ht: 华泰 FCNN 选股")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--predict", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--max-epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--top-k", type=int, default=None, help="每天只输出前 K 只")
    args = ap.parse_args()

    if not args.train and not args.predict:
        ap.error("请指定 --train 或 --predict")

    model, data = None, None

    if args.train:
        model, data = cmd_train(
            device=args.device, lr=args.lr, patience=args.patience,
            max_epochs=args.max_epochs, batch_size=args.batch_size,
        )

    if args.predict:
        if model is None:
            model = StockMLP()
            model_path = MODEL_DIR / "stock_mlp.pt"
            model.load_state_dict(torch.load(model_path, map_location=args.device))
            model = model.to(args.device)
            logger.info(f"加载模型: {model_path}")

            features, labels, can_train, factor_names, dates, stocks = load_long_table()
            data = preprocess(features, labels, can_train, dates, stocks)
            del features, labels, can_train

        cmd_predict(model, data, device=args.device, top_k=args.top_k)


if __name__ == "__main__":
    main()
