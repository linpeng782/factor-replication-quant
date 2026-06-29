"""CLI 入口：串联 dataset → train → predict → export。

训练 / 预测已解耦：训练走 can_train 池（含 label，二分类训练）；
预测走 can_predict 池（has_factor & can_buy，**不需要 label**）→
信号能覆盖到因子最新日（不再被 labels 截止日卡死）。

用法:
    # 训练（按原流程，存 model.pt + test_report.json）
    python ml_ht/run.py --train

    # 预测：全历史 can_predict 池 → 信号
    python ml_ht/run.py --predict

    # 预测：指定日期区间（适合回测补缺）
    python ml_ht/run.py --predict --start-date 2026-04-01 --end-date 2026-06-26

    # 预测：日频增量，只跑最后 N 个交易日
    python ml_ht/run.py --predict --latest-n 1

    # 自定义信号输出目录（默认 ml/ht_dquant/signals/）
    python ml_ht/run.py --predict --latest-n 5 --out-dir /tmp/ht_signals_test

    # 训练后立即预测（一起跑）
    python ml_ht/run.py --train --predict
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from core import config
from .dataset import (
    load_long_table,
    load_long_table_for_predict,
    latest_n_dates,
    build_predict_set,
    preprocess,
    build_loaders,
)
from .model import StockMLP
from .train import train
from .predict import predict
from .export_signal import export_signals
from .metrics import daily_rank_ic, daily_long_short, yearly_report

MODEL_DIR = config.ML_HT_BASE / "models"     # rq→ml/ht/models  dquant→ml/ht_dquant/models
SIGNAL_DIR = config.ML_HT_BASE / "signals"
RUNS_DIR = config.ML_HT_BASE / "runs"


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


def cmd_predict(
    model,
    device: str,
    top_k: int | None,
    start_date: str | None,
    end_date: str | None,
    latest_n: int | None,
    out_dir: Path | None,
):
    """预测 + 导出信号。

    走独立的预测流水线（与训练解耦）：
      1. 读长表（按日期区间 IO 层裁剪；--latest-n 取末 N 天）
      2. 派生 can_predict = has_factor & can_buy
      3. 逐日 MAD去极值 + zscore（与训练同款规则，统计量自 can_predict 池）
      4. 喂模型出 P(Y=1)
      5. 按日排序、写信号 txt

    特点：
      - 不读 label、不做 train/valid/test 时间切分
      - 不受 labels 截止日制约，信号覆盖到因子最新日
      - 历史天统计量池略广于训练，预期信号 ≈ bit 对齐；末端纯新增
    """
    logger.info("=== 预测（解耦路径：can_predict 池）===")

    # 解析日期区间
    if latest_n is not None:
        ds_list = latest_n_dates(latest_n)
        start_date, end_date = ds_list[0], ds_list[-1]
        logger.info(f"  --latest-n {latest_n} → 区间 {start_date} ~ {end_date}")
    else:
        logger.info(f"  区间: {start_date or '*'} ~ {end_date or '*'}")

    logger.info("  加载长表（仅截取区间，IO 层裁剪）...")
    features, has_factor, can_buy, factor_names, dates, stocks = load_long_table_for_predict(
        start_date=start_date, end_date=end_date,
    )
    n_loaded = len(features)
    n_can_predict = int((has_factor & can_buy).sum())
    logger.info(f"  加载 {n_loaded:,} 行 → can_predict={n_can_predict:,}")
    if n_can_predict == 0:
        logger.error("can_predict 池为空，无法预测。检查日期区间或长表覆盖。")
        return

    logger.info("  逐日标准化（MAD+5σ → zscore，can_predict 池）...")
    X_pred, pred_dates, pred_stocks = build_predict_set(
        features, has_factor, can_buy, dates, stocks,
    )
    del features, has_factor, can_buy
    logger.info(f"  标准化后样本: {len(X_pred):,} 行")

    logger.info("  喂模型出概率...")
    date_strs, stocks_per_day, probs_per_day = predict(
        model, X_pred, pred_dates, pred_stocks, device=device,
    )

    target_dir = out_dir if out_dir is not None else SIGNAL_DIR
    logger.info("=== 导出信号 ===")
    export_signals(date_strs, stocks_per_day, probs_per_day, target_dir, top_k=top_k)


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
    # 预测专用：日期区间 / 日频增量 / 输出目录
    ap.add_argument("--start-date", default=None, help="predict 起始日 YYYY-MM-DD")
    ap.add_argument("--end-date", default=None, help="predict 结束日 YYYY-MM-DD")
    ap.add_argument("--latest-n", type=int, default=None, help="predict 仅最后 N 个交易日（覆盖 --start/--end）")
    ap.add_argument("--out-dir", type=Path, default=None, help="信号输出目录（默认 ml/ht_dquant/signals/）")
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

        cmd_predict(
            model,
            device=args.device,
            top_k=args.top_k,
            start_date=args.start_date,
            end_date=args.end_date,
            latest_n=args.latest_n,
            out_dir=args.out_dir,
        )


if __name__ == "__main__":
    main()
