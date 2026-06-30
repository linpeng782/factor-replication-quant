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
    latest_n_dates,
    preprocess,
    build_loaders,
)
from .model import StockMLP
from .train import train
from .export_signal import export_signals
from .metrics import daily_rank_ic, daily_long_short, yearly_report

MODEL_DIR = config.ML_HT_BASE / "models"     # rq→ml/ht/models  dquant→ml/ht_dquant/models
SIGNAL_DIR = config.ML_HT_BASE / "signals"
RUNS_DIR = config.ML_HT_BASE / "runs"


def _mlp_feature_order() -> list[str]:
    """MLP serving 因子顺序：优先读 models/feature_names.json（固化的训练列序），
    回退长表 schema（向后兼容）。**serving 必须与训练同序**，否则输入列错位、静默出垃圾
    （alpha158 长表列序为 KLEN/KLOW/… 而非字母序，绝不能用 sorted 顶替）。"""
    import json
    fp = MODEL_DIR / "feature_names.json"
    if fp.exists():
        return json.loads(fp.read_text())["features"]
    import pyarrow.parquet as pq
    meta = {"has_factor", "can_buy", "has_label", "can_train"}
    lt = config.ML_HT_BASE / "alpha158_long.parquet"
    return [c for c in pq.read_schema(lt).names if c not in meta]


def _alpha158_source() -> str:
    """ml_ht 因子源目录名（跟随 ML_HT_BACKEND：ht_dquant→alpha158-dquant，ht→alpha158）。"""
    return "alpha158-dquant" if "dquant" in config.ML_HT_BASE.name else "alpha158"


def _panel_to_daily(panel):
    """(date×stock) 概率面板 → export_signals 所需的逐日 (date_strs, stocks, probs)。"""
    date_strs, stocks_per_day, probs_per_day = [], [], []
    for ts, row in panel.iterrows():
        r = row.dropna()
        if r.empty:
            continue
        date_strs.append(ts.strftime("%Y-%m-%d"))
        stocks_per_day.append(r.index.to_numpy())
        probs_per_day.append(r.to_numpy(dtype=np.float32))
    return date_strs, stocks_per_day, probs_per_day


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
    """预测 + 导出信号（因子组装走 ml_core 统一管线，脱离预物化长表）。

    流程（与训练解耦；已逐元素验证与旧长表路径一致，max_abs≈6e-8）：
      1. ml_core 底座 can_buy（不要 label）→ 现算因子矩阵（has_factor=ALL，缺一即排除）
         ⇒ 预测池 = can_buy & has_factor，等价旧 can_predict
      2. 逐日 MAD去极值(±5) + zscore（DailyCrossSectionMAD，统计量自预测池当日截面）
      3. MLP 出 P(Y=1) → (date×stock) 概率面板
      4. 按日排序、写信号 txt（沿用 export_signals，输出格式不变）

    要点：
      - feature_order 取自 models/feature_names.json（固化训练列序）→ 杜绝列错位
      - 不读 label、不做时间切分；信号覆盖到因子最新日
      - --latest-n 仍用长表日历定位末 N 天（日期解析未改，cron 语义不变）
    """
    from ml_core.features import HasFactorPolicy
    from ml_core.model import MLPAdapter
    from ml_core.pipeline import PipelineConfig
    from ml_core.pipeline import predict_live as core_predict_live
    from ml_core.scaling import DailyCrossSectionMAD

    logger.info("=== 预测（ml_core 现算组装：can_predict = can_buy & has_factor）===")

    # 解析日期区间（--latest-n 仍按长表日历取末 N 天，保持 cron 语义）
    if latest_n is not None:
        ds_list = latest_n_dates(latest_n)
        start_date, end_date = ds_list[0], ds_list[-1]
        logger.info(f"  --latest-n {latest_n} → 区间 {start_date} ~ {end_date}")
    else:
        logger.info(f"  区间: {start_date or '*'} ~ {end_date or '*'}")

    feat_order = _mlp_feature_order()
    cfg = PipelineConfig(sources=[_alpha158_source()], neu_sources=None,
                         has_factor_policy=HasFactorPolicy.ALL, horizon=20,
                         feature_order=feat_order)
    # 模型：训练后直传则包装其网络；否则从 MODEL_DIR 加载既有 stock_mlp.pt（自动剥 net. 前缀）
    adapter = MLPAdapter(n_features=len(feat_order), device=device)
    if model is not None:
        adapter.net = model.net
    else:
        adapter.load(MODEL_DIR)
        logger.info(f"加载模型: {MODEL_DIR / 'stock_mlp.pt'}")

    panel = core_predict_live(MODEL_DIR, adapter, DailyCrossSectionMAD(), cfg,
                              start=start_date, end=end_date)
    date_strs, stocks_per_day, probs_per_day = _panel_to_daily(panel)
    if not date_strs:
        logger.error("预测池为空，无法导出信号。检查日期区间或因子覆盖。")
        return

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
        # 模型加载交给 cmd_predict（经 MLPAdapter，自动兼容 stock_mlp.pt 的 net. 前缀）；
        # 训练后直跑时 model 为新训 StockMLP，直接复用其网络。
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
