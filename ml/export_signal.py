"""
导出回测可读信号：pred_panel(_live).parquet → 每日排序选股名单 txt
============================================================
回测项目（daily-realtime-backtest-pipeline）的「每日信号」口径：
  signal_dir 下每个交易日一份 `YYYY-MM-DD.txt`，每行 `YYYY-MM-DD_股票代码`，
  行序 = 选股优先级；回测按自己的 top_k 截取。回测只用排序、不用分数，故导出无损。
本脚本把 ŷ 宽表（date × code）逐日按预测分降序、取 top-N 写出。

两种面板来源（--source）：
  live（默认）：pred_panel_live.parquet —— 实盘口径，只过 can_buy_mask，覆盖到最新因子日；
  eval        ：pred_panel.parquet      —— 评估口径，过 label，止于可兑现末日(2026-03-31)。
两种布局（--layout）：
  daily（默认）：每日一份 YYYY-MM-DD.txt（回测实盘读这个）；
  merged       ：单个 signal.txt，全部日期合并（行同样是 YYYY-MM-DD_代码）。

**信号 = 交易流水，append-only 冻结**（daily 布局默认行为）：
  日更只**新增**信号目录里尚不存在的交易日 .txt，**绝不覆盖已有历史信号**——
  保证无前视(每天信号冻结在当日数据口径)、历史可复现、回测稳定。
  确需按当前因子口径重写历史(罕见 reconcile)才显式 `--rebuild`，与因子侧"日更增量 + 周期 reconcile"对称。

用法：
    python -m ml.export_signal --run-id full_gbdt_es200                 # 默认: 只补新增交易日(append-only)
    python -m ml.export_signal --run-id full_gbdt_es200 --rebuild       # 全段重写(显式 reconcile)
    python -m ml.export_signal --run-id full_gbdt --source eval --layout merged
产物：
    FACTOR_REPL_DATA_ROOT/ml/signals/<run_id>/YYYY-MM-DD.txt   （daily）
    FACTOR_REPL_DATA_ROOT/ml/signals/<run_id>/signal.txt       （merged）
"""
from __future__ import annotations

import argparse

import pandas as pd
from loguru import logger

from core import config

_PANEL_FILE = {"live": "pred_panel_live.parquet", "eval": "pred_panel.parquet"}


def export_signal(
    run_id: str = "full_gbdt",
    top_n: int = 500,
    source: str = "live",
    layout: str = "daily",
    start: str | None = None,
    end: str | None = None,
    rebuild: bool = False,
) -> str:
    pred_path = config.ML_PREDICTIONS_DIR / run_id / _PANEL_FILE[source]
    if not pred_path.exists():
        hint = "ml.predict_live" if source == "live" else "ml.run"
        raise FileNotFoundError(f"预测面板不存在：{pred_path}（先跑 python -m {hint} --run-id {run_id}）")

    panel = pd.read_parquet(pred_path)          # (T, N)  index=date, columns=code, 值=ŷ
    panel.index = pd.to_datetime(panel.index)
    if start:
        panel = panel.loc[panel.index >= pd.Timestamp(start)]
    if end:
        panel = panel.loc[panel.index <= pd.Timestamp(end)]
    if panel.empty:
        raise ValueError(f"过滤后无数据：start={start} end={end}（面板可用区间已超出）")

    out_dir = config.ML_SIGNALS_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    def _ranked_codes(row: pd.Series) -> list[str]:
        ranked = row.dropna().sort_values(ascending=False)
        return list(ranked.iloc[:top_n].index) if top_n else list(ranked.index)

    if layout == "merged":
        lines: list[str] = []
        for ts, row in panel.iterrows():
            ds = ts.strftime("%Y-%m-%d")
            lines.extend(f"{ds}_{code}" for code in _ranked_codes(row))
        out_path = out_dir / "signal.txt"
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        logger.success(f"[export] {run_id}({source}/merged): {len(panel)} 天 × top-{top_n} "
                       f"→ {len(lines):,} 行 → {out_path}")
        return str(out_path)

    # daily：每个交易日一份 YYYY-MM-DD.txt
    # 默认 append-only：跳过已存在的日期文件（冻结历史信号 = 无前视、可复现）；
    # --rebuild 才覆盖重写（罕见 reconcile）。
    n_new = n_skip = 0
    for ts, row in panel.iterrows():
        ds = ts.strftime("%Y-%m-%d")
        fpath = out_dir / f"{ds}.txt"
        if fpath.exists() and not rebuild:
            n_skip += 1
            continue
        codes = _ranked_codes(row)
        if not codes:
            continue
        fpath.write_text("\n".join(f"{ds}_{code}" for code in codes) + "\n", encoding="utf-8")
        n_new += 1
    mode = "全段重写" if rebuild else "append-only"
    msg = (f"[export] {run_id}({source}/daily,{mode}): 新增 {n_new} 份"
           + (f"，跳过已存在 {n_skip} 份（历史冻结）" if n_skip else "")
           + f" × top-{top_n} → {out_dir}")
    if n_new:
        logger.success(msg + f" | 新增区间 ~{panel.index.max().date()}")
    else:
        logger.info(msg + "（已最新，无新增交易日）")
    return str(out_dir)


def main() -> None:
    ap = argparse.ArgumentParser(description="导出回测可读信号（每日排序选股名单）")
    ap.add_argument("--run-id", default="full_gbdt")
    ap.add_argument("--top-n", type=int, default=500)
    ap.add_argument("--source", default="live", choices=["live", "eval"])
    ap.add_argument("--layout", default="daily", choices=["daily", "merged"])
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--rebuild", action="store_true",
                    help="全段重写覆盖历史信号（默认 append-only 只补新增交易日；reconcile 才用）")
    args = ap.parse_args()
    export_signal(args.run_id, args.top_n, args.source, args.layout, args.start, args.end, args.rebuild)


if __name__ == "__main__":
    main()
