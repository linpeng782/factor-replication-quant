"""
ml_core.concat_rolling_signals —— 拼接逐年滚动训练信号为一条连续样本外曲线
============================================================
7 年滚动训练（year=2019~2025，预测 2020~2026）的信号按年份拼接：
  year=2019 → 2020 全年
  year=2020 → 2021 全年
  ...
  year=2025 → 2026-01~06（截至因子覆盖末日）

拼接后输出：
  1. 合并 signals 目录（每日 txt，append-only，无重叠）
  2. 合并 pred_panel 面板（date×stock 宽表，连续分数）
  3. 拼接报告（每年覆盖区间 + 总天数）

用法：python -m ml_core.concat_rolling_signals
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
from loguru import logger

import config

YEARS = [2019, 2020, 2021, 2022, 2023, 2024, 2025]
OUT_RUN_ID = "lgbm_rolling_concat_2019_2025"


def main() -> None:
    out_model_dir = config.ML_PREDICTIONS_DIR / OUT_RUN_ID
    out_signals = out_model_dir / "signals"
    out_panel_path = out_model_dir / "pred_panel_live.parquet"

    # 清理旧产物
    if out_signals.exists():
        shutil.rmtree(out_signals)
    out_signals.mkdir(parents=True, exist_ok=True)

    panels = []
    total_days = 0
    report = []

    for year in YEARS:
        pred_year = year + 1
        run_id = f"lgbm_rolling_{year}"
        sig_dir = config.ML_PREDICTIONS_DIR / run_id / "signals"
        panel_path = config.ML_PREDICTIONS_DIR / run_id / "pred_panel_live.parquet"

        if not sig_dir.exists():
            logger.warning(f"跳过 {run_id}：信号目录不存在")
            continue

        # 信号 txt 复制
        txts = sorted(sig_dir.glob("*.txt"))
        for t in txts:
            shutil.copy2(t, out_signals / t.name)

        # 预测面板
        if panel_path.exists():
            p = pd.read_parquet(panel_path)
            panels.append(p)

        dates = sorted([t.stem for t in txts])
        report.append((year, pred_year, dates[0], dates[-1], len(txts)))
        total_days += len(txts)
        logger.info(f"year={year} → 预测{pred_year} | {dates[0]} ~ {dates[-1]} | {len(txts)} 份")

    # 合并面板
    if panels:
        concat = pd.concat(panels, axis=0).sort_index()
        # 去重（保留最后一份，理论上无重叠）
        concat = concat[~concat.index.duplicated(keep="last")]
        concat.to_parquet(out_panel_path)
        logger.info(f"合并面板 {concat.shape} → {out_panel_path}")

    # 报告
    logger.success(f"=== 拼接完成 ===")
    logger.info(f"总天数: {total_days} | 信号目录: {out_signals}")
    print("\n拼接报告：")
    print(f"{'year':>6} {'pred':>6} {'起':>12} {'止':>12} {'天数':>6}")
    for y, py, d0, d1, n in report:
        print(f"{y:>6} {py:>6} {d0:>12} {d1:>12} {n:>6}")
    all_dates = sorted((out_signals.glob("*.txt")))
    print(f"\n总计: {len(all_dates)} 份信号 | {all_dates[0].stem} ~ {all_dates[-1].stem}")


if __name__ == "__main__":
    main()
