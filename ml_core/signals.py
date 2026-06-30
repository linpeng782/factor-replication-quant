"""
ml_core.signals —— 回测可读信号导出（append-only 冻结历史，统一两条线口径）
============================================================
口径采用 ml.export_signal 的鲁棒做法（优于 ml_ht 的覆盖重写）：
  - 每个交易日一份 YYYY-MM-DD.txt，每行 `YYYY-MM-DD_股票代码`，行序=预测分降序选股优先级。
  - **append-only**：日更只新增尚不存在的交易日，绝不覆盖历史信号（无前视、可复现、回测稳定）。
  - 仅 --rebuild 时全段重写（罕见 reconcile）。
回测只用排序不用分数，故导出无损。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger


def panel_from_daily(date_strs: list[str], stocks_per_day: list[np.ndarray],
                     scores_per_day: list[np.ndarray]) -> pd.DataFrame:
    """把逐日 (日期, 股票数组, 分数数组) 拼成 (date × stock) 宽表 ŷ 面板（MLP 预测输出 → 面板）。"""
    series = {}
    for ds, stocks, scores in zip(date_strs, stocks_per_day, scores_per_day):
        series[pd.Timestamp(ds)] = pd.Series(scores, index=stocks)
    panel = pd.DataFrame(series).T.sort_index()
    return panel


def export_panel(panel: pd.DataFrame, out_dir: str | Path, top_n: int | None = 500,
                 rebuild: bool = False, start: str | None = None, end: str | None = None) -> str:
    """ŷ 面板 (date×stock) → 每日排序选股 txt（append-only）。"""
    panel = panel.copy()
    panel.index = pd.to_datetime(panel.index)
    if start:
        panel = panel.loc[panel.index >= pd.Timestamp(start)]
    if end:
        panel = panel.loc[panel.index <= pd.Timestamp(end)]
    if panel.empty:
        raise ValueError(f"过滤后无数据：start={start} end={end}")

    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    def _ranked(row: pd.Series) -> list[str]:
        r = row.dropna().sort_values(ascending=False)
        return list(r.iloc[:top_n].index) if top_n else list(r.index)

    n_new = n_skip = 0
    for ts, row in panel.iterrows():
        ds = ts.strftime("%Y-%m-%d")
        fpath = out_dir / f"{ds}.txt"
        if fpath.exists() and not rebuild:
            n_skip += 1
            continue
        codes = _ranked(row)
        if not codes:
            continue
        fpath.write_text("\n".join(f"{ds}_{c}" for c in codes) + "\n", encoding="utf-8")
        n_new += 1
    mode = "全段重写" if rebuild else "append-only"
    logger.success(f"[signals] {mode}：新增 {n_new} 份"
                   + (f"，跳过已存在 {n_skip} 份（历史冻结）" if n_skip else "")
                   + f" × top-{top_n} → {out_dir}")
    return str(out_dir)
