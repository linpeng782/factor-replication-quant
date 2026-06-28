"""信号文件导出：按日排序 → txt 文件，对接外部回测系统。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from loguru import logger


def export_signals(
    date_strs: list[str],
    stocks_per_day: list[np.ndarray],
    probs_per_day: list[np.ndarray],
    out_dir: str | Path,
    top_k: int | None = None,
) -> None:
    """每天按 P(Y=1) 降序排列，写信号文件。

    输出格式: out_dir/YYYY-MM-DD.txt，每行 YYYY-MM-DD_stockcode
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_written = 0
    for date_str, stocks, probs in zip(date_strs, stocks_per_day, probs_per_day):
        order = probs.argsort()[::-1]
        if top_k is not None:
            order = order[:top_k]

        path = out_dir / f"{date_str}.txt"
        with open(path, "w") as f:
            for idx in order:
                f.write(f"{date_str}_{stocks[idx]}\n")
        n_written += 1

    logger.success(f"信号导出完成: {n_written} 天 → {out_dir}/")
