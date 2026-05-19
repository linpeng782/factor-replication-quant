"""
因子 Sanity Check：分布统计 + 异常定位
产出：/nfs/ofs-prediction/peterzhenglinpeng/factor-replication/<factor>/sanity_check.txt
"""

from pathlib import Path
from typing import List, Tuple
import pandas as pd
import numpy as np


def analyze_factor(factor_df: pd.DataFrame, factor_name: str, output_dir: Path) -> str:
    """
    分析因子宽表，生成 sanity_check.txt
    """
    lines = []
    lines.append("=" * 60)
    lines.append(f"因子 Sanity Check: {factor_name}")
    lines.append(f"宽表维度: {factor_df.shape[0]} 天 × {factor_df.shape[1]} 只")
    lines.append("=" * 60)

    arr = factor_df.to_numpy(dtype=np.float64)
    total = arr.size

    # ── 1. 异常值统计 ──
    posinf_mask = np.isposinf(arr)
    neginf_mask = np.isneginf(arr)
    nan_mask = np.isnan(arr)
    finite_mask = ~(nan_mask | posinf_mask | neginf_mask)
    zero_mask = finite_mask & (arr == 0)

    n_posinf = int(posinf_mask.sum())
    n_neginf = int(neginf_mask.sum())
    n_nan = int(nan_mask.sum())
    n_zero = int(zero_mask.sum())
    n_valid = int(finite_mask.sum())

    lines.append("\n【异常值统计】")
    lines.append(f"  总单元格:     {total:,}")
    lines.append(f"  NaN:          {n_nan:,} ({n_nan/total:.2%})")
    lines.append(f"  +Inf:         {n_posinf:,}")
    lines.append(f"  -Inf:         {n_neginf:,}")
    lines.append(f"  零值:         {n_zero:,}")
    lines.append(f"  有效值:       {n_valid:,}")

    # ── 2. 分布（仅有效值） ──
    valid_arr = arr[finite_mask]
    if len(valid_arr) > 0:
        lines.append("\n【分布统计（排除 Inf/NaN）】")
        lines.append(f"  min:     {valid_arr.min():.6f}")
        lines.append(f"  max:     {valid_arr.max():.6f}")
        lines.append(f"  mean:    {valid_arr.mean():.6f}")
        lines.append(f"  median:  {np.median(valid_arr):.6f}")
        lines.append(f"  std:     {valid_arr.std():.6f}")
        lines.append(f"  p1:      {np.percentile(valid_arr, 1):.6f}")
        lines.append(f"  p5:      {np.percentile(valid_arr, 5):.6f}")
        lines.append(f"  p25:     {np.percentile(valid_arr, 25):.6f}")
        lines.append(f"  p75:     {np.percentile(valid_arr, 75):.6f}")
        lines.append(f"  p95:     {np.percentile(valid_arr, 95):.6f}")
        lines.append(f"  p99:     {np.percentile(valid_arr, 99):.6f}")
    else:
        lines.append("\n【分布统计】无有效值")

    cols_arr = factor_df.columns.to_numpy()
    idx_arr = factor_df.index.to_numpy()

    # ── 3. 定位 +Inf ──
    if n_posinf > 0:
        lines.append(f"\n【+Inf 定位】共 {n_posinf} 个")
        pos = _locate_by_mask(factor_df, posinf_mask, cols_arr, idx_arr)
        for stock, date in pos[:20]:
            lines.append(f"  {date}  {stock}")
        if len(pos) > 20:
            lines.append(f"  ... 还有 {len(pos)-20} 个")

    # ── 4. 定位 -Inf ──
    if n_neginf > 0:
        lines.append(f"\n【-Inf 定位】共 {n_neginf} 个")
        pos = _locate_by_mask(factor_df, neginf_mask, cols_arr, idx_arr)
        for stock, date in pos[:20]:
            lines.append(f"  {date}  {stock}")
        if len(pos) > 20:
            lines.append(f"  ... 还有 {len(pos)-20} 个")

    # ── 5. 定位 Min / Max（排除 Inf） ──
    if n_valid > 0:
        finite_arr = np.where(finite_mask, arr, np.nan)
        col_mins = np.nanmin(finite_arr, axis=0)
        col_maxs = np.nanmax(finite_arr, axis=0)
        min_val = float(np.nanmin(col_mins))
        max_val = float(np.nanmax(col_maxs))

        lines.append(f"\n【Min 定位】{min_val:.6f}")
        min_pos = _locate_value(factor_df, finite_arr, min_val, cols_arr, idx_arr)
        for stock, date in min_pos[:10]:
            lines.append(f"  {date}  {stock}")
        if len(min_pos) > 10:
            lines.append(f"  ... 还有 {len(min_pos)-10} 个")

        lines.append(f"\n【Max 定位】{max_val:.6f}")
        max_pos = _locate_value(factor_df, finite_arr, max_val, cols_arr, idx_arr)
        for stock, date in max_pos[:10]:
            lines.append(f"  {date}  {stock}")
        if len(max_pos) > 10:
            lines.append(f"  ... 还有 {len(max_pos)-10} 个")

    # ── 6. 零值定位 ──
    if n_zero > 0:
        lines.append(f"\n【零值定位】共 {n_zero} 个")
        zero_pos = _locate_by_mask(factor_df, zero_mask, cols_arr, idx_arr)
        for stock, date in zero_pos[:10]:
            lines.append(f"  {date}  {stock}")
        if len(zero_pos) > 10:
            lines.append(f"  ... 还有 {len(zero_pos)-10} 个")

    lines.append("\n" + "=" * 60)

    report_text = "\n".join(lines)

    # 保存到代码目录下的 sanity_check_report/，以因子名命名
    report_dir = Path("/nfs/volume-1593-1/peterzhenglinpeng/factor-repilcation-quant/sanity_check_report")
    report_dir.mkdir(parents=True, exist_ok=True)
    output_path = report_dir / f"{factor_name}.txt"
    output_path.write_text(report_text, encoding="utf-8")
    print(f"\n📄 Sanity check 报告已保存: {output_path}")
    return report_text


def _locate_by_mask(df: pd.DataFrame, mask: np.ndarray, cols_arr: np.ndarray, idx_arr: np.ndarray) -> List[Tuple[str, str]]:
    """根据 bool mask 定位 (stock, date)，使用 numpy 快速索引"""
    rows, cols = np.where(mask)
    return [(str(cols_arr[c]), str(idx_arr[r])[:10]) for r, c in zip(rows, cols)]


def _locate_value(df: pd.DataFrame, arr: np.ndarray, target: float, cols_arr: np.ndarray, idx_arr: np.ndarray) -> List[Tuple[str, str]]:
    """定位值约等于 target 的 (stock, date)"""
    mask = np.isclose(arr, target, rtol=1e-10, atol=1e-12) & ~np.isnan(arr)
    return _locate_by_mask(df, mask, cols_arr, idx_arr)
