"""
开源证券(kysec) 三组因子 IC 序列相关性
==========================================
对 raw/kysec 下三组因子（paper_03_smartmoney / paper_05_apm / paper_27_microstructure）：
  ① 每因子算日横截面 Rank IC 序列（spearman, vs forward_return_5d，方向校正）
  ② 算 IC 序列两两相关（同时赚钱? = alpha 冗余），并按三组分块汇总组内/组间均值

IC 口径与 scripts/factor_correlation.py 完全一致（复用 compute_ic_series + 方向校正）。

用法：python scripts/kysec_ic_correlation.py            # 5d, 全区间
      python scripts/kysec_ic_correlation.py --horizon 10d --start 2016-01-01
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import config  # noqa: E402
from alpha_shared.evaluation.ic import compute_ic_series  # noqa: E402

RAW_KYSEC = Path("/nfs/ofs-prediction/peterzhenglinpeng/factors/raw/kysec")
GROUPS = ["paper_03_smartmoney", "paper_05_apm", "paper_27_microstructure"]
OUT_ROOT = config.INVENTORY_ROOT / "correlation"


def discover() -> dict[str, list[tuple[str, Path]]]:
    """{组: [(因子名, 路径)]}"""
    out = {}
    for g in GROUPS:
        out[g] = [(p.stem, p) for p in sorted((RAW_KYSEC / g).glob("*.parquet"))]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", default="5d")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    groups = discover()
    flat = [(g, n, p) for g, items in groups.items() for n, p in items]
    factors = [n for _, n, _ in flat]
    fac2grp = {n: g for g, n, _ in flat}
    logger.info(
        f"因子: {len(factors)} 个 "
        + " | ".join(f"{g}={len(v)}" for g, v in groups.items())
    )

    fr = pd.read_parquet(config.LABELS_DIR / f"forward_return_{args.horizon}.parquet")
    fr.index = pd.to_datetime(fr.index)

    def work(item):
        g, n, p = item
        df = pd.read_parquet(p)
        df.index = pd.to_datetime(df.index)
        if args.start or args.end:
            df = df.loc[args.start : args.end]
        ic_raw = compute_ic_series(df, fr, method="spearman")
        d = -1 if float(ic_raw.dropna().mean()) < 0 else 1  # 方向校正
        ic = ic_raw if d == 1 else -ic_raw
        s = float(ic.std())
        icir = abs(float(ic.mean()) / s) if s else 0.0
        return n, ic, float(ic.mean()), icir

    logger.info(f"并行算日 IC 序列（{args.horizon}, spearman, {args.workers} 线程）...")
    ic_map, icmean, icir = {}, {}, {}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for n, ic, m, ir in ex.map(work, flat):
            ic_map[n], icmean[n], icir[n] = ic, m, ir

    # IC 序列相关矩阵（按组排序）
    C = pd.DataFrame(ic_map).dropna(how="all").corr()
    C = C.loc[factors, factors]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUT_ROOT / f"{ts}__kysec_3groups"
    out_dir.mkdir(parents=True, exist_ok=True)
    C.to_parquet(out_dir / "ic_corr_matrix.parquet")

    # 组内 / 组间均值
    def block_mean(rows, cols, diag):
        sub = C.loc[rows, cols].values
        if diag:
            n = len(rows)
            return (sub.sum() - n) / (n * (n - 1)) if n > 1 else float("nan")
        return float(np.nanmean(sub))

    L = [
        f"# 开源证券三组因子 — IC序列相关性 ({args.horizon}, spearman, 方向校正)",
        "",
        f"- 区间 {args.start or 'all'}~{args.end or 'all'}；IC定义同 factor_correlation.py",
        f"- 因子数 **{len(factors)}**："
        + "，".join(f"{g}({len(v)})" for g, v in groups.items()),
        "",
        "## 各因子 IC 概览（方向校正后 mean IC / ICIR）",
        "",
        "| 组 | 因子 | mean IC | ICIR |",
        "|---|---|---|---|",
    ]
    for g, n, _ in flat:
        L.append(f"| {g} | {n} | {icmean[n]:+.4f} | {icir[n]:.3f} |")

    L += ["", "## 组内 / 组间 IC序列相关均值", "", "| | " + " | ".join(GROUPS) + " |",
          "|" + "---|" * (len(GROUPS) + 1)]
    grp_facs = {g: [n for _, n, _ in flat if fac2grp[n] == g] for g in GROUPS}
    for gi in GROUPS:
        row = [gi]
        for gj in GROUPS:
            row.append(f"{block_mean(grp_facs[gi], grp_facs[gj], gi == gj):+.3f}")
        L.append("| " + " | ".join(row) + " |")
    L += ["", "（对角=组内平均ρ，非对角=组间平均ρ；方向已统一为正IC）", ""]

    # 高相关对（|ρ|>0.6, 跨因子）
    pairs = []
    for i in range(len(factors)):
        for j in range(i + 1, len(factors)):
            r = C.iloc[i, j]
            if abs(r) > 0.6:
                pairs.append((abs(r), factors[i], factors[j], r))
    pairs.sort(reverse=True)
    L += [f"## 高相关因子对（|ρ|>0.6，共 {len(pairs)} 对）", ""]
    for ar, a, b, r in pairs[:40]:
        ga, gb = fac2grp[a], fac2grp[b]
        tag = "组内" if ga == gb else f"**跨组** {ga.split('_')[1]}×{gb.split('_')[1]}"
        L.append(f"- {r:+.2f}  {a} × {b}  ({tag})")
    (out_dir / "summary.md").write_text("\n".join(L))

    # 热力图（按组排序，组边界画框）
    n = len(factors)
    fig, ax = plt.subplots(figsize=(max(10, n * 0.45),) * 2)
    im = ax.imshow(C.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(factors, rotation=90, fontsize=6)
    ax.set_yticklabels(factors, fontsize=6)
    bnd, acc = [0], 0
    for g in GROUPS:
        acc += len(grp_facs[g]); bnd.append(acc)
    for s, e in zip(bnd[:-1], bnd[1:]):
        ax.add_patch(plt.Rectangle((s - .5, s - .5), e - s, e - s,
                     fill=False, edgecolor="black", lw=1.6))
    plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="ρ (signed IC)")
    ax.set_title(f"kysec 3 groups — IC-series correlation ({args.horizon})", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_dir / "ic_corr_heatmap.png", dpi=130, bbox_inches="tight")
    plt.close()

    logger.info(f"✅ 完成 → {out_dir}")
    print("\n" + (out_dir / "summary.md").read_text())


if __name__ == "__main__":
    main()
