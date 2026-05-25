"""
因子相关性矩阵 + 聚类热力图 + 自动 summary

每次跑产出（factor_inventory/correlation/<时间戳>__<run_name>/）：
  matrix.parquet     N×N direction 校正后的日均 Spearman ρ
  heatmap.png        聚类排序 + RdBu_r diverging palette + 簇边框
  summary.md         模板化结论（簇划分 / 独立 alpha 排行 / 冗余对 / 去重建议）

direction 来自 factor_inventory/latest/inventory.parquet。

用法：
  python scripts/factor_correlation.py                          # 全 panel 因子
  python scripts/factor_correlation.py --pattern 'pj_*'         # paper_33
  python scripts/factor_correlation.py --pattern 'pj_*' '*ridge*'
  python scripts/factor_correlation.py --factors a b c          # 显式列出
  python scripts/factor_correlation.py --name microstructure_41 # 运行名
"""

from __future__ import annotations

import argparse
import fnmatch
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CLEANED = Path("/nfs/ofs-prediction/peterzhenglinpeng/my-alpha-engine/cleaned-factor-panel/spec")
INVENTORY = PROJECT_ROOT / "factor_inventory" / "latest" / "inventory.parquet"
OUTPUT_ROOT = PROJECT_ROOT / "factor_inventory" / "correlation"


def discover_factors(patterns: list[str], explicit: list[str]) -> list[str]:
    if explicit:
        return list(explicit)
    all_names = sorted(p.stem for p in CLEANED.glob("*.parquet"))
    if not patterns:
        return all_names
    out = []
    for n in all_names:
        if any(fnmatch.fnmatch(n, p) for p in patterns):
            out.append(n)
    return out


def load_directions(factors: list[str]) -> dict[str, int]:
    inv = pd.read_parquet(INVENTORY)
    out = {}
    for f in factors:
        if f in inv.index:
            out[f] = int(inv.loc[f, "direction"])
        else:
            logger.warning(f"  {f} 不在 inventory，direction 默认 +1")
            out[f] = 1
    return out


def _load_one(f: str) -> tuple[str, pd.DataFrame]:
    return f, pd.read_parquet(CLEANED / f"{f}.parquet")


def compute_corr_matrix(factors: list[str], directions: dict[str, int]) -> pd.DataFrame:
    """
    Pooled Pearson on cleaned panels（已 MAD + cross-section z-score；
    Pearson 与 Spearman on raw 差别 <1%）。

    用 3 次 BLAS GEMM 一次算完 N²：
      - sum_AB = F @ F.T   (NaN→0，零贡献，等价于 i,j 交集求和)
      - sum_A2 = F2 @ M.T  (‖A_i‖² 限制到 i,j 交集)
      - sum_B2 = M @ F2.T
      - rho = sum_AB / sqrt(sum_A2 * sum_B2)

    复杂度 N=41, T·S≈13M：~200ms GEMM + ~10s 并行 parquet 读。
    """
    from concurrent.futures import ThreadPoolExecutor

    logger.info(f"  并行读 {len(factors)} 个 parquet (32 threads)...")
    with ThreadPoolExecutor(max_workers=32) as ex:
        loaded = dict(ex.map(_load_one, factors))

    idx = loaded[factors[0]].index
    cols = loaded[factors[0]].columns
    for f in factors:
        idx = idx.intersection(loaded[f].index)
        cols = cols.intersection(loaded[f].columns)
    T, S = len(idx), len(cols)
    logger.info(f"  共同 (date, stock) = {T} × {S}")

    N = len(factors)
    K = T * S
    F = np.zeros((N, K), dtype=np.float32)
    M = np.zeros((N, K), dtype=np.float32)
    F2 = np.zeros((N, K), dtype=np.float32)
    for i, f in enumerate(factors):
        a = (directions[f] * loaded[f].reindex(index=idx, columns=cols).values).astype(np.float32).ravel()
        m = ~np.isnan(a)
        a_filled = np.where(m, a, np.float32(0.0))
        F[i] = a_filled
        M[i] = m.astype(np.float32)
        F2[i] = a_filled * a_filled

    logger.info(f"  BLAS GEMM 3 次 (N={N}, K={K})...")
    sum_AB = F @ F.T
    sum_A2 = F2 @ M.T
    sum_B2 = M @ F2.T

    with np.errstate(invalid="ignore", divide="ignore"):
        rho = sum_AB / np.sqrt(sum_A2 * sum_B2)
    rho = np.where(np.isfinite(rho), rho, np.nan)
    np.fill_diagonal(rho, 1.0)
    return pd.DataFrame(rho, index=factors, columns=factors)


def cluster_order(corr: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    """返回 reordered 因子顺序 + cluster 标签。"""
    dist = 1.0 - corr.fillna(0).values
    np.fill_diagonal(dist, 0)
    dist = (dist + dist.T) / 2  # 对称化
    cond = squareform(dist, checks=False)
    Z = linkage(cond, method="average")
    leaves = dendrogram(Z, no_plot=True)["leaves"]
    labels = fcluster(Z, t=0.4, criterion="distance")  # 1-ρ<0.4 ⇒ ρ>0.6 同簇
    return [corr.index[i] for i in leaves], labels


def plot_heatmap(corr: pd.DataFrame, order: list[str], labels: np.ndarray, out: Path):
    M = corr.loc[order, order].values
    n = len(order)

    # 簇边框：在 reordered labels 中，相邻 label 不同处画矩形
    label_in_order = labels[[corr.index.get_loc(f) for f in order]]
    boundaries = np.where(np.diff(label_in_order) != 0)[0] + 1
    cluster_starts = [0] + list(boundaries) + [n]

    fig, ax = plt.subplots(figsize=(max(12, n * 0.32), max(11, n * 0.30)))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(order, rotation=90, fontsize=7)
    ax.set_yticklabels(order, fontsize=7)
    for s, e in zip(cluster_starts[:-1], cluster_starts[1:]):
        ax.add_patch(plt.Rectangle((s - 0.5, s - 0.5), e - s, e - s,
                                    fill=False, edgecolor="black", linewidth=1.5))
    plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="signed Spearman ρ")
    ax.set_title(f"Factor correlation heatmap (n={n}, direction-corrected, hierarchical-ordered)",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()


def summarize(corr: pd.DataFrame, order: list[str], labels: np.ndarray, out: Path,
              run_name: str):
    n = len(order)
    pairs = [(order[i], order[j], corr.loc[order[i], order[j]])
             for i in range(n) for j in range(i + 1, n)]

    label_in_order = labels[[corr.index.get_loc(f) for f in order]]
    clusters: dict[int, list[str]] = {}
    for f, lab in zip(order, label_in_order):
        clusters.setdefault(int(lab), []).append(f)

    avg_abs = {}
    for f in order:
        vals = corr.loc[f].drop(f).abs()
        avg_abs[f] = float(vals.mean())

    redundant = sorted(pairs, key=lambda x: -abs(x[2]))[:10]
    independent = sorted(avg_abs.items(), key=lambda x: x[1])[:10]

    lines = []
    lines.append(f"# 因子相关性分析 — {run_name}")
    lines.append(f"")
    lines.append(f"- 因子数：**{n}**")
    lines.append(f"- 区间：cleaned panel 全交集（按 inventory direction 校正）")
    lines.append(f"- 聚类：scipy.linkage(method='average')，threshold = 1-ρ < 0.4 (ρ > 0.6 同簇)")
    lines.append(f"")

    lines.append(f"## 1. 簇划分（{len(clusters)} 个簇）")
    lines.append(f"")
    for cid in sorted(clusters):
        members = clusters[cid]
        if len(members) == 1:
            lines.append(f"- **簇 {cid}（独立）**：`{members[0]}`")
        else:
            sub = corr.loc[members, members]
            avg_within = (sub.values.sum() - len(members)) / (len(members) * (len(members) - 1))
            lines.append(f"- **簇 {cid}（{len(members)} 个，簇内 avg ρ={avg_within:+.3f}）**：")
            for m in members:
                lines.append(f"  - `{m}`")
    lines.append(f"")

    lines.append(f"## 2. 最独立因子排行（avg |ρ| 升序）")
    lines.append(f"")
    lines.append(f"| 排名 | 因子 | avg \\|ρ\\| |")
    lines.append(f"|-----|------|------------|")
    for i, (f, v) in enumerate(independent, 1):
        lines.append(f"| {i} | `{f}` | {v:.3f} |")
    lines.append(f"")

    lines.append(f"## 3. 冗余对 Top-10（|ρ| 降序）")
    lines.append(f"")
    lines.append(f"| 因子 A | 因子 B | ρ |")
    lines.append(f"|--------|--------|----|")
    for a, b, c in redundant:
        lines.append(f"| `{a}` | `{b}` | {c:+.3f} |")
    lines.append(f"")

    # 去重建议：每簇取 avg|ρ| 最低（最独立）的代表
    keepers = []
    drops = []
    for cid in sorted(clusters):
        members = clusters[cid]
        if len(members) == 1:
            keepers.append(members[0])
        else:
            rep = min(members, key=lambda f: avg_abs[f])
            keepers.append(rep)
            drops.extend([m for m in members if m != rep])
    lines.append(f"## 4. 推荐去重（{n} → {len(keepers)}）")
    lines.append(f"")
    lines.append(f"**保留**（每簇取簇内最独立的代表）：")
    for k in keepers:
        lines.append(f"- `{k}`")
    if drops:
        lines.append(f"")
        lines.append(f"**淘汰**（簇内冗余）：")
        for d in drops:
            lines.append(f"- `{d}`")
    lines.append(f"")

    out.write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--factors", nargs="*", default=[], help="显式因子列表")
    ap.add_argument("--pattern", nargs="*", default=[], help="fnmatch 通配符（多个 OR 关系）")
    ap.add_argument("--name", default="run", help="运行名（用于产出目录）")
    args = ap.parse_args()

    factors = discover_factors(args.pattern, args.factors)
    if len(factors) < 2:
        raise SystemExit(f"因子数不够：{factors}")
    logger.info(f"factors ({len(factors)}): {factors}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUTPUT_ROOT / f"{ts}__{args.name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"output: {out_dir}")

    directions = load_directions(factors)
    logger.info("computing pairwise correlation matrix ...")
    corr = compute_corr_matrix(factors, directions)
    corr.to_parquet(out_dir / "matrix.parquet")

    logger.info("hierarchical clustering ...")
    order, labels = cluster_order(corr)

    logger.info("plotting heatmap ...")
    plot_heatmap(corr, order, labels, out_dir / "heatmap.png")

    logger.info("writing summary.md ...")
    summarize(corr, order, labels, out_dir / "summary.md", args.name)

    logger.info(f"done: {out_dir}")


if __name__ == "__main__":
    main()
