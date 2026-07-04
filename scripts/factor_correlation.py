"""
因子相关性分析（双口径）+ 层次聚类 + ICIR 选代表 + 去重建议
============================================================
对全因子库(cxl+alpha158, neu 版)算两种"冗余"：
  ① 因子值相关 value：因子值是否"长得像"（结构同源）。流式读 + 固定抽样 cell + BLAS GEMM（控内存）。
  ② IC序列相关 ic   ：日 IC 序列是否"同时赚钱"（alpha 冗余）。直读 ic_series_neu_5d.parquet 算 corr。
两种各自层次聚类(ρ>0.6 同簇)，**每簇留 |ICIR| 最高的代表**（读 inventory icir5_n）。

产出 <DATA_ROOT>/factor-inventory/correlation/<ts>__<name>/：
  value_matrix.parquet / value_heatmap.png
  ic_matrix.parquet    / ic_heatmap.png
  summary.md           两种口径的簇划分 / 留-弃建议 / 冗余对

direction 与 ICIR 来自 factor-inventory/latest/inventory.parquet。

用法：python scripts/factor_correlation.py                 # 全 neu 因子(180)
      python scripts/factor_correlation.py --pattern 'QTL*' 'MA*'
      python scripts/factor_correlation.py --factors KMID MA60 CORD60
"""

from __future__ import annotations

import argparse
import fnmatch
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
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import config

NEU_BASE = config.NEU_FACTOR_BASE
INV_PATH = config.INVENTORY_ROOT / "latest" / "inventory.parquet"
IC_SERIES_PATH = config.INVENTORY_ROOT / "latest" / "ic_series_neu_5d.parquet"
OUT_ROOT = config.INVENTORY_ROOT / "correlation"
START, END = "2016-01-01", "2025-12-31"
SAMPLE_CELLS = 300_000  # 值相关抽样 cell 数（控内存：180×300K×4×3 ≈ 0.6GB）
CLUSTER_THRESH = 0.4  # 1-ρ < 0.4 ⇒ ρ > 0.6 同簇
SEED = 0


def discover_factors(patterns, explicit) -> dict[str, Path]:
    """返回 {因子裸名: neu parquet 路径}（裸名当前 180 个唯一）。"""
    pathmap = {p.stem: p for p in sorted(NEU_BASE.glob("*/*/*.parquet"))}
    if explicit:
        return {f: pathmap[f] for f in explicit if f in pathmap}
    if not patterns:
        return pathmap
    return {
        n: p
        for n, p in pathmap.items()
        if any(fnmatch.fnmatch(n, pat) for pat in patterns)
    }


def load_inventory(factors: list[str]) -> tuple[dict, dict]:
    """从 inventory 取 {factor: direction}, {factor: |icir5_n|}。"""
    inv = pd.read_parquet(INV_PATH).set_index("factor")
    dirs, icir = {}, {}
    for f in factors:
        if f in inv.index:
            dirs[f] = int(inv.loc[f, "dir"]) if not pd.isna(inv.loc[f, "dir"]) else 1
            icir[f] = (
                abs(float(inv.loc[f, "icir5_n"]))
                if "icir5_n" in inv.columns and not pd.isna(inv.loc[f, "icir5_n"])
                else 0.0
            )
        else:
            dirs[f], icir[f] = 1, 0.0
    return dirs, icir


def compute_corr(factors: list[str], pathmap: dict, workers: int = 16):
    """一次并行读盘 → 同时算 ①值相关(抽样GEMM) + ②IC序列相关，并现算方向/ICIR（自包含）。

    优化点：
      - **并行读** neu 面板（ThreadPool，I/O 瓶颈）；
      - 网格与 ref 相同则**跳过 reindex**（多数因子同网格，省大块拷贝）；
      - 每个 parquet **只读一遍**，同产 值采样向量 + 日 spearman IC 序列；
      - **方向 + ICIR 现算**（从 IC 序列符号定向）→ 不依赖 inventory，任意因子集即时可算。
    返回 (Cv 值相关, Ci IC序列相关, icir dict)。
    """
    from alpha_shared.evaluation.ic import compute_ic_series

    ref = pd.read_parquet(pathmap[factors[0]])
    ref.index = pd.to_datetime(ref.index)
    ref = ref.loc[START:END]
    dates, stocks = ref.index, ref.columns
    fr5 = pd.read_parquet(config.LABELS_DIR / "forward_return_5d.parquet")
    fr5.index = pd.to_datetime(fr5.index)
    fr5 = fr5.reindex(index=dates, columns=stocks)  # 对齐一次，供所有因子复用
    rng = np.random.default_rng(SEED)
    ri = rng.integers(0, len(dates), SAMPLE_CELLS)
    ci = rng.integers(0, len(stocks), SAMPLE_CELLS)

    def work(f):
        df = pd.read_parquet(pathmap[f])
        df.index = pd.to_datetime(df.index)
        df = df.loc[START:END]
        same = df.index.equals(dates) and df.columns.equals(stocks)
        g = df if same else df.reindex(index=dates, columns=stocks)
        ic_raw = compute_ic_series(g, fr5, method="spearman")
        d = -1 if float(ic_raw.dropna().mean()) < 0 else 1  # 现算方向
        ic = ic_raw if d == 1 else -ic_raw  # direction 校正后日 IC
        a = (d * g.values[ri, ci]).astype(np.float32)  # direction 校正后值采样
        s = float(ic.std())
        icir = abs(float(ic.mean()) / s) if s else 0.0
        return f, a, ic, icir

    logger.info(
        f"  并行读 {len(factors)} 因子（{workers} 线程，单读：值采样+日IC+方向+ICIR）..."
    )
    res = {}
    icir = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for f, a, ic, iciv in ex.map(work, factors):
            res[f] = (a, ic)
            icir[f] = iciv

    # ① 值相关：GEMM（NaN→0 掩码法；a 已 direction 校正）
    N, K = len(factors), SAMPLE_CELLS
    F = np.zeros((N, K), np.float32)
    M = np.zeros((N, K), np.float32)
    F2 = np.zeros((N, K), np.float32)
    for i, f in enumerate(factors):
        a = res[f][0]
        m = ~np.isnan(a)
        a = np.where(m, a, np.float32(0.0))
        F[i], M[i], F2[i] = a, m.astype(np.float32), a * a
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = (F @ F.T) / np.sqrt((F2 @ M.T) * (M @ F2.T))
    rho = np.where(np.isfinite(rho), rho, np.nan)
    np.fill_diagonal(rho, 1.0)
    Cv = pd.DataFrame(rho, index=factors, columns=factors)

    # ② IC 序列相关（IC 已 direction 校正）
    Ci = pd.DataFrame({f: res[f][1] for f in factors}).dropna(how="all").corr()
    return Cv, Ci, icir


def cluster(corr: pd.DataFrame):
    dist = 1.0 - corr.fillna(0).values
    np.fill_diagonal(dist, 0)
    dist = (dist + dist.T) / 2
    Z = linkage(squareform(dist, checks=False), method="average")
    order = [corr.index[i] for i in dendrogram(Z, no_plot=True)["leaves"]]
    labels = fcluster(Z, t=CLUSTER_THRESH, criterion="distance")
    return order, labels


def heatmap(corr, order, labels, out, title, icir=None):
    """聚类排序热力图：保留小字因子标签（放大可读）+ 每个簇块中心标注「代表因子 ×成员数」。"""
    icir = icir or {}
    Mx = corr.loc[order, order].values
    n = len(order)
    lab_o = labels[[corr.index.get_loc(f) for f in order]]
    bnd = [0] + list(np.where(np.diff(lab_o) != 0)[0] + 1) + [n]
    fs = max(3, min(7, int(900 / n)))  # 标签字号随因子数自适应
    fig, ax = plt.subplots(figsize=(min(46, max(10, n * 0.24)),) * 2)
    im = ax.imshow(Mx, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(order, rotation=90, fontsize=fs)
    ax.set_yticklabels(order, fontsize=fs)
    for s, e in zip(bnd[:-1], bnd[1:]):
        ax.add_patch(
            plt.Rectangle(
                (s - 0.5, s - 0.5), e - s, e - s, fill=False, edgecolor="black", lw=1.3
            )
        )
        if e - s >= 2:  # 多成员簇标注代表(最高ICIR)+成员数
            mem = order[s:e]
            rep = max(mem, key=lambda f: icir.get(f, 0.0))
            ax.text(
                (s + e - 1) / 2,
                (s + e - 1) / 2,
                f"{rep}\n×{e - s}",
                ha="center",
                va="center",
                fontsize=min(13, max(6, (e - s) * 0.6)),
                fontweight="bold",
                color="black",
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    facecolor="white",
                    alpha=0.75,
                    edgecolor="none",
                ),
            )
    plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="signed ρ")
    ax.set_title(
        f"{title} (n={n}, {len(bnd)-1} clusters) — 块内: 代表因子 ×成员数", fontsize=12
    )
    plt.tight_layout()
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()


def clusters_of(corr, order, labels):
    lab_o = labels[[corr.index.get_loc(f) for f in order]]
    cl = {}
    for f, l in zip(order, lab_o):
        cl.setdefault(int(l), []).append(f)
    return cl


def summarize(Cv, ov, lv, Ci, oi, li, icir, out, name, nfac):
    def keepers(corr, order, labels):
        cl = clusters_of(corr, order, labels)
        keep, drop = [], []
        for cid, mem in sorted(cl.items()):
            if len(mem) == 1:
                keep.append(mem[0])
            else:
                rep = max(mem, key=lambda f: icir.get(f, 0.0))
                keep.append(rep)
                drop += [m for m in mem if m != rep]
        return cl, keep, drop

    clv, kv, dv = keepers(Cv, ov, lv)
    cli, ki, di = keepers(Ci, oi, li)
    L = [
        f"# 因子相关性分析（双口径）— {name}",
        "",
        f"- 因子数：**{nfac}**；区间 {START}~{END}；聚类 ρ>0.6 同簇；每簇留 |ICIR| 最高代表",
        "",
    ]

    for tag, cl, keep, drop, corr in [
        ("因子值相关 value（长得像）", clv, kv, dv, Cv),
        ("IC序列相关 ic（同时赚钱）", cli, ki, di, Ci),
    ]:
        multi = {c: m for c, m in cl.items() if len(m) > 1}
        L += [
            f"## {tag}",
            "",
            f"- **{nfac} → {len(cl)} 簇**（其中 {len(multi)} 个多成员簇，{len(keep)} 个代表，淘汰 {len(drop)}）",
            "",
        ]
        L.append("### 多成员簇（簇内留⭐代表=最高ICIR，余淘汰）")
        for cid, mem in sorted(multi.items(), key=lambda x: -len(x[1])):
            sub = corr.loc[mem, mem]
            avg = (sub.values.sum() - len(mem)) / (len(mem) * (len(mem) - 1))
            rep = max(mem, key=lambda f: icir.get(f, 0.0))
            ms = "  ".join(
                (
                    f"⭐{m}({icir.get(m,0):.2f})"
                    if m == rep
                    else f"{m}({icir.get(m,0):.2f})"
                )
                for m in mem
            )
            L.append(f"- 簇{cid}（{len(mem)}个, 簇内avgρ={avg:+.2f}）: {ms}")
        L += ["", f"**保留 {len(keep)} 个代表**：`" + "`, `".join(keep) + "`", ""]
    out.write_text("\n".join(L))
    return kv, ki


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--factors", nargs="*", default=[])
    ap.add_argument("--pattern", nargs="*", default=[])
    ap.add_argument("--name", default="all")
    args = ap.parse_args()

    pathmap = discover_factors(args.pattern, args.factors)
    factors = sorted(pathmap)
    if len(factors) < 2:
        raise SystemExit(f"因子数不够：{factors}")
    logger.info(f"factors: {len(factors)}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUT_ROOT / f"{ts}__{args.name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"output: {out_dir}")

    Cv, Ci, icir = compute_corr(factors, pathmap)  # 一次并行读，自包含方向/ICIR
    Cv.to_parquet(out_dir / "value_matrix.parquet")
    Ci.to_parquet(out_dir / "ic_matrix.parquet")
    common = [f for f in factors if f in Ci.columns]
    Ci = Ci.loc[common, common]

    ov, lv = cluster(Cv)
    oi, li = cluster(Ci)
    heatmap(Cv, ov, lv, out_dir / "value_heatmap.png", "Factor-VALUE correlation", icir)
    heatmap(Ci, oi, li, out_dir / "ic_heatmap.png", "IC-SERIES correlation", icir)
    kv, ki = summarize(
        Cv, ov, lv, Ci, oi, li, icir, out_dir / "summary.md", args.name, len(factors)
    )

    logger.info(f"\n✅ 完成 → {out_dir}")
    logger.info(f"   值相关:   {len(factors)} → {len(set(lv))} 簇，留 {len(kv)} 代表")
    logger.info(f"   IC序列相关: {len(common)} → {len(set(li))} 簇，留 {len(ki)} 代表")


if __name__ == "__main__":
    main()
