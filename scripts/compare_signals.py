"""
ml_ht 新旧信号对比器
====================

回答重构后（predict 用 can_predict 池、去 label 依赖）信号差异是否符合预期。

对比三个维度（仅重叠日期）：
  1. 股票池：每日 old/new 池集合差异（new 应 ≥ old，因 can_predict ⊇ can_train）
  2. 排序：在两版都出现的股票上比较排名——Spearman 秩相关 + Top-K 重叠率
  3. 新增股票侵入 Top-K 的情况：新增股票里有多少进了 Top-100/500/1000

输入：--old-dir / --new-dir 各含 YYYY-MM-DD.txt 信号文件（每行 "<date>_<stock>"）
输出：终端统计 + 双图（daily Spearman 时序、daily Top-K 重叠）
      + summary.md（百分位表）

用法：
    python scripts/compare_signals.py \
        --old-dir /nfs/ofs-prediction/peterzhenglinpeng/ml/ht_dquant/dquant-signals \
        --new-dir /nfs/ofs-prediction/peterzhenglinpeng/ml/ht_dquant/signals-new

设计自包含：不依赖 config，路径全由命令行参数控制。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ────────────────────── 信号文件读写 ──────────────────────


def load_signals(dir_: Path) -> dict[str, list[str]]:
    """读 YYYY-MM-DD.txt → {date_str: [stock_codes in ranked order]}。"""
    out: dict[str, list[str]] = {}
    for f in sorted(dir_.glob("*.txt")):
        date_str = f.stem
        stocks = [line.rsplit("_", 1)[-1].strip() for line in f.read_text().splitlines() if line.strip()]
        out[date_str] = stocks
    return out


def parse_date(s: str) -> pd.Timestamp:
    return pd.Timestamp(s.replace("-", ""))


# ────────────────────── 每日比较 ──────────────────────


def _ranks_zero_based(stocks: list[str]) -> dict[str, int]:
    return {s: i for i, s in enumerate(stocks)}


def compare_one_day(old: list[str], new: list[str], top_ks: tuple[int, ...]):
    """一天的比较。

    Returns dict:
        n_old, n_new, n_inter, n_only_new
        spearman         — 交集股票上 rank 的 Spearman ρ
        topk_overlap[k]  — top-K 集合交并比 Jaccard（k 是绝对值，min(实际数, k)）
        topk_new_infiltration[k] — 新增股票进入 new Top-K 的数量
        days_with_no_overlap — bool (handled by caller)
    """
    ro = _ranks_zero_based(old)
    rn = _ranks_zero_based(new)
    set_old, set_new = set(old), set(new)
    inter = set_old & set_new
    only_new = set_new - set_old  # 新池多出来的股票
    out = {
        "n_old": len(old),
        "n_new": len(new),
        "n_inter": len(inter),
        "n_only_new": len(only_new),
    }

    # Spearman (rank correlation) on intersection
    if len(inter) >= 5:
        x = np.array([ro[s] for s in inter], dtype=np.float64)
        y = np.array([rn[s] for s in inter], dtype=np.float64)
        out["spearman"] = _spearman_naive(x, y)
        out["mean_rank_delta"] = float(np.mean(np.abs(x - y)))
    else:
        out["spearman"] = np.nan
        out["mean_rank_delta"] = np.nan

    # Top-K 对比（new 里前 K 名 vs old 里前 K 名）
    for k in top_ks:
        ko = min(k, len(old))
        kn = min(k, len(new))
        if ko == 0 or kn == 0:
            out[f"topk_overlap_{k}"] = np.nan
            out[f"topk_infiltration_{k}"] = np.nan
            continue
        old_topk = set(old[:ko])
        new_topk = set(new[:kn])
        u = old_topk | new_topk
        out[f"topk_overlap_{k}"] = len(old_topk & new_topk) / len(u) if u else np.nan
        # 新增股票里有多少是新预测池的"独有股"（而不是两池都有但新池里排前了）
        # 这里只统计：出现在 new 的 Top-K、且不在 old 池里的股票数量
        out[f"topk_infiltration_{k}"] = sum(1 for s in new[:kn] if s not in set_old)
    return out


def _spearman_naive(x: np.ndarray, y: np.ndarray) -> float:
    """直接对已经是 rank 的整数序列算 Pearson（即 Spearman）。"""
    x = x - x.mean(); y = y - y.mean()
    d = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / d) if d > 0 else np.nan


# ────────────────────── 主流程 ──────────────────────


def main():
    ap = argparse.ArgumentParser(description="ml_ht 新旧信号对比")
    ap.add_argument("--old-dir", type=Path, required=True)
    ap.add_argument("--new-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("output/signal_compare"))
    ap.add_argument("--top-ks", type=int, nargs="*", default=[100, 500, 1000])
    args = ap.parse_args()

    top_ks = tuple(args.top_ks)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"加载 old 信号: {args.old_dir}")
    old = load_signals(args.old_dir)
    logger.info(f"  {len(old)} 天 | {min(old)} ~ {max(old)}")

    logger.info(f"加载 new 信号: {args.new_dir}")
    new = load_signals(args.new_dir)
    logger.info(f"  {len(new)} 天 | {min(new)} ~ {max(new)}")

    overlap_dates = sorted(set(old) & set(new))
    only_new_dates = sorted(set(new) - set(old))
    only_old_dates = sorted(set(old) - set(new))
    logger.info(f"重叠日期: {len(overlap_dates)} 天 ({overlap_dates[0]} ~ {overlap_dates[-1]})")
    if only_new_dates:
        logger.info(f"仅 new 有的日期(尾部新增): {len(only_new_dates)} ({only_new_dates[0]} ~ {only_new_dates[-1]})")
    if only_old_dates:
        logger.warning(f"仅 old 有的日期: {len(only_old_dates)} → 这种情况不该发生")

    # 逐日比较
    rows = []
    for ds in overlap_dates:
        comp = compare_one_day(old[ds], new[ds], top_ks)
        comp["date"] = ds
        rows.append(comp)
    df = pd.DataFrame(rows)
    df["date_ts"] = pd.to_datetime(df["date"])
    df = df.sort_values("date_ts").reset_index(drop=True)

    # ── 汇总统计 ──
    def stats(s):
        s = s.dropna()
        return dict(
            mean=float(s.mean()),
            median=float(s.median()),
            p10=float(s.quantile(0.10)),
            p90=float(s.quantile(0.90)),
            min=float(s.min()),
            max=float(s.max()),
            n=int(len(s)),
        )

    logger.info("=" * 72)
    logger.info("重叠期差异统计")
    logger.info("=" * 72)
    summary_lines = [
        f"# ml_ht 信号对比\n",
        f"- old: `{args.old_dir}`  ({len(old)} 天, {min(old)} ~ {max(old)})",
        f"- new: `{args.new_dir}`  ({len(new)} 天, {min(new)} ~ {max(new)})",
        f"- 重叠: {len(overlap_dates)} 天 ({overlap_dates[0]} ~ {overlap_dates[-1]})",
        f"- 仅 new 有的尾部日期: {len(only_new_dates)} 天",
        "",
        "## 每日差异百分位表\n",
        f"| 指标 | mean | median | p10 | p90 | min | max |",
        f"|---|---|---|---|---|---|---|",
    ]
    metric_specs = [
        ("n_old",          "old 池规模"),
        ("n_new",          "new 池规模"),
        ("n_only_new",     "new 独有股票数"),
        ("spearman",       "Spearman 秩相关"),
        ("mean_rank_delta","平均 |rankΔ|"),
    ]
    for k in top_ks:
        metric_specs.append((f"topk_overlap_{k}", f"Top-{k} Jaccard 重叠"))
    for k in top_ks:
        metric_specs.append((f"topk_infiltration_{k}", f"Top-{k} 内新增股数"))

    for key, label in metric_specs:
        if key not in df:
            continue
        st = stats(df[key])
        logger.info(
            f"  {label:<22} | mean={st['mean']:.4f} median={st['median']:.4f} "
            f"p10={st['p10']:.4f} p90={st['p90']:.4f}"
        )
        summary_lines.append(
            f"| {label} | {st['mean']:.4f} | {st['median']:.4f} | "
            f"{st['p10']:.4f} | {st['p90']:.4f} | {st['min']:.4f} | {st['max']:.4f} |"
        )

    # 新增日期样本数
    if only_new_dates:
        n_new_days_data = pd.DataFrame([
            {"date": ds, "n_new": len(new[ds])}
            for ds in only_new_dates
        ])
        logger.info("-" * 72)
        logger.info(f"仅 new 有的尾部新增日期（{len(only_new_dates)} 天）池规模:")
        for _, r in n_new_days_data.head(5).iterrows():
            logger.info(f"  {r['date']}: {r['n_new']} 股")
        if len(n_new_days_data) > 5:
            logger.info(f"  ... (共 {len(n_new_days_data)} 天)")

    # ── 图 ──
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    ax = axes[0]
    ax.plot(df["date_ts"], df["spearman"], lw=0.5, alpha=0.7, label="Spearman ρ")
    if df["spearman"].notna().any():
        ax.axhline(df["spearman"].median(), color="r", ls="--", lw=1, label=f"median={df['spearman'].median():.4f}")
    ax.set_ylabel("Spearman rho\n(rank corr on intersection)")
    ax.set_title("Old vs New signal divergence over time")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(df["date_ts"], df["n_old"], lw=0.5, alpha=0.6, label="old pool")
    ax.plot(df["date_ts"], df["n_new"], lw=0.5, alpha=0.6, label="new pool")
    ax.plot(df["date_ts"], df["n_only_new"], lw=0.8, alpha=0.8, color="g", label="new-only stocks")
    ax.set_ylabel("# stocks")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[2]
    for k in top_ks:
        col = f"topk_overlap_{k}"
        if col in df:
            ax.plot(df["date_ts"], df[col], lw=0.6, alpha=0.7, label=f"Top-{k} Jaccard")
    ax.set_ylabel("Top-K Jaccard overlap")
    ax.set_xlabel("Date")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plot_path = args.out_dir / "signal_compare.png"
    fig.savefig(plot_path, dpi=140)
    logger.success(f"图: {plot_path}")

    # 落 csv 便于复盘
    csv_path = args.out_dir / "daily_compare.csv"
    df.to_csv(csv_path, index=False)
    logger.success(f"每日明细: {csv_path}")

    # 落 summary
    summary_path = args.out_dir / "summary.md"
    summary_path.write_text("\n".join(summary_lines))
    logger.success(f"摘要: {summary_path}")

    # 判定
    logger.info("=" * 72)
    if df["spearman"].median() >= 0.995:
        logger.success(f"✅ Spearman 中位数 {df['spearman'].median():.4f} ≥ 0.995 → 信号基本不变，方案通过")
    elif df["spearman"].median() >= 0.98:
        logger.info(f"✅/⚠️  Spearman 中位数 {df['spearman'].median():.4f} ∈ [0.98, 0.995) → 信号几乎一致，可接受")
    else:
        logger.warning(f"⚠️  Spearman 中位数 {df['spearman'].median():.4f} < 0.98 → 排序有明显变化，请人工检查")


if __name__ == "__main__":
    main()