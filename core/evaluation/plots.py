"""
因子评估可视化
------------------------------------------------------------
每个因子生成一张 2x2 综合报告图（全英文字体）：
    [0, 0] Standardized factor distribution
    [0, 1] Group cumulative NAV (G1~Gg + LongShort)
    [1, 0] Daily IC time series + 20d rolling mean
    [1, 1] Summary table (Ann Return / Ann Vol / Sharpe / Turnover)
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无界面环境
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger


# 统一字体（英文）+ 负号显示
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.unicode_minus": False,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    }
)


def _plot_factor_distribution(ax, factor_clean: pd.DataFrame):
    """左上：标准化因子值分布"""
    values = factor_clean.values.ravel()
    values = values[~np.isnan(values)]
    ax.hist(
        values,
        bins=80,
        density=True,
        color="steelblue",
        alpha=0.75,
        edgecolor="white",
        linewidth=0.3,
    )
    ax.axvline(0, color="red", linestyle="--", alpha=0.6, label="zero")
    ax.set_title("Standardized Factor Distribution")
    ax.set_xlabel("Factor value (z-score)")
    ax.set_ylabel("Density")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    # 标注关键统计
    txt = (
        f"N = {len(values):,}\n"
        f"mean = {values.mean():+.4f}\n"
        f"std  = {values.std():.4f}\n"
        f"min  = {values.min():+.3f}\n"
        f"max  = {values.max():+.3f}"
    )
    ax.text(
        0.02,
        0.97,
        txt,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lemonchiffon", alpha=0.7),
    )


def _get_group_styling(g: int):
    """
    根据分组数 g 生成颜色 / 线宽 / 透明度
    使用 tab10 高对比度离散色，每组颜色差异明显
    """
    # tab10 高饱和度离散色，对比强烈
    tab10 = [
        "#d62728",  # 红
        "#ff7f0e",  # 橙
        "#2ca02c",  # 绿
        "#1f77b4",  # 蓝
        "#9467bd",  # 紫
    ]
    colors = np.array(tab10[:g])
    linewidths = np.full(g, 1.8)
    alphas = np.full(g, 0.85)
    # G1(最差) 和 Gg(最好) 加粗突出
    linewidths[0] = linewidths[-1] = 2.6
    alphas[0] = alphas[-1] = 1.0
    return colors, linewidths, alphas


def _plot_group_nav(ax, group_nav: pd.DataFrame):
    """右上：分组累计净值曲线（仅显示 G1~Gg，不显示 LongShort）"""
    g_cols = [c for c in group_nav.columns if c != "LongShort"]
    g = len(g_cols)
    colors, lws, alphas = _get_group_styling(g)
    for i, col in enumerate(g_cols):
        ax.plot(
            group_nav.index,
            group_nav[col].values,
            label=col,
            color=colors[i],
            linewidth=lws[i],
            alpha=alphas[i],
        )
    ax.axhline(1, color="gray", alpha=0.4, linewidth=0.8)
    ax.set_title("Group Cumulative NAV")
    ax.set_ylabel("NAV")
    ax.legend(loc="best", ncol=2, fontsize=9)
    ax.grid(alpha=0.3)


def _plot_ic_time_series(ax, ic_series: pd.Series, horizon: int, ic_row: pd.Series):
    """左下：日 IC（fill 柱图）+ 累计 IC（右轴）"""
    ic = ic_series.dropna()
    pos = ic.where(ic > 0, 0)
    neg = ic.where(ic < 0, 0)
    # 左轴：日 IC fill
    ax.fill_between(
        ic.index, 0, pos.values, color="red", alpha=0.30, linewidth=0, label="IC > 0"
    )
    ax.fill_between(
        ic.index, 0, neg.values, color="green", alpha=0.30, linewidth=0, label="IC < 0"
    )
    ax.axhline(0, color="black", alpha=0.5, linewidth=0.8)
    ax.set_title(f"Daily IC + Cumulative IC ({horizon}d horizon)")
    ax.set_ylabel("Daily IC")
    ax.grid(alpha=0.3)

    # 右轴：累计 IC（IC 时序的 cumsum）
    ax2 = ax.twinx()
    cum_ic = ic.cumsum()
    ax2.plot(
        cum_ic.index,
        cum_ic.values,
        color="navy",
        linewidth=1.8,
        label="Cumulative IC",
    )
    ax2.axhline(0, color="navy", alpha=0.3, linewidth=0.6, linestyle=":")
    ax2.set_ylabel("Cumulative IC", color="navy")
    ax2.tick_params(axis="y", labelcolor="navy")

    # 合并两轴的图例
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)

    # IC 指标文本框（放右下角避免和右轴字重叠）
    txt = (
        f"IC mean   = {ic_row['ic_mean']:+.4f}\n"
        f"IC std    = {ic_row['ic_std']:.4f}\n"
        f"ICIR      = {ic_row['icir']:+.3f}\n"
        f"t-stat    = {ic_row['ic_t']:+.2f}\n"
        f"positive% = {ic_row['pct_positive']:.1%}\n"
        f"n_days    = {int(ic_row['n_days'])}"
    )
    ax.text(
        0.99,
        0.02,
        txt,
        transform=ax.transAxes,
        va="bottom",
        ha="right",
        fontsize=9,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lemonchiffon", alpha=0.80),
    )


def _plot_summary_table(
    ax,
    layered_result: dict,
    ic_summary_df: pd.DataFrame,
    direction: int = 1,
):
    """
    右下：上半 Layered Summary Table + 下半 IC Summary Table
    两表格样式统一（header 深蓝 + 行高亮）
    """
    ax.axis("off")
    ax.set_title("Summary", fontweight="bold")

    # ==================== 上半：Layered Summary ====================
    summary = layered_result["summary"]
    g_cols = [c for c in summary.index if c != "LongShort"]
    col_labels1 = ["Metric"] + g_cols
    table_data1 = [
        ["Ann Return", *[f"{summary.loc[c, 'ann_return']:+.2%}" for c in g_cols]],
        ["Ann Vol", *[f"{summary.loc[c, 'ann_vol']:.2%}" for c in g_cols]],
        ["Sharpe", *[f"{summary.loc[c, 'sharpe']:+.3f}" for c in g_cols]],
        ["Turnover", *[f"{summary.loc[c, 'mean_turnover']:.1%}" for c in g_cols]],
    ]
    table1 = ax.table(
        cellText=table_data1,
        colLabels=col_labels1,
        loc="upper center",
        cellLoc="center",
        bbox=[0.0, 0.54, 1.0, 0.40],
    )
    table1.auto_set_font_size(False)
    table1.set_fontsize(10)
    for j in range(len(col_labels1)):
        table1[(0, j)].set_facecolor("#4472C4")
        table1[(0, j)].set_text_props(color="white", fontweight="bold")
    for i in range(1, len(table_data1) + 1):
        table1[(i, 0)].set_text_props(fontweight="bold")

    # ==================== 下半：IC Summary ====================
    # 列：Metric + 每个 horizon（比如 2d / 5d）
    ic_cols = ["Metric"] + list(ic_summary_df.index)
    horizons = list(ic_summary_df.index)
    ic_data = [
        ["IC Direction", *[f"{direction:+d}" for _ in horizons]],
        ["IC Mean", *[f"{ic_summary_df.loc[h, 'ic_mean']:+.4f}" for h in horizons]],
        ["IC Std", *[f"{ic_summary_df.loc[h, 'ic_std']:.4f}" for h in horizons]],
        ["ICIR", *[f"{ic_summary_df.loc[h, 'icir']:+.3f}" for h in horizons]],
        ["t-stat", *[f"{ic_summary_df.loc[h, 'ic_t']:+.2f}" for h in horizons]],
        [
            "Positive %",
            *[f"{ic_summary_df.loc[h, 'pct_positive']:.1%}" for h in horizons],
        ],
    ]
    table2 = ax.table(
        cellText=ic_data,
        colLabels=ic_cols,
        loc="lower center",
        cellLoc="center",
        bbox=[0.0, 0.00, 1.0, 0.46],
    )
    table2.auto_set_font_size(False)
    table2.set_fontsize(10)
    for j in range(len(ic_cols)):
        table2[(0, j)].set_facecolor("#4472C4")
        table2[(0, j)].set_text_props(color="white", fontweight="bold")
    # IC Direction 行高亮
    for j in range(len(ic_cols)):
        table2[(1, j)].set_facecolor("#FFE699")
    for i in range(1, len(ic_data) + 1):
        table2[(i, 0)].set_text_props(fontweight="bold")


def plot_factor_report(
    factor_name: str,
    factor_clean: pd.DataFrame,
    ic_series_dict: dict,
    ic_summary_df: pd.DataFrame,
    layered_result: dict,
    output_path,
    primary_ic_horizon: int = 5,
    direction: int = 1,
    figsize=(16, 10),
):
    """
    生成单因子 2x2 综合评估图

    参数:
        factor_name        : 因子名（图标题）
        factor_clean       : (T, N) 清洗后因子（用于分布图，不翻转）
        ic_series_dict     : {horizon -> pd.Series}，已按 direction 翻转后的 IC
        ic_summary_df      : IC 汇总表（已按 direction 翻转后）
        layered_result     : layered_backtest 返回 dict（已按 direction 翻转）
        output_path        : 图片保存路径
        primary_ic_horizon : 左下子图使用哪个 horizon
        direction          : IC 方向（1 或 -1），用于标题标注
        figsize            : 图大小

    注意：factor_distribution 使用原始因子（不翻转），保留分布形态特征
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    title = f"Factor Evaluation Report: {factor_name}"
    if direction == -1:
        title += "   [Flipped: IC direction = -1]"
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.995)

    # [0,0] 分布（始终用原始 factor_clean，不翻转）
    _plot_factor_distribution(axes[0, 0], factor_clean)
    # [0,1] 分组净值
    _plot_group_nav(axes[0, 1], layered_result["group_nav"])
    # [1,0] IC 时序
    h = primary_ic_horizon
    if h in ic_series_dict and f"{h}d" in ic_summary_df.index:
        _plot_ic_time_series(
            axes[1, 0], ic_series_dict[h], h, ic_summary_df.loc[f"{h}d"]
        )
    else:
        axes[1, 0].axis("off")
        axes[1, 0].text(0.5, 0.5, f"IC series for {h}d not available", ha="center")
    # [1,1] summary
    _plot_summary_table(axes[1, 1], layered_result, ic_summary_df, direction=direction)

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"[plots] Saved: {output_path}")
