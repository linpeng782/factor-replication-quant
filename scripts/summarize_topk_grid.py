"""
topk 网格结果汇总 —— 从 account_history.csv 复算绩效，口径对齐回测引擎
================================================================================
为什么不直接读 periodic_returns.txt：那里的区间是信号全长（到 2026-08-07），
而与 joey 横向对比的既有表格截断在 2026-05-26。本脚本统一截断后复算，
保证同一张表内所有行可比。

复算公式与 core/performance_utils.py + core/metrics_calculator.py 一致：
  年化    = (1 + 累计收益) ** (252 / 交易日数) - 1        （几何年化 EAR）
  年化波动 = daily_return.std() * sqrt(252)
  夏普    = (年化 - 3%) / 年化波动                        （无风险利率 3%）
  年化换手 = 调仓日平均换手 * (252 / rebalance_interval)
  每期收益 = (1 + 累计) ** (1 / 期数) - 1，期数 = 交易日数 / interval

参数写在下方 PARAMS，不走命令行。
用法：PYTHONPATH=. python scripts/summarize_topk_grid.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ============================== PARAMS ==============================
RESULTS_ROOT = Path("/nfs/ofs-prediction/peterzhenglinpeng/ml_experiments/topk_grid/results/dr")
END = "2026-05-26"          # 截断日，对齐 joey 数据末日
INTERVAL = 2                # 调仓间隔，与回测配置一致
RISK_FREE = 0.03

# 展示名 → 结果目录名中的 run_id 前缀
RUNS = {
    "E0 基线 20d": "lgbm_shap128_csrank5_dq",
    "E1 纯 2d":    "lgbm_shap128_h2_csrank5_dq",
}
TOPKS = (20, 30, 50, 100)
# ====================================================================


def perf(path: Path) -> dict:
    """从单个 account_history.csv 复算绩效指标。"""
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    df = df.loc[:pd.Timestamp(END)]
    ndays = len(df) - 1                                   # 首行是建仓前基准日，无收益
    total = df["total_asset"].iloc[-1] / df["total_asset"].iloc[0] - 1
    ann = (1 + total) ** (252 / ndays) - 1
    r = df["daily_return"].dropna()
    vol = r.std() * np.sqrt(252)
    nav = df["total_asset"] / df["total_asset"].iloc[0]
    mdd = (nav / nav.cummax() - 1).min()
    reb = df["turnover"][df["turnover"] > 0]
    ann_turnover = reb.mean() * (252 / INTERVAL)
    nperiods = ndays / INTERVAL
    per_period = (1 + total) ** (1 / nperiods) - 1
    return {
        "年化": ann, "夏普": (ann - RISK_FREE) / vol, "最大回撤": mdd,
        "年化换手": ann_turnover, "每期": per_period,
        # 每单位换手收益：每期收益 ÷ 每期换手（每期换手 = 年化换手 / 年调仓次数）
        "单位换手": per_period / (ann_turnover / (252 / INTERVAL)),
        "天数": ndays,
    }


def main() -> None:
    rows = []
    for label, run_id in RUNS.items():
        for k in TOPKS:
            hits = list(RESULTS_ROOT.glob(f"{run_id}_*_topk{k}_*/account_history.csv"))
            # 前缀 glob 会让 E0 的 run_id 命中 E1 目录（h2 是 E0 名字的扩展），需精确过滤
            hits = [h for h in hits if h.parent.name.split("_20200103_")[0] == run_id]
            if not hits:
                print(f"[WARN] 缺 {run_id} topk{k}")
                continue
            p = perf(hits[0])
            rows.append({
                "模型": label, "top_k": k,
                "年化": f"{p['年化']*100:.2f}%", "夏普": f"{p['夏普']:.2f}",
                "最大回撤": f"{p['最大回撤']*100:.1f}%",
                "年化换手": f"{p['年化换手']:.1f}",
                "每期": f"{p['每期']*1e4:.1f}bp",
                "单位换手收益": f"{p['单位换手']*1e4:.1f}bp",
            })
    print(f"\n区间 2020-01-02 ~ {END}，interval={INTERVAL}，0bp 滑点\n")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
