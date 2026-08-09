"""
ML top-N 周度调仓基线 —— 2020-2026 长样本（全程样本外 test 段）
============================================================
口径：
  - 每周最后一个交易日 D 收盘：取当日信号 txt 前 N 名
  - 收益 = forward_return_5d[D] 均值（D+1 周一 vwap 买 → D+6 下周一 vwap 卖）
  - N ∈ {5, 10, 20, 50, 100}，考察集中度 → 收益/风险的完整曲线
  - 换手率 = 1 - 相邻两周持仓交集比例；费率 = 买 0.03% + 卖 0.08%（佣金+印花+过户）
  - 净值周度复利，统计年化收益/波动/Sharpe/最大回撤/周胜率/逐年拆分

运行：仓库根目录 python scripts/research_ml_topn_weekly_longterm.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ==================== 参数（手动修改，不走命令行） ====================
RUN_ID = "lgbm_a158_p27_style_shap_all_dq"
SIGNAL_DIR = Path(f"/nfs/ofs-prediction/peterzhenglinpeng/ml/predictions/{RUN_ID}/signals")
FWD_RET_PATH = Path("/nfs/ofs-prediction/peterzhenglinpeng/market-data/labels/forward_return_5d.parquet")
OUT_DIR = Path("/nfs/ofs-prediction/peterzhenglinpeng/tmp/hl_surge_research")

STUDY_START = "2020-01-01"
STUDY_END = "2026-07-15"
TOP_NS = [5, 10, 20, 50, 100]
COST_ROUNDTRIP = 0.0011        # 单次全换手成本：买 0.03% + 卖 0.08%
WEEKS_PER_YEAR = 52


def load_signals() -> dict[pd.Timestamp, list[str]]:
    result = {}
    for f in sorted(SIGNAL_DIR.glob("*.txt")):
        date = pd.Timestamp(f.stem)
        if not (pd.Timestamp(STUDY_START) <= date <= pd.Timestamp(STUDY_END)):
            continue
        codes = [line.strip().split("_", 1)[1] for line in f.read_text().splitlines() if line.strip()]
        result[date] = codes
    logger.info(f"加载信号：{len(result)} 个交易日，{min(result).date()} ~ {max(result).date()}")
    return result


def max_drawdown(nav: pd.Series) -> float:
    return float((nav / nav.cummax() - 1).min())


def perf_stats(r: pd.Series) -> dict:
    """r: 周收益序列（已按时间排序）。"""
    nav = (1 + r).cumprod()
    n = len(r)
    ann = nav.iloc[-1] ** (WEEKS_PER_YEAR / n) - 1
    vol = r.std() * np.sqrt(WEEKS_PER_YEAR)
    return {
        "周数": n, "周均": r.mean(), "周胜率": (r > 0).mean(),
        "年化收益": ann, "年化波动": vol, "Sharpe": ann / vol if vol > 0 else np.nan,
        "最大回撤": max_drawdown(nav), "累计净值": nav.iloc[-1],
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    signals = load_signals()
    fwd = pd.read_parquet(FWD_RET_PATH)

    dates = sorted(signals)
    week_key = pd.Series(dates, index=dates).dt.to_period("W-FRI")
    decision_days = [list(ds)[-1] for _, ds in pd.Series(dates, index=dates).groupby(week_key)]

    # 逐 N 计算周收益序列 + 换手率
    weekly = {}
    turnover = {}
    for n in TOP_NS:
        rets, tos, idx = [], [], []
        prev_set = None
        for D in decision_days:
            if D not in fwd.index:
                continue
            picks = signals[D][:n]
            r = fwd.loc[D].reindex(picks).dropna()
            if len(r) == 0:
                continue
            cur_set = set(picks)
            if prev_set is not None:
                tos.append(1 - len(cur_set & prev_set) / n)
            prev_set = cur_set
            rets.append(r.mean())
            idx.append(D)
        weekly[n] = pd.Series(rets, index=pd.DatetimeIndex(idx), name=f"top{n}")
        turnover[n] = float(np.mean(tos))

    pd.set_option("display.width", 250)
    print("\n" + "=" * 110)
    print(f"【ML top-N 周度调仓 · 长样本汇总】{STUDY_START} ~ {STUDY_END}（全程样本外 test 段；毛收益，未扣费）")
    print("=" * 110)
    rows = []
    for n in TOP_NS:
        s = perf_stats(weekly[n])
        s["周换手"] = turnover[n]
        # 费后：每周成本 = 换手率 × 全换手成本
        r_net = weekly[n] - turnover[n] * COST_ROUNDTRIP
        s["费后年化"] = perf_stats(r_net)["年化收益"]
        rows.append(pd.Series(s, name=f"top{n}"))
    summary = pd.DataFrame(rows)
    fmt = {c: "{:+.2%}" for c in ["周均", "年化收益", "年化波动", "最大回撤", "费后年化"]}
    fmt.update({"周胜率": "{:.1%}", "周换手": "{:.1%}", "Sharpe": "{:.2f}", "累计净值": "{:.2f}", "周数": "{:.0f}"})
    print(summary.to_string(formatters={k: v.format for k, v in fmt.items()}))

    # 逐年拆分（top5 与 top100 对照）
    for n in [5, 100]:
        r = weekly[n]
        by_year = r.groupby(r.index.year)
        print("\n" + "=" * 110)
        print(f"【top{n} 逐年拆分】")
        print("=" * 110)
        yr_rows = []
        for y, ry in by_year:
            s = perf_stats(ry.sort_index())
            yr_rows.append(pd.Series(
                {"周数": s["周数"], "周均": s["周均"], "周胜率": s["周胜率"],
                 "年收益": (1 + ry).prod() - 1, "最大回撤": s["最大回撤"]}, name=y))
        ydf = pd.DataFrame(yr_rows)
        yfmt = {"周均": "{:+.2%}", "周胜率": "{:.1%}", "年收益": "{:+.2%}", "最大回撤": "{:+.2%}", "周数": "{:.0f}"}
        print(ydf.to_string(formatters={k: v.format for k, v in yfmt.items()}))

    # top5 最惨的 10 周 + 最好的 10 周
    r5 = weekly[5]
    print("\n" + "=" * 110)
    print("【top5 极端周】")
    print("=" * 110)
    worst = r5.nsmallest(10)
    best = r5.nlargest(10)
    ext = pd.DataFrame({"最差10周": worst.map(lambda x: f"{x:+.2%}").values,
                        "日期W": [d.date() for d in worst.index],
                        "最好10周": best.map(lambda x: f"{x:+.2%}").values,
                        "日期B": [d.date() for d in best.index]})
    print(ext.to_string())

    # 落盘周收益序列
    out = pd.DataFrame(weekly)
    out.to_csv(OUT_DIR / "ml_topn_weekly_2020_2026.csv")
    print(f"\n周收益序列已落盘：{OUT_DIR}/ml_topn_weekly_2020_2026.csv")


if __name__ == "__main__":
    main()
