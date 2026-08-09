"""
因子地图冒烟测试 —— 3 因子 + 模型锚,完整跑一遍"体检四组指标"
============================================================
目的:在铺开全库 ~300 因子之前,用一行地图展示体检的判别力:
  eruption_followup_ratio  p27 主力,预期"强但脆"
  npf_mrq_sue8             基本面动量(SUE),假说"全天候"
  CMC                      联合动量,已知故事做校准锚
  MODEL(ensemble)          模型预测分本身,"强但脆"的定义基准

四组指标:
  ① 平均效力   IC_full / ICIR(2020~2025,20d RankIC,方向调整后)
  ② 压力窗口   2021抱团瓦解 / 2024微盘危机 / 2026动量行情 的窗口 IC 与留存率
  ③ 赢家体检   2026H1 赢家池截面分位(方向调整后的"看多程度") vs 模型入选组
  ④ regime     动量月/反转月分别的 IC、与模型 IC 的时序相关

运行:仓库根 PYTHONPATH=. python scripts/factor_map_smoke.py
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from loguru import logger

# ── 参数(手动改这里)─────────────────────────────────────────
_FB = "/nfs/ofs-prediction/peterzhenglinpeng/factors/raw-dquant"
FACTORS = {
    "eruption_followup": f"{_FB}/kysec-dquant/paper_27_microstructure/eruption_followup_ratio.parquet",
    "npf_mrq_sue8": f"{_FB}/cxl-dquant/npf_series/npf_mrq_sue8.parquet",
    "CMC": f"{_FB}/guosen/co_momentum/CMC.parquet",
}
PRED_PANEL = "/nfs/ofs-prediction/peterzhenglinpeng/ml/predictions/ensemble_seed5_dq/pred_panel_live.parquet"
LABELS_FR20 = "/nfs/ofs-prediction/peterzhenglinpeng/market-data/labels/forward_return_20d.parquet"
VWAP_PANEL = "/nfs/ofs-prediction/peterzhenglinpeng/market-data/labels/vwap_panel.parquet"
WINNERS_PARQUET = "/nfs/ofs-prediction/peterzhenglinpeng/tmp/diagnose_2026h1_winners.parquet"
FULL_START, FULL_END = "2020-01-02", "2025-12-31"      # 平均效力段
STRESS = {"2021抱团瓦解": ("2021-02-01", "2021-04-30"),
          "2024微盘危机": ("2024-01-01", "2024-02-29"),
          "2026动量行情": ("2026-01-05", "2026-05-31")}
WIN_START = "2026-01-05"                                # 赢家体检窗口起点
pd.set_option("display.width", 220)


def daily_ic(f: pd.DataFrame, r: pd.DataFrame) -> pd.Series:
    """逐日截面 Spearman RankIC(向量化:掩码共同有效 → 行内秩 → 皮尔逊)。"""
    valid = f.notna() & r.notna()
    x = f.where(valid).rank(axis=1)
    y = r.where(valid).rank(axis=1)
    xm = x.sub(x.mean(axis=1), axis=0)
    ym = y.sub(y.mean(axis=1), axis=0)
    num = (xm * ym).sum(axis=1)
    den = np.sqrt((xm ** 2).sum(axis=1) * (ym ** 2).sum(axis=1))
    return (num / den).replace([np.inf, -np.inf], np.nan)


def load_panel(path: str, grid_idx, grid_cols) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    return df.reindex(index=grid_idx, columns=grid_cols)


def main():
    fr20 = pd.read_parquet(LABELS_FR20); fr20.index = pd.to_datetime(fr20.index)
    pred = pd.read_parquet(PRED_PANEL); pred.index = pd.to_datetime(pred.index)
    grid_idx = pred.index                                  # 2020-01-02 起,与模型池对齐
    grid_cols = pred.columns
    fr = fr20.reindex(index=grid_idx, columns=grid_cols)
    pool = pred.notna()                                    # 模型 eligible 池
    logger.info(f"网格 {pred.shape} | labels 有效末日 {fr.dropna(how='all').index[-1].date()}")

    # regime 标尺:Ret20 月度 IC 符号(>0 动量月,<0 反转月)
    vwap = load_panel(VWAP_PANEL, grid_idx, grid_cols)
    ret20 = vwap.pct_change(20).where(pool)
    mom_ic_m = daily_ic(ret20, fr).groupby(grid_idx.to_period("M")).mean().dropna()
    mom_months = mom_ic_m[mom_ic_m > 0].index
    rev_months = mom_ic_m[mom_ic_m <= 0].index
    logger.info(f"regime 标尺:动量月 {len(mom_months)} 个 | 反转月 {len(rev_months)} 个 | "
                f"2026 年动量月: {[str(m) for m in mom_months if m.year == 2026]}")

    # 模型月度 IC(regime 相关性的参照序列)
    model_ic_d = daily_ic(pred, fr)
    model_ic_m = model_ic_d.groupby(grid_idx.to_period("M")).mean()

    winners = pd.read_parquet(WINNERS_PARQUET)
    rk = pred.loc[WIN_START:].rank(axis=1, ascending=False, method="first")
    pick_mask = rk <= 100

    rows = {}
    panels = {**{k: load_panel(p, grid_idx, grid_cols).where(pool) for k, p in FACTORS.items()},
              "MODEL(ensemble)": pred}
    for name, f in panels.items():
        ic_d = daily_ic(f, fr)
        ic_full = ic_d.loc[FULL_START:FULL_END]
        direction = np.sign(ic_full.mean())
        adj = lambda s: s * direction                      # 方向调整:全样本 IC 归一为正
        row = {
            "方向": int(direction),
            "IC_full(20-25)": adj(ic_full).mean(),
            "ICIR_full": adj(ic_full).mean() / ic_full.std(),
        }
        # ② 压力窗口
        for wname, (s, e) in STRESS.items():
            w = adj(ic_d.loc[s:e]).mean()
            row[f"IC_{wname}"] = w
            row[f"留存_{wname}"] = w / row["IC_full(20-25)"]
        # ③ 赢家体检(方向调整后的"看多程度"分位,>0.5=喜欢赢家)
        pct = f.loc[WIN_START:].rank(axis=1, pct=True)
        if direction < 0:
            pct = 1 - pct
        row["赢家看多分位"] = pct.reindex(columns=winners.index).stack().median()
        row["入选组看多分位"] = pct.where(pick_mask.reindex(index=pct.index, columns=pct.columns)).stack().median()
        # ④ regime 分解 + 与模型 IC 时序相关
        ic_m = adj(ic_d.groupby(grid_idx.to_period("M")).mean())
        row["IC_动量月"] = ic_m.reindex(mom_months).mean()
        row["IC_反转月"] = ic_m.reindex(rev_months).mean()
        row["与模型IC月相关"] = ic_m.corr(model_ic_m)
        rows[name] = row
        logger.info(f"{name} 完成")

    out = pd.DataFrame(rows)
    print("\n" + "=" * 100)
    print("因子地图冒烟测试(每列 = 地图中的一行;IC 均为方向调整后)")
    print("=" * 100)
    print(out.round(4).to_string())
    out.to_parquet("/nfs/ofs-prediction/peterzhenglinpeng/tmp/factor_map_smoke.parquet")


if __name__ == "__main__":
    main()
