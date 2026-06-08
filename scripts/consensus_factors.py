"""
卖方一致预期因子 producer（lib + CLI）
============================================================
从缓存的 comp_indicators / consensus_reports 计算因子面板，
直接调 core.evaluation.evaluate_single_factor 出 IC/ICIR（namespace=cxl/consensus_series）。

先期只实现 comp_indicators 类（pe_fy1_new）；明细类(adj_pct/up_ratio/...)后续接入。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

import core.config as config

_T_COLS = {
    1: "comp_con_net_profit_t1",
    2: "comp_con_net_profit_t2",
    3: "comp_con_net_profit_t3",
}


def compute_forward_np(comp: pd.DataFrame) -> pd.DataFrame:
    """forward-12m 滚动一致预期净利润。

    report_year_t=ry → 一致预期对应日历年 ry+1/ry+2/ry+3 = t1/t2/t3。
    当前日历年 Y，forward 窗口 [date, date+1yr] 线性插值：
      w = 当年剩余天数/365（落在 Y 的比例），forward_np = w*NP(Y) + (1-w)*NP(Y+1)。
    NP(Y) 取 t{Y-ry}，NP(Y+1) 取 t{Y-ry+1}（offset 超出 1..3 置 NaN）。
    """
    d = comp.dropna(subset=["report_year_t"]).copy()
    d["report_year_t"] = d["report_year_t"].astype(int)
    Y = d["date"].dt.year.values
    ry = d["report_year_t"].values
    off_curr = Y - ry
    off_next = off_curr + 1

    def pick(off):
        out = np.full(len(d), np.nan)
        for k, col in _T_COLS.items():
            m = off == k
            out[m] = d[col].values[m]
        return out

    np_curr = pick(off_curr)
    np_next = pick(off_next)
    dec31 = pd.to_datetime(d["date"].dt.year.astype(str) + "-12-31")
    w = ((dec31 - d["date"]).dt.days / 365.0).clip(0, 1).values
    fwd = w * np_curr + (1 - w) * np_next
    d["forward_np"] = fwd
    return d[["order_book_id", "date", "forward_np"]]


def _mcap_panel_yuan() -> pd.DataFrame:
    mc = pd.read_parquet(config.MARKET_CAP_PANEL_PATH)
    mc.index = pd.to_datetime(mc.index)
    return mc * 1e8  # 面板单位为亿元 → 元


def build_pe_fy1_new(comp: pd.DataFrame) -> pd.DataFrame:
    """pe_fy1_new = 总市值 / forward 滚动一致预期净利润（要求 np>0）。返回宽表 date×stock。"""
    fwd = compute_forward_np(comp)
    fwd = fwd[fwd["forward_np"] > 0]
    fwd_w = fwd.pivot(index="date", columns="order_book_id", values="forward_np")
    mcap = _mcap_panel_yuan()
    idx = fwd_w.index
    cols = fwd_w.columns.intersection(mcap.columns)
    mcap_a = mcap.reindex(index=idx, columns=cols)
    fwd_a = fwd_w.reindex(columns=cols)
    pe = mcap_a / fwd_a
    return pe


# ════════════════ 明细类因子（盈利预测调整）════════════════

def clean_reports(rep: pd.DataFrame) -> pd.DataFrame:
    """报告级明细清洗：剔非个股报告 + 提取首序分析师。

    - 非个股报告：report_main_id 的数字主体 ≠ order_book_id 数字主体（如挂错股票的
      跨市场/行业报告，report_main_id 常为 None）→ 剔除。
    - 首序分析师 lead = author 逗号分隔的第一位。
    """
    d = rep.copy()
    code_num = d["order_book_id"].str.split(".").str[0]
    rid = d["report_main_id"].astype(str).str.extract(r"(\d+)")[0]
    rid = rid.str.zfill(6)
    d = d[rid == code_num].copy()
    d["lead"] = d["author"].fillna("").astype(str).str.split(",").str[0].str.strip()
    d = d[d["lead"] != ""]
    d["date"] = pd.to_datetime(d["date"])
    return d


def build_events(d: pd.DataFrame, value_col: str = "net_profit_t",
                 dup_pct: float = 0.01) -> pd.DataFrame:
    """每首序分析师的预测调整事件：相对其上一篇的调整幅度 + 方向。

    - 同 (stock, lead, date) 多篇取最新（last）。
    - 1% 重复剔除：相对上一保留值偏离 < dup_pct 视为重复报告，剔除后再算调整。
    返回事件表 [order_book_id, date, lead, np, adj, is_up]。
    """
    d = d.dropna(subset=[value_col])
    d = (d.sort_values(["order_book_id", "lead", "date"])
           .groupby(["order_book_id", "lead", "date"], as_index=False)[value_col].last())
    out = []
    for (oid, lead), g in d.groupby(["order_book_id", "lead"], sort=False):
        prev = None
        for dt, np_val in zip(g["date"].values, g[value_col].values):
            if prev is None:
                prev = np_val
                continue
            rel = (np_val - prev) / abs(prev) if prev != 0 else np.nan
            if pd.isna(rel) or abs(rel) < dup_pct:
                # 视为重复/无实质调整：不产出事件，也不更新基准
                continue
            out.append((oid, dt, lead, np_val, rel, 1.0 if rel > 0 else 0.0))
            prev = np_val
    return pd.DataFrame(out, columns=["order_book_id", "date", "lead", "np", "adj", "is_up"])


def _trading_grid() -> pd.DatetimeIndex:
    mc = pd.read_parquet(config.MARKET_CAP_PANEL_PATH)
    return pd.to_datetime(mc.index)


def _stock_snapshots(ev_s: pd.DataFrame, lookback_days: int, min_n: int):
    """单股事件 → 快照序列 [(date, up_ratio, adj_pct)]。
    在每个事件日重算「过去 lookback 天内每首序分析师最新观点」的截面聚合。
    """
    ev_s = ev_s.sort_values("date")
    last_seen: dict = {}      # lead -> (date, is_up, adj)
    snaps = []
    for dt, lead, is_up, adj in zip(ev_s["date"], ev_s["lead"], ev_s["is_up"], ev_s["adj"]):
        last_seen[lead] = (dt, is_up, adj)
        cutoff = dt - pd.Timedelta(days=lookback_days)
        active = [(u, a) for (d0, u, a) in last_seen.values() if d0 >= cutoff]
        if len(active) >= min_n:
            up_ratio = float(np.mean([u for u, _ in active]))
            adj_pct = float(np.mean([a for _, a in active]))
        else:
            up_ratio = np.nan
            adj_pct = np.nan
        snaps.append((dt, up_ratio, adj_pct))
    return snaps


def build_revision_panels(fiscal_years, lookback_days: int = 180, min_n: int = 3,
                          dup_pct: float = 0.01, value_col: str = "net_profit_t"):
    """跨 fiscal_year 构建 up_ratio / adj_pct 日频面板。

    value_col='net_profit_t' → fy1（对 fiscal_year 当年的预测）；
    value_col='net_profit_t1' → fy2（对 fiscal_year+1 年的预测）。

    fiscal_year=Y 的报告（对 Y 年净利润的预测）只在 Y 作为 fy1 的区间内贡献：
    近似 [Y-05-01, (Y+1)-04-30]（年报披露截止 4/30 后切换预测期，与 pe 的 report_year_t 滚动一致）。
    """
    from data_fetching.consensus import CONSENSUS_DIR
    grid = _trading_grid()
    up_rows, adj_rows = [], []   # (date, oid, value)
    for Y in fiscal_years:
        cache = CONSENSUS_DIR / f"reports_fy{Y}_rice_create_tm.parquet"
        if not cache.exists():
            logger.warning(f"reports fy{Y} 缺失，跳过"); continue
        rep = pd.read_parquet(cache)
        ev = build_events(clean_reports(rep), value_col=value_col, dup_pct=dup_pct)
        if ev.empty:
            continue
        vstart = pd.Timestamp(f"{Y}-05-01")
        vend = pd.Timestamp(f"{Y+1}-04-30")
        valid_grid = grid[(grid >= vstart) & (grid <= vend)]
        for oid, ev_s in ev.groupby("order_book_id", sort=False):
            snaps = _stock_snapshots(ev_s, lookback_days, min_n)
            if not snaps:
                continue
            ss = pd.DataFrame(snaps, columns=["date", "up", "adj"]).set_index("date")
            ss = ss[~ss.index.duplicated(keep="last")]
            # 快照 ffill 到有效交易日；过期：距最近事件 > lookback 置 NaN
            last_evt = pd.Series(ss.index, index=ss.index).reindex(valid_grid, method="ffill")
            up = ss["up"].reindex(valid_grid, method="ffill")
            adj = ss["adj"].reindex(valid_grid, method="ffill")
            expired = (valid_grid - last_evt.values) > pd.Timedelta(days=lookback_days)
            up = up.where(~expired); adj = adj.where(~expired)
            up = up.dropna(); adj = adj.dropna()
            for dt, v in up.items():
                up_rows.append((dt, oid, v))
            for dt, v in adj.items():
                adj_rows.append((dt, oid, v))
    def _to_wide(rows):
        df = pd.DataFrame(rows, columns=["date", "order_book_id", "v"])
        return df.pivot_table(index="date", columns="order_book_id", values="v", aggfunc="last")
    return _to_wide(up_rows), _to_wide(adj_rows)


def build_count_panel(fiscal_years, window_days: int = 90):
    """cnts_ana_rpt90：过去 window 天内对 fy1 做过盈利预测的首序分析师数。

    clean_reports（剔非个股+首序）后，每股每交易日统计 [D-window, D] 内不同 lead 数。
    fy1 滚动同 revision：fiscal_year=Y 在 [Y-05-01, (Y+1)-04-30] 区间贡献。
    在月末交易日快照（覆盖数月频变化平缓），再 ffill 到日频有效区间。
    """
    from data_fetching.consensus import CONSENSUS_DIR
    grid = _trading_grid()
    out_cols = {}
    for Y in fiscal_years:
        cache = CONSENSUS_DIR / f"reports_fy{Y}_rice_create_tm.parquet"
        if not cache.exists():
            logger.warning(f"reports fy{Y} 缺失，跳过"); continue
        c = clean_reports(pd.read_parquet(cache))
        c = c[["order_book_id", "date", "lead"]].drop_duplicates()
        vstart, vend = pd.Timestamp(f"{Y}-05-01"), pd.Timestamp(f"{Y+1}-04-30")
        valid_grid = grid[(grid >= vstart) & (grid <= vend)]
        # 月末快照点
        snap_pts = (pd.Series(valid_grid, index=valid_grid)
                    .groupby([valid_grid.year, valid_grid.month]).last().values)
        snap_pts = pd.DatetimeIndex(snap_pts)
        for oid, g in c.groupby("order_book_id", sort=False):
            dates = g["date"].values
            leads = g["lead"].values
            vals = []
            for D in snap_pts:
                lo = np.datetime64(D - pd.Timedelta(days=window_days))
                m = (dates > lo) & (dates <= np.datetime64(D))
                vals.append(len(set(leads[m])) if m.any() else np.nan)
            s = pd.Series(vals, index=snap_pts).reindex(valid_grid, method="ffill").dropna()
            if not s.empty:
                out_cols.setdefault(oid, []).append(s)
    data = {oid: pd.concat(parts).sort_index() for oid, parts in out_cols.items()}
    return pd.DataFrame(data)


# ════════════════════════════ CLI ════════════════════════════

def _evaluate(panel: pd.DataFrame, factor: str, eval_start: str, eval_end: str):
    logger.info(f"{factor} 面板 shape={panel.shape} 非空={panel.notna().sum().sum()}")
    raw_dir = config.RAW_FACTOR_BASE / "cxl/consensus_series"
    raw_dir.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(raw_dir / f"{factor}.parquet")
    from core.evaluation import evaluate_single_factor
    res = evaluate_single_factor(
        factor_name=factor, factor_df=panel,
        start_date=eval_start, end_date=eval_end,
        namespace="cxl/consensus_series",
    )
    print(res["ic_summary"])
    return res


# ════════════════ reg_pe_fy1（久期调整预期 PE）════════════════

def fy1_np_panel(comp: pd.DataFrame) -> pd.DataFrame:
    """fy1 一致预期净利润（当前日历年 Y 的一致预期，不做 forward 滚动）= 宽表 date×stock。"""
    d = comp.dropna(subset=["report_year_t"]).copy()
    d["report_year_t"] = d["report_year_t"].astype(int)
    Y = d["date"].dt.year.values
    off = Y - d["report_year_t"].values
    out = np.full(len(d), np.nan)
    for k, col in _T_COLS.items():
        m = off == k
        out[m] = d[col].values[m]
    d["fy1_np"] = out
    d = d[d["fy1_np"] > 0]
    return d.pivot(index="date", columns="order_book_id", values="fy1_np")


def build_duration_panels(fiscal_years, lookback_days: int = 180):
    """久期面板 + fy1 报告数面板（月末快照 + ffill）。

    久期 = (n_fy1·1 + n_fy2·2 + n_fy3·3) / (n_fy1+n_fy2+n_fy3)，n_fyk = 可回溯区间内对
    第 k 预测年（net_profit_t/t1/t2 非空）做过预测的首序分析师数。衡量分析师"看多远"。
    """
    from data_fetching.consensus import CONSENSUS_DIR
    grid = _trading_grid()
    dur_cols, n1_cols = {}, {}
    for Y in fiscal_years:
        cache = CONSENSUS_DIR / f"reports_fy{Y}_rice_create_tm.parquet"
        if not cache.exists():
            continue
        c = clean_reports(pd.read_parquet(cache))
        vstart, vend = pd.Timestamp(f"{Y}-05-01"), pd.Timestamp(f"{Y+1}-04-30")
        valid_grid = grid[(grid >= vstart) & (grid <= vend)]
        snap_pts = pd.DatetimeIndex(
            pd.Series(valid_grid, index=valid_grid).groupby([valid_grid.year, valid_grid.month]).last().values)
        for oid, g in c.groupby("order_book_id", sort=False):
            dts = g["date"].values
            t0 = g["net_profit_t"].notna().values
            t1 = g["net_profit_t1"].notna().values
            t2 = g["net_profit_t2"].notna().values
            leads = g["lead"].values
            dvals, n1vals = [], []
            for D in snap_pts:
                lo = np.datetime64(D - pd.Timedelta(days=lookback_days))
                m = (dts > lo) & (dts <= np.datetime64(D))
                if not m.any():
                    dvals.append(np.nan); n1vals.append(np.nan); continue
                n1 = len(set(leads[m & t0])); n2 = len(set(leads[m & t1])); n3 = len(set(leads[m & t2]))
                tot = n1 + n2 + n3
                dvals.append((n1 + 2*n2 + 3*n3)/tot if tot > 0 else np.nan)
                n1vals.append(n1)
            sd = pd.Series(dvals, index=snap_pts).reindex(valid_grid, method="ffill")
            sn = pd.Series(n1vals, index=snap_pts).reindex(valid_grid, method="ffill")
            dur_cols.setdefault(oid, []).append(sd.dropna())
            n1_cols.setdefault(oid, []).append(sn.dropna())
    dur = pd.DataFrame({o: pd.concat(p).sort_index() for o, p in dur_cols.items()})
    n1 = pd.DataFrame({o: pd.concat(p).sort_index() for o, p in n1_cols.items()})
    return dur, n1


def _xsec_residual(y: pd.DataFrame, x: pd.DataFrame) -> pd.DataFrame:
    """逐日截面 OLS y ~ a + b·x，返回残差面板（对齐 y 的网格）。"""
    x = x.reindex_like(y)
    res = pd.DataFrame(index=y.index, columns=y.columns, dtype=float)
    for dt in y.index:
        yi = y.loc[dt].values.astype(float)
        xi = x.loc[dt].values.astype(float)
        m = np.isfinite(yi) & np.isfinite(xi)
        if m.sum() < 30:
            continue
        xa = np.column_stack([np.ones(m.sum()), xi[m]])
        beta, *_ = np.linalg.lstsq(xa, yi[m], rcond=None)
        r = yi[m] - xa @ beta
        res.loc[dt, np.array(y.columns)[m]] = r
    return res


def build_reg_pe_fy1(comp, fiscal_years, lookback_days: int = 180, diff_periods: int = 60):
    """reg_pe_fy1：log(pe_fy1) 与 log(久期) 各取 60 日 diff 后，截面回归 Δlogpe ~ Δlogdur 的残差。"""
    fy1np = fy1_np_panel(comp)
    mcap = _mcap_panel_yuan()
    cols = fy1np.columns.intersection(mcap.columns)
    pe = (mcap.reindex(index=fy1np.index, columns=cols) / fy1np.reindex(columns=cols))
    dur, n1 = build_duration_panels(fiscal_years, lookback_days=lookback_days)
    # 对齐网格
    idx = pe.index.intersection(dur.index)
    cols2 = pe.columns.intersection(dur.columns)
    pe = pe.reindex(index=idx, columns=cols2)
    dur = dur.reindex(index=idx, columns=cols2)
    n1 = n1.reindex(index=idx, columns=cols2)
    # 连续报告数<3 或 pe<=0 置 nan
    pe = pe.where((n1 >= 3) & (pe > 0))
    dur = dur.where(dur > 0)
    log_pe = np.log(pe)
    log_dur = np.log(dur)
    d_pe = log_pe.diff(diff_periods)
    d_dur = log_dur.diff(diff_periods)
    return _xsec_residual(d_pe, d_dur)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--factor", default="pe_fy1_new")
    p.add_argument("--eval-start", default="20160101")
    p.add_argument("--eval-end", default="20251231")
    p.add_argument("--lookback", type=int, default=180)
    p.add_argument("--min-analysts", type=int, default=3)
    p.add_argument("--fy", default="2015-2025", help="明细因子 fiscal_year 范围, 如 2015-2025")
    a = p.parse_args()

    if a.factor == "pe_fy1_new":
        from data_fetching.consensus import load_or_fetch_comp_indicators
        panel = build_pe_fy1_new(load_or_fetch_comp_indicators())
    elif a.factor in ("up_ratio_fy1", "adj_pct_fy1", "up_ratio_fy2", "adj_pct_fy2"):
        lo, hi = (int(x) for x in a.fy.split("-"))
        vcol = "net_profit_t1" if a.factor.endswith("fy2") else "net_profit_t"
        up, adj = build_revision_panels(list(range(lo, hi + 1)), lookback_days=a.lookback,
                                        min_n=a.min_analysts, value_col=vcol)
        panel = up if a.factor.startswith("up_ratio") else adj
    elif a.factor == "cnts_ana_rpt90":
        lo, hi = (int(x) for x in a.fy.split("-"))
        panel = build_count_panel(list(range(lo, hi + 1)), window_days=90)
    elif a.factor == "reg_pe_fy1":
        from data_fetching.consensus import load_or_fetch_comp_indicators
        lo, hi = (int(x) for x in a.fy.split("-"))
        panel = build_reg_pe_fy1(load_or_fetch_comp_indicators(),
                                 list(range(lo, hi + 1)), lookback_days=a.lookback)
    else:
        raise SystemExit(f"未实现: {a.factor}")
    _evaluate(panel, a.factor, a.eval_start, a.eval_end)
