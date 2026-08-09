"""
市场层广播特征 构建/日更（dquant 后端）
============================================================
设计文档：docs/20260807_market_broadcast_features_design.md（口径唯一依据，改口径先改文档）

产出「每日一个标量、广播到全市场所有股票」的市场层特征，供 LGBM 截面模型
作为 regime / 因子择时输入。第一批 10 个，全部只依赖现有数据，零新建数据管线。

两级产物
--------
  1. 标量序列（本脚本主产物，步骤1）
     factors/helpers/market_broadcast_series.parquet   (T × 10)
     供画图、相关性体检、PIT 审计用，**不进模型**。

  2. 广播宽表（步骤3，由 WRITE_BROADCAST 开关控制，默认关）
     factors/raw-dquant/market-dquant/<group>/<name>.parquet   (T × N，每行同值)
     与 style-dquant/calendar/week_of_year.parquet 同款格式，ml_core 零改动直读。

⚠️ 两条硬约束（详见设计文档 §6）
--------------------------------
  1. 广播特征**只能用于 LGBM 线**。MLP 线的 DailyCrossSectionMAD 逐日截面 zscore
     会把「当日恒定」的列静默抹成全 0 且不报错。启用 market-dquant 源时 model 必须是 lgbm。
  2. 广播特征在长表里被复制约 5000 倍，SHAP 重要度系统性虚高，
     **不可与截面因子放同一张表排序**。评估只看回测 / 分年度 IC / 消融。

10 个特征
---------
  group=vol      mkt_disp20_pct     全市场截面离散度（选股环境好坏的直接度量）
                 mkt_avgcorr20_pct  平均两两相关性（抱团/羊群度）
                 mkt_rv20_pct       市场已实现波动
                 mkt_rv_ts          波动期限结构 RV20/RV60-1（波动加速度）
                 ind_disp20_pct     中信一级行业收益离散度（行业分化度）
  group=regime   mkt_dd250          距 250 日高点回撤
                 mkt_amt_exp        量能扩张 AMT/MA60-1
  group=breadth  mkt_adl_div        ADL 广度与市场收益的背离
                 mkt_median_gap20   赚钱效应偏离（个股中位数 − 市值加权）
  group=alphaenv fac_absic20        alpha 环境强度（12 个代表因子近 20 日 |IC| 均值）

两种模式
--------
  python data_fetching/market_broadcast.py             # 增量日更（默认）
  python data_fetching/market_broadcast.py --full      # 全量重建

业务参数（POOL 口径 / 窗口 / 代表因子集 / 是否落广播宽表）一律在下方常量区手改，
不走命令行——命令行只保留 --full / --output-dir 两个开关（与 style_ln_market_cap.py 一致）。
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

import config as cfg
from alpha_shared.evaluation.ic import compute_ic_series

# ==================== 常量区（业务参数在这里手改） ====================

DTYPE = "float32"

# ── POOL 统计口径（设计文档 §4.2）──
# 全部市场层统计只在 POOL 内做。
#   不剔 ST     ：ST 股本身是市场情绪的一部分，剔掉会损失信息
#   不剔涨跌停  ：涨跌停是情绪信号本身
# ⚠️ 绝对不要改用 limit-dquant/normal_day_panel.parquet —— 那个把涨跌停也剔了，
#    是给动量因子用的口径，用在广度统计上会系统性削掉极端行情样本。
POOL_EXCLUDE_ST = False          # True 则 POOL 额外剔除 ST
MIN_POOL_SIZE = 100              # 当日 POOL 有效股票数 < 此值 → 当日全部指标置 NaN

# ── 滚动窗口 ──
WIN_SHORT = 20                   # 短窗（离散度/波动/广度斜率/赚钱效应 的平滑与计算窗）
WIN_LONG = 60                    # 长窗（波动期限结构分母、量能基准）
WIN_DD = 250                     # 回撤高点回看窗
WIN_PCT = 500                    # 滚动分位窗（约 2 年，与研报口径一致）
MIN_PCT_PERIODS = 250            # 滚动分位最小样本（不足置 NaN）
ANN = 252.0                      # 年化因子

# ── M10 alpha 环境强度：12 个固定代表因子 ──
# 固定不变、不依赖模型入选结果，保证 A/B 干净（设计文档 §4.4 M10）。
IC_HORIZON = 20                  # 用 forward_return_20d
# forward_return_20d[T] = (vwap[T+21]-vwap[T+1])/vwap[T+1]，故 T 日的 IC 到 T+21 才实现。
# 再叠加 1 日安全垫 → 总滞后 22 个交易日。这是 M10 唯一的前视防线，改动需重跑 PIT 验证。
IC_AVAIL_LAG = IC_HORIZON + 1    # 21：IC 实现所需交易日
IC_SAFETY_LAG = 1                # 1 ：额外安全垫
REPRESENTATIVE_FACTORS = [
    "rolling/ROC20", "rolling/STD20", "rolling/MA20", "rolling/RSQR20",
    "rolling/MAX20", "rolling/CORR20", "rolling/RANK20",
    "volume/VMA20", "volume/VSTD20", "volume/WVMA20",
]                                # 上列 10 个在 alpha158-dquant 下
EXTRA_FACTORS = {                # 另外 2 个在别的源下（相对 RAW_FACTOR_BASE）
    "ln_market_cap": "style-dquant/size/ln_market_cap.parquet",
    "wgt_return_1m": "htsec/paper_04_momentum/wgt_return_1m.parquet",
}

# ── 特征清单：name -> group（决定落盘子目录，便于按维度做消融）──
FEATURE_GROUPS = {
    "mkt_disp20_pct": "vol",
    "mkt_avgcorr20_pct": "vol",
    "mkt_rv20_pct": "vol",
    "mkt_rv_ts": "vol",
    "ind_disp20_pct": "vol",
    "mkt_dd250": "regime",
    "mkt_amt_exp": "regime",
    "mkt_adl_div": "breadth",
    "mkt_median_gap20": "breadth",
    "fac_absic20": "alphaenv",
}

# ── 落盘开关 ──
# 步骤1 只产标量序列；体检（步骤2）通过后再把这里改 True 跑步骤3 落广播宽表。
WRITE_BROADCAST = True
BROADCAST_SOURCE = "market-dquant"

# ── 训练段方差断言（设计文档 §6.1）──
# WholeSetRobustZ 只在 train 段 fit：若某列在 train 段内恒定或大面积 NaN，
# 该列 scale=NaN → 全程失效且不报错。落盘前必须挡住。
TRAIN_END = "2017-11-30"
MIN_TRAIN_NUNIQUE = 50           # train 段内不同取值数下限
MAX_TRAIN_NAN_RATIO = 0.20       # train 段内 NaN 占比上限

IO_WORKERS = 32                  # per-day 成交额文件并行读线程数

# ── 末尾 ffill（实盘可用性，设计文档 §6.6）──
# 各源末日不齐：ret1 到 2026-07-31，但 industry_index_return 只到 07-17，
# forward_return_20d（M10 的原料）因需 vwap[T+21] 只到 05-29 → 叠加 22 日 PIT 滞后后到 07-15。
# ml_core 的 predict.end=null 取「入选因子共同覆盖末日」，不处理的话整个推理会被卡掉 12 个交易日。
# ffill 的语义是「沿用最近一次可测值」，只会让特征更**陈旧**、不会更新鲜，**不引入前视**。
# 用 limit 封顶，避免数据管线长期断更时静默用一个几个月前的老值。
FFILL_LIMIT = 30                 # 末尾最多前向填充的交易日数；0 = 关闭


# ==================== 通用小工具 ====================

def _pct_roll(s: pd.Series) -> pd.Series:
    """滚动分位（含当日、纯历史窗口、不含未来），窗口 WIN_PCT。"""
    return s.rolling(WIN_PCT, min_periods=MIN_PCT_PERIODS).rank(pct=True)


def _z_roll(s: pd.Series) -> pd.Series:
    """滚动 z-score（含当日、纯历史窗口），窗口 WIN_PCT。"""
    mu = s.rolling(WIN_PCT, min_periods=MIN_PCT_PERIODS).mean()
    sd = s.rolling(WIN_PCT, min_periods=MIN_PCT_PERIODS).std()
    return (s - mu) / sd.where(sd > 0)


def _read_panel(path: Path, name: str) -> pd.DataFrame:
    """读宽表面板并规范 index 为 DatetimeIndex。"""
    if not Path(path).exists():
        raise FileNotFoundError(f"{name} 面板不存在: {path}")
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    logger.info(f"  {name}: shape={df.shape} | {df.index.min().date()}~{df.index.max().date()}")
    return df


# ==================== POOL ====================

def load_pool(dates: pd.DatetimeIndex, stocks: pd.Index) -> pd.DataFrame:
    """构建统计口径池 POOL (T×N bool)：非停牌 & 非新股（可选剔 ST）。

    缺记录的格子（未上市/已退市）一律 False，不进池。
    """
    logger.info("[POOL] 加载 combo_mask / new_stock_mask …")
    cols = ["order_book_id", "datetime", "is_suspended"] + (["is_st"] if POOL_EXCLUDE_ST else [])
    combo = pd.read_parquet(cfg.COMBO_MASK_PATH, columns=cols)
    combo["datetime"] = pd.to_datetime(combo["datetime"])
    new = pd.read_parquet(
        cfg.NEW_STOCK_MASK_PATH, columns=["order_book_id", "datetime", "is_new_stock"]
    )
    new["datetime"] = pd.to_datetime(new["datetime"])

    def _wide(df: pd.DataFrame, col: str) -> pd.DataFrame:
        # 缺格补 True（状态未知 → 阻断，保守）
        w = df.pivot(index="datetime", columns="order_book_id", values=col)
        return w.reindex(index=dates, columns=stocks).fillna(True).astype(bool)

    bad = _wide(combo, "is_suspended") | _wide(new, "is_new_stock")
    if POOL_EXCLUDE_ST:
        bad |= _wide(combo, "is_st")
    pool = ~bad
    n = pool.sum(axis=1)
    logger.success(
        f"[POOL] shape={pool.shape} | 剔ST={POOL_EXCLUDE_ST} | "
        f"日均在池={n[n > 0].mean():.0f} 只 | 通过率={pool.values.mean():.2%}"
    )
    return pool


# ==================== 成交额（M7 用） ====================

def load_amount_panel(dates: pd.DatetimeIndex, stocks: pd.Index) -> pd.DataFrame:
    """从 per-day 分片读 total_turnover，拼成 (T×N) 成交额面板（不复权，单位元）。"""
    per_day_dir = Path(cfg.RAW_OHLCV_DIR).parent / "per-day"
    if not per_day_dir.exists():
        raise FileNotFoundError(f"per-day 目录不存在: {per_day_dir}")
    files = {
        pd.Timestamp(p.stem): p
        for p in per_day_dir.glob("*.parquet")
        if pd.Timestamp(p.stem) in set(dates)
    }
    logger.info(f"[AMT] per-day 分片 {len(files)} 个，{IO_WORKERS} 线程并行读 …")

    def _one(item):
        d, p = item
        df = pd.read_parquet(p, columns=["order_book_id", "total_turnover"])
        return d, df.set_index("order_book_id")["total_turnover"]

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=IO_WORKERS) as ex:
        rows = dict(ex.map(_one, files.items()))
    amt = pd.DataFrame(rows).T.sort_index()
    amt = amt.reindex(index=dates, columns=stocks).astype(DTYPE)
    logger.success(f"[AMT] shape={amt.shape} | 耗时 {time.time() - t0:.1f}s")
    return amt


# ==================== M10：alpha 环境强度 ====================

def build_alpha_env(pool: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.Series:
    """12 个固定代表因子近 20 日截面 |rank-IC| 的均值，严格 PIT 滞后。

    PIT 两道保险（设计文档 §4.4 M10）：
      1. ic[T] 用 forward_return_20d[T]，到 T+21 才实现 → 整体 shift(IC_AVAIL_LAG)
      2. 再 shift(IC_SAFETY_LAG) 安全垫
    """
    fwd_path = cfg.LABELS_DIR / f"forward_return_{IC_HORIZON}d.parquet"
    fwd = _read_panel(fwd_path, f"forward_return_{IC_HORIZON}d")
    fwd = fwd.reindex(index=dates, columns=pool.columns).where(pool)

    paths = {p.split("/")[-1]: cfg.ALPHA158_RAW_BASE / f"{p}.parquet"
             for p in REPRESENTATIVE_FACTORS}
    paths.update({k: cfg.RAW_FACTOR_BASE / v for k, v in EXTRA_FACTORS.items()})

    ics = {}
    for name, path in paths.items():
        f = _read_panel(path, f"因子 {name}")
        f = f.reindex(index=dates, columns=pool.columns).where(pool)
        ics[name] = compute_ic_series(f, fwd, method="spearman").abs()
        logger.info(f"  [M10] {name}: |IC| 均值={ics[name].mean():.4f}")

    absic = pd.DataFrame(ics).mean(axis=1)                       # 跨因子平均 |IC|
    absic = absic.rolling(WIN_SHORT, min_periods=WIN_SHORT // 2).mean()
    out = absic.shift(IC_AVAIL_LAG + IC_SAFETY_LAG)              # PIT 滞后
    logger.success(
        f"[M10] fac_absic20 完成 | 因子数={len(paths)} | "
        f"总滞后={IC_AVAIL_LAG + IC_SAFETY_LAG} 交易日 | 有效={out.notna().sum()} 天"
    )
    return out


# ==================== 主计算：10 条标量序列 ====================

def build_series() -> pd.DataFrame:
    """一次扫描算出全部 10 条日频标量序列，返回 (T × 10) DataFrame。"""
    logger.info("=" * 72)
    logger.info("[1/4] 加载公共面板")
    ret = _read_panel(cfg.RET1_PANEL_PATH, "ret1")
    dates, stocks = ret.index, ret.columns
    mcap = _read_panel(cfg.MARKET_CAP_PANEL_PATH, "market_cap").reindex(
        index=dates, columns=stocks)
    indret = _read_panel(cfg.INDUSTRY_INDEX_RETURN_PATH, "industry_index_return")

    logger.info("[2/4] 构建 POOL 与成交额面板")
    pool = load_pool(dates, stocks)
    amt = load_amount_panel(dates, stocks)

    # 池内收益：池外一律 NaN，后续所有截面统计自动忽略
    retp = ret.where(pool)
    n_pool = pool.sum(axis=1)
    thin = n_pool < MIN_POOL_SIZE          # 有效股票太少的日子，指标不可信
    logger.info(f"  有效股票数 < {MIN_POOL_SIZE} 的交易日：{int(thin.sum())} 天（将置 NaN）")

    logger.info("[3/4] 计算 10 条标量序列")
    out = {}

    # ── 市场基准序列 ──
    mkt_ew = retp.mean(axis=1)                                   # 等权市场收益
    nav = (1.0 + mkt_ew.fillna(0.0)).cumprod()                   # 等权净值

    # M1 全市场截面离散度
    disp = retp.std(axis=1)
    out["mkt_disp20_pct"] = _pct_roll(disp.rolling(WIN_SHORT).mean())

    # M2 平均两两相关性（由等权组合方差反解 implied average correlation）
    #   avgcorr = (σ_p² − Σw²σ_i²) / ((Σwσ_i)² − Σw²σ_i²)，等权 w=1/N
    #   注：mkt_ew 的成分随 POOL 变动，σ_p 是近似（非固定组合），设计文档已注明
    sig_i = retp.rolling(WIN_SHORT).std()                        # (T,N) 个股 20 日波动
    sig_p = mkt_ew.rolling(WIN_SHORT).std()                      # 组合 20 日波动
    n_eff = sig_i.notna().sum(axis=1).replace(0, np.nan)
    sum_w2_si2 = (sig_i ** 2).sum(axis=1) / n_eff ** 2
    sum_w_si = sig_i.sum(axis=1) / n_eff
    denom = sum_w_si ** 2 - sum_w2_si2
    avgcorr = (sig_p ** 2 - sum_w2_si2) / denom.where(denom > 0)
    avgcorr = avgcorr.where((avgcorr >= -1) & (avgcorr <= 1))     # 越界置 NaN
    out["mkt_avgcorr20_pct"] = _pct_roll(avgcorr)
    del sig_i

    # M3 市场已实现波动（年化）
    rv20 = mkt_ew.rolling(WIN_SHORT).std() * np.sqrt(ANN)
    rv60 = mkt_ew.rolling(WIN_LONG).std() * np.sqrt(ANN)
    out["mkt_rv20_pct"] = _pct_roll(rv20)

    # M4 波动期限结构（本身无量纲，不再取分位）
    out["mkt_rv_ts"] = rv20 / rv60.where(rv60 > 0) - 1.0

    # M5 行业收益离散度（33 个中信一级行业）
    ind_disp = indret.std(axis=1).reindex(dates)
    out["ind_disp20_pct"] = _pct_roll(ind_disp.rolling(WIN_SHORT).mean())

    # M6 距 250 日高点回撤（本身有界 (-1,0]，不取分位）
    out["mkt_dd250"] = nav / nav.rolling(WIN_DD, min_periods=WIN_DD // 2).max() - 1.0

    # M7 量能扩张
    amt_tot = amt.where(pool).sum(axis=1, min_count=MIN_POOL_SIZE)
    amt_ma = amt_tot.rolling(WIN_LONG, min_periods=WIN_LONG // 2).mean()
    out["mkt_amt_exp"] = amt_tot / amt_ma.where(amt_ma > 0) - 1.0

    # M8 ADL 广度背离：广度斜率 z 值 − 市场收益 z 值，< 0 表示指数涨但广度没跟上
    ad = ((retp > 0).sum(axis=1) - (retp < 0).sum(axis=1)) / n_pool.replace(0, np.nan)
    adl = ad.fillna(0.0).cumsum()
    slope = adl - adl.shift(WIN_SHORT)
    mom = nav / nav.shift(WIN_SHORT) - 1.0
    out["mkt_adl_div"] = _z_roll(slope) - _z_roll(mom)

    # M9 赚钱效应偏离：个股涨幅中位数 − 市值加权市场涨幅
    mcap_v = mcap.where(retp.notna())
    mkt_vw = (retp * mcap_v).sum(axis=1) / mcap_v.sum(axis=1).replace(0, np.nan)
    gap = retp.median(axis=1) - mkt_vw
    out["mkt_median_gap20"] = gap.rolling(WIN_SHORT).mean()

    # M10 alpha 环境强度
    out["fac_absic20"] = build_alpha_env(pool, dates)

    logger.info("[4/4] 汇总与体检")
    df = pd.DataFrame(out, index=dates)[list(FEATURE_GROUPS)]

    # 末尾对齐：各源末日不齐，ffill 沿用最近可测值（只会更陈旧，不引入前视）
    if FFILL_LIMIT > 0:
        before = {c: df[c].last_valid_index() for c in df}
        df = df.ffill(limit=FFILL_LIMIT)
        for c in df.columns:
            a, b = before[c], df[c].last_valid_index()
            if a != b:
                logger.info(f"  [ffill] {c}: 末日 {a.date()} → {b.date()}"
                            f"（+{df.index.get_loc(b) - df.index.get_loc(a)} 交易日）")

    df[thin] = np.nan                                            # 薄日整行作废
    df = df.astype(DTYPE)
    df.index.name = "date"
    _report(df)
    return df


def _report(df: pd.DataFrame) -> None:
    """打印描述统计 + 训练段方差断言 + 两两相关，供步骤2 体检。"""
    logger.info("-" * 72)
    logger.info("各特征描述统计：")
    desc = pd.DataFrame({
        "首个有效日": [df[c].first_valid_index() for c in df],
        "有效天数": df.notna().sum().values,
        "NaN占比": df.isna().mean().round(3).values,
        "均值": df.mean().round(4).values,
        "标准差": df.std().round(4).values,
        "最小": df.min().round(4).values,
        "最大": df.max().round(4).values,
    }, index=df.columns)
    logger.info("\n" + desc.to_string())

    logger.info("-" * 72)
    logger.info(f"训练段（≤{TRAIN_END}）方差断言：")
    tr = df.loc[:TRAIN_END]
    bad = []
    for c in df.columns:
        nu, nan_r = tr[c].nunique(), tr[c].isna().mean()
        ok = nu >= MIN_TRAIN_NUNIQUE and nan_r <= MAX_TRAIN_NAN_RATIO
        logger.info(f"  {c:<20} 取值数={nu:>5} NaN占比={nan_r:.1%}  {'OK' if ok else '✗ 不合格'}")
        if not ok:
            bad.append(c)
    if bad:
        raise ValueError(
            f"以下特征在训练段内恒定或大面积缺失，WholeSetRobustZ 会使其全程失效：{bad}\n"
            "（详见 docs/20260807_market_broadcast_features_design.md §6.1）"
        )
    logger.success("训练段方差断言全部通过")

    logger.info("-" * 72)
    logger.info("特征两两相关（|ρ|>0.8 需考虑二选一）：")
    corr = df.corr()
    logger.info("\n" + corr.round(2).to_string())
    hi = [(a, b, round(corr.loc[a, b], 3))
          for i, a in enumerate(corr.columns) for b in corr.columns[i + 1:]
          if abs(corr.loc[a, b]) > 0.8]
    logger.warning(f"高相关对子：{hi}") if hi else logger.success("无 |ρ|>0.8 的对子")


# ==================== 广播落盘（步骤3，默认关闭） ====================

def write_broadcast(series: pd.DataFrame, stocks: pd.Index) -> None:
    """把标量序列沿列方向 tile 成 (T×N) 宽表落盘，格式对齐 week_of_year.parquet。"""
    logger.info("=" * 72)
    logger.info(f"落广播宽表 → {cfg.RAW_FACTOR_BASE / BROADCAST_SOURCE}")
    for name, group in FEATURE_GROUPS.items():
        panel = pd.DataFrame(
            np.repeat(series[[name]].to_numpy(dtype=DTYPE), len(stocks), axis=1),
            index=series.index, columns=stocks,
        )
        panel.index.name, panel.columns.name = "date", "stock"
        path = cfg.RAW_FACTOR_BASE / BROADCAST_SOURCE / group / f"{name}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        panel.to_parquet(path)
        logger.success(f"  {group}/{name}.parquet | shape={panel.shape} | "
                       f"{path.stat().st_size / 1024**2:.1f}MB")


# ==================== 入口 ====================

def build(output: Path, full: bool = False, write_broadcast_panels: bool = WRITE_BROADCAST):
    """构建标量序列（可选落广播宽表），返回序列 DataFrame。

    全量重建成本约数分钟；增量日更同样全量重算（纯本地计算、零 API），
    但只在末日推进时才重写文件，避免无谓 I/O。
    """
    output = Path(output)
    if not full and output.exists():
        existing = pd.read_parquet(output)
        existing.index = pd.to_datetime(existing.index)
        ret_last = pd.read_parquet(cfg.RET1_PANEL_PATH).index.max()
        if pd.Timestamp(ret_last) <= existing.index.max():
            logger.success(f"[日更] 已最新（到 {existing.index.max().date()}），无需更新")
            return existing
        logger.info(f"[日更] 现有到 {existing.index.max().date()}，ret1 已到 "
                    f"{pd.Timestamp(ret_last).date()} → 全量重算")

    t0 = time.time()
    series = build_series()
    output.parent.mkdir(parents=True, exist_ok=True)
    series.to_parquet(output)
    logger.success(
        f"已保存 {output} | shape={series.shape} | "
        f"{series.index.min().date()}~{series.index.max().date()} | "
        f"耗时 {time.time() - t0:.1f}s"
    )

    if write_broadcast_panels:
        stocks = pd.read_parquet(cfg.RET1_PANEL_PATH).columns
        write_broadcast(series, stocks)
    else:
        logger.info("WRITE_BROADCAST=False → 只产标量序列，未落广播宽表"
                    "（体检通过后把常量改 True 再跑）")
    return series


def main():
    ap = argparse.ArgumentParser(
        description="市场层广播特征 构建/日更（业务参数见脚本常量区，不走命令行）"
    )
    ap.add_argument("--full", action="store_true", help="全量重建（默认增量日更）")
    ap.add_argument("--output", type=Path,
                    default=cfg.RET1_PANEL_PATH.parent / "market_broadcast_series.parquet",
                    help="标量序列输出路径")
    args = ap.parse_args()
    logger.info(f"输出: {args.output} | 模式: {'全量重建' if args.full else '增量日更'}")
    build(args.output, full=args.full)


if __name__ == "__main__":
    main()
