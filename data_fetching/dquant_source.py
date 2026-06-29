"""
dquant 单日取数 helper
================================
供 raw_ohlcv_dquant.py / ex_factors_jy.py 复用。

设计原则:
  - 父进程一次 `from dquant import data as ddata` + `ddata.get_trading_dates` 预热,
    子进程 fork 继承连接, 不重复 init。
  - 每个函数处理"单个交易日", 返回长表 DataFrame, 不做文件 IO。
  - 不依赖 rqdatac, 100% dquant。

口径(对齐 T0 结果):
  - 价格: get_price(1d, source="rq"), 未复权; 冒烟验证与 rq bit-exact。
  - 股池: all_instruments(None, D, D) 筛 type=="CS"。
  - 复权(见 T3): get_adj_factor(source="jy" 主, "ricequant" 兜底), 返回累计 adjfactor。
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from dquant import data as ddata


# ==================== 股票池 ====================

def get_cs_codes_for_day(date_str: str) -> list[str]:
    """当日全部 A 股(type==CS)代码列表。date_str 格式 'YYYY-MM-DD'。"""
    inst = ddata.all_instruments(None, date_str, date_str)
    if inst is None or len(inst) == 0:
        return []
    return inst[inst["type"] == "CS"]["order_book_id"].tolist()


# ==================== OHLCV ====================

OHLCV_KEEP = ["order_book_id", "open", "high", "low", "close", "volume", "total_turnover"]


def get_ohlcv_for_day(date_str: str) -> pd.DataFrame:
    """单日未复权 OHLCV 长表。

    返回列: order_book_id, open, high, low, close, volume, total_turnover(date as 'date' 列)
    缺失股票/没有行情的格子 = NaN。空行情返回空 DataFrame。
    """
    dd = date_str.replace("-", "")
    codes = get_cs_codes_for_day(date_str)
    if not codes:
        return pd.DataFrame(columns=OHLCV_KEEP + ["date"])

    df = ddata.get_price(
        order_book_ids=codes, start_date=dd, end_date=dd,
        frequency="1d", source="rq",
    )
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=OHLCV_KEEP + ["date"])

    # dquant 返回列含 amount, 重命名为 total_turnover(与现有 raw_ohlcv 一致)
    df = df.rename(columns={"amount": "total_turnover"})
    keep = [c for c in OHLCV_KEEP if c in df.columns]
    df = df[keep].copy()
    df["date"] = pd.Timestamp(date_str)
    return df


# ==================== 复权因子 (T3 用) ====================

def get_adj_factor_for_day(codes: list[str], date_str: str) -> pd.DataFrame:
    """单日复权因子长表。

    jy 主、ricequant 兜底(覆盖互补, 经验互补 ~0.5%)。
    返回: order_book_id, adjfactor (=累计复权因子, T0.3 已验证等价 rq ex_cum_factor)
    """
    dd = date_str.replace("-", "")
    adj_map: dict[str, float] = {}
    for src in ("jy", "ricequant"):
        missing = [c for c in codes if c not in adj_map]
        if not missing:
            break
        try:
            adj_df = ddata.get_adj_factor(order_book_ids=missing, trade_date=dd, source=src)
            if adj_df is not None and len(adj_df):
                adj_map.update(dict(zip(adj_df["order_book_id"], adj_df["adjfactor"])))
        except Exception:
            pass
    out = pd.DataFrame({"order_book_id": list(adj_map.keys()),
                        "adjfactor": list(adj_map.values())})
    return out


# ==================== 交易日历 ====================

def get_trading_days(start: str, end: str) -> list[str]:
    """[start, end] 内交易日列表, 格式 'YYYY-MM-DD'。"""
    s = start.replace("-", "")
    e = end.replace("-", "")
    dates = ddata.get_trade_dates(s, e, market="SH")
    return [d.strftime("%Y-%m-%d") for d in dates]


def latest_trading_date() -> str:
    """最近一个已收盘的交易日(17:00 前退到前一交易日)。"""
    latest = pd.Timestamp(ddata.get_latest_trading_date())
    now = pd.Timestamp.now()
    if now < latest.replace(hour=17, minute=0):
        prev = ddata.get_previous_trading_date(latest)
        return pd.Timestamp(prev).strftime("%Y-%m-%d")
    return latest.strftime("%Y-%m-%d")