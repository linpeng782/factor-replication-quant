"""
中信一级行业指数 日收益面板 构建 / 日更 —— dquant 版
============================================================
与 industry_index.py（rqdatac）平行；默认写独立路径，不覆盖旧面板。

输出 (T 交易日 × 33 行业) 宽表面板：
  index   : DatetimeIndex, name='date'
  columns : 行业名（str），name='industry'
  values  : 该行业指数当日收益率（dquant get_index_price 的 pctchange）

数据源：dquant.get_index_price(CI005001~CI005030)
  - 字段 pctchange 与 close.pct_change() 差异 ≤1e-6，直接用
  - dquant 不支持 instruments(.INDX)，行业名用硬编码映射

命名对齐（与旧脚本一致）：
  额外 3 个旧名别名列，值复制对应新名：
    电力设备   → 电力设备及新能源 (CI005011)
    电子元器件 → 电子            (CI005025)
    餐饮旅游   → 消费者服务      (CI005015)
  输出 33 列 = 30 新名 + 3 旧名别名

两种模式
--------
  python industry_index_dquant.py          # 增量日更（默认）
  python industry_index_dquant.py --full   # 全量重建（2005-至今）

输出路径（优先级）：
  --output >
  $INDUSTRY_INDEX_RETURN_DQUANT_PATH >
  $FACTOR_REPL_DATA_ROOT/market-data/industry/industry_index_return_dquant.parquet
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
from loguru import logger

from data_fetching.dquant_source import latest_trading_date

FULL_START = "2005-01-01"
CITIC_L1_CODES = [f"CI0050{i:02d}.INDX" for i in range(1, 31)]

# dquant get_index_price 返回的 order_book_id 无 .INDX 后缀
CODE2NAME = {
    "CI005001": "石油石化",
    "CI005002": "煤炭",
    "CI005003": "有色金属",
    "CI005004": "电力及公用事业",
    "CI005005": "钢铁",
    "CI005006": "基础化工",
    "CI005007": "建筑",
    "CI005008": "建材",
    "CI005009": "轻工制造",
    "CI005010": "机械",
    "CI005011": "电力设备及新能源",
    "CI005012": "国防军工",
    "CI005013": "汽车",
    "CI005014": "商贸零售",
    "CI005015": "消费者服务",
    "CI005016": "家电",
    "CI005017": "纺织服装",
    "CI005018": "医药",
    "CI005019": "食品饮料",
    "CI005020": "农林牧渔",
    "CI005021": "银行",
    "CI005022": "非银行金融",
    "CI005023": "房地产",
    "CI005024": "交通运输",
    "CI005025": "电子",
    "CI005026": "通信",
    "CI005027": "计算机",
    "CI005028": "传媒",
    "CI005029": "综合",
    "CI005030": "综合金融",
}

LEGACY_ALIASES = {
    "电力设备": "电力设备及新能源",
    "电子元器件": "电子",
    "餐饮旅游": "消费者服务",
}


def _default_output() -> Path:
    explicit = os.environ.get("INDUSTRY_INDEX_RETURN_DQUANT_PATH")
    if explicit:
        return Path(explicit)
    root = Path(os.environ.get("FACTOR_REPL_DATA_ROOT", "/nfs/ofs-prediction/peterzhenglinpeng"))
    return root / "market-data" / "industry" / "industry_index_return_dquant.parquet"


def _normalize_code(c: str) -> str:
    return str(c).replace(".INDX", "")


def _fetch_ret_chunk(start: str, end: str) -> pd.DataFrame:
    """单次拉取一段区间的原始长表。"""
    from dquant import data as ddata

    s = start.replace("-", "")
    e = end.replace("-", "")
    raw = ddata.get_index_price(order_book_ids=CITIC_L1_CODES, start_date=s, end_date=e)
    if raw is None or len(raw) == 0:
        return pd.DataFrame()
    return raw


def _fetch_ret(start: str, end: str) -> pd.DataFrame:
    """取 30 个中信一级行业指数日收益 → (date × 行业名) 宽表。

    全量区间按年分块拉取，避免 dquant MySQL 超时（2013 Lost connection）。
    """
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    parts = []
    for y in range(start_ts.year, end_ts.year + 1):
        s = max(start_ts, pd.Timestamp(f"{y}-01-01")).strftime("%Y-%m-%d")
        e = min(end_ts, pd.Timestamp(f"{y}-12-31")).strftime("%Y-%m-%d")
        logger.info(f"  拉取 {s} ~ {e}")
        chunk = _fetch_ret_chunk(s, e)
        if chunk is not None and len(chunk):
            parts.append(chunk)

    if not parts:
        return pd.DataFrame()

    raw = pd.concat(parts, ignore_index=True)
    raw = raw.copy()
    raw["trade_date"] = pd.to_datetime(raw["trade_date"])
    raw["code"] = raw["order_book_id"].map(_normalize_code)
    raw["pctchange"] = pd.to_numeric(raw["pctchange"], errors="coerce")
    # 同年边界可能重复拉，按日期+code 去重
    raw = raw.drop_duplicates(subset=["trade_date", "code"], keep="last")

    ret = raw.pivot(index="trade_date", columns="code", values="pctchange")
    ret = ret.rename(columns=CODE2NAME)
    keep = [n for n in CODE2NAME.values() if n in ret.columns]
    ret = ret[keep].sort_index()

    for old, new in LEGACY_ALIASES.items():
        if new in ret.columns:
            ret[old] = ret[new]

    ret = ret.reindex(columns=sorted(ret.columns))
    ret.index.name = "date"
    ret.columns.name = "industry"
    return ret


def build(output: Path, full: bool = False) -> pd.DataFrame:
    end = latest_trading_date()
    output = Path(output)

    if full or not output.exists():
        logger.info(f"[全量] 重建 {FULL_START} ~ {end}")
        ret = _fetch_ret(FULL_START, end)
        if ret.empty:
            raise RuntimeError("dquant get_index_price 返回空")
    else:
        existing = pd.read_parquet(output)
        existing.index = pd.to_datetime(existing.index)
        last = existing.index.max()
        if pd.Timestamp(end) <= last:
            logger.success(f"[日更] 已最新（到 {last.date()}），无需更新")
            return existing
        # 多取缓冲日，防节假日边界
        buf_start = (last - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
        new_ret = _fetch_ret(buf_start, end)
        new_ret = new_ret[new_ret.index > last]
        if new_ret.empty:
            logger.success(f"[日更] 现有到 {last.date()}，无新交易日数据")
            return existing
        logger.info(f"[日更] 现有到 {last.date()}，补齐 {len(new_ret)} 个新交易日")
        ret = pd.concat([existing, new_ret])
        ret = ret[~ret.index.duplicated(keep="last")].sort_index()

    output.parent.mkdir(parents=True, exist_ok=True)
    ret.to_parquet(output)
    size_kb = output.stat().st_size / 1024
    logger.success(
        f"已保存 {output} | shape={ret.shape} | "
        f"{ret.index.min().date()}~{ret.index.max().date()} | "
        f"行业列={ret.shape[1]}（含3旧名别名）| {size_kb:.0f}KB"
    )
    return ret


def main():
    ap = argparse.ArgumentParser(description="中信一级行业指数日收益面板 构建/日更（dquant）")
    ap.add_argument("--full", action="store_true", help="全量重建（默认增量日更）")
    ap.add_argument("--output", type=Path, default=_default_output(), help="输出 parquet 路径")
    args = ap.parse_args()
    logger.info(f"输出: {args.output} | 模式: {'全量重建' if args.full else '增量日更'}")
    build(args.output, full=args.full)


if __name__ == "__main__":
    main()
