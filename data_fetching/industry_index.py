"""
中信一级行业指数 日收益面板 构建 / 日更
============================================================
输出 (T 交易日 × 33 行业) 宽表面板：
  index   : DatetimeIndex, name='date'
  columns : 行业名（str，去"中信"前缀；如 '石油石化'/'银行'），name='industry'
  values  : 该行业指数当日收益率（close.pct_change()）

数据源：rqdatac 中信一级行业指数 CI005001~CI005030.INDX（共 30 个）。
  历史起点 2010-06（少数 2010-04），与联合动量研报回测区间 2010.01-2023.12 吻合。

命名对齐（关键）：industry_panel_zx 跨 2019-12-02 混用中信新旧行业名，含 3 个旧名
  指数侧只有 30 个新名。为让消费端零特判，本面板额外加 3 个【旧名别名列】，
  其值复制对应新名指数收益：
      电力设备   → 电力设备及新能源 (CI005011)
      电子元器件 → 电子            (CI005025)
      餐饮旅游   → 消费者服务      (CI005015)
  故输出 33 列 = 30 新名 + 3 旧名别名。

两种模式
--------
  python industry_index.py          # 增量日更（默认）：仅追加缺失的近期交易日
  python industry_index.py --full   # 全量重建（2010-至今）

输出路径（优先级）：--output > $INDUSTRY_INDEX_RETURN_PATH > $FACTOR_REPL_DATA_ROOT/market-data/industry/industry_index_return.parquet
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
import rqdatac
from loguru import logger

FULL_START = "2010-01-01"          # 中信一级行业指数最早 2010-04
CITIC_L1_CODES = [f"CI0050{i:02d}.INDX" for i in range(1, 31)]  # CI005001~CI005030
INCR_BUFFER_DAYS = 10              # 增量时多取的缓冲日（算 pct_change 首日需前一日收盘）

# industry_panel_zx 旧名 → 对应中信新名（指数侧名）
LEGACY_ALIASES = {
    "电力设备": "电力设备及新能源",
    "电子元器件": "电子",
    "餐饮旅游": "消费者服务",
}


def _default_output() -> Path:
    explicit = os.environ.get("INDUSTRY_INDEX_RETURN_PATH")
    if explicit:
        return Path(explicit)
    root = Path(os.environ.get("FACTOR_REPL_DATA_ROOT", "/nfs/ofs-prediction/peterzhenglinpeng"))
    return root / "market-data" / "industry" / "industry_index_return.parquet"


def _fetch_close(start: str, end: str) -> pd.DataFrame:
    """取 30 个中信一级行业指数收盘价 → (date × 行业名) 宽表（列名去'中信'前缀）。"""
    px = rqdatac.get_price(
        CITIC_L1_CODES, start_date=start, end_date=end,
        frequency="1d", fields=["close"],
    )
    if px is None or len(px) == 0:
        return pd.DataFrame()
    close = px["close"].unstack("order_book_id")        # (date × code)
    close.index = pd.to_datetime(close.index)
    # code → 行业名（去"中信"前缀）
    code2name = {}
    for c in close.columns:
        sym = rqdatac.instruments(c).symbol            # 如 '中信石油石化'
        code2name[c] = sym[2:] if sym.startswith("中信") else sym
    close = close.rename(columns=code2name).sort_index()
    close.index.name = "date"
    return close


def _close_to_return(close: pd.DataFrame) -> pd.DataFrame:
    """收盘价 → 日收益；加 3 个旧名别名列。"""
    ret = close.pct_change()
    for old, new in LEGACY_ALIASES.items():
        if new in ret.columns:
            ret[old] = ret[new]
    ret = ret.reindex(columns=sorted(ret.columns))
    ret.index.name = "date"
    ret.columns.name = "industry"
    return ret


def build(output: Path, full: bool = False) -> pd.DataFrame:
    rqdatac.init()
    today = pd.Timestamp.today().normalize().strftime("%Y-%m-%d")
    output = Path(output)

    if full or not output.exists():
        logger.info(f"[全量] 重建 {FULL_START} ~ {today}")
        close = _fetch_close(FULL_START, today)
        if close.empty:
            raise RuntimeError("取指数收盘价为空")
        ret = _close_to_return(close)
        ret = ret.iloc[1:]   # 首日 pct_change 全 NaN，丢弃
    else:
        existing = pd.read_parquet(output)
        existing.index = pd.to_datetime(existing.index)
        last = existing.index.max()
        new_dates = pd.to_datetime(rqdatac.get_trading_dates(
            (last + pd.Timedelta(days=1)).strftime("%Y-%m-%d"), today))
        if len(new_dates) == 0:
            logger.success(f"[日更] 已最新（到 {last.date()}），无需更新")
            return existing
        # 多取缓冲日算新日的 pct_change 首日
        buf_start = (last - pd.Timedelta(days=INCR_BUFFER_DAYS)).strftime("%Y-%m-%d")
        close = _fetch_close(buf_start, today)
        new_ret = _close_to_return(close)
        new_ret = new_ret[new_ret.index > last]   # 只保留严格新的日
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
    ap = argparse.ArgumentParser(description="中信一级行业指数日收益面板 构建/日更")
    ap.add_argument("--full", action="store_true", help="全量重建（默认增量日更）")
    ap.add_argument("--output", type=Path, default=_default_output(), help="输出 parquet 路径")
    args = ap.parse_args()
    logger.info(f"输出: {args.output} | 模式: {'全量重建' if args.full else '增量日更'}")
    build(args.output, full=args.full)


if __name__ == "__main__":
    main()
