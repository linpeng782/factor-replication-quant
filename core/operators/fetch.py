"""
fetch 算子
============================================================
从米筐拉数据，规范化为 long 格式后挂到 ctx.dataframes[output_dataframe]。

契约：
  - output_dataframe : str, 默认 "data"
  - 单字段: output_column: str
  - 多字段: output_columns: {rq_field: column_name}（一个 fetch 步并行拉多列）
  - 同名 output_dataframe 已存在 → 走 merge step 合并，不要重复 fetch 然后 silently 拼接

支持的 api：
  - get_factor               日频因子（单只或区间）
  - get_pit_financials_ex    PIT 财务（按 quarter）
  - custom                   自定义内部命令（含 __internal__zx2019_industry 自动日频化）
"""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd
from loguru import logger

from . import Context, OpRegistry


@OpRegistry.register("fetch")
def op_fetch(ctx: Context, step: Dict, fetcher: Any) -> None:
    target_df = step.get("output_dataframe", "data")
    if ctx.has_df(target_df):
        raise ValueError(
            f"fetch: DataFrame {target_df!r} 已存在；要扩列请用 merge step，不要重复 fetch"
        )

    field_to_col = _resolve_output_columns(step)
    if not ctx.universe:
        raise ValueError("fetch 调用前必须先确定 ctx.universe（股票池）")

    api = step.get("api", "get_pit_financials_ex")

    if api == "get_factor":
        df = _fetch_get_factor(ctx, fetcher, list(field_to_col.keys()))
    elif api == "get_pit_financials_ex":
        df = _fetch_pit(ctx, fetcher, list(field_to_col.keys()), step)
    elif api == "custom":
        df = _fetch_custom(ctx, fetcher, step)
    else:
        raise ValueError(f"fetch: 暂不支持的 api={api!r}")

    # 字段重命名为 spec 声明的 column 名
    rename_map = {f: c for f, c in field_to_col.items() if f != c and f in df.columns}
    if rename_map:
        df = df.rename(columns=rename_map)

    # 保留 [order_book_id, date, *output_cols] 列序
    output_cols = list(field_to_col.values())
    keep_cols = [c for c in ("order_book_id", "date") if c in df.columns] + output_cols
    df = df.loc[:, [c for c in keep_cols if c in df.columns]].copy()

    # 排序保证 groupby+diff/rolling 等下游算子语义稳定
    if "order_book_id" in df.columns and "date" in df.columns:
        df = df.sort_values(["order_book_id", "date"]).reset_index(drop=True)

    ctx.set_df(target_df, df)


# ── helpers ────────────────────────────────────────────────


def _resolve_output_columns(step: Dict) -> Dict[str, str]:
    """返回 {rq_field: column_name} 映射。spec_schema 已保证至少有一个。"""
    if "output_columns" in step:
        return dict(step["output_columns"])
    out = step["output_column"]
    if step.get("api") == "custom":
        # custom api 执行命令而非取米筐字段，没有 rq_field 概念；
        # output_column 即输出列名，与 _fetch_custom 内部产出列对齐
        return {out: out}
    fields = step.get("fields", [])
    if not fields:
        raise ValueError("fetch: 必须显式声明 fields")
    if len(fields) != 1:
        raise ValueError(
            "fetch: fields 长度 > 1 时必须用 output_columns dict 显式映射，"
            "不能用 output_column 单字段形式"
        )
    return {fields[0]: out}


def _fetch_get_factor(
    ctx: Context, fetcher: Any, fields: List[str]
) -> pd.DataFrame:
    """get_factor：日频因子。一次拉所有字段（米筐 API 原生支持 fields list）。"""
    if not (ctx.start_date and ctx.end_date) and not ctx.trade_date:
        raise ValueError("get_factor 需要 ctx.start_date+end_date 或 ctx.trade_date")

    if ctx.start_date and ctx.end_date:
        df = fetcher.get_factor(
            ctx.universe,
            fields,
            start_date=ctx.start_date,
            end_date=ctx.end_date,
        )
    else:
        df = fetcher.get_factor(ctx.universe, fields, date=ctx.trade_date)

    if df is None or len(df) == 0:
        raise ValueError(f"get_factor 无返回: fields={fields}")
    return _normalize_to_long(df)


def _fetch_pit(
    ctx: Context, fetcher: Any, fields: List[str], step: Dict
) -> pd.DataFrame:
    """PIT 财务数据：按 quarter 拉，包含 info_date / quarter 等元列。"""
    statements = step.get("statements", "latest")
    df = fetcher.fetch_pit(
        ctx.universe,
        fields,
        ctx.start_quarter,
        ctx.end_quarter,
        statements=statements,
    )
    df = _normalize_to_long(df)
    return df


def _fetch_custom(ctx: Context, fetcher: Any, step: Dict) -> pd.DataFrame:
    """自定义命令；目前只规范化 __internal__zx2019_industry 一种。"""
    command = step.get("command", "")
    if not command:
        raise ValueError("fetch api=custom 时必须指定 command")

    raw = fetcher._rq.client.get_client().execute(command)
    columns = step.get("columns_raw")
    df = pd.DataFrame(raw, columns=columns) if columns else pd.DataFrame(raw)

    if command == "__internal__zx2019_industry":
        df = _zx_industry_to_daily(df, fetcher, ctx)
        return df

    raise ValueError(f"fetch api=custom: 未规范化的 command={command!r}")


def _zx_industry_to_daily(
    df: pd.DataFrame, fetcher: Any, ctx: Context
) -> pd.DataFrame:
    """中信 2019 一级行业分类原始数据 → (order_book_id, date, first_industry_name)"""
    if "start_date" not in df.columns:
        raise ValueError("zx2019_industry: 缺少 start_date 列")
    df["start_date"] = pd.to_datetime(df["start_date"])

    value_col = "first_industry_name"
    if value_col not in df.columns:
        cands = [c for c in df.columns if "industry" in c.lower()]
        if not cands:
            raise ValueError(f"zx2019_industry: 找不到行业名称列；现有 {list(df.columns)}")
        value_col = cands[0]

    wide = (
        df.sort_values(["order_book_id", "start_date"])
        .pivot(index="start_date", columns="order_book_id", values=value_col)
        .ffill()
    )

    if not (ctx.start_date and ctx.end_date):
        raise ValueError("zx2019_industry 日频化需要 ctx.start_date+end_date")
    try:
        trading_dates = pd.to_datetime(
            fetcher.get_trading_dates(ctx.start_date, ctx.end_date)
        )
    except Exception as e:
        logger.warning(f"获取交易日失败: {e}; 退回 date_range")
        trading_dates = pd.date_range(ctx.start_date, ctx.end_date)

    wide = wide.reindex(index=trading_dates).ffill()
    long = (
        wide.reset_index()
        .melt(id_vars=["index"], var_name="order_book_id", value_name=value_col)
        .rename(columns={"index": "date"})
        .dropna(subset=[value_col])
    )
    return long.loc[:, ["order_book_id", "date", value_col]]


def _normalize_to_long(df: pd.DataFrame) -> pd.DataFrame:
    """米筐返回值规范为 (order_book_id, date, *fields) long 表"""
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()
    elif df.index.name in ("order_book_id", "quarter", "date"):
        df = df.reset_index()

    # 米筐 PIT 接口返回列叫 quarter 而非 date，下游 transform.diff_quarterly 等
    # 需要 quarter；但通用 long 输出统一以 date 为索引列。这里只做必要 rename。
    if "datetime" in df.columns and "date" not in df.columns:
        df = df.rename(columns={"datetime": "date"})
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    return df
