"""数据获取操作"""

import pandas as pd
from typing import Any, Dict

from . import OpRegistry


@OpRegistry.register("fetch")
def op_fetch(ctx: Dict, step: Dict, fetcher: Any) -> pd.DataFrame:
    """
    action: fetch
    从米筐获取数据
    """
    stocks = ctx.get("_universe", [])
    if not stocks:
        raise ValueError("fetch 操作前需要先确定股票池（universe）")

    fields = step.get("fields", [])
    api = step.get("api", "get_pit_financials_ex")
    output_name = step.get("output", "data")

    start_q = ctx.get("_start_quarter", "2020q1")
    end_q = ctx.get("_end_quarter", "2024q4")

    if api == "get_pit_financials_ex":
        df = fetcher.fetch_pit(stocks, fields, start_q, end_q)
        if isinstance(df.index, pd.MultiIndex):
            df = df.reset_index()
        elif df.index.name in ("order_book_id", "quarter"):
            df = df.reset_index()

    elif api == "get_factor":
        start_date = ctx.get("_start_date")
        end_date = ctx.get("_end_date")
        trade_date = ctx.get("_trade_date")

        field_dfs = []
        for field in fields:
            try:
                if start_date and end_date:
                    df_f = fetcher.get_factor(stocks, field, start_date=start_date, end_date=end_date)
                elif trade_date:
                    df_f = fetcher.get_factor(stocks, field, date=trade_date)
                else:
                    raise ValueError("get_factor 需要指定 trade_date 或 start_date+end_date")

                if df_f is not None and len(df_f) > 0:
                    if isinstance(df_f.index, pd.MultiIndex):
                        df_f = df_f.reset_index()
                    elif df_f.index.name == "order_book_id":
                        df_f = df_f.reset_index()
                    field_dfs.append(df_f)
            except Exception as e:
                print(f"     ⚠️ get_factor({field}) 失败: {e}")
                continue

        if not field_dfs:
            raise ValueError(f"get_factor 未获取到任何数据: fields={fields}")

        df = field_dfs[0]
        merge_cols = [c for c in ["order_book_id", "date"] if c in df.columns]
        for other in field_dfs[1:]:
            other_cols = [c for c in merge_cols if c in other.columns]
            other_field_cols = [c for c in other.columns if c not in other_cols]
            df = df.merge(other[other_cols + other_field_cols], on=other_cols, how="outer")

    elif api == "custom":
        # 自定义 API 调用（如米筐内部接口）
        command = step.get("command", "")
        if not command:
            raise ValueError("api=custom 时需要指定 command 参数")

        try:
            raw = fetcher._rq.client.get_client().execute(command)
        except Exception as e:
            raise ValueError(f"自定义 API 调用失败: {command}, 错误: {e}")

        columns = step.get("columns")
        if columns:
            df = pd.DataFrame(raw, columns=columns)
        else:
            df = pd.DataFrame(raw)

        # 特殊处理：中信行业分类自动日频化
        if command == "__internal__zx2019_industry":
            df = _process_zx_industry(df, fetcher, ctx)

    else:
        raise ValueError(f"暂不支持的 API: {api}")

    ctx[output_name] = df
    return df


def _process_zx_industry(df: pd.DataFrame, fetcher: Any, ctx: Dict) -> pd.DataFrame:
    """
    将 __internal__zx2019_industry 原始数据转换为日频 long 格式。

    输入: (first_industry_name, order_book_id, start_date)
    输出: (order_book_id, date, first_industry_name)
    """
    if "start_date" not in df.columns:
        raise ValueError("行业分类数据缺少 start_date 列")

    df["start_date"] = pd.to_datetime(df["start_date"])
    df = df.sort_values(["order_book_id", "start_date"])

    # pivot 为 wide: start_date × order_book_id
    value_col = "first_industry_name"
    if value_col not in df.columns:
        # 尝试找行业名称列
        candidates = [c for c in df.columns if "industry" in c.lower()]
        if candidates:
            value_col = candidates[0]
        else:
            raise ValueError(f"行业分类数据缺少行业名称列，现有列: {list(df.columns)}")

    id_col = "order_book_id"
    df_wide = df.pivot(index="start_date", columns=id_col, values=value_col)
    df_wide = df_wide.ffill()

    # 获取交易日列表
    start_date = ctx.get("_start_date", "2016-01-01")
    end_date = ctx.get("_end_date", "2025-12-31")
    try:
        trading_dates = fetcher.get_trading_dates(start_date, end_date)
        trading_dates = pd.to_datetime(trading_dates)
    except Exception as e:
        print(f"⚠️ 获取交易日列表失败: {e}，使用日期范围代替")
        trading_dates = pd.date_range(start_date, end_date)

    # reindex 到交易日，ffill
    df_wide = df_wide.reindex(index=trading_dates)
    df_wide = df_wide.ffill()

    # melt 为 long 格式
    df_long = df_wide.reset_index().melt(
        id_vars=["index"], var_name=id_col, value_name=value_col
    )
    df_long = df_long.rename(columns={"index": "date"})
    df_long = df_long.dropna(subset=[value_col])

    print(f"   行业分类已转换为日频: {len(df_long)} 行, {df_long[value_col].nunique()} 个行业")
    return df_long
