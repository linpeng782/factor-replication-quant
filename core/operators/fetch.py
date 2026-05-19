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
    else:
        raise ValueError(f"暂不支持的 API: {api}")

    ctx[output_name] = df
    return df
