"""
因子: reg_pe_hist — 经过历史增速调整的历史 PE 变化（情绪性估值变化）

定义（双层截面回归）：
  log_yoy   = log(net_profit_mrq_0 / net_profit_mrq_4)   单季同比对数
  log_npttm = log(net_profitTTM)                          TTM 净利润对数
  log_pe    = log(pe_ratio_ttm)                           PE_TTM 对数

  对每只股票按米筐返回的交易日序列做 60 期差分：
    delta_log_yoy   = log_yoy.diff(60)
    delta_log_npttm = log_npttm.diff(60)
    delta_log_pe    = log_pe.diff(60)

  第一层截面回归：delta_log_yoy = a · delta_log_npttm + b + residual_1
  第二层截面回归：delta_log_pe  = a' · residual_1 + b' + reg_pe_hist

预过滤：保留 net_profit_mrq_0 > 0 & net_profit_mrq_4 > 0 & net_profitTTM > 0 & pe_ratio_ttm > 0

方向：负向；分类：价值

实现说明：用 long format 处理，与生产管线 long 表语义一致——每只股票的 diff(60)
基于该股票实际有数据的交易日序列，而非整个市场的日历。
"""

import numpy as np
import pandas as pd
import rqdatac


def cross_sectional_regress(df: pd.DataFrame, y_col: str, x_cols: list,
                              add_intercept: bool = True,
                              min_samples: int = 30) -> pd.Series:
    """
    每天做横截面 OLS：y = X @ beta + intercept + residual
    返回 residual Series（与 df.index 同长，无效位置 NaN）。

    df 必须有 'date' 列。
    """
    p = len(x_cols) + (1 if add_intercept else 0)
    threshold = max(p + 5, min_samples)

    out = pd.Series(np.nan, index=df.index, dtype="float64")

    for dt, grp in df.groupby("date", sort=False):
        valid_mask = grp[y_col].notna() & grp[x_cols].notna().all(axis=1)
        if valid_mask.sum() < threshold:
            continue
        sub = grp.loc[valid_mask]
        Y = sub[y_col].to_numpy(dtype=np.float64)
        X = sub[x_cols].to_numpy(dtype=np.float64)
        if add_intercept:
            X = np.column_stack([X, np.ones(len(X))])
        try:
            beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
        except np.linalg.LinAlgError:
            continue
        resid = Y - X @ beta
        out.loc[sub.index] = resid

    return out


def fetch_long(stocks, fields, start_date, end_date):
    """拉多字段 long 表 [order_book_id, date, *fields]。"""
    df = rqdatac.get_factor(stocks, fields, start_date=start_date, end_date=end_date)
    if isinstance(df, pd.Series):
        df = df.to_frame(name=fields[0] if isinstance(fields, list) else fields)
    df = df.reset_index()
    df.columns = ["order_book_id", "date"] + list(df.columns[2:])
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["order_book_id", "date"]).reset_index(drop=True)


def main():
    start_date, end_date = "20100101", "20260527"
    rqdatac.init()
    stocks = rqdatac.all_instruments(type="CS")["order_book_id"].tolist()

    print("拉 4 个原料字段 ...")
    fields = ["net_profit_mrq_0", "net_profit_mrq_4", "net_profitTTM", "pe_ratio_ttm"]
    df = fetch_long(stocks, fields, start_date, end_date)

    # 正值过滤：drop 行（必须删行而非置 NaN，否则后续 groupby.diff(60) 行为不同）
    df = df.query(
        "net_profit_mrq_0 > 0 and net_profit_mrq_4 > 0 "
        "and net_profitTTM > 0 and pe_ratio_ttm > 0"
    ).reset_index(drop=True)

    # log
    df["log_yoy"]   = np.log(df["net_profit_mrq_0"] / df["net_profit_mrq_4"])
    df["log_npttm"] = np.log(df["net_profitTTM"])
    df["log_pe"]    = np.log(df["pe_ratio_ttm"])

    # 60 期 diff（按 order_book_id 分组，对应该股票的交易日序列）
    print("60 期 diff（per stock）...")
    for col in ["log_yoy", "log_npttm", "log_pe"]:
        df[f"delta_{col}"] = df.groupby("order_book_id")[col].diff(60)

    # 第一层截面回归：delta_log_yoy ~ delta_log_npttm
    print("第一层截面回归 ...")
    df["residual_1"] = cross_sectional_regress(
        df, y_col="delta_log_yoy", x_cols=["delta_log_npttm"],
        add_intercept=True, min_samples=30
    )

    # 第二层截面回归：delta_log_pe ~ residual_1
    print("第二层截面回归 ...")
    df["reg_pe_hist"] = cross_sectional_regress(
        df, y_col="delta_log_pe", x_cols=["residual_1"],
        add_intercept=True, min_samples=30
    )

    # 转宽表
    panel = df.pivot(index="date", columns="order_book_id", values="reg_pe_hist")
    panel = panel.sort_index()

    print(f"因子 shape: {panel.shape}, 非空率 {panel.notna().mean().mean():.2%}")
    panel.to_parquet("reg_pe_hist.parquet")


if __name__ == "__main__":
    main()
