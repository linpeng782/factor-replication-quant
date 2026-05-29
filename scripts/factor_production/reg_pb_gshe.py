"""
因子: reg_pb_gshe — 经过 EP 中位数分组的 PB 估值残差因子

定义：
  原料：pb_ratio_lf, net_profit_mrq_0, total_equity_mrq_0, market_cap_3
  衍生：
    roe_mrq = net_profit_mrq_0 / total_equity_mrq_0
    ep      = net_profit_mrq_0 / market_cap_3
    log_pb  = log(pb)
  过滤：剔除 pb <= 0 或 roe <= 0 的行（**删行**，非置 NaN）
  滚动：ep_median = ep.rolling(252, min_periods=60).median()  逐股（按 long 表实际行序）
  截面 MAD 去极值（n=3, scale=1.4826）：ep_median, roe, log_pb 各自每日做 MAD clip
  分组：按 ep_median_w 升序截面分 10 组（pct rank → 1..10）
  回归：每个 (date, ep_group) 内 OLS：log_pb_w = a · roe_w + b + residual
       样本数 < max(p+5, min_samples) = max(7, 15) = 15 时该组 NaN

方向：负向；分类：价值

实现：用 long format，与生产管线 long 表语义完全一致。
"""

import numpy as np
import pandas as pd
import rqdatac


def fetch_long(stocks, fields, start_date, end_date):
    """拉多字段 long 表 [order_book_id, date, *fields]。"""
    df = rqdatac.get_factor(stocks, fields, start_date=start_date, end_date=end_date)
    if isinstance(df, pd.Series):
        df = df.to_frame(name=fields[0] if isinstance(fields, list) else fields)
    df = df.reset_index()
    df.columns = ["order_book_id", "date"] + list(df.columns[2:])
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["order_book_id", "date"]).reset_index(drop=True)


def mad_winsorize(s: pd.Series, n: float = 3.0) -> pd.Series:
    """对单组 series 做 MAD clip。"""
    med = s.median()
    mad = (s - med).abs().median() * 1.4826
    if mad == 0:
        return s
    return s.clip(lower=med - n * mad, upper=med + n * mad)


def cross_sectional_regress(df: pd.DataFrame, y_col: str, x_cols: list,
                             group_keys: list,
                             add_intercept: bool = True,
                             min_samples: int = 15) -> pd.Series:
    """
    每个 (date, group_id) 内做 OLS，返回 residual Series。

    样本数不足 max(p+5, min_samples) 时整组 NaN。
    """
    p = len(x_cols) + (1 if add_intercept else 0)
    threshold = max(p + 5, min_samples)

    out = pd.Series(np.nan, index=df.index, dtype="float64")

    for _key, sub in df.groupby(group_keys, sort=False):
        valid_mask = sub[y_col].notna() & sub[x_cols].notna().all(axis=1)
        n_valid = int(valid_mask.sum())
        if n_valid < threshold:
            continue
        sub_v = sub.loc[valid_mask]
        Y = sub_v[y_col].to_numpy(dtype=np.float64)
        X = sub_v[x_cols].to_numpy(dtype=np.float64)
        if add_intercept:
            X = np.column_stack([X, np.ones(len(X))])
        try:
            beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
        except np.linalg.LinAlgError:
            continue
        out.loc[sub_v.index] = Y - X @ beta

    return out


def main():
    start_date, end_date = "20100101", "20260527"
    rqdatac.init()
    stocks = rqdatac.all_instruments(type="CS")["order_book_id"].tolist()

    print("拉 4 个原料字段 ...")
    fields = ["pb_ratio_lf", "net_profit_mrq_0", "total_equity_mrq_0", "market_cap_3"]
    df = fetch_long(stocks, fields, start_date, end_date)

    # 衍生
    df["roe_mrq"] = df["net_profit_mrq_0"] / df["total_equity_mrq_0"]
    df["ep"]      = df["net_profit_mrq_0"] / df["market_cap_3"]
    df["log_pb"]  = np.log(df["pb_ratio_lf"])

    # 过滤：drop 行（pb > 0 and roe > 0）
    df = df.query("pb_ratio_lf > 0 and roe_mrq > 0").reset_index(drop=True)

    # EP 过去 252 日中位数（per stock，按 long 表实际行序）
    print("EP 过去 252 日中位数 ...")
    df["ep_median"] = df.groupby("order_book_id")["ep"].transform(
        lambda x: x.rolling(window=252, min_periods=60).median()
    )

    # 截面 MAD 去极值：ep_median, roe_mrq, log_pb
    print("截面 MAD 去极值 ...")
    df["ep_median_w"] = df.groupby("date")["ep_median"].transform(mad_winsorize)
    df["roe_w"]       = df.groupby("date")["roe_mrq"].transform(mad_winsorize)
    df["log_pb_w"]    = df.groupby("date")["log_pb"].transform(mad_winsorize)

    # EP 中位数截面分 10 组（升序 pct rank 映射 1..10）
    print("EP 中位数截面分 10 组 ...")
    pct = df.groupby("date")["ep_median_w"].rank(pct=True, ascending=True, method="average")
    df["ep_group"] = np.minimum(np.floor(pct * 10) + 1, 10)
    df["ep_group"] = df["ep_group"].astype("Int64")

    # 分组截面回归
    print("分组截面回归（每个 (date, ep_group) 组）...")
    df["reg_pb_gshe"] = cross_sectional_regress(
        df, y_col="log_pb_w", x_cols=["roe_w"],
        group_keys=["date", "ep_group"],
        add_intercept=True, min_samples=15
    )

    # 转宽表
    panel = df.pivot(index="date", columns="order_book_id", values="reg_pb_gshe")
    panel = panel.sort_index()

    print(f"因子 shape: {panel.shape}, 非空率 {panel.notna().mean().mean():.2%}")
    panel.to_parquet("reg_pb_gshe.parquet")


if __name__ == "__main__":
    main()
