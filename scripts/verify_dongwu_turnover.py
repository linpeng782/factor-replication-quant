"""东吴换手率四因子（turn20/str20/pct_turn20/gtr20）单股冒烟验证

直接从换手率面板用 numpy/pandas 手算，与 raw parquet 逐单元对照。
"""
import numpy as np
import pandas as pd

import config as cfg

STOCKS = ["000001.XSHE", "600519.XSHG", "300750.XSHE"]
RAW = cfg.RAW_FACTOR_BASE  # 需按 source/group 拼

turnover = pd.read_parquet(cfg.TURNOVER_RATE_PANEL_PATH)
turnover.index = pd.to_datetime(turnover.index)

paths = {
    "turn20": "dongwu/paper_07_stable_turnover/turn20.parquet",
    "str20": "dongwu/paper_07_stable_turnover/str20.parquet",
    "pct_turn20": "dongwu/paper_07_stable_turnover/pct_turn20.parquet",
    "gtr20": "dongwu/paper_13_gtr/gtr20.parquet",
}
base = cfg.RAW_FACTOR_BASE.parent if hasattr(cfg.RAW_FACTOR_BASE, "parent") else None
root = str(cfg.RAW_FACTOR_BASE).split("/dongwu")[0]
root = root.rstrip("/")

for name, rel in paths.items():
    fac = pd.read_parquet(f"{root}/{rel}")
    fac.index = pd.to_datetime(fac.index)
    for s in STOCKS:
        if s not in turnover.columns or s not in fac.columns:
            print(f"[skip] {name} {s} 不在面板")
            continue
        # 手算：先剔除 NaN 日（与 load_panel 的 stack 语义一致），再滚动
        t = turnover[s].dropna()
        if name == "turn20":
            expect = t.rolling(20, min_periods=20).mean()
        elif name == "str20":
            expect = t.rolling(20, min_periods=20).std()
        elif name == "gtr20":
            g = t / t.shift(1) - 1
            g = g.replace([np.inf, -np.inf], np.nan)
            expect = g.rolling(20, min_periods=20).std()
        else:
            base40 = t.rolling(40, min_periods=40).mean().shift(1)
            p = t / base40 - 1
            p = p.replace([np.inf, -np.inf], np.nan)
            expect = p.rolling(20, min_periods=20).mean()

        got = fac[s].reindex(expect.index)
        cmp = pd.DataFrame({"expect": expect, "got": got}).dropna(how="all")
        both = cmp.dropna()
        diff = (both["expect"] - both["got"]).abs()
        n_only_expect = cmp["got"].isna().sum()
        n_only_got = cmp["expect"].isna().sum()
        print(
            f"{name:12s} {s}: 对照 {len(both):5d} 个单元, 最大绝对差={diff.max():.3e}, "
            f"仅手算有={n_only_expect}, 仅产出有={n_only_got}"
        )
