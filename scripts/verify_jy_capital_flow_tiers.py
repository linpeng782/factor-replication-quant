# -*- coding: utf-8 -*-
"""jy 分单资金流 value_range 门槛交叉验证

背景：研报（开源系列12）按挂单金额分四档：超大单(>100万)、大单(20-100万)、
中单(4-20万)、小单(<4万)。API 文档称 value_range 1/2/3/4 = 特大/大/中/小，
但 600519 初探显示 value_range=1 金额几乎为 0、=4 金额最大，与文档方向矛盾
（茅台股价下 <4万 的单只能是零股，不可能占大头）。本脚本用结构不可能性检验
严格确定映射方向，并验证数据完整性（配平）、起始日期与覆盖率。

检验逻辑（关键）：
  股价 P >= 420 元时，最小整手 100 股 >= 4.2万 > 4万，"小单(<4万)"档只可能
  包含零股成交，金额占比必然趋近 0。反之低价股(P<=5)中"特大单(>100万)"需
  单笔 >= 20万股，占比应明显偏低。据此可唯一确定 1..4 与四档的对应方向。
"""
import random
from pathlib import Path

import numpy as np
import pandas as pd

from dquant.data import get_capital_flow, get_price_daily

# ============ 参数区（手动修改） ============
VERIFY_DATE = "2024-06-03"  # 主验证日
BALANCE_SAMPLE_N = 300      # 配平检验抽样股票数
BAND_SAMPLE_N = 60          # 每个价格带抽样股票数上限
COVERAGE_DATES = ["2013-01-07", "2016-01-05", "2019-01-07", "2022-01-05", "2024-06-03"]
EARLIEST_PROBE_YEARS = list(range(2005, 2015))  # 起始日期逐年探测
PROBE_STOCKS = ["600519.XSHG", "000001.XSHE", "600000.XSHG"]
OHLCV_DIR = Path("/nfs/ofs-prediction/peterzhenglinpeng/market-data/daily-dquant/stock-ohlcv-dquant")
# ==========================================

pd.set_option("display.width", 220)
pd.set_option("display.float_format", lambda x: f"{x:,.2f}")
random.seed(42)

all_stocks = sorted(p.stem for p in OHLCV_DIR.glob("*.parquet"))
print(f"本地股票清单: {len(all_stocks)} 只\n")

# ---------- 一、当日全市场行情（价格带划分 + 配平基准） ----------
px = get_price_daily(all_stocks, VERIFY_DATE, VERIFY_DATE)
px = px[["order_book_id", "close", "volume", "amount", "deals"]].set_index("order_book_id")
px = px.apply(pd.to_numeric, errors="coerce")  # get_price_daily 返回字符串列，统一转数值
print(f"[{VERIFY_DATE}] 有行情股票: {len(px)} 只")

# ---------- 二、quote_type / 行数结构 + 配平检验 ----------
bal_ids = random.sample(sorted(px.index), BALANCE_SAMPLE_N)
cf = get_capital_flow(bal_ids, VERIFY_DATE, VERIFY_DATE, source="jy")
print(f"\n== 结构检查 ==\nquote_type 取值: {sorted(cf['quote_type'].unique())}")
rows_per_stock = cf.groupby("order_book_id").size()
print(f"每股行数分布: {rows_per_stock.value_counts().to_dict()}（4=四档齐全）")
print(f"value_range 取值: {sorted(cf['value_range'].unique())}")

agg = cf.groupby("order_book_id").agg(
    cf_value=("buy_value", "sum"), cf_sell=("sell_value", "sum"),
    cf_vol_b=("buy_volume", "sum"), cf_vol_s=("sell_volume", "sum"))
agg["cf_amount"] = agg["cf_value"] + agg["cf_sell"]
agg["cf_volume"] = agg["cf_vol_b"] + agg["cf_vol_s"]
agg = agg.join(px[["amount", "volume"]], how="inner")
amt_err = (agg["cf_amount"] / agg["amount"] - 1).abs()
vol_err = (agg["cf_volume"] / agg["volume"] - 1).abs()
print(f"\n== 配平检验（{len(agg)} 只，四档buy+sell 对比当日成交额/量） ==")
print(f"金额相对误差: 中位 {amt_err.median():.2e} | 均值 {amt_err.mean():.2e} | 最大 {amt_err.max():.2e} | >1% 占比 {(amt_err > 0.01).mean():.1%}")
print(f"数量相对误差: 中位 {vol_err.median():.2e} | 均值 {vol_err.mean():.2e} | 最大 {vol_err.max():.2e} | >1% 占比 {(vol_err > 0.01).mean():.1%}")

# ---------- 三、门槛方向性验证（结构不可能性检验） ----------
bands = {
    "P>=420(小单档结构性为0)": px.index[px["close"] >= 420],
    "200<=P<400": px.index[(px["close"] >= 200) & (px["close"] < 400)],
    "20<=P<50": px.index[(px["close"] >= 20) & (px["close"] < 50)],
    "P<=5(特大单档应偏低)": px.index[px["close"] <= 5],
}
band_ids = {k: random.sample(sorted(v), min(BAND_SAMPLE_N, len(v))) for k, v in bands.items()}
flat_ids = sorted({i for v in band_ids.values() for i in v})
cfb = get_capital_flow(flat_ids, VERIFY_DATE, VERIFY_DATE, source="jy")
cfb["total"] = cfb["buy_value"] + cfb["sell_value"]
share = (cfb.pivot_table(index="order_book_id", columns="value_range", values="total", aggfunc="sum")
         .pipe(lambda d: d.div(d.sum(axis=1), axis=0)))

print(f"\n== 各价格带 x value_range 金额占比均值（{VERIFY_DATE}） ==")
rows = []
for name, ids in band_ids.items():
    sub = share.reindex([i for i in ids if i in share.index]).mean()
    rows.append(pd.Series(sub, name=f"{name} (n={len(ids)})"))
band_mat = pd.DataFrame(rows)
band_mat.columns = [f"range{c}" for c in band_mat.columns]
print(band_mat.to_string(float_format=lambda x: f"{x:.2%}"))

# 零股签名：P>=420 组 range1 的成交量应 <100 股/日（只可能是零股）
hi_ids = [i for i in band_ids["P>=420(小单档结构性为0)"] if i in share.index]
r1 = cfb[(cfb["value_range"] == 1) & cfb["order_book_id"].isin(hi_ids)]
r1_vol = r1["buy_volume"] + r1["sell_volume"]
print(f"\n零股签名（P>=420 组 range1 单日总成交量）: 中位 {r1_vol.median():.0f} 股 | 最大 {r1_vol.max():.0f} 股 | <100股 占比 {(r1_vol < 100).mean():.1%}")

# ---------- 四、极端个股显微镜：600519 @ 2021-02-18（P约2500） ----------
print("\n== 600519 2021-02-18（P约2500：小单档仅容纳<16股零股，中单档仅16-80股零股） ==")
mt = get_capital_flow(["600519.XSHG"], "2021-02-18", "2021-02-18", source="jy")
print(mt[["value_range", "buy_volume", "sell_volume", "buy_value", "sell_value"]].to_string(index=False))

# ---------- 五、数据起始日期探测 ----------
print("\n== 起始日期探测（逐年首周） ==")
for y in EARLIEST_PROBE_YEARS:
    d0, d1 = f"{y}-01-01", f"{y}-01-15"
    n = len(get_capital_flow(PROBE_STOCKS, d0, d1, source="jy"))
    print(f"{y}: {'有数据 ' + str(n) + ' 行' if n else '无数据'}")
    if n:
        break

# ---------- 六、覆盖率 ----------
print("\n== 覆盖率（jy 有资金流数据股票数 / 当日有行情股票数） ==")
for d in COVERAGE_DATES:
    pxd = get_price_daily(all_stocks, d, d)
    cfd = get_capital_flow(sorted(pxd["order_book_id"].unique()), d, d, source="jy")
    print(f"{d}: {cfd['order_book_id'].nunique()} / {pxd['order_book_id'].nunique()}"
          f" = {cfd['order_book_id'].nunique() / max(len(pxd), 1):.1%}")

print("\n验证完成。")
