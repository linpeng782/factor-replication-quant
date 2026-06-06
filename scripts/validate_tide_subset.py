"""潮汐因子 子集快验(直接内存计算,不走缓存)：确认方向/量级 ~ 论文 RankIC -7.09%。"""
import glob, os
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from core import config
from core.minute_data import load_adjusted_minute_window
from core.operators.minute_tide import TideReducer

START, END = "2024-01-01", "2026-06-05"
N_STOCKS = 250

# 取最近一个日文件里的股票，前 N 只（保证有数据）
last = sorted(glob.glob(str(config.MINUTE_RAW_DIR / "[0-9]*.parquet")))[-1]
stocks = sorted(pd.read_parquet(last, columns=["order_book_id"])["order_book_id"].unique())[:N_STOCKS]
print(f"子集 {len(stocks)} 股 | 窗口 {START}~{END}")

print("读分钟+复权（一次性，~2min）...")
m = load_adjusted_minute_window(START, END, stocks=stocks)
print(f"  {len(m):,} 行")

red = TideReducer.from_step({"cache_key": "tide_v1"})
parts = [red.reduce(ob, g) for ob, g in m.groupby("order_book_id", sort=False)]
daily = pd.concat([p for p in parts if not p.empty], ignore_index=True)
panel = daily.pivot(index="date", columns="order_book_id", values="full_tide_rate").sort_index()
print(f"日频速率面板 {panel.shape}, 非空率 {panel.notna().mean().mean():.0%}")

# 全潮汐 = 20日均值
tide_full = panel.rolling(20, min_periods=15).mean()

# forward_return_20d 标签
lab = pd.read_parquet(config.LABELS_DIR / "forward_return_20d.parquet")
cols = tide_full.columns.intersection(lab.columns)
dates = tide_full.index.intersection(lab.index)
tf = tide_full.loc[dates, cols]
fr = lab.loc[dates, cols]

ics = []
for d in dates:
    a, b = tf.loc[d].to_numpy(float), fr.loc[d].to_numpy(float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() >= 30:
        ics.append(spearmanr(a[ok], b[ok]).correlation)
ics = np.array([x for x in ics if np.isfinite(x)])
print(f"\n=== 潮汐速率(原始,未翻转) 的 RankIC over {len(ics)} 天 ===")
print(f"  RankIC 均值 = {ics.mean():+.4f}  (论文全潮汐 -0.0709)")
print(f"  RankICIR    = {ics.mean()/ics.std():+.2f}")
print(f"  负占比      = {(ics<0).mean():.0%}")
print(f"  方向{'✅ 与论文一致(负)' if ics.mean()<0 else '❌ 方向相反'} | "
      f"量级{'✅ 同档' if abs(ics.mean())>0.02 else '⚠️ 偏弱(子集/短窗/无mask)'}")
