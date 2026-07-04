"""验证：牛市 valid 的截面超额信号是否真的更弱。

对比三段的截面信号强度指标：
  1. 截面超额收益的逐日方差（越大=个股分化越大=因子可解释空间越大）
  2. 截面超额收益的逐日标准差均值
  3. 绝对收益的逐日均值（判断牛/熊/震荡）
  4. 截面 top-bottom 分位差（top10% - bottom10% 均值，衡量可分选性）

三段：
  train : 2015-01-01 ~ 2024-10-31
  valid : 2024-12-01 ~ 2025-11-30  （大涨年）
  bt2026: 2026-01-01 ~ 2026-06-30  （回测段）
"""
import numpy as np
import pandas as pd
from core import config
from ml_core.labels import load_forward_return
from alpha_shared.cleaning.mask_loader import load_filter_masks

HORIZON = 20
ret = load_forward_return(HORIZON)
ret.index = pd.to_datetime(ret.index)

can_buy_mask, _ = load_filter_masks(
    combo_mask_path=config.COMBO_MASK_PATH,
    new_stock_mask_path=config.NEW_STOCK_MASK_PATH,
)
can_buy_mask = can_buy_mask.reindex(index=ret.index, columns=ret.columns).fillna(False).to_numpy(dtype=bool)
ret_arr = ret.to_numpy(dtype=np.float32)
dates = ret.index
stocks = ret.columns

# 截面 demean（对齐 ExcessReturn：仅 can_buy 计入市场均值）
ret_for_mean = np.where(can_buy_mask, ret_arr, np.nan)
mkt = np.nanmean(ret_for_mean, axis=1, keepdims=True)  # (T,1)
excess = ret_arr - mkt

SEGMENTS = {
    "train  (2015-01 ~ 2024-10)": ("2015-01-01", "2024-10-31"),
    "valid  (2024-12 ~ 2025-11)": ("2024-12-01", "2025-11-30"),
    "bt2026 (2026-01 ~ 2026-06)": ("2026-01-01", "2026-06-30"),
}

print(f"horizon={HORIZON}d | 全局 {dates[0].date()}~{dates[-1].date()} | {len(stocks)} 股票\n")
print(f"{'段':<30} {'天数':>5} {'日均绝对收益':>12} {'截面std均值':>12} "
      f"{'截面var均值':>12} {'top10%-bot10%':>14}")
print("-" * 90)

for name, (lo, hi) in SEGMENTS.items():
    mask = (dates >= pd.Timestamp(lo)) & (dates <= pd.Timestamp(hi))
    seg_ret = ret_arr[mask]
    seg_excess = excess[mask]
    seg_pm = can_buy_mask[mask]

    # 仅 can_buy 且非 NaN 的样本
    valid = seg_pm & np.isfinite(seg_ret)

    # 日均绝对收益（can_buy 等权）
    daily_abs = np.array([
        np.nanmean(seg_ret[t][seg_pm[t] & np.isfinite(seg_ret[t])])
        for t in range(seg_ret.shape[0])
    ])

    # 逐日截面超额 std（只算 can_buy 且非 NaN）
    daily_cs_std = np.array([
        np.nanstd(seg_excess[t][seg_pm[t] & np.isfinite(seg_excess[t])])
        for t in range(seg_excess.shape[0])
    ])

    # 逐日截面超额 var
    daily_cs_var = daily_cs_std ** 2

    # 逐日 top10% - bottom10% 分位差（超额收益的可分选性）
    daily_spread = []
    for t in range(seg_excess.shape[0]):
        vals = seg_excess[t][seg_pm[t] & np.isfinite(seg_excess[t])]
        if len(vals) < 20:
            continue
        q90 = np.nanpercentile(vals, 90)
        q10 = np.nanpercentile(vals, 10)
        daily_spread.append(q90 - q10)
    daily_spread = np.array(daily_spread)

    print(f"{name:<30} {mask.sum():>5} "
          f"{daily_abs.mean()*100:>10.3f}%  "
          f"{daily_cs_std.mean():>12.4f} "
          f"{daily_cs_var.mean():>12.4f} "
          f"{daily_spread.mean():>13.4f}")

print("\n解读：")
print("  截面std/var 越大 → 个股分化越大 → 因子可解释空间越大（信号越强）")
print("  top10%-bot10% 越大 → 截面可分选性越强 → 模型能区分好/坏股的空间越大")
print("  如果 valid(2024-12~2025-11) 的截面std/var 明显小于 train 和 bt2026，")
print("  则证实'牛市 valid 截面信号弱 → 早停砍树过多 → 184棵欠拟合'的推断。")
