"""验证：公共网格对"未上市/已退市"股票的处理。

挑3只股票：
  000001.XSHE  平安银行（1991年上市，全程存在）
  688981.XSHG  中芯国际（2020-07-16上市，科创板新股）
  002681.XSHE  奋达科技（2010-12-31 之前不存在）

挑2个日期：
  2010-01-04   （中芯国际、奋达科技都还没上市）
  2024-06-03   （3只都已上市）

验证每个 (日期, 股票) 格子在三层防线中的状态：
  1. can_buy   — can_buy_mask reindex 后是 True 还是 False
  2. has_label — forward_return_20d reindex 后是 NaN 还是有值
  3. 因子值    — KMID 因子宽表 reindex 后是 NaN 还是有值
"""
import numpy as np
import pandas as pd
from loguru import logger
import config
from ml_core.universe import build_universe, dquant_grid
from ml_core.features import discover_features, load_factor_grid

# 关掉 loguru 噪音
logger.remove()

# 1. 建公共网格（全集）
dates, stocks = dquant_grid()
print(f"公共网格: {len(dates)} 天 × {len(stocks)} 股票")
print(f"日期范围: {dates[0].date()} ~ {dates[-1].date()}\n")

# 2. 挑3只股票 + 2个日期
test_stocks = ["000001.XSHE", "688981.XSHG", "002681.XSHE"]
test_dates = [pd.Timestamp("2010-01-04"), pd.Timestamp("2024-06-03")]

# 确认这些股票/日期在网格里
for s in test_stocks:
    print(f"  {s} 在网格列里: {s in stocks}")
for d in test_dates:
    print(f"  {d.date()} 在网格行里: {d in dates}")
print()

# 3. 建 universe（全集，不裁剪）
u = build_universe(horizon=20)
print(f"can_buy 总数: {u.can_buy.sum():,}")
print(f"has_label 总数: {u.has_label.sum():,}\n")

# 4. 读 KMID 因子
raw_map = discover_features(["alpha158-dquant"], stage="raw")
kmid_path = raw_map.get("KMID")
print(f"KMID 因子路径: {kmid_path}")
kmid_arr = load_factor_grid(kmid_path, u.dates, u.stocks)
print(f"KMID 网格形状: {kmid_arr.shape}\n")

# 5. 逐格子检查
print(f"{'日期':<12} {'股票':<14} {'can_buy':>8} {'has_label':>10} {'KMID值':>10} {'判定':>20}")
print("-" * 80)

for d in test_dates:
    di = dates.get_loc(d)
    for s in test_stocks:
        si = stocks.get_loc(s)
        cb = u.can_buy[di, si]
        hl = u.has_label[di, si]
        kv = kmid_arr[di, si]
        kv_str = f"{kv:.4f}" if np.isfinite(kv) else "NaN"

        # 判定这个格子会不会进训练/推理
        if not cb and not hl and not np.isfinite(kv):
            verdict = "三层全False/NaN（丢弃）"
        elif cb and hl and np.isfinite(kv):
            verdict = "三层全True（保留）"
        elif not cb:
            verdict = "can_buy=False（丢弃）"
        else:
            verdict = f"部分True（cb={cb} hl={hl} kv={np.isfinite(kv)})"

        print(f"{d.date():<12} {s:<14} {str(cb):>8} {str(hl):>10} {kv_str:>10} {verdict:>20}")

print()

# 6. 额外验证：中芯国际上市日附近的 can_buy 变化
print("=== 中芯国际 688981.XSHG 上市日附近 can_buy 变化 ===")
smic_si = stocks.get_loc("688981.XSHG")
# 找 can_buy 从 False 变 True 的第一个日期
smic_cb = u.can_buy[:, smic_si]
first_true_idx = np.argmax(smic_cb)  # 第一个 True
if smic_cb[first_true_idx]:
    first_true_date = dates[first_true_idx]
    print(f"  can_buy 首次 True: {first_true_date.date()}")
    # 看上市日前几天
    for offset in [-3, -2, -1, 0, 1, 2]:
        idx = first_true_idx + offset
        if 0 <= idx < len(dates):
            print(f"    {dates[idx].date()}: can_buy={smic_cb[idx]}, "
                  f"has_label={u.has_label[idx, smic_si]}, "
                  f"KMID={'NaN' if not np.isfinite(kmid_arr[idx, smic_si]) else f'{kmid_arr[idx, smic_si]:.4f}'}")
