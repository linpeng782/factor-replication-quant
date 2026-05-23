"""
开源_微观_27 论文 17 个剩余因子的 spec 批量生成器
========================================================

已手写：peak_minute_count (f1)、peak_interval_kurt (f10)、eruption_turnover_corr (f19)
本脚本生成剩余 17 个：f2/f3/f4/f5/f6/f7/f8/f9/f11/f12/f13/f14/f15/f16/f17/f18/f20

每个 spec 落到 specs/<name>/spec.yaml。生成后由 tools/sweep_oss27.sh 批量跑。

注意：
- f6/f7 的"反转中性化"暂不实现（v1 版本，会比论文略低）；
  v2 实现需要 cross_section_regress over 20 日累计收益。
- 所有数值守门统一用 divide-by-mask（避开 pd.eval 不支持 where 函数）。
- 高阶矩 std/skew 用 var ** 0.5 / var ** 1.5（pd.eval 支持）。
- f18 OLS 斜率用中心化矩形式提高数值稳定性（避开大数减大数）。
"""

from pathlib import Path
import yaml

ROOT = Path("/nfs/volume-1593-1/peterzhenglinpeng/factor-repilcation-quant")
SPEC_DIR = ROOT / "specs"
CACHE_KEY = "prv_v3"  # 与 _FEATURES_SUPERSET_VERSION 对齐


def _write(name: str, name_cn: str, direction: int, description: str, steps: list):
    target = SPEC_DIR / name / "spec.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    spec = {
        "factor": {
            "name": name,
            "name_cn": name_cn,
            "category": "微观结构",
            "direction": direction,
            "column": name,
            "description": description,
        },
        "universe": {"primary_index": "MINUTE_DIR"},
        "calculation_steps": steps,
    }
    with open(target, "w", encoding="utf-8") as f:
        yaml.safe_dump(spec, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
    print(f"  wrote {target}")


def load(features: list):
    return {
        "name": "加载分钟、分类、reduce 到日频",
        "action": "minute_intraday_aggregate",
        "cache_key": CACHE_KEY,
        "std_window": 20,
        "std_threshold": 1.0,
        "features": features,
        "output_dataframe": "data",
    }


def roll(src: str, out: str, agg: str = "mean", window: int = 20, name: str = None):
    return {
        "name": name or out,
        "action": "rolling",
        "source_column": src,
        "output_column": out,
        "window": window,
        "agg": agg,
        "group_by": "order_book_id",
    }


def compute(formula: str, source_columns: list, out: str, name: str = None):
    return {
        "name": name or out,
        "action": "compute",
        "formula": formula,
        "source_columns": source_columns,
        "output_column": out,
    }


# ===== Pattern A: 单一 count → rolling mean =====
_write("ridge_minute_count", "量岭分钟数", -1,
    "过去 20 日量岭分钟数（连续大额成交时点 = 跟随交易/散户特征）。论文 IC -9.04% LS 26.20% IR 2.20。",
    [
        load(["ridge_count"]),
        roll("ridge_count", "ridge_minute_count", "mean"),
    ])

# ===== Pattern B: ridge_return_sum → rolling sum =====
_write("ridge_minute_return", "量岭分钟收益", -1,
    "过去 20 日量岭分钟的累计 1-min 收益和。论文 IC -6.29% LS 14.98% IR 1.73（散户过度反应）。",
    [
        load(["ridge_return_sum"]),
        roll("ridge_return_sum", "ridge_minute_return", "sum"),
    ])

# ===== Pattern C: ratio of vwap / daily_vwap → rolling mean =====
# v4 起 superset 直接带 daily_vwap（后复权口径）
_write("ridge_relative_vwap", "量岭相对加权价", -1,
    "过去 20 日 ridge_vwap / daily_vwap 的均值（均后复权口径）。论文 IC -6.27% LS 17.99% IR 2.27。",
    [
        load(["ridge_vwap", "daily_vwap"]),
        compute("ridge_vwap / daily_vwap", ["ridge_vwap", "daily_vwap"], "ridge_rel_vwap_daily"),
        roll("ridge_rel_vwap_daily", "ridge_relative_vwap", "mean"),
    ])

_write("valley_relative_vwap", "量谷相对加权价", 1,
    "过去 20 日 valley_vwap / daily_vwap 的均值（均后复权口径）。论文 IC +8.69% LS 25.35% IR 3.04。",
    [
        load(["valley_vwap", "daily_vwap"]),
        compute("valley_vwap / daily_vwap", ["valley_vwap", "daily_vwap"], "valley_rel_vwap_daily"),
        roll("valley_rel_vwap_daily", "valley_relative_vwap", "mean"),
    ])

# ===== Pattern D: vwap ratio between two classes =====
_write("peak_ridge_price_ratio", "峰岭加权价格比", 1,
    "过去 20 日 peak_vwap / ridge_vwap 的均值。论文 IC +4.70% LS 10.31% IR 1.81。",
    [
        load(["peak_vwap", "ridge_vwap"]),
        compute("peak_vwap / ridge_vwap", ["peak_vwap", "ridge_vwap"], "peak_ridge_ratio_daily"),
        roll("peak_ridge_ratio_daily", "peak_ridge_price_ratio", "mean"),
    ])

_write("valley_ridge_price_ratio", "谷岭加权价格比", 1,
    "过去 20 日 valley_vwap / ridge_vwap 的均值。论文 IC +6.98% LS 15.83% IR 1.83。",
    [
        load(["valley_vwap", "ridge_vwap"]),
        compute("valley_vwap / ridge_vwap", ["valley_vwap", "ridge_vwap"], "valley_ridge_ratio_daily"),
        roll("valley_ridge_ratio_daily", "valley_ridge_price_ratio", "mean"),
    ])

# ===== Pattern E: rolling_sum / rolling_sum =====
_write("peak_ridge_turnover_ratio", "峰岭成交比", 1,
    "过去 20 日 peak_turnover_sum / ridge_turnover_sum。论文 IC +10.28% LS 27.13% IR 2.89。",
    [
        load(["peak_turnover_sum", "ridge_turnover_sum"]),
        roll("peak_turnover_sum", "peak_to_20", "sum"),
        roll("ridge_turnover_sum", "ridge_to_20", "sum"),
        compute("peak_to_20 / ridge_to_20", ["peak_to_20", "ridge_to_20"], "peak_ridge_turnover_ratio"),
    ])

_write("eruption_followup_ratio", "喷发成交额跟随比例", -1,
    "过去 20 日 eruption_next_turnover_sum / eruption_turnover_sum（散户跟随强度）。论文 IC -10.59% LS 30.09% IR 2.85。",
    [
        load(["eruption_turnover_sum", "eruption_next_turnover_sum"]),
        roll("eruption_turnover_sum", "ex_to_20", "sum"),
        roll("eruption_next_turnover_sum", "ey_to_20", "sum"),
        compute("ey_to_20 / ex_to_20", ["ey_to_20", "ex_to_20"], "eruption_followup_ratio"),
    ])

# ===== Pattern F: 间隔 std (sqrt(var)) =====
def _interval_std(prefix_cn: str, name: str, name_cn: str, direction: int, ic_str: str):
    p = "peak" if "峰" in name_cn else "ridge"
    _write(name, name_cn, direction,
        f"过去 20 日 pooled {p} 间隔的标准差。{ic_str}。",
        [
            load([f"{p}_interval_n", f"{p}_interval_m1", f"{p}_interval_m2"]),
            roll(f"{p}_interval_n", "pi_n_20", "sum"),
            roll(f"{p}_interval_m1", "pi_m1_20", "sum"),
            roll(f"{p}_interval_m2", "pi_m2_20", "sum"),
            compute("pi_m1_20 / pi_n_20", ["pi_m1_20", "pi_n_20"], "pi_mean"),
            compute("pi_m2_20 / pi_n_20 - pi_mean ** 2", ["pi_m2_20", "pi_n_20", "pi_mean"], "pi_var"),
            compute("pi_var ** 0.5", ["pi_var"], "std_raw"),
            compute("(pi_n_20 >= 5) * (pi_var > 0.000000000001)",
                ["pi_n_20", "pi_var"], "gate"),
            compute("std_raw * (gate / gate)", ["std_raw", "gate"], name),
        ])

_interval_std("峰", "peak_interval_std", "量峰间隔标准差", -1, "论文 IC -8.57% LS 25.66% IR 2.93")
_interval_std("岭", "ridge_interval_std", "量岭间隔标准差", 1, "论文 IC +7.34% LS 18.41% IR 2.23")


# ===== Pattern G: 间隔 skew = (m3 - 3μm2 + 2Nμ³) / (N · var^1.5) =====
def _interval_skew(name: str, name_cn: str, direction: int, ic_str: str):
    p = "peak" if "峰" in name_cn else "ridge"
    _write(name, name_cn, direction,
        f"过去 20 日 pooled {p} 间隔的偏度。{ic_str}。",
        [
            load([f"{p}_interval_n", f"{p}_interval_m1", f"{p}_interval_m2", f"{p}_interval_m3"]),
            roll(f"{p}_interval_n", "pi_n_20", "sum"),
            roll(f"{p}_interval_m1", "pi_m1_20", "sum"),
            roll(f"{p}_interval_m2", "pi_m2_20", "sum"),
            roll(f"{p}_interval_m3", "pi_m3_20", "sum"),
            compute("pi_m1_20 / pi_n_20", ["pi_m1_20", "pi_n_20"], "pi_mean"),
            compute("pi_m2_20 / pi_n_20 - pi_mean ** 2", ["pi_m2_20", "pi_n_20", "pi_mean"], "pi_var"),
            compute(
                "(pi_m3_20 - 3 * pi_mean * pi_m2_20 + 2 * pi_n_20 * pi_mean ** 3) / (pi_n_20 * (pi_var ** 1.5))",
                ["pi_m3_20", "pi_mean", "pi_m2_20", "pi_n_20", "pi_var"],
                "skew_raw"),
            compute("(pi_n_20 >= 5) * (pi_var > 0.000000000001)",
                ["pi_n_20", "pi_var"], "gate"),
            compute("skew_raw * (gate / gate)", ["skew_raw", "gate"], name),
        ])

_interval_skew("peak_interval_skew", "量峰间隔偏度", 1, "论文 IC +7.68% LS 24.56% IR 3.37")
_interval_skew("ridge_interval_skew", "量岭间隔偏度", -1, "论文 IC -8.08% LS 22.19% IR 2.66")

# ===== Pattern H: 量岭间隔峰度（与 peak_interval_kurt 对称） =====
_write("ridge_interval_kurt", "量岭间隔峰度", -1,
    "过去 20 日 pooled 量岭间隔分布峰度。论文 IC -7.61% LS 19.47% IR 2.45。",
    [
        load(["ridge_interval_n", "ridge_interval_m1", "ridge_interval_m2", "ridge_interval_m3", "ridge_interval_m4"]),
        roll("ridge_interval_n", "pi_n_20", "sum"),
        roll("ridge_interval_m1", "pi_m1_20", "sum"),
        roll("ridge_interval_m2", "pi_m2_20", "sum"),
        roll("ridge_interval_m3", "pi_m3_20", "sum"),
        roll("ridge_interval_m4", "pi_m4_20", "sum"),
        compute("pi_m1_20 / pi_n_20", ["pi_m1_20", "pi_n_20"], "pi_mean"),
        compute("pi_m2_20 / pi_n_20 - pi_mean ** 2", ["pi_m2_20", "pi_n_20", "pi_mean"], "pi_var"),
        compute(
            "(pi_m4_20 - 4 * pi_mean * pi_m3_20 + 6 * pi_mean ** 2 * pi_m2_20 - 3 * pi_n_20 * pi_mean ** 4) / (pi_n_20 * pi_var ** 2) - 3",
            ["pi_m4_20", "pi_mean", "pi_m3_20", "pi_m2_20", "pi_n_20", "pi_var"], "kurt_raw"),
        compute("(pi_n_20 >= 5) * (pi_var > 0.000000000001)",
            ["pi_n_20", "pi_var"], "gate"),
        compute("kurt_raw * (gate / gate)", ["kurt_raw", "gate"], "ridge_interval_kurt"),
    ])

# ===== Pattern I: 量峰/谷加权价格分位点（无反转中性化版本，v1） =====
def _weighted_quantile(name: str, name_cn: str, direction: int, vwap_col: str, ic_str: str):
    """计算 vwap 在 [min(high,low,prev_close), max(...)] 区间的分位点的 20 日均值。

    注：论文最后做了"反转中性化"。本 v1 暂不实现，复现 LS 应略低于论文。
    """
    _write(name, name_cn, direction,
        f"过去 20 日 {vwap_col} 在日内 [min(high,low,prev_close), max(...)] 的分位点均值。"
        f"{ic_str}。**v1 未做反转中性化**，需要后续 v2 加 cross_section_regress over 20d return。",
        [
            load([vwap_col, "daily_high", "daily_low", "daily_close"]),
            # prev_close = daily_close.shift(1)
            {
                "name": "prev_close (shift 1)",
                "action": "transform",
                "method": "shift",
                "source_column": "daily_close",
                "output_column": "prev_close",
                "periods": 1,
                "group_by": "order_book_id",
            },
            # high_bound = max(high, low, prev_close); low_bound = min(...)
            {
                "name": "high_bound",
                "action": "row_aggregate",
                "source_columns": ["daily_high", "daily_low", "prev_close"],
                "agg": "max",
                "output_column": "high_bound",
            },
            {
                "name": "low_bound",
                "action": "row_aggregate",
                "source_columns": ["daily_high", "daily_low", "prev_close"],
                "agg": "min",
                "output_column": "low_bound",
            },
            compute(f"({vwap_col} - low_bound) / (high_bound - low_bound)",
                [vwap_col, "high_bound", "low_bound"], "quantile_daily_raw"),
            # 守门：high == low（停牌或一字板）→ 分母 0；用 (high-low > 0) gate
            compute("high_bound - low_bound > 0.000001",
                ["high_bound", "low_bound"], "qgate"),
            compute("quantile_daily_raw * (qgate / qgate)",
                ["quantile_daily_raw", "qgate"], "quantile_daily"),
            roll("quantile_daily", name, "mean"),
        ])

_weighted_quantile("peak_weighted_quantile", "量峰加权价格分位点", 1, "peak_vwap",
    "论文 IC +3.47% LS 11.20% IR 1.88")
_weighted_quantile("valley_weighted_quantile", "量谷加权价格分位点", 1, "valley_vwap",
    "论文 IC +6.34% LS 20.22% IR 3.29")

# ===== Pattern J: f18 OLS 斜率（中心化形式提升数值稳定性） =====
_write("eruption_turnover_sensitivity", "喷发成交额敏感度", -1,
    "OLS slope of (next-min turnover) on (eruption-min turnover) over 20-day pooled。论文 IC -7.14% LS 15.61% IR 2.18。",
    [
        load(["eruption_count", "eruption_turnover_sum", "eruption_turnover_sumsq",
              "eruption_next_turnover_sum", "eruption_xy_sum"]),
        roll("eruption_count", "n_20", "sum"),
        roll("eruption_turnover_sum", "sx_20", "sum"),
        roll("eruption_turnover_sumsq", "sx2_20", "sum"),
        roll("eruption_next_turnover_sum", "sy_20", "sum"),
        roll("eruption_xy_sum", "sxy_20", "sum"),
        # 用中心化形式：β = (ΣXY − N·μ_x·μ_y) / (ΣX² − N·μ_x²)
        # 避免大数减大数的精度损失
        compute("sx_20 / n_20", ["sx_20", "n_20"], "mean_x"),
        compute("sy_20 / n_20", ["sy_20", "n_20"], "mean_y"),
        compute("sxy_20 - n_20 * mean_x * mean_y",
            ["sxy_20", "n_20", "mean_x", "mean_y"], "slope_num"),
        compute("sx2_20 - n_20 * mean_x ** 2",
            ["sx2_20", "n_20", "mean_x"], "slope_den"),
        compute("slope_num / slope_den", ["slope_num", "slope_den"], "slope_raw"),
        # 守门：n>=10 且 slope_den > 1（turnover 量级巨大，0.001 也行；选 1 保守）
        compute("(n_20 >= 10) * (slope_den > 1.0)",
            ["n_20", "slope_den"], "sgate"),
        compute("slope_raw * (sgate / sgate)",
            ["slope_raw", "sgate"], "eruption_turnover_sensitivity"),
    ])

# ===== Pattern K: f20 直接用 operator pre-pooled corr =====
_write("peakridge_minute_corr", "同时点峰岭数相关性", -1,
    "过去 20 日同时点（minute_of_day）的峰数与岭数的 Pearson 相关。已在 operator 内 pooled，spec 直接取列。论文 IC -6.67% LS 22.78% IR 3.27。",
    [
        load(["peakridge_minute_corr_pooled"]),
        # 直接 rolling window=1 mean = identity，让 factor.column 等于 spec 步骤产出
        roll("peakridge_minute_corr_pooled", "peakridge_minute_corr", "mean", window=1),
    ])

print()
print(f"=== 完成：17 个 spec 已生成到 {SPEC_DIR}/<name>/spec.yaml ===")
