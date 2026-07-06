"""
Panel 级算子模块

所有算子的输入输出都是 Panel 格式的 DataFrame：
  - 行索引: date
  - 列索引: stock_code
  - 值:     字段值（后复权）

设计原则：
  1. 时间序列算子（沿 axis=0）：每只股票独立计算，默认行为
  2. 横截面算子（沿 axis=1）：每个时间点对所有股票计算（如 Rank）
  3. 所有算子都是向量化的（DataFrame 级 rolling/shift），性能远超 Series 级逐股票循环

参考：
  - 借鉴 alpha-191 的 Alpha191OperatorsPanel 架构
  - 适配 Alpha158 的算子需求，重点补充 Slope/IdxMax/IdxMin 等
"""

import numpy as np
import pandas as pd
from typing import Union
from numpy.lib.stride_tricks import sliding_window_view


class PanelOperators:
    """Panel 级因子算子集合（所有方法都是 DataFrame 级向量化）"""

    # ==================== 时间序列算子 ====================
    # 默认 axis=0，沿时间轴，每只股票独立计算

    @staticmethod
    def Ref(df: pd.DataFrame, period: int) -> pd.DataFrame:
        """滞后 period 天（等价于 alpha-191 的 Delay）"""
        return df.shift(period)

    @staticmethod
    def Mean(df: pd.DataFrame, window: int) -> pd.DataFrame:
        """window 日滚动均值"""
        return df.rolling(window, min_periods=1).mean()

    @staticmethod
    def Std(df: pd.DataFrame, window: int) -> pd.DataFrame:
        """window 日滚动标准差"""
        return df.rolling(window, min_periods=1).std()

    @staticmethod
    def Max(df: pd.DataFrame, window: int) -> pd.DataFrame:
        """window 日滚动最大值"""
        return df.rolling(window, min_periods=1).max()

    @staticmethod
    def Min(df: pd.DataFrame, window: int) -> pd.DataFrame:
        """window 日滚动最小值"""
        return df.rolling(window, min_periods=1).min()

    @staticmethod
    def Sum(df: pd.DataFrame, window: int) -> pd.DataFrame:
        """window 日滚动求和"""
        return df.rolling(window, min_periods=1).sum()

    @staticmethod
    def Quantile(df: pd.DataFrame, window: int, q: float) -> pd.DataFrame:
        """window 日滚动分位数"""
        return df.rolling(window, min_periods=1).quantile(q)

    @staticmethod
    def Slope(df: pd.DataFrame, window: int) -> pd.DataFrame:
        """
        window 日滚动线性回归斜率（完全向量化）

        斜率公式（x = [0, 1, ..., window-1] 为时间索引）：
            β = Σ((x - x̄)·y) / Σ((x - x̄)²)
        由于 x 固定，分母是常数，只需计算分子的滚动值。

        实现：用 sliding_window_view 切出滑动窗口后矩阵乘 x_centered

        Args:
            df:     DataFrame (时间 × 股票)
            window: 滚动窗口长度

        Returns:
            DataFrame，形状与输入相同，前 window-1 行为 NaN
        """
        n = window
        x_centered = np.arange(n) - (n - 1) / 2.0
        denom = (x_centered**2).sum()  # = n*(n²-1)/12

        values = df.values.astype(np.float64, copy=False)
        T, N = values.shape

        if T < n:
            # 数据太短，返回全 NaN
            return pd.DataFrame(
                np.full_like(values, np.nan), index=df.index, columns=df.columns
            )

        # sliding_window_view 沿时间轴切窗：形状 (T-n+1, N, n)
        windows = sliding_window_view(values, window_shape=n, axis=0)

        # 滑窗 × x_centered，得到分子 (T-n+1, N)
        # 注意：窗口内若有 NaN，结果也会是 NaN（符合预期）
        numerator = windows @ x_centered  # 形状 (T-n+1, N)

        slopes = numerator / denom

        # 前 window-1 行填 NaN，保持形状对齐
        result = np.full_like(values, np.nan)
        result[n - 1 :] = slopes

        return pd.DataFrame(result, index=df.index, columns=df.columns)

    @staticmethod
    def Rsquare(df: pd.DataFrame, window: int) -> pd.DataFrame:
        """
        滚动线性回归 R²（完全向量化）

        公式：R² = [Σ((x-x̄)(y-ȳ))]² / [Σ((x-x̄)²) · Σ((y-ȳ)²)]
        其中 x = [0, 1, ..., window-1] 固定；利用 sliding_window_view 一次算完

        返回值域 [0, 1]，前 window-1 行为 NaN
        """
        n = window
        x_centered = np.arange(n) - (n - 1) / 2.0
        x_var_sum = (x_centered**2).sum()

        values = df.values.astype(np.float64, copy=False)
        T, N = values.shape
        if T < n:
            return pd.DataFrame(
                np.full_like(values, np.nan), index=df.index, columns=df.columns
            )

        windows = sliding_window_view(values, window_shape=n, axis=0)  # (T-n+1, N, n)

        # x 已中心化，故 Σ(x·y_centered) = Σ(x·y)
        num_sum = windows @ x_centered  # (T-n+1, N)

        # y_var_sum = Σ((y - ȳ)²)，用恒等式避免再复制一份矩阵
        y_mean = windows.mean(axis=-1)  # (T-n+1, N)
        y_var_sum = (windows**2).sum(axis=-1) - n * (y_mean**2)  # (T-n+1, N)

        denom = x_var_sum * y_var_sum
        with np.errstate(divide="ignore", invalid="ignore"):
            r2 = np.where(denom > 1e-12, (num_sum**2) / denom, np.nan)

        result = np.full_like(values, np.nan)
        result[n - 1 :] = r2
        return pd.DataFrame(result, index=df.index, columns=df.columns)

    @staticmethod
    def Resi(df: pd.DataFrame, window: int) -> pd.DataFrame:
        """
        滚动线性回归残差（完全向量化）

        返回：窗口内最后一个点的残差 = y_last − predicted
        其中 predicted = ȳ + slope · (x_last − x̄)，x_last = n-1，x̄ = (n-1)/2

        化简：resi = y_last − ȳ − slope · (n-1)/2
        """
        n = window
        x_centered = np.arange(n) - (n - 1) / 2.0
        x_var_sum = (x_centered**2).sum()
        x_mean = (n - 1) / 2.0

        values = df.values.astype(np.float64, copy=False)
        T, N = values.shape
        if T < n:
            return pd.DataFrame(
                np.full_like(values, np.nan), index=df.index, columns=df.columns
            )

        windows = sliding_window_view(values, window_shape=n, axis=0)  # (T-n+1, N, n)
        y_last = windows[..., -1]
        y_mean = windows.mean(axis=-1)
        num_sum = windows @ x_centered
        slope = num_sum / x_var_sum
        resi = y_last - y_mean - slope * x_mean

        result = np.full_like(values, np.nan)
        result[n - 1 :] = resi
        return pd.DataFrame(result, index=df.index, columns=df.columns)

    @staticmethod
    def TsRank(df: pd.DataFrame, window: int) -> pd.DataFrame:
        """
        时序百分位排名（与原版 Alpha158 Rank 对齐）

        对每个时间点 t：
          取 [t-window+1, t] 的窗口值，返回 current = values[t] 相对于
          历史（非 NaN 且不含 current）的严格小于比例。
          - current 为 NaN 时返回 NaN
          - 窗口内非 NaN 值 < 2 时返回 0.5
        """
        n = window
        values = df.values.astype(np.float64, copy=False)
        T, N = values.shape
        if T < n:
            return pd.DataFrame(
                np.full_like(values, np.nan), index=df.index, columns=df.columns
            )

        windows = sliding_window_view(values, window_shape=n, axis=0)  # (T-n+1, N, n)
        current = windows[..., -1]  # (T-n+1, N)，窗口末尾
        past = windows[..., :-1]  # (T-n+1, N, n-1)

        # 有效计数：窗口内非 NaN 的数量（含 current）
        valid_count = (~np.isnan(windows)).sum(axis=-1)  # (T-n+1, N)

        # 历史中严格小于 current 的数量（NaN < x 始终为 False，自动忽略 NaN）
        less_count = (past < current[..., None]).sum(axis=-1)  # (T-n+1, N)

        with np.errstate(divide="ignore", invalid="ignore"):
            rank = np.where(
                np.isnan(current),
                np.nan,
                np.where(
                    valid_count < 2,
                    0.5,
                    less_count / np.maximum(valid_count - 1, 1),
                ),
            )

        result = np.full_like(values, np.nan)
        result[n - 1 :] = rank
        return pd.DataFrame(result, index=df.index, columns=df.columns)

    @staticmethod
    def IdxMax(df: pd.DataFrame, window: int) -> pd.DataFrame:
        """
        window 日内最大值的位置（1-based，从窗口起点计数）

        全 NaN 窗口返回 NaN（常见于新股上市前 / 长期停牌）
        """
        values = df.values.astype(np.float64, copy=False)
        T, N = values.shape

        if T < window:
            return pd.DataFrame(
                np.full_like(values, np.nan), index=df.index, columns=df.columns
            )

        windows = sliding_window_view(values, window_shape=window, axis=0)

        # 把 NaN 替换为 -inf 以便 argmax，然后把全 NaN 位置手动标 NaN
        all_nan = np.isnan(windows).all(axis=-1)  # (T-window+1, N)
        windows_filled = np.where(np.isnan(windows), -np.inf, windows)
        idx = np.argmax(windows_filled, axis=-1).astype(np.float64)  # (T-window+1, N)
        idx = np.where(all_nan, np.nan, idx + 1)  # 1-based

        result = np.full_like(values, np.nan)
        result[window - 1 :] = idx

        return pd.DataFrame(result, index=df.index, columns=df.columns)

    @staticmethod
    def IdxMin(df: pd.DataFrame, window: int) -> pd.DataFrame:
        """window 日内最小值的位置（1-based），全 NaN 窗口返回 NaN"""
        values = df.values.astype(np.float64, copy=False)
        T, N = values.shape

        if T < window:
            return pd.DataFrame(
                np.full_like(values, np.nan), index=df.index, columns=df.columns
            )

        windows = sliding_window_view(values, window_shape=window, axis=0)

        all_nan = np.isnan(windows).all(axis=-1)
        windows_filled = np.where(np.isnan(windows), np.inf, windows)
        idx = np.argmin(windows_filled, axis=-1).astype(np.float64)
        idx = np.where(all_nan, np.nan, idx + 1)

        result = np.full_like(values, np.nan)
        result[window - 1 :] = idx

        return pd.DataFrame(result, index=df.index, columns=df.columns)

    # ==================== 统计算子 ====================

    @staticmethod
    def RollingCorr(x: pd.DataFrame, y: pd.DataFrame, window: int) -> pd.DataFrame:
        """
        window 日滚动相关系数（每只股票独立）

        pandas 的 DataFrame.rolling().corr(DataFrame) 会自动按列对齐
        """
        return x.rolling(window, min_periods=max(2, window // 2)).corr(y)

    @staticmethod
    def RollingCov(x: pd.DataFrame, y: pd.DataFrame, window: int) -> pd.DataFrame:
        """window 日滚动协方差（每只股票独立）"""
        return x.rolling(window, min_periods=max(2, window // 2)).cov(y)

    # ==================== 横截面算子 ====================
    # axis=1，沿股票轴操作，每个时间点计算

    @staticmethod
    def CrossRank(df: pd.DataFrame) -> pd.DataFrame:
        """
        横截面排名（每个时间点对所有股票百分位排名，pct=True）

        用于 alpha-191 风格的 Rank(...) 算子
        """
        return df.rank(axis=1, method="min", pct=True)

    # ==================== 数学算子 ====================
    # element-wise 操作

    @staticmethod
    def Abs(df: pd.DataFrame) -> pd.DataFrame:
        """绝对值"""
        return df.abs()

    @staticmethod
    def Log(df: pd.DataFrame) -> pd.DataFrame:
        """自然对数"""
        return np.log(df)

    @staticmethod
    def Sign(df: pd.DataFrame) -> pd.DataFrame:
        """符号函数"""
        return np.sign(df)

    @staticmethod
    def Greater(
        a: Union[pd.DataFrame, float], b: Union[pd.DataFrame, float]
    ) -> Union[pd.DataFrame, float]:
        """逐元素取较大值"""
        return np.maximum(a, b)

    @staticmethod
    def Less(
        a: Union[pd.DataFrame, float], b: Union[pd.DataFrame, float]
    ) -> Union[pd.DataFrame, float]:
        """逐元素取较小值"""
        return np.minimum(a, b)
