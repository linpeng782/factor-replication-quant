"""
Alpha158 Panel 级因子实现（完整版 158 个因子）

设计：
  - 继承 PanelOperators，以算子组合表达每个因子
  - 所有因子一次性对全市场股票向量化计算，性能远超 Series 级逐股票循环
  - 公式与原版 Alpha158MigratedFixed (operators.py + alpha158.py) 保持一致

因子清单（共 158 个）：
  K线形态 (9):  KMID, KLEN, KMID2, KUP, KUP2, KLOW, KLOW2, KSFT, KSFT2
  价格     (4): OPEN0, HIGH0, LOW0, VWAP0
  Rolling (23×5=115)，windows 默认 [5, 10, 20, 30, 60]：
    ROC, MA, STD, BETA, RSQR, RESI, MAX, MIN, QTLU, QTLD, RANK, RSV,
    IMAX, IMIN, IMXD, CORR, CORD, CNTP, CNTN, CNTD, SUMP, SUMN, SUMD
  成交量  (6×5=30): VMA, VSTD, WVMA, VSUMP, VSUMN, VSUMD
"""

from typing import Dict, List, Optional
from loguru import logger

import numpy as np
import pandas as pd

from .panel_operators import PanelOperators


class Alpha158Panel(PanelOperators):
    """
    Alpha158 Panel 级因子计算器

    输入：
        panels (dict[str, DataFrame]): 字段 -> Panel 数据
            必需字段：open, high, low, close, volume
            可选字段：vwap (缺失时 VWAP0 使用 (H+L+C)/3 近似)

    用法：
        alpha = Alpha158Panel(panels)
        factor_dict = alpha.compute_all(windows=[5, 10, 20, 30, 60])
    """

    SOURCE_NAME = "alpha158"

    # 所有因子计算需要的最小字段集合
    REQUIRED_FIELDS: List[str] = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "vwap",
    ]

    # 默认滚动窗口
    DEFAULT_WINDOWS: List[int] = [5, 10, 20, 30, 60]

    def __init__(self, panels: Dict[str, pd.DataFrame]):
        missing = [
            f for f in ("open", "high", "low", "close", "volume") if f not in panels
        ]
        if missing:
            raise ValueError(f"缺少必要字段：{missing}")

        self.open = panels["open"]
        self.high = panels["high"]
        self.low = panels["low"]
        self.close = panels["close"]
        self.volume = panels["volume"]
        self.vwap = panels.get("vwap", None)

        logger.info(
            f"Alpha158Panel 初始化完成：shape={self.close.shape}, "
            f"时间={self.close.index.min()} ~ {self.close.index.max()}"
        )

    # ==================== K 线形态因子 (9个) ====================

    def KMID(self) -> pd.DataFrame:
        """(close - open) / open"""
        return (self.close - self.open) / self.open

    def KLEN(self) -> pd.DataFrame:
        """(high - low) / open"""
        return (self.high - self.low) / self.open

    def KMID2(self) -> pd.DataFrame:
        """(close - open) / (high - low + ε)"""
        return (self.close - self.open) / (self.high - self.low + 1e-12)

    def KUP(self) -> pd.DataFrame:
        """(high - max(open, close)) / open — 上影线相对开盘"""
        return (self.high - np.maximum(self.open, self.close)) / self.open

    def KUP2(self) -> pd.DataFrame:
        """(high - max(open, close)) / (high - low + ε)"""
        return (self.high - np.maximum(self.open, self.close)) / (
            self.high - self.low + 1e-12
        )

    def KLOW(self) -> pd.DataFrame:
        """(min(open, close) - low) / open — 下影线相对开盘"""
        return (np.minimum(self.open, self.close) - self.low) / self.open

    def KLOW2(self) -> pd.DataFrame:
        """(min(open, close) - low) / (high - low + ε)"""
        return (np.minimum(self.open, self.close) - self.low) / (
            self.high - self.low + 1e-12
        )

    def KSFT(self) -> pd.DataFrame:
        """(2*close - high - low) / open — 偏移度"""
        return (2 * self.close - self.high - self.low) / self.open

    def KSFT2(self) -> pd.DataFrame:
        """(2*close - high - low) / (high - low + ε)"""
        return (2 * self.close - self.high - self.low) / (self.high - self.low + 1e-12)

    # ==================== 价格因子 (4个) ====================

    def OPEN0(self) -> pd.DataFrame:
        """open / close"""
        return self.open / self.close

    def HIGH0(self) -> pd.DataFrame:
        """high / close"""
        return self.high / self.close

    def LOW0(self) -> pd.DataFrame:
        """low / close"""
        return self.low / self.close

    def VWAP0(self) -> pd.DataFrame:
        """vwap / close；vwap 缺失时用 (H+L+C)/3 近似"""
        if self.vwap is not None:
            return self.vwap / self.close
        logger.warning("vwap 字段缺失，VWAP0 使用 (H+L+C)/3 近似")
        return (self.high + self.low + self.close) / 3 / self.close

    # ==================== Rolling 因子 (23 × window 数) ====================

    def ROC(self, w: int) -> pd.DataFrame:
        """Ref(close, w) / close"""
        return self.Ref(self.close, w) / self.close

    def MA(self, w: int) -> pd.DataFrame:
        """Mean(close, w) / close"""
        return self.Mean(self.close, w) / self.close

    def STD(self, w: int) -> pd.DataFrame:
        """Std(close, w) / close"""
        return self.Std(self.close, w) / self.close

    def BETA(self, w: int) -> pd.DataFrame:
        """Slope(close, w) / close — 滚动回归斜率归一化（注：原版 BETA 与 RSTR 同公式）"""
        return self.Slope(self.close, w) / self.close

    def RSQR(self, w: int) -> pd.DataFrame:
        """Rsquare(close, w) — 滚动回归 R²"""
        return self.Rsquare(self.close, w)

    def RESI(self, w: int) -> pd.DataFrame:
        """Resi(close, w) / close — 滚动回归残差归一化"""
        return self.Resi(self.close, w) / self.close

    def MAX(self, w: int) -> pd.DataFrame:
        """Max(high, w) / close"""
        return self.Max(self.high, w) / self.close

    def MIN(self, w: int) -> pd.DataFrame:
        """Min(low, w) / close"""
        return self.Min(self.low, w) / self.close

    def QTLU(self, w: int) -> pd.DataFrame:
        """Quantile(close, w, 0.8) / close — 上分位数"""
        return self.Quantile(self.close, w, 0.8) / self.close

    def QTLD(self, w: int) -> pd.DataFrame:
        """Quantile(close, w, 0.2) / close — 下分位数"""
        return self.Quantile(self.close, w, 0.2) / self.close

    def RANK(self, w: int) -> pd.DataFrame:
        """TsRank(close, w) — 时序排名"""
        return self.TsRank(self.close, w)

    def RSV(self, w: int) -> pd.DataFrame:
        """(close - Min(low, w)) / (Max(high, w) - Min(low, w) + ε)"""
        return (self.close - self.Min(self.low, w)) / (
            self.Max(self.high, w) - self.Min(self.low, w) + 1e-12
        )

    def IMAX(self, w: int) -> pd.DataFrame:
        """IdxMax(high, w) / w — 最大值位置归一化"""
        return self.IdxMax(self.high, w) / w

    def IMIN(self, w: int) -> pd.DataFrame:
        """IdxMin(low, w) / w — 最小值位置归一化"""
        return self.IdxMin(self.low, w) / w

    def IMXD(self, w: int) -> pd.DataFrame:
        """(IdxMax(high, w) - IdxMin(low, w)) / w"""
        return (self.IdxMax(self.high, w) - self.IdxMin(self.low, w)) / w

    def CORR(self, w: int) -> pd.DataFrame:
        """RollingCorr(close, log(volume+1), w)"""
        return self.RollingCorr(self.close, self.Log(self.volume + 1), w)

    def CORD(self, w: int) -> pd.DataFrame:
        """RollingCorr(close / Ref(close,1), log(volume / Ref(volume,1) + 1), w)"""
        close_ret = self.close / self.Ref(self.close, 1)
        volume_ret = self.Log(self.volume / self.Ref(self.volume, 1) + 1)
        return self.RollingCorr(close_ret, volume_ret, w)

    def CNTP(self, w: int) -> pd.DataFrame:
        """Mean((close > Ref(close,1)), w) — 上涨天数占比"""
        up = (self.close > self.Ref(self.close, 1)).astype(float)
        return self.Mean(up, w)

    def CNTN(self, w: int) -> pd.DataFrame:
        """Mean((close < Ref(close,1)), w) — 下跌天数占比"""
        down = (self.close < self.Ref(self.close, 1)).astype(float)
        return self.Mean(down, w)

    def CNTD(self, w: int) -> pd.DataFrame:
        """CNTP - CNTN"""
        return self.CNTP(w) - self.CNTN(w)

    def SUMP(self, w: int) -> pd.DataFrame:
        """
        Sum(Greater(close-Ref(close,1), 0), w) / (Sum(|close-Ref(close,1)|, w) + ε)
        上涨幅度占总绝对变动的比例
        """
        pc = self.close - self.Ref(self.close, 1)
        pos = pc * (pc > 0).astype(float)
        return self.Sum(pos, w) / (self.Sum(self.Abs(pc), w) + 1e-12)

    def SUMN(self, w: int) -> pd.DataFrame:
        """Sum(Greater(Ref(close,1)-close, 0), w) / (Sum(|Δ|, w) + ε)"""
        pc = self.close - self.Ref(self.close, 1)
        neg = (-pc) * (pc < 0).astype(float)
        return self.Sum(neg, w) / (self.Sum(self.Abs(pc), w) + 1e-12)

    def SUMD(self, w: int) -> pd.DataFrame:
        """SUMP - SUMN"""
        return self.SUMP(w) - self.SUMN(w)

    # ==================== 成交量因子 (6 × window 数) ====================

    def VMA(self, w: int) -> pd.DataFrame:
        """Mean(volume, w) / (volume + ε)"""
        return self.Mean(self.volume, w) / (self.volume + 1e-12)

    def VSTD(self, w: int) -> pd.DataFrame:
        """Std(volume, w) / (volume + ε)"""
        return self.Std(self.volume, w) / (self.volume + 1e-12)

    def WVMA(self, w: int) -> pd.DataFrame:
        """
        成交量加权价格变化波动率:
          weighted = |close/Ref(close,1) - 1| * volume
          WVMA = Std(weighted, w) / (Mean(weighted, w) + ε)
        """
        weighted = self.Abs(self.close / self.Ref(self.close, 1) - 1) * self.volume
        return self.Std(weighted, w) / (self.Mean(weighted, w) + 1e-12)

    def VSUMP(self, w: int) -> pd.DataFrame:
        """Sum(Greater(volume - Ref(volume,1), 0), w) / (Sum(|Δvol|, w) + ε)"""
        vc = self.volume - self.Ref(self.volume, 1)
        pos = self.Greater(vc, 0)
        return self.Sum(pos, w) / (self.Sum(self.Abs(vc), w) + 1e-12)

    def VSUMN(self, w: int) -> pd.DataFrame:
        """Sum(Greater(Ref(volume,1)-volume, 0), w) / (Sum(|Δvol|, w) + ε)"""
        vc = self.volume - self.Ref(self.volume, 1)
        neg = self.Greater(-vc, 0)
        return self.Sum(neg, w) / (self.Sum(self.Abs(vc), w) + 1e-12)

    def VSUMD(self, w: int) -> pd.DataFrame:
        """
        (Sum(Greater(Δvol, 0), w) - Sum(Greater(-Δvol, 0), w)) / (Sum(|Δvol|, w) + ε)
        == VSUMP - VSUMN（等价式）
        """
        vc = self.volume - self.Ref(self.volume, 1)
        sum_pos = self.Sum(self.Greater(vc, 0), w)
        sum_neg = self.Sum(self.Greater(-vc, 0), w)
        return (sum_pos - sum_neg) / (self.Sum(self.Abs(vc), w) + 1e-12)

    # ==================== 批量计算入口 ====================

    # 分类标识（方便外部按类别筛选）
    KLINE_FACTORS = [
        "KMID",
        "KLEN",
        "KMID2",
        "KUP",
        "KUP2",
        "KLOW",
        "KLOW2",
        "KSFT",
        "KSFT2",
    ]
    PRICE_FACTORS = ["OPEN0", "HIGH0", "LOW0", "VWAP0"]
    ROLLING_FAMILIES = [
        "ROC",
        "MA",
        "STD",
        "BETA",
        "RSQR",
        "RESI",
        "MAX",
        "MIN",
        "QTLU",
        "QTLD",
        "RANK",
        "RSV",
        "IMAX",
        "IMIN",
        "IMXD",
        "CORR",
        "CORD",
        "CNTP",
        "CNTN",
        "CNTD",
        "SUMP",
        "SUMN",
        "SUMD",
    ]
    VOLUME_FAMILIES = ["VMA", "VSTD", "WVMA", "VSUMP", "VSUMN", "VSUMD"]

    @classmethod
    def list_factor_names(
        cls,
        windows: Optional[List[int]] = None,
        include_kline: bool = True,
        include_price: bool = True,
        include_rolling: bool = True,
        include_volume: bool = True,
    ) -> List[str]:
        """
        列出当前配置下会产出的所有因子名（不真正计算，仅按命名规则展开）

        命名规则：
            - K 线 / 价格类：直接用 KLINE_FACTORS / PRICE_FACTORS 中的名字（无窗口）
            - Rolling / Volume：f"{fam}{w}"，如 ROC5、VMA20

        与 compute_all 完全对应，便于 build_factors 做"已存在则跳过"的增量判断
        """
        if windows is None:
            windows = cls.DEFAULT_WINDOWS

        names: List[str] = []
        if include_kline:
            names.extend(cls.KLINE_FACTORS)
        if include_price:
            names.extend(cls.PRICE_FACTORS)
        if include_rolling:
            names.extend(f"{fam}{w}" for w in windows for fam in cls.ROLLING_FAMILIES)
        if include_volume:
            names.extend(f"{fam}{w}" for w in windows for fam in cls.VOLUME_FAMILIES)
        return names

    def compute_all(
        self,
        windows: Optional[List[int]] = None,
        include_kline: bool = True,
        include_price: bool = True,
        include_rolling: bool = True,
        include_volume: bool = True,
        only_names: Optional[List[str]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        批量计算 Alpha158 全量因子

        参数：
            windows:         Rolling/Volume 因子的窗口列表，默认 [5,10,20,30,60]
            include_*:       按类别开关（便于只算部分类别做调试）
            only_names:      仅计算名字在该集合中的因子（增量模式专用）。
                             None 表示不过滤，全部计算

        返回：
            dict[因子名 -> DataFrame(时间 × 股票)]
        """
        if windows is None:
            windows = self.DEFAULT_WINDOWS

        only_set = set(only_names) if only_names is not None else None

        factors: Dict[str, pd.DataFrame] = {}

        logger.info(
            f"开始批量计算 Alpha158Panel 因子：windows={windows}, "
            f"kline={include_kline}, price={include_price}, "
            f"rolling={include_rolling}, volume={include_volume}, "
            f"only_names={'全部' if only_set is None else f'{len(only_set)} 个'}"
        )

        # K 线形态
        if include_kline:
            for name in self.KLINE_FACTORS:
                if only_set is not None and name not in only_set:
                    continue
                factors[name] = getattr(self, name)()

        # 价格
        if include_price:
            for name in self.PRICE_FACTORS:
                if only_set is not None and name not in only_set:
                    continue
                factors[name] = getattr(self, name)()

        # Rolling（时序类）
        if include_rolling:
            for w in windows:
                for fam in self.ROLLING_FAMILIES:
                    name = f"{fam}{w}"
                    if only_set is not None and name not in only_set:
                        continue
                    factors[name] = getattr(self, fam)(w)

        # Volume（成交量类）
        if include_volume:
            for w in windows:
                for fam in self.VOLUME_FAMILIES:
                    name = f"{fam}{w}"
                    if only_set is not None and name not in only_set:
                        continue
                    factors[name] = getattr(self, fam)(w)

        # 统一把 inf 替换为 nan
        for name, df in factors.items():
            factors[name] = df.replace([np.inf, -np.inf], np.nan)

        logger.success(f"Alpha158Panel 因子计算完成，共 {len(factors)} 个")
        return factors
