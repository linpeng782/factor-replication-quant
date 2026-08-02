"""
ml_core.labels —— 标签变换策略（模型驱动，可插拔）
============================================================
两套刻意不同的标签口径，做成策略对象，由入口选用（绝不强行统一）：

  ExcessReturn  （LGBM/ml）：截面等权 demean → 超额收益（回归目标）
      mkt[t]   = mean_{j: can_buy[t,j]}( ret[t,j] )     # 当日可投 universe 等权市场
      excess   = ret - mkt                              # 对全网格减市场（保留全网格）
      （后续再由 scaling 的 WholeSetRobustZ 标准化；与中证全指超额相关 0.94、IC 等价）

  BinaryMedian  （MLP/ml_ht）：逐日截面中位数二分类
      median[t] = median( ret[t, pool[t]] 非NaN )       # pool=样本池（如 can_train）
      y[t,j]    = 1 if ret[t,j] > median[t] else 0      # 当日 > 中位数 → 1

  CSRankNormCDFRobust （LGBM 备选）：逐日截面 秩→正态分位→稳健Z→clip
      纯单调重标定（日内 Spearman 与 ExcessReturn 恒为 1），只改 L2 损失的样本权重：
      把「高波动日 / 极端收益样本吃掉大半梯度」拉平成等权高斯目标。

前两者口径分别对齐 ml.labels.build_excess_label 与 ml_ht.dataset.binarize_labels。
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from loguru import logger

import config


def load_forward_return(horizon: int = 20) -> pd.DataFrame:
    """读 forward_return_{horizon}d 宽表 (T,N)。"""
    path = config.LABELS_DIR / f"forward_return_{horizon}d.parquet"
    if not path.exists():
        raise FileNotFoundError(f"标签不存在：{path}")
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


class LabelTransform(ABC):
    """标签变换接口：把远期收益 ret 面板变成目标面板 (T,N)。"""

    name: str
    is_regression: bool   # True=回归；False=分类(0/1)
    needs_y_scaling: bool = True   # 目标是否还需再套一次全集 RobustZ（自带标准化的变换设 False）

    @abstractmethod
    def build_panel(self, ret: np.ndarray, can_buy: np.ndarray, pool: np.ndarray) -> np.ndarray:
        """ret/can_buy/pool 均 (T,N)；返回目标面板 (T,N) float32（缺失=NaN）。"""
        ...


class ExcessReturn(LabelTransform):
    """截面等权 demean 超额收益（对齐 ml.labels.build_excess_label，demean universe=can_buy）。"""

    name = "excess_return"
    is_regression = True

    def build_panel(self, ret: np.ndarray, can_buy: np.ndarray, pool: np.ndarray) -> np.ndarray:
        ret_for_mean = np.where(can_buy, ret, np.nan)            # 仅 can_buy 计入市场均值
        mkt = np.nanmean(ret_for_mean, axis=1, keepdims=True)   # (T,1) 当日市场等权收益
        excess = (ret - mkt).astype(np.float32)
        logger.info(f"[label] ExcessReturn：demean over can_buy（{int(np.isfinite(excess).sum()):,} 有效）")
        return excess


class BinaryMedian(LabelTransform):
    """逐日截面中位数二分类（对齐 ml_ht.dataset.binarize_labels，median pool=样本池）。"""

    name = "binary_median"
    is_regression = False
    needs_y_scaling = False

    def build_panel(self, ret: np.ndarray, can_buy: np.ndarray, pool: np.ndarray) -> np.ndarray:
        ret_pool = np.where(pool & np.isfinite(ret), ret, np.nan)     # 仅样本池内算中位数
        med = np.nanmedian(ret_pool, axis=1, keepdims=True)          # (T,1) 当日池中位数
        with np.errstate(invalid="ignore"):
            binary = (ret > med).astype(np.float32)                  # 全网格 > 当日中位数 → 1
        binary[~np.isfinite(med).repeat(ret.shape[1], axis=1)] = np.nan  # 当日无有效池 → NaN
        logger.info(f"[label] BinaryMedian：逐日池中位数二分类（正样本 {int(np.nansum(binary)):,}）")
        return binary


class CSRankNormCDFRobust(LabelTransform):
    """逐日截面 秩 → 正态分位(ppf) → 稳健Z(MAD) → clip（对齐 joey csrank_normcdf_robust）。

    与 ExcessReturn 的日内排序完全一致（Spearman≡1），差别只在数值刻度：
      - 每个交易日的目标被拉到同一尺度 → 高波动日不再垄断 L2 梯度
      - 目标近似标准正态、尾部被 clip → 3.9% 的极端样本不再贡献 45% 的平方损失
    自带标准化，needs_y_scaling=False（不再叠加全集 RobustZ）。
    """

    name = "csrank_normcdf_robust"
    is_regression = True
    needs_y_scaling = False

    def __init__(self, clip: float = 3.0) -> None:
        self.clip = clip

    def build_panel(self, ret: np.ndarray, can_buy: np.ndarray, pool: np.ndarray) -> np.ndarray:
        from scipy.stats import norm

        valid = pool & np.isfinite(ret)                                # 仅样本池内参与排名
        r = np.where(valid, ret, np.nan)
        rank = pd.DataFrame(r).rank(axis=1, method="average").to_numpy()   # 逐日截面平均秩
        n = valid.sum(axis=1, keepdims=True).astype(np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            pct = np.clip((rank - 0.5) / n, 1e-7, 1 - 1e-7)            # 分位点，避开 ppf 端点发散
            z = np.where(valid, norm.ppf(pct), np.nan)                 # 正态分位 → 高斯化
            med = np.nanmedian(z, axis=1, keepdims=True)
            mad = np.nanmedian(np.abs(z - med), axis=1, keepdims=True) * 1.4826
            out = np.clip((z - med) / np.where(mad > 1e-9, mad, np.nan), -self.clip, self.clip)
        out = out.astype(np.float32)
        logger.info(f"[label] CSRankNormCDFRobust：逐日 秩→ppf→稳健Z→clip±{self.clip}"
                    f"（{int(np.isfinite(out).sum()):,} 有效）")
        return out
