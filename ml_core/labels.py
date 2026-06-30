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

两者口径分别对齐 ml.labels.build_excess_label 与 ml_ht.dataset.binarize_labels。
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from loguru import logger

from core import config


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
    is_regression: bool   # True=回归(需 y 标准化)；False=分类(0/1，不标准化)

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

    def build_panel(self, ret: np.ndarray, can_buy: np.ndarray, pool: np.ndarray) -> np.ndarray:
        ret_pool = np.where(pool & np.isfinite(ret), ret, np.nan)     # 仅样本池内算中位数
        med = np.nanmedian(ret_pool, axis=1, keepdims=True)          # (T,1) 当日池中位数
        with np.errstate(invalid="ignore"):
            binary = (ret > med).astype(np.float32)                  # 全网格 > 当日中位数 → 1
        binary[~np.isfinite(med).repeat(ret.shape[1], axis=1)] = np.nan  # 当日无有效池 → NaN
        logger.info(f"[label] BinaryMedian：逐日池中位数二分类（正样本 {int(np.nansum(binary)):,}）")
        return binary
