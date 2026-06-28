"""全市场日频预测。"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from loguru import logger


@torch.no_grad()
def predict(
    model: nn.Module,
    X_test: np.ndarray,
    test_dates: np.ndarray,
    test_stocks: np.ndarray,
    batch_size: int = 8192,
    device: str = "cuda",
) -> tuple[list[str], list[np.ndarray], list[np.ndarray]]:
    """对测试集预测 P(Y=1)，按日分组返回。

    Returns:
        date_strs:  日期字符串列表
        stocks_per_day:  每天对应的股票代码数组
        probs_per_day:   每天的预测概率数组
    """
    model.eval()
    X_tensor = torch.from_numpy(X_test)
    n = len(X_test)

    # 批量预测
    all_probs = np.empty(n, dtype=np.float32)
    for i in range(0, n, batch_size):
        batch = X_tensor[i : i + batch_size].to(device)
        all_probs[i : i + batch_size] = torch.sigmoid(model(batch)).cpu().numpy().ravel()

    # 按日分组
    unique_dates = np.unique(test_dates)
    date_strs, stocks_per_day, probs_per_day = [], [], []
    for d in unique_dates:
        mask = test_dates == d
        stocks_per_day.append(test_stocks[mask])
        probs_per_day.append(all_probs[mask])
        date_strs.append(str(np.datetime64(d, "D")))

    logger.info(f"预测完成: {len(unique_dates)} 个交易日, {n:,} 样本")
    return date_strs, stocks_per_day, probs_per_day
