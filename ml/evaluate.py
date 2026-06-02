"""
模型评估：样本外 IC（对齐国金之十图表1 口径）
============================================================
  ŷ = model.predict(X_test) → 还原 (T,N) 面板
  模型 IC：逐日截面 Spearman corr(ŷ, 真实超额) → 序列；均值=IC，均值/标准差=ICIR
注：特征虽全集标准化，IC 仍按【逐日截面】算（输入口径 vs 评价口径是两件事）。
   真实超额用 meta['excess_raw']（未标准化）；ŷ 仅看排序，无需还原尺度。
"""
from __future__ import annotations

import pandas as pd


def predict_panel(model, X_test: pd.DataFrame, best_iteration: int | None = None) -> pd.DataFrame:
    """model.predict(X_test) → 还原成 (date, stock) 宽表 ŷ 面板。X_test 为 MultiIndex(date,stock)。"""
    pred = model.predict(X_test, num_iteration=best_iteration)
    return pd.Series(pred, index=X_test.index, name="yhat").unstack("stock")   # (T, N)


def model_ic(pred_panel: pd.DataFrame, excess_raw: pd.DataFrame, min_stocks: int = 30) -> pd.Series:
    """逐日截面 Spearman IC(ŷ, 真实超额) → 日 IC 序列（均值=IC，均值/std=ICIR）。"""
    ex = excess_raw.reindex(index=pred_panel.index, columns=pred_panel.columns)
    ic = {}
    for d in pred_panel.index:
        a = pred_panel.loc[d]; b = ex.loc[d]
        m = a.notna() & b.notna()
        if m.sum() >= min_stocks:
            ic[d] = a[m].rank().corr(b[m].rank())     # Spearman = rank 后 Pearson
    return pd.Series(ic).sort_index()
