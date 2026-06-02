"""
LightGBM 因子合成训练流水线
============================================================
消费 factor-rep 的因子库 + 标签 + mask，产出合成选股信号 ŷ。
与单因子评估（core/evaluation.py）解耦。设计见 docs/ml_pipeline_plan.md。

模块：
  labels      超额标签构造（截面 demean）
  preprocess  RobustZScoreScaler（全集 per-factor，train-only fit）
  dataset     特征矩阵 + 标签 + 时间划分（含 embargo）+ mask 对齐
  train       LightGBM(GBDT/MSE) + 早停
  evaluate    模型 IC + 分层回测
  run         CLI 入口
"""
