"""未来收益率面板（thin wrapper → alpha_shared.evaluation.returns；本仓默认 horizons）。"""

from alpha_shared.evaluation.returns import build_forward_returns as _impl


def build_forward_returns(price_panel, horizons=(1, 5, 10, 20)):
    """与抽取前签名一致：horizons 默认 (1, 5, 10, 20)。"""
    return _impl(price_panel, horizons=horizons)
