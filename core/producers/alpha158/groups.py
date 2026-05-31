"""因子名 → 组（落盘 factors/<stage>/alpha158/<group>/ 的 group）。

四组对应 Alpha158Panel 的四类：kline / price / rolling / volume。
rolling/volume 因子名形如 '<家族><窗口>'（MA20、VSUMP5），去掉尾部窗口即家族。
"""
from __future__ import annotations

import re

from .factors import Alpha158Panel

_ROLLING = set(Alpha158Panel.ROLLING_FAMILIES)
_VOLUME = set(Alpha158Panel.VOLUME_FAMILIES)
_KLINE = set(Alpha158Panel.KLINE_FACTORS)
_PRICE = set(Alpha158Panel.PRICE_FACTORS)


def factor_group(name: str) -> str:
    """返回 kline / price / rolling / volume；无法归类抛错（防新增因子漏归组）。"""
    if name in _KLINE:
        return "kline"
    if name in _PRICE:
        return "price"
    family = re.sub(r"\d+$", "", name)  # 去掉尾部窗口数字
    if family in _ROLLING:
        return "rolling"
    if family in _VOLUME:
        return "volume"
    raise ValueError(f"无法为因子 {name!r} 归组（family={family!r}）")
