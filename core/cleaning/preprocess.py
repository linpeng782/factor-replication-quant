"""清洗（thin wrapper → alpha_shared.cleaning.preprocess）。"""

from alpha_shared.cleaning.preprocess import (  # noqa: F401
    prepare_factor,
    standardize_zscore,
    winsorize_mad,
)
