"""Optional dependency detection for TA-Lib."""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import talib  # type: ignore[import-untyped]

    HAS_TALIB = True
except ImportError:
    talib = None  # type: ignore[assignment]
    HAS_TALIB = False


def _f64(series: pd.Series) -> np.ndarray:  # type: ignore[type-arg]
    """Extract a float64 ndarray from a pandas Series for TA-Lib compatibility."""
    return np.asarray(series, dtype=np.float64)
