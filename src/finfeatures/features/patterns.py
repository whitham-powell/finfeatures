"""
Candlestick pattern recognition features.

When TA-Lib is available, all 61 CDL* pattern functions are used.
Otherwise, a small set of common patterns is computed in pure pandas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from finfeatures.core._compat import HAS_TALIB, talib
from finfeatures.core.base import Columns, Feature


def _talib_pattern_names() -> list[str]:
    """Return sorted CDL* function names from TA-Lib."""
    groups = talib.get_function_groups()  # type: ignore[union-attr]
    return sorted(groups["Pattern Recognition"])


class CandlePatterns(Feature):
    """
    Candlestick pattern recognition.

    With TA-Lib: produces one integer column per CDL* pattern (61 patterns).
    Without: produces 6 common patterns in pure pandas.
    Values: +100 (bullish), -100 (bearish), 0 (no pattern).
    """

    name = "candle_patterns"
    required_cols = [Columns.OPEN, Columns.HIGH, Columns.LOW, Columns.CLOSE]
    description = "Candlestick pattern recognition"

    _PANDAS_PATTERNS = [
        "cdl_doji",
        "cdl_hammer",
        "cdl_inverted_hammer",
        "cdl_engulfing",
        "cdl_harami",
        "cdl_morning_star",
    ]

    @property
    def min_periods(self) -> int:
        return 5

    @property
    def output_cols(self) -> list[str]:
        if HAS_TALIB:
            # e.g. CDL2CROWS -> cdl_2crows
            return [fn.lower() for fn in _talib_pattern_names()]
        return list(self._PANDAS_PATTERNS)

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        o = np.asarray(df[Columns.OPEN], dtype=np.float64)
        h = np.asarray(df[Columns.HIGH], dtype=np.float64)
        lo = np.asarray(df[Columns.LOW], dtype=np.float64)
        c = np.asarray(df[Columns.CLOSE], dtype=np.float64)

        if HAS_TALIB:
            for fn_name in _talib_pattern_names():
                func = getattr(talib, fn_name)
                out[fn_name.lower()] = func(o, h, lo, c)
        else:
            self._pandas_patterns(out, o, h, lo, c)
        return out

    @staticmethod
    def _pandas_patterns(
        out: pd.DataFrame,
        o: np.ndarray,
        h: np.ndarray,
        lo: np.ndarray,
        c: np.ndarray,
    ) -> None:
        body = c - o
        abs_body = np.abs(body)
        hl_range = h - lo
        upper_wick = h - np.maximum(o, c)
        lower_wick = np.minimum(o, c) - lo

        # Doji: body is tiny relative to range
        doji_thresh = 0.05 * hl_range
        out["cdl_doji"] = np.where((abs_body <= doji_thresh) & (hl_range > 0), 100, 0)

        # Hammer: small body near top, long lower wick
        out["cdl_hammer"] = np.where(
            (lower_wick >= 2 * abs_body) & (upper_wick <= abs_body * 0.3) & (hl_range > 0),
            100,
            0,
        )

        # Inverted hammer: small body near bottom, long upper wick
        out["cdl_inverted_hammer"] = np.where(
            (upper_wick >= 2 * abs_body) & (lower_wick <= abs_body * 0.3) & (hl_range > 0),
            100,
            0,
        )

        # Engulfing: current body fully engulfs previous body
        prev_body = np.roll(body, 1)
        prev_body[0] = 0
        engulfing_bull = (body > 0) & (prev_body < 0) & (abs_body > np.abs(prev_body))
        engulfing_bear = (body < 0) & (prev_body > 0) & (abs_body > np.abs(prev_body))
        out["cdl_engulfing"] = np.where(engulfing_bull, 100, np.where(engulfing_bear, -100, 0))

        # Harami: current body is contained within previous body
        prev_abs = np.abs(prev_body)
        harami_bull = (body > 0) & (prev_body < 0) & (abs_body < prev_abs)
        harami_bear = (body < 0) & (prev_body > 0) & (abs_body < prev_abs)
        out["cdl_harami"] = np.where(harami_bull, 100, np.where(harami_bear, -100, 0))

        # Morning star (simplified 3-bar): down bar, small bar, up bar
        prev2_body = np.roll(body, 2)
        prev2_body[:2] = 0
        prev_abs_body = np.roll(abs_body, 1)
        prev_abs_body[0] = 0
        out["cdl_morning_star"] = np.where(
            (prev2_body < 0)
            & (prev_abs_body < abs_body * 0.3)
            & (body > 0)
            & (c > (np.roll(o, 2) + np.roll(c, 2)) / 2),
            100,
            0,
        )
