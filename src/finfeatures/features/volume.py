"""
Volume-based features.

On-Balance Volume, VWAP, volume z-score, and rolling participation ratios.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from finfeatures.core._compat import HAS_TALIB, _f64, talib
from finfeatures.core.base import Columns, Feature, _sma_seeded_ema, _validate_window, safe_divide


class VolumeFeatures(Feature):
    """
    Bundle of basic volume features:
      - volume_return:   day-over-day % change in volume
      - volume_zscore:   rolling z-score of volume
      - volume_sma_ratio: volume / rolling SMA (relative volume)
    """

    name = "volume_features"
    required_cols = [Columns.VOLUME]
    description = "Volume return, z-score, and relative volume"

    def __init__(self, window: int = 20) -> None:
        _validate_window(window)
        self.window = window

    @property
    def min_periods(self) -> int:
        return self.window

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        vol = df[Columns.VOLUME]
        out["volume_return"] = vol.pct_change()
        vol_sma = vol.rolling(self.window).mean()
        vol_std = vol.rolling(self.window).std()
        out[f"volume_zscore_{self.window}"] = safe_divide(vol - vol_sma, vol_std)
        out[f"volume_rel_{self.window}"] = safe_divide(vol, vol_sma)
        return out


class OnBalanceVolume(Feature):
    """
    On-Balance Volume (Granville, 1963).
    OBV accumulates volume: +vol on up days, -vol on down days.
    """

    name = "obv"
    required_cols = [Columns.CLOSE, Columns.VOLUME]
    description = "On-Balance Volume"

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if HAS_TALIB:
            out["obv"] = talib.OBV(
                _f64(df[Columns.CLOSE]),
                _f64(df[Columns.VOLUME]),
            )
        else:
            direction = np.sign(df[Columns.CLOSE].diff()).fillna(0)
            out["obv"] = (direction * df[Columns.VOLUME]).cumsum()
        return out


class VWAP(Feature):
    """
    Rolling Volume-Weighted Average Price.
    vwap_N = sum(typical_price * volume, window) / sum(volume, window)
    Note: This is a rolling VWAP, not session VWAP.
    """

    name = "vwap"
    required_cols = [Columns.HIGH, Columns.LOW, Columns.CLOSE, Columns.VOLUME]
    description = "Rolling Volume-Weighted Average Price"

    def __init__(self, window: int = 20) -> None:
        _validate_window(window)
        self.window = window

    @property
    def min_periods(self) -> int:
        return self.window

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        tp = (df[Columns.HIGH] + df[Columns.LOW] + df[Columns.CLOSE]) / 3
        vol = df[Columns.VOLUME]
        vwap = safe_divide((tp * vol).rolling(self.window).sum(), vol.rolling(self.window).sum())
        out[f"vwap_{self.window}"] = vwap
        out[f"vwap_ratio_{self.window}"] = safe_divide(df[Columns.CLOSE], vwap) - 1
        return out


class ChaikinMoneyFlow(Feature):
    """
    Chaikin Money Flow.
    CMF = sum(MFV, window) / sum(Volume, window)
    MFV = ((Close - Low) - (High - Close)) / (High - Low) * Volume
    """

    name = "chaikin_mf"
    required_cols = [Columns.HIGH, Columns.LOW, Columns.CLOSE, Columns.VOLUME]
    description = "Chaikin Money Flow"

    def __init__(self, window: int = 20) -> None:
        _validate_window(window)
        self.window = window

    @property
    def min_periods(self) -> int:
        return self.window

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        hl = df[Columns.HIGH] - df[Columns.LOW]
        mf_multiplier = safe_divide(
            (df[Columns.CLOSE] - df[Columns.LOW]) - (df[Columns.HIGH] - df[Columns.CLOSE]),
            hl,
        )
        mfv = mf_multiplier * df[Columns.VOLUME]
        out[f"cmf_{self.window}"] = safe_divide(
            mfv.rolling(self.window).sum(),
            df[Columns.VOLUME].rolling(self.window).sum(),
        )
        return out


class AccumulationDistribution(Feature):
    """
    Accumulation/Distribution Line.
    AD = cumsum(CLV * volume) where CLV = ((C-L) - (H-C)) / (H-L).
    """

    name = "ad"
    required_cols = [Columns.HIGH, Columns.LOW, Columns.CLOSE, Columns.VOLUME]
    description = "Accumulation/Distribution Line"

    @property
    def min_periods(self) -> int:
        return 1

    @property
    def output_cols(self) -> list[str]:
        return ["ad_line"]

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        h, lo, c, v = df[Columns.HIGH], df[Columns.LOW], df[Columns.CLOSE], df[Columns.VOLUME]
        if HAS_TALIB:
            out["ad_line"] = talib.AD(_f64(h), _f64(lo), _f64(c), _f64(v))
        else:
            clv = safe_divide((c - lo) - (h - c), h - lo)
            out["ad_line"] = (clv * v).cumsum()
        return out


class ChaikinADOscillator(Feature):
    """
    Chaikin A/D Oscillator = EMA(AD, fast) - EMA(AD, slow).
    """

    name = "adosc"
    required_cols = [Columns.HIGH, Columns.LOW, Columns.CLOSE, Columns.VOLUME]
    description = "Chaikin A/D Oscillator"

    def __init__(self, fast: int = 3, slow: int = 10) -> None:
        _validate_window(fast, "fast")
        _validate_window(slow, "slow")
        self.fast = fast
        self.slow = slow

    @property
    def min_periods(self) -> int:
        return self.slow + 1

    @property
    def output_cols(self) -> list[str]:
        return [f"adosc_{self.fast}_{self.slow}"]

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        h, lo, c, v = df[Columns.HIGH], df[Columns.LOW], df[Columns.CLOSE], df[Columns.VOLUME]
        if HAS_TALIB:
            out[f"adosc_{self.fast}_{self.slow}"] = talib.ADOSC(
                _f64(h),
                _f64(lo),
                _f64(c),
                _f64(v),
                fastperiod=self.fast,
                slowperiod=self.slow,
            )
        else:
            clv = safe_divide((c - lo) - (h - c), h - lo)
            ad = (clv * v).cumsum()
            out[f"adosc_{self.fast}_{self.slow}"] = _sma_seeded_ema(
                ad, self.fast
            ) - _sma_seeded_ema(ad, self.slow)
        return out
