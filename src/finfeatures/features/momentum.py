"""
Momentum features.

RSI, Rate of Change, Stochastic Oscillator, Williams %R, and CCI.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from finfeatures.core._compat import HAS_TALIB, _f64, talib
from finfeatures.core.base import (
    Columns,
    Feature,
    _sma_seeded_ema,
    _validate_window,
    _wilder_smooth,
    safe_divide,
)


class RSI(Feature):
    """
    Relative Strength Index (Wilder, 1978).
    RSI ranges 0-100.  Overbought > 70, oversold < 30.
    """

    name = "rsi"
    required_cols = [Columns.CLOSE]
    description = "Relative Strength Index"

    def __init__(self, window: int = 14) -> None:
        _validate_window(window)
        self.window = window

    @property
    def min_periods(self) -> int:
        return self.window + 1

    @property
    def output_cols(self) -> list[str]:
        return [f"rsi_{self.window}"]

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        close = df[Columns.CLOSE]
        if HAS_TALIB:
            out[f"rsi_{self.window}"] = talib.RSI(_f64(close), timeperiod=self.window)
        else:
            delta = close.diff()
            gain = delta.clip(lower=0)
            loss = (-delta).clip(lower=0)
            avg_gain = _wilder_smooth(gain.iloc[1:], self.window).reindex(close.index)
            avg_loss = _wilder_smooth(loss.iloc[1:], self.window).reindex(close.index)
            rs = safe_divide(avg_gain, avg_loss)
            out[f"rsi_{self.window}"] = 100 - (100 / (1 + rs))
        return out


class RateOfChange(Feature):
    """
    Rate of Change: (P_t / P_{t-n}) - 1.
    Measures momentum over n periods.
    """

    name = "rate_of_change"
    required_cols = [Columns.CLOSE]
    description = "N-period Rate of Change (momentum)"

    def __init__(self, window: int = 10) -> None:
        _validate_window(window)
        self.window = window

    @property
    def min_periods(self) -> int:
        return self.window + 1

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        close = df[Columns.CLOSE]
        if HAS_TALIB:
            # TA-Lib ROC returns percentage, divide by 100 to match our fractional convention
            out[f"roc_{self.window}"] = talib.ROC(_f64(close), timeperiod=self.window) / 100
        else:
            out[f"roc_{self.window}"] = safe_divide(close, close.shift(self.window)) - 1
        return out


class StochasticOscillator(Feature):
    """
    Stochastic Oscillator (%K and %D).
    %K = (Close - LowestLow) / (HighestHigh - LowestLow) * 100
    %D = SMA(%K, d_window)
    """

    name = "stochastic"
    required_cols = [Columns.HIGH, Columns.LOW, Columns.CLOSE]
    description = "Stochastic Oscillator %K and %D"

    def __init__(self, k_window: int = 14, d_window: int = 3) -> None:
        _validate_window(k_window, "k_window")
        _validate_window(d_window, "d_window")
        self.k_window = k_window
        self.d_window = d_window

    @property
    def min_periods(self) -> int:
        return self.k_window + self.d_window

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if HAS_TALIB:
            slowk, slowd = talib.STOCH(
                _f64(df[Columns.HIGH]),
                _f64(df[Columns.LOW]),
                _f64(df[Columns.CLOSE]),
                fastk_period=self.k_window,
                slowk_period=self.d_window,
                slowk_matype=0,  # type: ignore[arg-type]  # MA_Type.SMA
                slowd_period=self.d_window,
                slowd_matype=0,  # type: ignore[arg-type]  # MA_Type.SMA
            )
            out[f"stoch_k_{self.k_window}"] = slowk
            out[f"stoch_d_{self.k_window}"] = slowd
        else:
            low_min = df[Columns.LOW].rolling(self.k_window).min()
            high_max = df[Columns.HIGH].rolling(self.k_window).max()
            pct_k = 100 * safe_divide(df[Columns.CLOSE] - low_min, high_max - low_min)
            out[f"stoch_k_{self.k_window}"] = pct_k
            out[f"stoch_d_{self.k_window}"] = pct_k.rolling(self.d_window).mean()
        return out


class WilliamsR(Feature):
    """
    Williams %R.
    %R = (HighestHigh - Close) / (HighestHigh - LowestLow) * -100
    Ranges -100 to 0.  Overbought > -20, oversold < -80.
    """

    name = "williams_r"
    required_cols = [Columns.HIGH, Columns.LOW, Columns.CLOSE]
    description = "Williams %R oscillator"

    def __init__(self, window: int = 14) -> None:
        _validate_window(window)
        self.window = window

    @property
    def min_periods(self) -> int:
        return self.window

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if HAS_TALIB:
            out[f"williams_r_{self.window}"] = talib.WILLR(
                _f64(df[Columns.HIGH]),
                _f64(df[Columns.LOW]),
                _f64(df[Columns.CLOSE]),
                timeperiod=self.window,
            )
        else:
            high_max = df[Columns.HIGH].rolling(self.window).max()
            low_min = df[Columns.LOW].rolling(self.window).min()
            out[f"williams_r_{self.window}"] = -100 * safe_divide(
                high_max - df[Columns.CLOSE], high_max - low_min
            )
        return out


class CommodityChannelIndex(Feature):
    """
    Commodity Channel Index (Lambert, 1980).
    CCI = (TypicalPrice - SMA(TP)) / (0.015 * MeanDeviation)
    """

    name = "cci"
    required_cols = [Columns.HIGH, Columns.LOW, Columns.CLOSE]
    description = "Commodity Channel Index"

    def __init__(self, window: int = 20) -> None:
        _validate_window(window)
        self.window = window

    @property
    def min_periods(self) -> int:
        return self.window

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if HAS_TALIB:
            out[f"cci_{self.window}"] = talib.CCI(
                _f64(df[Columns.HIGH]),
                _f64(df[Columns.LOW]),
                _f64(df[Columns.CLOSE]),
                timeperiod=self.window,
            )
        else:
            tp = (df[Columns.HIGH] + df[Columns.LOW] + df[Columns.CLOSE]) / 3
            sma = tp.rolling(self.window).mean()
            mad = tp.rolling(self.window).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
            out[f"cci_{self.window}"] = safe_divide(tp - sma, 0.015 * mad)
        return out


class MomentumScore(Feature):
    """
    Composite normalised momentum score — average of standardised
    multi-period returns (1m, 3m, 6m, 12m).  Used in cross-sectional
    factor models and regime filters.
    """

    name = "momentum_score"
    required_cols = [Columns.CLOSE]
    description = "Multi-period normalised momentum factor score"

    def __init__(self, periods: list[int] | None = None) -> None:
        self.periods = periods or [21, 63, 126, 252]
        for p in self.periods:
            _validate_window(p, "periods element")

    @property
    def min_periods(self) -> int:
        return max(self.periods) + 252

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        scores = []
        for p in self.periods:
            ret = df[Columns.CLOSE].pct_change(p)
            col = f"mom_ret_{p}"
            out[col] = ret
            # Z-score within a rolling 252-day window
            z = (ret - ret.rolling(252).mean()) / ret.rolling(252).std()
            z_col = f"mom_zscore_{p}"
            out[z_col] = z
            scores.append(z)
        if scores:
            out["momentum_composite"] = pd.concat(scores, axis=1).mean(axis=1)
        return out


class MoneyFlowIndex(Feature):
    """
    Money Flow Index — volume-weighted RSI.
    Ranges 0-100. Overbought > 80, oversold < 20.
    """

    name = "mfi"
    required_cols = [Columns.HIGH, Columns.LOW, Columns.CLOSE, Columns.VOLUME]
    description = "Money Flow Index (volume-weighted RSI)"

    def __init__(self, window: int = 14) -> None:
        _validate_window(window)
        self.window = window

    @property
    def min_periods(self) -> int:
        return self.window + 1

    @property
    def output_cols(self) -> list[str]:
        return [f"mfi_{self.window}"]

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if HAS_TALIB:
            out[f"mfi_{self.window}"] = talib.MFI(
                _f64(df[Columns.HIGH]),
                _f64(df[Columns.LOW]),
                _f64(df[Columns.CLOSE]),
                _f64(df[Columns.VOLUME]),
                timeperiod=self.window,
            )
        else:
            tp = (df[Columns.HIGH] + df[Columns.LOW] + df[Columns.CLOSE]) / 3
            raw_mf = tp * df[Columns.VOLUME]
            delta = tp.diff()
            pos_mf = raw_mf.where(delta > 0, 0.0).rolling(self.window).sum()
            neg_mf = raw_mf.where(delta <= 0, 0.0).rolling(self.window).sum()
            ratio = safe_divide(pos_mf, neg_mf)
            out[f"mfi_{self.window}"] = 100 - (100 / (1 + ratio))
        return out


class Aroon(Feature):
    """
    Aroon indicator — trend onset detection.
    Aroon Up/Down range 0-100, oscillator ranges -100 to 100.
    """

    name = "aroon"
    required_cols = [Columns.HIGH, Columns.LOW]
    description = "Aroon Up, Down, and Oscillator"

    def __init__(self, window: int = 25) -> None:
        _validate_window(window)
        self.window = window

    @property
    def min_periods(self) -> int:
        return self.window

    @property
    def output_cols(self) -> list[str]:
        w = self.window
        return [f"aroon_up_{w}", f"aroon_down_{w}", f"aroon_osc_{w}"]

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        w = self.window
        if HAS_TALIB:
            down, up = talib.AROON(_f64(df[Columns.HIGH]), _f64(df[Columns.LOW]), timeperiod=w)
            out[f"aroon_up_{w}"] = up
            out[f"aroon_down_{w}"] = down
            out[f"aroon_osc_{w}"] = talib.AROONOSC(
                _f64(df[Columns.HIGH]), _f64(df[Columns.LOW]), timeperiod=w
            )
        else:
            high_roll = df[Columns.HIGH].rolling(w + 1)
            low_roll = df[Columns.LOW].rolling(w + 1)
            aroon_up = high_roll.apply(lambda x: x.argmax(), raw=True) / w * 100
            aroon_down = low_roll.apply(lambda x: x.argmin(), raw=True) / w * 100
            out[f"aroon_up_{w}"] = aroon_up
            out[f"aroon_down_{w}"] = aroon_down
            out[f"aroon_osc_{w}"] = aroon_up - aroon_down
        return out


class ChandeMomentumOscillator(Feature):
    """
    Chande Momentum Oscillator.
    CMO = 100 * (sum_gains - sum_losses) / (sum_gains + sum_losses)
    Ranges -100 to 100.
    """

    name = "cmo"
    required_cols = [Columns.CLOSE]
    description = "Chande Momentum Oscillator"

    def __init__(self, window: int = 14) -> None:
        _validate_window(window)
        self.window = window

    @property
    def min_periods(self) -> int:
        return self.window + 1

    @property
    def output_cols(self) -> list[str]:
        return [f"cmo_{self.window}"]

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        close = df[Columns.CLOSE]
        if HAS_TALIB:
            out[f"cmo_{self.window}"] = talib.CMO(_f64(close), timeperiod=self.window)
        else:
            delta = close.diff()
            gains = delta.clip(lower=0).rolling(self.window).sum()
            losses = (-delta).clip(lower=0).rolling(self.window).sum()
            out[f"cmo_{self.window}"] = safe_divide(100 * (gains - losses), gains + losses)
        return out


class UltimateOscillator(Feature):
    """
    Ultimate Oscillator — multi-period momentum (Williams, 1985).
    Ranges 0-100.
    """

    name = "ultimate_oscillator"
    required_cols = [Columns.HIGH, Columns.LOW, Columns.CLOSE]
    description = "Ultimate Oscillator"

    def __init__(self, period1: int = 7, period2: int = 14, period3: int = 28) -> None:
        _validate_window(period1, "period1")
        _validate_window(period2, "period2")
        _validate_window(period3, "period3")
        self.period1 = period1
        self.period2 = period2
        self.period3 = period3

    @property
    def min_periods(self) -> int:
        return self.period3 + 1

    @property
    def output_cols(self) -> list[str]:
        return ["ultimate_osc"]

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        h, lo, c = df[Columns.HIGH], df[Columns.LOW], df[Columns.CLOSE]
        if HAS_TALIB:
            out["ultimate_osc"] = talib.ULTOSC(
                _f64(h),
                _f64(lo),
                _f64(c),
                timeperiod1=self.period1,
                timeperiod2=self.period2,
                timeperiod3=self.period3,
            )
        else:
            prev_c = c.shift(1)
            bp = c - pd.concat([lo, prev_c], axis=1).min(axis=1)
            tr = pd.concat([h - lo, (h - prev_c).abs(), (lo - prev_c).abs()], axis=1).max(axis=1)
            avg1 = safe_divide(bp.rolling(self.period1).sum(), tr.rolling(self.period1).sum())
            avg2 = safe_divide(bp.rolling(self.period2).sum(), tr.rolling(self.period2).sum())
            avg3 = safe_divide(bp.rolling(self.period3).sum(), tr.rolling(self.period3).sum())
            out["ultimate_osc"] = 100 * (4 * avg1 + 2 * avg2 + avg3) / 7
        return out


class StochasticRSI(Feature):
    """
    Stochastic RSI — RSI run through the stochastic formula.
    Ranges 0-100. More sensitive than plain RSI.

    Outputs: stochrsi_k_N, stochrsi_d_N
    """

    name = "stochastic_rsi"
    required_cols = [Columns.CLOSE]
    description = "Stochastic RSI"

    def __init__(
        self,
        window: int = 14,
        k_period: int = 5,
        d_period: int = 3,
    ) -> None:
        _validate_window(window)
        _validate_window(k_period, "k_period")
        _validate_window(d_period, "d_period")
        self.window = window
        self.k_period = k_period
        self.d_period = d_period

    @property
    def min_periods(self) -> int:
        return self.window + self.k_period + self.d_period

    @property
    def output_cols(self) -> list[str]:
        w = self.window
        return [f"stochrsi_k_{w}", f"stochrsi_d_{w}"]

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        close = df[Columns.CLOSE]
        w = self.window
        if HAS_TALIB:
            fastk, fastd = talib.STOCHRSI(
                _f64(close),
                timeperiod=w,
                fastk_period=self.k_period,
                fastd_period=self.d_period,
                fastd_matype=0,  # type: ignore[arg-type]  # MA_Type.SMA
            )
            out[f"stochrsi_k_{w}"] = fastk
            out[f"stochrsi_d_{w}"] = fastd
        else:
            # Compute RSI first (Wilder smoothing to match TA-Lib)
            delta = close.diff()
            gain = delta.clip(lower=0)
            loss = (-delta).clip(lower=0)
            avg_gain = _wilder_smooth(gain.iloc[1:], w).reindex(close.index)
            avg_loss = _wilder_smooth(loss.iloc[1:], w).reindex(close.index)
            rs = safe_divide(avg_gain, avg_loss)
            rsi = 100 - (100 / (1 + rs))
            # Apply stochastic formula to RSI
            rsi_min = rsi.rolling(self.k_period).min()
            rsi_max = rsi.rolling(self.k_period).max()
            stoch_k = 100 * safe_divide(rsi - rsi_min, rsi_max - rsi_min)
            out[f"stochrsi_k_{w}"] = stoch_k
            out[f"stochrsi_d_{w}"] = stoch_k.rolling(self.d_period).mean()
        return out


class TRIX(Feature):
    """
    TRIX — 1-day rate of change of a triple-smoothed EMA.
    Good noise filter; oscillates around zero.
    """

    name = "trix"
    required_cols = [Columns.CLOSE]
    description = "Triple-smoothed EMA rate of change"

    def __init__(self, window: int = 30) -> None:
        _validate_window(window)
        self.window = window

    @property
    def min_periods(self) -> int:
        return 3 * self.window

    @property
    def output_cols(self) -> list[str]:
        return [f"trix_{self.window}"]

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        close = df[Columns.CLOSE]
        if HAS_TALIB:
            out[f"trix_{self.window}"] = talib.TRIX(_f64(close), timeperiod=self.window)
        else:
            ema1 = _sma_seeded_ema(close, self.window)
            ema2 = _sma_seeded_ema(ema1.dropna(), self.window).reindex(close.index)
            ema3 = _sma_seeded_ema(ema2.dropna(), self.window).reindex(close.index)
            out[f"trix_{self.window}"] = ema3.pct_change() * 100
        return out


class PPO(Feature):
    """
    Percentage Price Oscillator — MACD expressed as a percentage.
    PPO = (EMA_fast - EMA_slow) / EMA_slow * 100
    """

    name = "ppo"
    required_cols = [Columns.CLOSE]
    description = "Percentage Price Oscillator"

    def __init__(self, fast: int = 12, slow: int = 26) -> None:
        _validate_window(fast, "fast")
        _validate_window(slow, "slow")
        if fast >= slow:
            raise ValueError(f"'fast' must be < 'slow', got fast={fast}, slow={slow}")
        self.fast = fast
        self.slow = slow

    @property
    def min_periods(self) -> int:
        return self.slow

    @property
    def output_cols(self) -> list[str]:
        return [f"ppo_{self.fast}_{self.slow}"]

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        close = df[Columns.CLOSE]
        if HAS_TALIB:
            out[f"ppo_{self.fast}_{self.slow}"] = talib.PPO(
                _f64(close),
                fastperiod=self.fast,
                slowperiod=self.slow,
                matype=0,  # type: ignore[arg-type]  # MA_Type.SMA
            )
        else:
            ema_fast = _sma_seeded_ema(close, self.fast)
            ema_slow = _sma_seeded_ema(close, self.slow)
            out[f"ppo_{self.fast}_{self.slow}"] = safe_divide(100 * (ema_fast - ema_slow), ema_slow)
        return out
