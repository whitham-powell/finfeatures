"""
Statistical features.

Rolling distributional statistics: z-score normalisation, rolling higher
moments (skewness, kurtosis), autocorrelation, and cross-asset correlation.
Useful across a wide range of downstream tasks including regime detection,
risk monitoring, and general time-series modelling.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from finfeatures.core._compat import HAS_TALIB, _f64, talib
from finfeatures.core.base import Columns, Feature, _validate_window, safe_divide


class RollingZScore(Feature):
    """
    Rolling z-score of any named column.
    z_t = (x_t - μ_{t,window}) / σ_{t,window}
    """

    name = "rolling_zscore"
    description = "Rolling z-score normalisation of a target column"

    def __init__(self, column: str = Columns.LOG_RETURN, window: int = 21) -> None:
        _validate_window(window)
        self.column = column
        self.window = window
        self.required_cols = [self.column]

    @property
    def min_periods(self) -> int:
        return self.window

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        col = df[self.column]
        mu = col.rolling(self.window).mean()
        sig = col.rolling(self.window).std()
        out[f"{self.column}_zscore_{self.window}"] = safe_divide(col - mu, sig)
        return out


class RollingSkewKurt(Feature):
    """
    Rolling skewness and excess kurtosis of any named column.
    Used to characterise tail risk in the return distribution.
    """

    name = "rolling_skew_kurt"
    description = "Rolling skewness and excess kurtosis"

    def __init__(self, column: str = Columns.LOG_RETURN, window: int = 63) -> None:
        _validate_window(window)
        self.column = column
        self.window = window
        self.required_cols = [self.column]

    @property
    def min_periods(self) -> int:
        return self.window

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        col = df[self.column]
        w = self.window
        out[f"{self.column}_skew_{w}"] = col.rolling(w).skew()
        out[f"{self.column}_kurt_{w}"] = col.rolling(w).kurt()  # excess kurtosis
        return out


class RollingMoments(Feature):
    """
    Full set of rolling moments for a column: mean, std, skewness, kurtosis.
    """

    name = "rolling_moments"
    description = "Full rolling distributional moments (mean, std, skew, kurt)"

    def __init__(self, column: str = Columns.LOG_RETURN, window: int = 63) -> None:
        _validate_window(window)
        self.column = column
        self.window = window
        self.required_cols = [self.column]

    @property
    def min_periods(self) -> int:
        return self.window

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        col = df[self.column]
        w = self.window
        out[f"{self.column}_mean_{w}"] = col.rolling(w).mean()
        out[f"{self.column}_std_{w}"] = col.rolling(w).std()
        out[f"{self.column}_skew_{w}"] = col.rolling(w).skew()
        out[f"{self.column}_kurt_{w}"] = col.rolling(w).kurt()
        # Value at Risk (historical, non-parametric)
        out[f"{self.column}_var5_{w}"] = col.rolling(w).quantile(0.05)
        out[f"{self.column}_cvar5_{w}"] = col.rolling(w).apply(
            lambda x: x[x <= np.quantile(x, 0.05)].mean(), raw=True
        )
        return out


class RollingAutocorrelation(Feature):
    """
    Rolling autocorrelation at a given lag.
    Elevated autocorrelation may indicate trending or mean-reverting behaviour.
    """

    name = "rolling_autocorr"
    description = "Rolling autocorrelation at a given lag"

    def __init__(
        self,
        column: str = Columns.LOG_RETURN,
        window: int = 63,
        lag: int = 1,
    ) -> None:
        _validate_window(window)
        _validate_window(lag, "lag")
        self.column = column
        self.window = window
        self.lag = lag
        self.required_cols = [self.column]

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        col = df[self.column]
        out[f"{self.column}_autocorr_lag{self.lag}_{self.window}"] = col.rolling(self.window).apply(
            lambda x: pd.Series(x).autocorr(lag=self.lag), raw=True
        )
        return out


class RollingCorrelation(Feature):
    """
    Rolling correlation between two named columns.
    Useful for tracking dynamic cross-asset relationships.
    """

    name = "rolling_correlation"
    description = "Rolling Pearson correlation between two columns"

    def __init__(
        self,
        col_a: str = Columns.LOG_RETURN,
        col_b: str = "realized_vol_21",
        window: int = 63,
    ) -> None:
        _validate_window(window)
        self.col_a = col_a
        self.col_b = col_b
        self.window = window
        self.required_cols = [self.col_a, self.col_b]

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out[f"corr_{self.col_a}_{self.col_b}_{self.window}"] = (
            df[self.col_a].rolling(self.window).corr(df[self.col_b])
        )
        return out


class CrossAssetCorrelation(Feature):
    """
    Rolling Pearson correlation between a column in the input DataFrame
    and a column from an external reference asset's DataFrame.

    The reference DataFrame is provided at construction time and aligned
    by DatetimeIndex. Mismatched dates are forward-filled.

    Output columns:
      - Same column both sides: ``corr_{ref}_{col}_{w}``
      - Different columns: ``corr_{ref}_{col}_vs_{ref_col}_{w}``
    """

    name = "cross_asset_correlation"
    description = "Rolling correlation against an external reference asset"

    def __init__(
        self,
        reference: pd.DataFrame,
        reference_name: str,
        column: str,
        windows: int | list[int],
        reference_column: str | None = None,
    ) -> None:
        if reference_column is None:
            reference_column = column
        if isinstance(windows, int):
            windows = [windows]
        for w in windows:
            _validate_window(w, "windows element")
        if reference_column not in reference.columns:
            raise ValueError(
                f"reference_column '{reference_column}' not found in reference DataFrame. "
                f"Available: {list(reference.columns)}"
            )
        self.reference_name = reference_name
        self.column = column
        self.reference_column = reference_column
        self.windows = windows
        self.required_cols = [self.column]
        # Store the aligned reference series
        self._ref_series: pd.Series = reference[reference_column]  # type: ignore[type-arg]

    @property
    def min_periods(self) -> int:
        return max(self.windows)

    @property
    def output_cols(self) -> list[str]:
        return [self._col_name(w) for w in self.windows]

    def _col_name(self, w: int) -> str:
        ref = self.reference_name
        if self.column == self.reference_column:
            return f"corr_{ref}_{self.column}_{w}"
        return f"corr_{ref}_{self.column}_vs_{self.reference_column}_{w}"

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        primary = df[self.column]
        # Align reference to the input's index, forward-fill gaps
        ref_aligned = self._ref_series.reindex(primary.index, method="ffill")
        for w in self.windows:
            if HAS_TALIB:
                out[self._col_name(w)] = talib.CORREL(
                    _f64(primary), _f64(ref_aligned), timeperiod=w
                )
            else:
                out[self._col_name(w)] = primary.rolling(w).corr(ref_aligned)
        return out


class HurstExponent(Feature):
    """
    Rolling Hurst exponent via Rescaled Range (R/S) analysis.

    Measures long-term memory of a time series:
      - H > 0.5: trending / persistent
      - H ≈ 0.5: random walk
      - H < 0.5: mean-reverting

    Output column: ``{column}_hurst_{window}``
    """

    name = "hurst_exponent"
    description = "Rolling Hurst exponent via R/S analysis"

    def __init__(self, column: str = Columns.LOG_RETURN, window: int = 100) -> None:
        _validate_window(window, minimum=20)
        self.column = column
        self.window = window
        self.required_cols = [self.column]

    @property
    def min_periods(self) -> int:
        return self.window

    @property
    def output_cols(self) -> list[str]:
        return [f"{self.column}_hurst_{self.window}"]

    @staticmethod
    def _rs_hurst(x: np.ndarray) -> float:
        """Estimate Hurst exponent from a 1-D array using R/S analysis."""
        n = len(x)
        # Use sub-series lengths that are divisors or near-divisors of n
        # At least 3 distinct sizes are needed for a meaningful regression
        max_k = n // 2
        sizes = []
        k = 10
        while k <= max_k:
            sizes.append(k)
            k = int(k * 1.5) or k + 1
        if len(sizes) < 3:
            return np.nan

        log_sizes = []
        log_rs = []
        for size in sizes:
            n_chunks = n // size
            if n_chunks < 1:
                continue
            rs_values = np.empty(n_chunks)
            for i in range(n_chunks):
                chunk = x[i * size : (i + 1) * size]
                mean = chunk.mean()
                deviate = np.cumsum(chunk - mean)
                r = deviate.max() - deviate.min()
                s = chunk.std(ddof=1)
                if s < 1e-12:
                    rs_values[i] = np.nan
                else:
                    rs_values[i] = r / s
            rs_mean = np.nanmean(rs_values)
            if np.isnan(rs_mean) or rs_mean <= 0:
                continue
            log_sizes.append(np.log(size))
            log_rs.append(np.log(rs_mean))

        if len(log_sizes) < 3:
            return np.nan

        # OLS slope of log(R/S) vs log(size)
        log_sizes_arr = np.array(log_sizes)
        log_rs_arr = np.array(log_rs)
        x_mean = log_sizes_arr.mean()
        y_mean = log_rs_arr.mean()
        slope = np.dot(log_sizes_arr - x_mean, log_rs_arr - y_mean) / np.dot(
            log_sizes_arr - x_mean, log_sizes_arr - x_mean
        )
        return float(slope)

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        col_name = f"{self.column}_hurst_{self.window}"
        out[col_name] = (
            df[self.column]
            .rolling(self.window)
            .apply(lambda x: self._rs_hurst(np.asarray(x)), raw=True)
        )
        return out


class LinearRegressionSlope(Feature):
    """
    Rolling linear regression slope.
    Measures the trend direction and strength via least-squares fit.
    """

    name = "linreg_slope"
    description = "Rolling linear regression slope"

    def __init__(self, column: str = Columns.CLOSE, window: int = 14) -> None:
        _validate_window(window, minimum=2)
        self.column = column
        self.window = window
        self.required_cols = [self.column]

    @property
    def min_periods(self) -> int:
        return self.window

    @property
    def output_cols(self) -> list[str]:
        return [f"{self.column}_linreg_slope_{self.window}"]

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        col_name = f"{self.column}_linreg_slope_{self.window}"
        vals = df[self.column]
        if HAS_TALIB and self.column == Columns.CLOSE:
            out[col_name] = talib.LINEARREG_SLOPE(_f64(vals), timeperiod=self.window)
        else:
            x = np.arange(self.window, dtype=float)
            x_mean = x.mean()
            x_var = ((x - x_mean) ** 2).sum()
            out[col_name] = vals.rolling(self.window).apply(
                lambda y: np.dot(y - y.mean(), x - x_mean) / x_var, raw=True
            )
        return out
