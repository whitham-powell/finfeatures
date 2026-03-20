"""
Core abstract base classes for the finfeatures library.

Design principle: features are pure, stateless transformations of a DataFrame.
The FeaturePipeline owns state (which features to apply, in what order).
"""

from __future__ import annotations

import abc
from typing import Any, ClassVar, Protocol, runtime_checkable

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Validation / utility helpers
# ---------------------------------------------------------------------------


def _validate_window(value: int, name: str = "window", minimum: int = 1) -> None:
    """Raise ValueError if *value* is not an int >= *minimum*."""
    if not isinstance(value, int) or value < minimum:
        raise ValueError(f"'{name}' must be an integer >= {minimum}, got {value!r}")


def _sma_seeded_ema(series: pd.Series, span: int) -> pd.Series:
    """EMA matching TA-Lib: SMA of first *span* values as seed, then standard EMA.

    pandas ``ewm(span=..., adjust=False)`` uses the first value as the seed,
    producing a small but persistent offset from TA-Lib's output.  This helper
    replicates TA-Lib's initialisation exactly.
    """
    alpha = 2.0 / (span + 1)
    vals = series.values.astype(float)
    n = len(vals)
    out = np.full(n, np.nan)
    if n < span:
        return pd.Series(out, index=series.index)
    out[span - 1] = np.mean(vals[:span])
    for i in range(span, n):
        out[i] = alpha * vals[i] + (1 - alpha) * out[i - 1]
    return pd.Series(out, index=series.index)


def _wilder_smooth(series: pd.Series, window: int) -> pd.Series:
    """Wilder's smoothing matching TA-Lib: SMA seed then ``prev*(n-1)/n + cur/n``.

    Used by RSI, ATR, ADX, and related Wilder-family indicators.
    Equivalent to EMA with ``alpha=1/window`` but with an SMA seed.
    """
    alpha = 1.0 / window
    vals = series.values.astype(float)
    n = len(vals)
    out = np.full(n, np.nan)
    if n < window:
        return pd.Series(out, index=series.index)
    out[window - 1] = np.mean(vals[:window])
    for i in range(window, n):
        out[i] = alpha * vals[i] + (1 - alpha) * out[i - 1]
    return pd.Series(out, index=series.index)


def safe_divide(
    numerator: pd.Series | np.ndarray,
    denominator: pd.Series | np.ndarray,
) -> pd.Series:
    """Division returning NaN where denominator is zero or result is ±inf."""
    result = numerator / pd.Series(denominator).replace(0, np.nan)
    return result.replace([np.inf, -np.inf], np.nan)


# ---------------------------------------------------------------------------
# Column name constants — canonical OHLCV column names expected by all features
# ---------------------------------------------------------------------------


class Columns:
    OPEN = "open"
    HIGH = "high"
    LOW = "low"
    CLOSE = "close"
    VOLUME = "volume"
    RETURN = "return"  # simple return
    LOG_RETURN = "log_return"  # log return

    OHLCV: ClassVar[list[str]] = ["open", "high", "low", "close", "volume"]
    PRICE: ClassVar[list[str]] = ["open", "high", "low", "close"]


# ---------------------------------------------------------------------------
# DataFrame adapter protocol — makes the library swap-able to polars, etc.
# ---------------------------------------------------------------------------


@runtime_checkable
class DataFrameAdapter(Protocol):
    """
    Minimal structural protocol a DataFrame-like object must satisfy.
    Implementing this allows finfeatures to work with pandas, polars,
    modin, cuDF, or any future columnar format.

    For now pandas is the default concrete implementation.
    """

    def __getitem__(self, key: Any) -> Any: ...
    def __setitem__(self, key: Any, value: Any) -> None: ...
    def __contains__(self, key: Any) -> bool: ...
    def copy(self) -> DataFrameAdapter: ...


# ---------------------------------------------------------------------------
# Abstract Feature
# ---------------------------------------------------------------------------


class Feature(abc.ABC):
    """
    Base class for all features.

    A Feature is a **pure, named transformation** from a DataFrame to one or
    more new columns appended to that DataFrame.  It must not mutate the
    input; it must return a *new* DataFrame that contains both the original
    columns and any derived columns.

    Subclass contract:
        - Override `name` (class variable) with a unique slug.
        - Override `required_cols` with the column names the feature reads.
        - Implement `compute(df)` → pd.DataFrame.
    """

    #: Unique, human-readable slug identifying the feature group.
    name: ClassVar[str] = ""

    #: Columns that must be present in the input DataFrame.
    required_cols: list[str] = []

    #: Optional human-readable description.
    description: ClassVar[str] = ""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if cls.name and not cls.name.startswith("_"):
            FeatureRegistry.register(cls)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the feature and return a new DataFrame."""
        self._validate(df)
        result = self.compute(df)
        # Guarantee source columns are never lost
        for col in df.columns:
            if col not in result.columns:
                result[col] = df[col]
        return result

    @abc.abstractmethod
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute feature columns.

        Args:
            df: Input DataFrame with at minimum the columns listed in
                `required_cols`.  Must not be mutated.

        Returns:
            A new DataFrame containing all original columns plus the derived
            feature columns.
        """

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def min_periods(self) -> int:
        """Minimum rows before this feature produces non-NaN output."""
        return 0

    @property
    def output_cols(self) -> list[str]:
        """Columns this feature adds. Override for dependency validation."""
        return []

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def _validate(self, df: pd.DataFrame) -> None:
        missing = [c for c in self.required_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"Feature '{self.name}' requires columns {missing} "
                f"which are missing from the input DataFrame. "
                f"Available columns: {list(df.columns)}"
            )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"


# ---------------------------------------------------------------------------
# Abstract DataSource
# ---------------------------------------------------------------------------


class DataSource(abc.ABC):
    """
    Abstract base class for market data providers.

    Implementations must return a DataFrame whose columns are normalised to
    the Columns constants above (lowercase ohlcv).  The index must be a
    DatetimeIndex.
    """

    @abc.abstractmethod
    def fetch(
        self,
        symbol: str,
        start: str | None = None,
        end: str | None = None,
        interval: str = "1d",
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Fetch raw OHLCV data for a single symbol.

        Returns a pd.DataFrame with columns: open, high, low, close, volume
        and a DatetimeIndex.
        """

    @abc.abstractmethod
    def fetch_multiple(
        self,
        symbols: list[str],
        start: str | None = None,
        end: str | None = None,
        interval: str = "1d",
        **kwargs: Any,
    ) -> dict[str, pd.DataFrame]:
        """Fetch raw OHLCV data for multiple symbols.  Returns {symbol: df}."""


# ---------------------------------------------------------------------------
# Feature Registry — global, auto-populated via __init_subclass__
# ---------------------------------------------------------------------------


class FeatureRegistry:
    """
    Auto-populated registry of all Feature subclasses.

    Features register themselves on class definition; there is no need to
    explicitly call register() from user code.
    """

    _registry: ClassVar[dict[str, type[Feature]]] = {}

    @classmethod
    def register(cls, feature_cls: type[Feature]) -> None:
        if feature_cls.name in cls._registry:
            existing = cls._registry[feature_cls.name]
            if existing is not feature_cls:
                raise ValueError(
                    f"Feature name '{feature_cls.name}' is already registered "
                    f"by {existing.__qualname__}.  Choose a unique name."
                )
        cls._registry[feature_cls.name] = feature_cls

    @classmethod
    def get(cls, name: str) -> type[Feature]:
        if name not in cls._registry:
            raise KeyError(
                f"No feature named '{name}' is registered.  Available: {sorted(cls._registry)}"
            )
        return cls._registry[name]

    @classmethod
    def all(cls) -> dict[str, type[Feature]]:
        return dict(cls._registry)

    @classmethod
    def list(cls) -> list[str]:
        return sorted(cls._registry)
