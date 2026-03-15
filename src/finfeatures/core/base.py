"""
Core abstract base classes for the finfeatures library.

Design principle: features are pure, stateless transformations of a DataFrame.
The FeaturePipeline owns state (which features to apply, in what order).
"""

from __future__ import annotations

import abc
from typing import Any, ClassVar, Protocol, runtime_checkable

import pandas as pd

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
    required_cols: ClassVar[list[str]] = []

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
    # Validation helpers
    # ------------------------------------------------------------------

    def _validate(self, df: pd.DataFrame) -> None:
        missing = [c for c in self.required_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"Feature '{self.name}' requires columns {missing} "
                f"which are missing from the input DataFrame."
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
