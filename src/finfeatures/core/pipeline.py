"""
FeaturePipeline — ordered, composable feature computation.

Usage
-----
    from finfeatures.core import FeaturePipeline
    from finfeatures.features.price import Returns, LogReturns
    from finfeatures.features.volatility import RollingVolatility
    from finfeatures.features.momentum import RSI

    pipeline = FeaturePipeline(
        Returns(),
        LogReturns(),
        RollingVolatility(window=21),
        RSI(window=14),
    )

    # df is a raw OHLCV DataFrame
    enriched = pipeline.transform(df)
    # enriched contains ALL original columns + all derived feature columns
"""

from __future__ import annotations

import pandas as pd

from .base import Feature, FeatureRegistry


class FeaturePipeline:
    """
    Ordered sequence of Features applied left-to-right.

    Each feature receives the full accumulated DataFrame (raw columns + all
    previously computed feature columns), so later features can depend on
    earlier ones.

    The raw source columns are **always preserved** in the output.

    Parameters
    ----------
    *features:
        Feature instances to apply, in order.  May also contain string
        names to look up in the FeatureRegistry.
    """

    def __init__(self, *features: Feature | str) -> None:
        self._steps: list[Feature] = [
            FeatureRegistry.get(f)() if isinstance(f, str) else f for f in features
        ]

    # ------------------------------------------------------------------
    # Composition helpers
    # ------------------------------------------------------------------

    def add(self, feature: Feature | str) -> FeaturePipeline:
        """Return a new pipeline with the given feature appended."""
        if isinstance(feature, str):
            feature = FeatureRegistry.get(feature)()
        return FeaturePipeline(*self._steps, feature)

    def __add__(self, other: FeaturePipeline | Feature) -> FeaturePipeline:
        if isinstance(other, Feature):
            return self.add(other)
        return FeaturePipeline(*self._steps, *other._steps)

    # ------------------------------------------------------------------
    # Core transform
    # ------------------------------------------------------------------

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all features sequentially.

        Parameters
        ----------
        df:
            Raw OHLCV DataFrame (DatetimeIndex, lowercase column names).

        Returns
        -------
        pd.DataFrame
            Original columns plus all computed feature columns.
            Columns are deduplicated; if two features write the same column
            the later one wins (with a warning).
        """
        result = df.copy()
        raw_cols = list(df.columns)

        for feature in self._steps:
            try:
                result = feature(result)
            except Exception as exc:
                raise RuntimeError(
                    f"Feature '{feature.name}' failed during pipeline execution: {exc}"
                ) from exc

        # Guarantee raw columns survive
        for col in raw_cols:
            if col not in result.columns:
                result[col] = df[col]

        return result

    def transform_many(
        self,
        data: dict[str, pd.DataFrame],
    ) -> dict[str, pd.DataFrame]:
        """Apply the pipeline to a dict of {symbol: df}.  Returns same shape."""
        return {symbol: self.transform(df) for symbol, df in data.items()}

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def steps(self) -> list[Feature]:
        return list(self._steps)

    @property
    def feature_names(self) -> list[str]:
        return [f.name for f in self._steps]

    def describe(self) -> pd.DataFrame:
        """Return a DataFrame summarising the pipeline steps."""
        return pd.DataFrame(
            [
                {
                    "step": i,
                    "name": f.name,
                    "class": f.__class__.__name__,
                    "required_cols": ", ".join(f.required_cols),
                    "description": f.description,
                }
                for i, f in enumerate(self._steps)
            ]
        )

    def __repr__(self) -> str:
        steps = " → ".join(f.name for f in self._steps)
        return f"FeaturePipeline([{steps}])"

    def __len__(self) -> int:
        return len(self._steps)


# ---------------------------------------------------------------------------
# Preset pipelines — convenient starting points
# ---------------------------------------------------------------------------


def minimal_pipeline() -> FeaturePipeline:
    """Returns + log returns only.  Suitable as the base for everything."""
    from finfeatures.features.price import LogReturns, Returns

    return FeaturePipeline(Returns(), LogReturns())


def standard_pipeline() -> FeaturePipeline:
    """
    A comprehensive baseline covering returns, vol, trend and momentum.
    Safe default for regime detection and backtesting inputs.
    """
    from finfeatures.features.momentum import RSI, RateOfChange
    from finfeatures.features.price import LogReturns, PriceRange, Returns, TypicalPrice
    from finfeatures.features.statistical import RollingSkewKurt, RollingZScore
    from finfeatures.features.trend import (
        MACD,
        ExponentialMovingAverage,
        SimpleMovingAverage,
    )
    from finfeatures.features.volatility import (
        AverageTrueRange,
        BollingerBands,
        RollingVolatility,
    )
    from finfeatures.features.volume import VolumeFeatures

    return FeaturePipeline(
        # Layer 1: price transforms
        Returns(),
        LogReturns(),
        PriceRange(),
        TypicalPrice(),
        # Layer 2: volatility
        RollingVolatility(window=21),
        BollingerBands(window=20),
        AverageTrueRange(window=14),
        # Layer 3: trend
        SimpleMovingAverage(windows=[10, 20, 50, 200]),
        ExponentialMovingAverage(windows=[12, 26]),
        MACD(),
        # Layer 4: momentum
        RSI(window=14),
        RateOfChange(window=10),
        # Layer 5: volume
        VolumeFeatures(window=20),
        # Layer 6: statistical
        RollingZScore(column="log_return", window=21),
        RollingSkewKurt(column="log_return", window=63),
    )


def regime_pipeline() -> FeaturePipeline:
    """
    Extended pipeline with distributional and structural features useful as
    inputs to regime detectors (HMM, iHMM, BOCPD, threshold classifiers, etc.).
    Superset of standard_pipeline.

    Note: algorithm-specific features (e.g. RollingMMD, BOCPD run-length
    posterior) are NOT included here — they belong in your detection project.
    Extend this pipeline there:

        from finfeatures import regime_pipeline
        from mmd_regime.features import RollingMMD

        pipeline = regime_pipeline().add(RollingMMD(window=63))
        enriched = pipeline.transform(raw)
    """
    from finfeatures.features.price import (
        CandleShape,
        CrossDay,
        LogTransform,
        ShapeDynamics,
    )
    from finfeatures.features.regime import (
        DistributionShiftScore,
        DrawdownFeatures,
        RegimeIndicators,
    )
    from finfeatures.features.statistical import (
        RollingSkewKurt,
        RollingZScore,
    )
    from finfeatures.features.volatility import MovingTrueRange, RollingVolatility

    base = standard_pipeline()
    return base + FeaturePipeline(
        # Log-space transforms (must precede candle/cross/dynamics)
        LogTransform(),
        CandleShape(),
        CrossDay(),
        ShapeDynamics(),
        # Multi-horizon raw (non-annualized) realized vol
        RollingVolatility(window=10, annualize=False),
        RollingVolatility(window=30, annualize=False),
        RollingVolatility(window=60, annualize=False),
        RollingVolatility(window=90, annualize=False),
        # Moving true range
        MovingTrueRange(windows=[20, 50, 200]),
        # Statistical & regime
        RollingZScore(column="realized_vol_21", window=63),
        RollingSkewKurt(column="log_return", window=126),
        DistributionShiftScore(column="log_return", window=21),
        DrawdownFeatures(),
        RegimeIndicators(),
    )
