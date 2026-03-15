"""
finfeatures — Financial Timeseries Feature Library
===================================================

A source-agnostic, pipeline-based library for deriving features from raw
OHLCV market data.  Raw columns are never lost; all features are additive.

Quick start
-----------
    from finfeatures import FeaturePipeline, standard_pipeline
    from finfeatures.sources import YFinanceSource

    # 1. Fetch raw data
    source = YFinanceSource()
    raw = source.fetch("SPY", start="2020-01-01", end="2024-12-31")

    # 2. Build a pipeline
    pipeline = standard_pipeline()

    # 3. Derive features
    enriched = pipeline.transform(raw)
    print(enriched.columns.tolist())

    # 4. Use in any downstream task (regime detection, backtesting, ML)
    feature_matrix = enriched.dropna()

Layers
------
    DataSource   →  raw OHLCV DataFrame
    Feature      →  pure transformation: DataFrame → DataFrame (additive)
    FeaturePipeline →  ordered composition of Features
    DataFrameAdapter → thin wrapper for backend portability (pandas default)
"""

# Import features subpackage to trigger auto-registration of all features
import finfeatures.features  # noqa: F401
from finfeatures.core import (
    Columns,
    DataSource,
    Feature,
    FeaturePipeline,
    FeatureRegistry,
    extended_pipeline,
    minimal_pipeline,
    standard_pipeline,
)

__version__ = "0.1.0"

__all__ = [
    # Core
    "Columns",
    "Feature",
    "FeatureRegistry",
    "DataSource",
    "FeaturePipeline",
    # Preset pipelines
    "minimal_pipeline",
    "standard_pipeline",
    "extended_pipeline",
    # Version
    "__version__",
]
