"""
finfeatures — Financial Timeseries Feature Library
===================================================

A source-agnostic, pipeline-based library for deriving features from raw
OHLCV market data.  Raw columns are never lost; all features are additive.

Quick start
-----------
    import pandas as pd
    from finfeatures import standard_pipeline

    # 1. Load raw OHLCV data (any source — CSV, database, API, …)
    raw = pd.read_csv("spy.csv", index_col="date", parse_dates=True)

    # 2. Derive features
    enriched = standard_pipeline().transform(raw)
    print(enriched.columns.tolist())

    # 3. Use in any downstream task (backtesting, ML, …)
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
    Feature,
    FeaturePipeline,
    FeatureRegistry,
    extended_pipeline,
    minimal_pipeline,
    standard_pipeline,
)

__version__ = "0.3.0"

__all__ = [
    # Core
    "Columns",
    "Feature",
    "FeatureRegistry",
    "FeaturePipeline",
    # Preset pipelines
    "minimal_pipeline",
    "standard_pipeline",
    "extended_pipeline",
    # Version
    "__version__",
]
