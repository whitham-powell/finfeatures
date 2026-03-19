"""Tests for finfeatures.core — base classes, registry, pipeline."""

from __future__ import annotations

import pandas as pd
import pytest

from finfeatures.core import (
    Columns,
    Feature,
    FeaturePipeline,
    FeatureRegistry,
    extended_pipeline,
    minimal_pipeline,
    standard_pipeline,
)
from finfeatures.features.price import LogReturns, Returns
from finfeatures.features.volatility import RollingVolatility

# ---------------------------------------------------------------------------
# Columns constants
# ---------------------------------------------------------------------------


def test_columns_ohlcv_list():
    assert set(Columns.OHLCV) == {"open", "high", "low", "close", "volume"}


# ---------------------------------------------------------------------------
# Feature base class contract
# ---------------------------------------------------------------------------


def test_feature_preserves_raw_columns(ohlcv_daily, raw_cols):
    """Every feature must preserve all source columns in its output."""
    feat = Returns()
    out = feat(ohlcv_daily)
    for col in raw_cols:
        assert col in out.columns, f"Raw column '{col}' was lost after Returns()"


def test_feature_does_not_mutate_input(ohlcv_daily):
    """Features must not mutate the input DataFrame."""
    original_cols = list(ohlcv_daily.columns)
    original_values = ohlcv_daily["close"].copy()
    Returns()(ohlcv_daily)
    assert list(ohlcv_daily.columns) == original_cols
    pd.testing.assert_series_equal(ohlcv_daily["close"], original_values)


def test_feature_missing_required_col_raises():
    """Missing required column must raise ValueError with informative message."""
    df = pd.DataFrame({"open": [1, 2, 3]})  # no 'close'
    with pytest.raises(ValueError, match="requires columns.*Available columns"):
        Returns()(df)


# ---------------------------------------------------------------------------
# FeatureRegistry
# ---------------------------------------------------------------------------


def test_registry_contains_builtin_features():
    registry = FeatureRegistry.all()
    expected = ["returns", "log_returns", "rsi", "macd", "rolling_volatility"]
    for name in expected:
        assert name in registry, f"Expected '{name}' in registry"


def test_registry_get_known_feature():
    cls = FeatureRegistry.get("returns")
    assert cls is Returns


def test_registry_get_unknown_raises():
    with pytest.raises(KeyError, match="No feature named"):
        FeatureRegistry.get("does_not_exist_xyz")


def test_registry_list_is_sorted():
    names = FeatureRegistry.list()
    assert names == sorted(names)


def test_registry_duplicate_name_raises():
    """Registering two different classes with the same name must raise."""
    with pytest.raises(ValueError, match="already registered"):

        class DuplicateReturns(Feature):
            name = "returns"  # already taken
            required_cols = [Columns.CLOSE]

            def compute(self, df):
                return df.copy()


# ---------------------------------------------------------------------------
# FeaturePipeline
# ---------------------------------------------------------------------------


def test_pipeline_applies_features_in_order(ohlcv_daily, raw_cols):
    pipeline = FeaturePipeline(Returns(), LogReturns())
    out = pipeline.transform(ohlcv_daily)
    assert Columns.RETURN in out.columns
    assert Columns.LOG_RETURN in out.columns
    for col in raw_cols:
        assert col in out.columns


def test_pipeline_preserves_raw_cols_through_all_steps(ohlcv_daily, raw_cols):
    pipeline = FeaturePipeline(Returns(), LogReturns(), RollingVolatility(window=21))
    out = pipeline.transform(ohlcv_daily)
    for col in raw_cols:
        assert col in out.columns


def test_pipeline_addition(ohlcv_daily):
    p1 = FeaturePipeline(Returns())
    p2 = FeaturePipeline(LogReturns())
    combined = p1 + p2
    out = combined.transform(ohlcv_daily)
    assert Columns.RETURN in out.columns
    assert Columns.LOG_RETURN in out.columns


def test_pipeline_add_single_feature(ohlcv_daily):
    pipeline = FeaturePipeline(Returns()).add(LogReturns())
    out = pipeline.transform(ohlcv_daily)
    assert Columns.RETURN in out.columns
    assert Columns.LOG_RETURN in out.columns


def test_pipeline_describe_returns_dataframe():
    pipeline = FeaturePipeline(Returns(), LogReturns())
    desc = pipeline.describe()
    assert isinstance(desc, pd.DataFrame)
    assert len(desc) == 2
    assert "name" in desc.columns


def test_pipeline_len():
    p = FeaturePipeline(Returns(), LogReturns(), RollingVolatility())
    assert len(p) == 3


def test_pipeline_transform_many(ohlcv_daily):
    pipeline = FeaturePipeline(Returns(), LogReturns())
    data = {"SPY": ohlcv_daily, "QQQ": ohlcv_daily.copy()}
    result = pipeline.transform_many(data)
    assert set(result.keys()) == {"SPY", "QQQ"}
    for df in result.values():
        assert Columns.RETURN in df.columns


def test_pipeline_lookup_by_string(ohlcv_daily):
    """Pipeline must accept feature names as strings for registry lookup."""
    pipeline = FeaturePipeline("returns", "log_returns")
    out = pipeline.transform(ohlcv_daily)
    assert Columns.RETURN in out.columns
    assert Columns.LOG_RETURN in out.columns


# ---------------------------------------------------------------------------
# Preset pipelines smoke tests
# ---------------------------------------------------------------------------


def test_minimal_pipeline_runs(ohlcv_daily):
    out = minimal_pipeline().transform(ohlcv_daily)
    assert Columns.RETURN in out.columns
    assert Columns.LOG_RETURN in out.columns


def test_standard_pipeline_runs(ohlcv_daily):
    out = standard_pipeline().transform(ohlcv_daily)
    # Should produce many columns
    assert len(out.columns) > 20


def test_extended_pipeline_runs(ohlcv_daily):
    out = extended_pipeline().transform(ohlcv_daily)
    # extended_pipeline adds distribution shift and drawdown on top of standard
    assert "dist_shift_log_return_21" in out.columns
    assert "drawdown" in out.columns
    assert "stress_score" in out.columns


def test_pipeline_no_raw_column_loss_standard(ohlcv_daily, raw_cols):
    out = standard_pipeline().transform(ohlcv_daily)
    for col in raw_cols:
        assert col in out.columns, f"'{col}' lost after standard_pipeline"


# ---------------------------------------------------------------------------
# DatetimeIndex validation (Item 1)
# ---------------------------------------------------------------------------


def test_pipeline_rejects_range_index():
    """Pipeline must reject a DataFrame with a RangeIndex."""
    df = pd.DataFrame(
        {
            "open": [1, 2],
            "high": [2, 3],
            "low": [0.5, 1.5],
            "close": [1.5, 2.5],
            "volume": [100, 200],
        }
    )
    pipeline = FeaturePipeline(Returns())
    with pytest.raises(ValueError, match="DatetimeIndex"):
        pipeline.transform(df)


def test_pipeline_accepts_datetime_index(ohlcv_daily):
    """Existing DatetimeIndex fixtures must still work."""
    pipeline = FeaturePipeline(Returns())
    out = pipeline.transform(ohlcv_daily)
    assert "return" in out.columns


# ---------------------------------------------------------------------------
# min_periods (Item 4)
# ---------------------------------------------------------------------------


def test_feature_min_periods_default():
    """Base Feature.min_periods returns 0."""
    from finfeatures.core.base import Feature

    class Dummy(Feature):
        name = "_dummy_test_min_periods"
        required_cols: list[str] = []

        def compute(self, df: pd.DataFrame) -> pd.DataFrame:
            return df.copy()

    assert Dummy().min_periods == 0


def test_pipeline_min_periods():
    """Pipeline.min_periods should be max across steps."""
    pipeline = FeaturePipeline(Returns(), LogReturns(), RollingVolatility(window=21))
    assert pipeline.min_periods == 22  # window + 1


# ---------------------------------------------------------------------------
# Dependency validation (Item 5)
# ---------------------------------------------------------------------------


def test_pipeline_validates_missing_dependency():
    """CompositeScores without MACD/RSI/Vol upstream must raise."""
    from finfeatures.features.composite import CompositeScores

    pipeline = FeaturePipeline(Returns(), CompositeScores())
    df = pd.DataFrame(
        {"open": [1.0], "high": [2.0], "low": [0.5], "close": [1.5], "volume": [100]},
        index=pd.to_datetime(["2020-01-01"]),
    )
    with pytest.raises(ValueError, match="requires columns"):
        pipeline.transform(df)


def test_pipeline_valid_preset_passes(ohlcv_daily):
    """extended_pipeline().transform(ohlcv_daily) must work end-to-end."""
    out = extended_pipeline().transform(ohlcv_daily)
    assert "stress_score" in out.columns
