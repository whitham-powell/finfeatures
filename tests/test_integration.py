"""
End-to-end integration test.

Simulates the full workflow: raw OHLCV → pipeline → feature matrix ready
for downstream consumption (regime detector, backtester, ML model).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from finfeatures import minimal_pipeline, regime_pipeline, standard_pipeline
from finfeatures.core import Columns
from finfeatures.io import PandasAdapter


class TestEndToEnd:
    """Full pipeline → adapter → downstream consumption."""

    def test_standard_pipeline_output_shape(self, ohlcv_daily):
        out = standard_pipeline().transform(ohlcv_daily)
        assert out.shape[0] == len(ohlcv_daily)
        assert out.shape[1] > len(ohlcv_daily.columns)

    def test_feature_matrix_for_ml(self, ohlcv_daily):
        """
        Simulate extracting a clean feature matrix for an ML model.
        All rows with NaN (warm-up period) are dropped.
        """
        out = standard_pipeline().transform(ohlcv_daily)
        adapter = PandasAdapter(out)
        feature_cols = adapter.feature_columns(exclude_raw=True)
        clean = adapter.dropna_features()

        # Should still have a meaningful number of rows
        assert len(clean) > 200
        # All selected feature columns should be non-null
        assert clean[feature_cols].isna().sum().sum() == 0

    def test_regime_pipeline_features_present(self, ohlcv_daily):
        out = regime_pipeline().transform(ohlcv_daily)
        assert "dist_shift_log_return_21" in out.columns
        assert "drawdown" in out.columns
        assert "stress_score" in out.columns

    def test_pipeline_is_composable_with_custom_feature(self, ohlcv_daily):
        """User-defined features can be appended to preset pipelines."""
        from finfeatures.core import Feature

        class HighLowMidpoint(Feature):
            name = "_test_hl_midpoint"
            required_cols = [Columns.HIGH, Columns.LOW]
            description = "Midpoint of high and low"

            def compute(self, df: pd.DataFrame) -> pd.DataFrame:
                out = df.copy()
                out["hl_midpoint"] = (df[Columns.HIGH] + df[Columns.LOW]) / 2
                return out

        pipeline = minimal_pipeline().add(HighLowMidpoint())
        out = pipeline.transform(ohlcv_daily)
        assert "hl_midpoint" in out.columns
        assert Columns.RETURN in out.columns

    def test_multi_asset_pipeline(self, ohlcv_daily):
        """Pipeline applies independently and correctly to each asset."""
        assets = {
            "SPY": ohlcv_daily,
            "QQQ": ohlcv_daily * 1.1,  # synthetic second asset
            "IWM": ohlcv_daily * 0.9,
        }
        results = minimal_pipeline().transform_many(assets)
        assert set(results.keys()) == {"SPY", "QQQ", "IWM"}
        for _symbol, df in results.items():
            assert Columns.RETURN in df.columns
            assert Columns.LOG_RETURN in df.columns

    def test_output_index_unchanged(self, ohlcv_daily):
        """The DatetimeIndex must be identical after pipeline transforms."""
        out = standard_pipeline().transform(ohlcv_daily)
        pd.testing.assert_index_equal(out.index, ohlcv_daily.index)

    def test_raw_values_unchanged(self, ohlcv_daily):
        """Raw OHLCV values must be byte-for-byte identical after pipeline."""
        out = standard_pipeline().transform(ohlcv_daily)
        for col in Columns.OHLCV:
            pd.testing.assert_series_equal(
                ohlcv_daily[col],
                out[col],
                obj=f"column '{col}' was altered by the pipeline",
            )

    def test_pandas_adapter_to_numpy(self, ohlcv_daily):
        out = standard_pipeline().transform(ohlcv_daily)
        adapter = PandasAdapter(out)
        matrix = adapter.to_numpy(["close", "realized_vol_21"])
        assert matrix.shape == (len(ohlcv_daily), 2)
        assert matrix.dtype == float

    def test_pipeline_describe(self, ohlcv_daily):
        p = standard_pipeline()
        desc = p.describe()
        assert isinstance(desc, pd.DataFrame)
        assert "name" in desc.columns
        assert len(desc) == len(p)


class TestDownstreamConsumption:
    """
    Simulate downstream tasks consuming the feature matrix.
    These are smoke tests only — not testing the downstream logic.
    """

    def test_regime_detector_input_format(self, ohlcv_daily):
        """
        Verify the feature matrix has the shape and columns
        expected by a rolling-window regime detector (e.g. HMM).
        """
        out = regime_pipeline().transform(ohlcv_daily)
        adapter = PandasAdapter(out)
        clean = adapter.dropna_features()

        # Select typical regime detection inputs
        regime_cols = [
            c
            for c in clean.columns
            if any(
                tag in c
                for tag in [
                    "log_return",
                    "realized_vol",
                    "dist_shift",
                    "zscore",
                    "drawdown",
                ]
            )
        ]
        assert len(regime_cols) >= 3

        X = clean[regime_cols].to_numpy(dtype=float)
        assert not np.any(np.isnan(X))
        assert X.shape[0] > 100

    def test_backtest_signal_input_format(self, ohlcv_daily):
        """
        Verify the feature matrix has columns usable as backtest signals.
        Signals: MACD histogram, RSI, rolling z-score.
        """
        out = standard_pipeline().transform(ohlcv_daily)
        for col in ["macd_hist", "rsi_14", "log_return_zscore_21"]:
            assert col in out.columns

    def test_ml_feature_matrix_finite(self, ohlcv_daily):
        """After dropna, the feature matrix should contain no inf or nan."""
        out = standard_pipeline().transform(ohlcv_daily)
        clean = PandasAdapter(out).dropna_features()
        feature_cols = [c for c in clean.columns if c not in Columns.OHLCV]
        X = clean[feature_cols].to_numpy(dtype=float)
        assert np.all(np.isfinite(X)), "Feature matrix contains inf or nan after dropna"
