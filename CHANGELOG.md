# Changelog

All notable changes to this project will be documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Changed
- Renamed `regime.py` module → `composite.py`
- Renamed `RegimeIndicators` class → `CompositeScores` (registry name: `composite_scores`)
- Renamed `VolatilityRegime` class → `VolatilityRatio` (registry name: `volatility_ratio`)
- Renamed `regime_pipeline()` → `extended_pipeline()`
- Renamed output columns: `vol_regime_ratio` → `vol_ratio`, `vol_regime_zscore` → `vol_ratio_zscore`
- Reworded docstrings across feature modules to use general-purpose language

## [0.1.0] - Initial release
### Added
- `Feature` base class with auto-registration via `FeatureRegistry`
- `FeaturePipeline` — ordered, composable feature composition
- `DataSource` ABC — swap any market data provider
- Feature modules: `price`, `volatility`, `trend`, `momentum`, `volume`,
  `statistical`, `composite`
- `YFinanceSource` reference implementation (optional dependency)
- `PandasAdapter` portability wrapper
- Preset pipelines: `minimal_pipeline`, `standard_pipeline`, `extended_pipeline`
- 81 tests, zero network dependencies in default suite
