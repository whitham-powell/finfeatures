# Changelog

All notable changes to this project will be documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - Initial release
### Added
- `Feature` base class with auto-registration via `FeatureRegistry`
- `FeaturePipeline` — ordered, composable feature composition
- `DataSource` ABC — swap any market data provider
- Feature modules: `price`, `volatility`, `trend`, `momentum`, `volume`,
  `statistical`, `regime`
- `YFinanceSource` reference implementation (optional dependency)
- `PandasAdapter` portability wrapper
- Preset pipelines: `minimal_pipeline`, `standard_pipeline`, `regime_pipeline`
- 81 tests, zero network dependencies in default suite
