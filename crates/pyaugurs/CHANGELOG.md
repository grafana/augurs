# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0-alpha.0](https://github.com/grafana/augurs/releases/tag/pyaugurs-v0.1.0-alpha.0) - 2023-09-08

### Other
- Add workspace metadata and use in all the subpackages ([#33](https://github.com/grafana/augurs/pull/33))
- (cargo-release) version 0.1.0-alpha.1
- Use -alpha.0 suffix in crate versions
- Bump all versions to latest ([#26](https://github.com/grafana/augurs/pull/26))
- Add __repr__ for pyaugurs structs
- Accept &[f64] in MSTL::fit, rather than owned Vec ([#17](https://github.com/grafana/augurs/pull/17))
- Bump pyo3 and numpy to 0.19.0 ([#16](https://github.com/grafana/augurs/pull/16))
- Add type stubs to pyaugurs
- Update docstring on PyTrendModel
- Fix doc comment on pyaugurs Forecast
- Add numpy as dependency to pyaugurs
- Bump maturin build-system version to stable
- Initial commit
