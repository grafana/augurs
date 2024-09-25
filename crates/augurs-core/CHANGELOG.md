# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.0](https://github.com/grafana/augurs/compare/augurs-core-v0.3.1...augurs-core-v0.4.0) - 2024-09-25

### Added

- add 'augurs' convenience crate, re-exporting other crates ([#117](https://github.com/grafana/augurs/pull/117))
- add 'Forecast::chain' method to chain two forecasts together ([#115](https://github.com/grafana/augurs/pull/115))
- add `augurs-dtw` crate with dynamic time warping implementation ([#98](https://github.com/grafana/augurs/pull/98))

### Fixed

- [**breaking**] add serde derives for more types ([#112](https://github.com/grafana/augurs/pull/112))

## [0.3.1](https://github.com/grafana/augurs/compare/augurs-core-v0.3.0...augurs-core-v0.3.1) - 2024-07-30

No notable changes in this release.

## [0.3.0](https://github.com/grafana/augurs/compare/augurs-core-v0.2.0...augurs-core-v0.3.0) - 2024-07-30

### Other
- Remove unsupported .github/workflows/bencher subdirectory and old benchmark workflow ([#90](https://github.com/grafana/augurs/pull/90))

## [0.2.0](https://github.com/grafana/augurs/compare/augurs-core-v0.1.2...augurs-core-v0.2.0) - 2024-06-05

### Added
- [**breaking**] add transformations and high-level forecasting API ([#65](https://github.com/grafana/augurs/pull/65))

## [0.1.2](https://github.com/grafana/augurs/compare/augurs-core-v0.1.1...augurs-core-v0.1.2) - 2024-02-20

### Added
- *(core)* add interpolation iterator adaptor ([#63](https://github.com/grafana/augurs/pull/63))

## [0.1.1](https://github.com/grafana/augurs/compare/augurs-core-v0.1.0...augurs-core-v0.1.1) - 2024-02-15

### Other
- Add license files to repo root and symlinks in crate directories ([#43](https://github.com/grafana/augurs/pull/43))
- Add repository to sub-crate Cargo.tomls ([#42](https://github.com/grafana/augurs/pull/42))

## [0.1.0-alpha.0](https://github.com/grafana/augurs/releases/tag/augurs-core-v0.1.0-alpha.0) - 2023-09-08

### Other
- Add workspace metadata and use in all the subpackages ([#33](https://github.com/grafana/augurs/pull/33))
- (cargo-release) version 0.1.0-alpha.1
- Use -alpha.0 suffix in crate versions
- Bump all versions to latest ([#26](https://github.com/grafana/augurs/pull/26))
- Short circuit where horizon == 0 in predict methods
- Initial commit
