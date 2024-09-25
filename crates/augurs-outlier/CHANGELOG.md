# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.0](https://github.com/grafana/augurs/compare/augurs-outlier-v0.3.1...augurs-outlier-v0.4.0) - 2024-09-25

### Added

- add 'augurs' convenience crate, re-exporting other crates ([#117](https://github.com/grafana/augurs/pull/117))
- export the data types used by MAD and DBSCAN detectors ([#117](https://github.com/grafana/augurs/pull/117))

### Fixed

- [**breaking**] add serde derives for more types ([#112](https://github.com/grafana/augurs/pull/112))
- [**breaking**] make `cluster_band` optional, undefined if no cluster is found ([#105](https://github.com/grafana/augurs/pull/105))

### Changed

- rename `DBSCANDetector` to `DbscanDetector` ([#117](https://github.com/grafana/augurs/pull/117))
- `DbscanDetector::parallelize` now takes self by value rather than mutable reference to encourage chaining ([#117](https://github.com/grafana/augurs/pull/117))

## [0.3.1](https://github.com/grafana/augurs/compare/augurs-outlier-v0.3.0...augurs-outlier-v0.3.1) - 2024-07-30

No notable changes in this release.

## [0.3.0](https://github.com/grafana/augurs/compare/augurs-outlier-v0.2.0...augurs-outlier-v0.3.0) - 2024-07-30

### Fixed
- use more accurate series count to avoid panic ([#101](https://github.com/grafana/augurs/pull/101))

### Other
- Add MAD outlier algorithm ([#89](https://github.com/grafana/augurs/pull/89))
- Update rustc-hash requirement from 1.1.0 to 2.0.0 ([#95](https://github.com/grafana/augurs/pull/95))
- Remove unsupported .github/workflows/bencher subdirectory and old benchmark workflow ([#90](https://github.com/grafana/augurs/pull/90))

## [0.2.0](https://github.com/grafana/augurs/compare/augurs-outlier-v0.1.2...augurs-outlier-v0.2.0) - 2024-06-05

### Added
- add outlier crate with DBSCAN implementation ([#79](https://github.com/grafana/augurs/pull/79))

### Other
- Add empty CHANGELOGs

