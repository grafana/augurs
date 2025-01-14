# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.9.0](https://github.com/grafana/augurs/compare/augurs-outlier-v0.8.1...augurs-outlier-v0.9.0) - 2025-01-14

### Added

- *(forecaster)* add NaN handling to MinMaxScaler and StandardScaler (#227)

## [0.8.1](https://github.com/grafana/augurs/compare/augurs-outlier-v0.8.0...augurs-outlier-v0.8.1) - 2025-01-07

### Other

- update Cargo.toml dependencies

## [0.8.0](https://github.com/grafana/augurs/compare/augurs-outlier-v0.7.0...augurs-outlier-v0.8.0) - 2024-12-23

### Other

- *(deps)* update rv requirement from 0.17.0 to 0.18.0 (#198)

## [0.7.0](https://github.com/grafana/augurs/compare/augurs-outlier-v0.6.3...augurs-outlier-v0.7.0) - 2024-11-25

### Other

- update Cargo.toml dependencies

## [0.5.1](https://github.com/grafana/augurs/compare/augurs-outlier-v0.5.0...augurs-outlier-v0.5.1) - 2024-10-24

### Other

- define lints in Cargo.toml instead of each crate's lib.rs ([#138](https://github.com/grafana/augurs/pull/138))

## [0.5.0](https://github.com/grafana/augurs/compare/augurs-outlier-v0.5.0...augurs-outlier-v0.4.3) - 2024-10-18

No changes to the Rust crate; this version bump is due to breaking changes in the
Javascript package.

## [0.4.0](https://github.com/grafana/augurs/compare/augurs-outlier-v0.3.1...augurs-outlier-v0.4.0) - 2024-10-16

### Added

- add cmdstan-based optimizer for augurs-prophet ([#121](https://github.com/grafana/augurs/pull/121))
- add 'augurs' convenience crate, re-exporting other crates ([#117](https://github.com/grafana/augurs/pull/117))

### Fixed

- [**breaking**] add serde derives for more types ([#112](https://github.com/grafana/augurs/pull/112))
- [**breaking**] make `cluster_band` optional, undefined if no cluster is found ([#105](https://github.com/grafana/augurs/pull/105))

### Other

- Add Prophet algorithm in `augurs-prophet` crate ([#118](https://github.com/grafana/augurs/pull/118))

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

