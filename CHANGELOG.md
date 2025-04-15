# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## `augurs-prophet` - [0.9.1](https://github.com/grafana/augurs/compare/augurs-prophet-v0.9.0...augurs-prophet-v0.9.1) - 2025-04-15

### Other
- pin zip to 2.5.0 for now ([#268](https://github.com/grafana/augurs/pull/268))
- Add JS bindings for Prophet make_future_dataframe ([#257](https://github.com/grafana/augurs/pull/257))
- *(deps)* update ureq requirement from 2.10.1 to 3.0.0 ([#245](https://github.com/grafana/augurs/pull/245))
- *(deps)* update wasmtime requirement from 28 to 29 ([#247](https://github.com/grafana/augurs/pull/247))

## `augurs-outlier` - [0.9.1](https://github.com/grafana/augurs/compare/augurs-outlier-v0.9.0...augurs-outlier-v0.9.1) - 2025-04-15

### Added
- *(outlier)* add setters for parameters ([#253](https://github.com/grafana/augurs/pull/253))

## `augurs-ets` - [0.9.1](https://github.com/grafana/augurs/compare/augurs-ets-v0.9.0...augurs-ets-v0.9.1) - 2025-04-15

### Other
- pin zip to 2.5.0 for now ([#268](https://github.com/grafana/augurs/pull/268))

## `augurs-clustering` - [0.9.1](https://github.com/grafana/augurs/compare/augurs-clustering-v0.9.0...augurs-clustering-v0.9.1) - 2025-04-15

### Other
- pin zip to 2.5.0 for now ([#268](https://github.com/grafana/augurs/pull/268))

## `augurs-changepoint` - [0.9.0](https://github.com/grafana/augurs/compare/augurs-changepoint-v0.8.1...augurs-changepoint-v0.9.0) - 2025-01-14

No changes in this release.

## `augurs-clustering` - [0.9.0](https://github.com/grafana/augurs/compare/augurs-clustering-v0.8.1...augurs-clustering-v0.9.0) - 2025-01-14

### Changed
- *(clustering)* [**breaking**] use new DbscanCluster type instead of isize ([#233](https://github.com/grafana/augurs/pull/233))

This changes the return type of `DbscanClusterer::fit` from `Vec<isize>` to `Vec<DbscanCluster>`, which is a more self-explanatory type.
The ID of the first cluster is now `1` (instead of `0`), and the IDs of the subsequent clusters are incremented by `1`.

## `augurs-core` - [0.9.0](https://github.com/grafana/augurs/compare/augurs-core-v0.8.1...augurs-core-v0.9.0) - 2025-01-14

### Added
- exposed `FloatIterExt` with helper methods for calculating summary statistics on iterators over floats ([#227](https://github.com/grafana/augurs/pull/227))

## `augurs-dtw` - [0.9.0](https://github.com/grafana/augurs/compare/augurs-dtw-v0.8.1...augurs-dtw-v0.9.0) - 2025-01-14

No changes in this release.

## `augurs-ets` - [0.9.0](https://github.com/grafana/augurs/compare/augurs-ets-v0.8.1...augurs-ets-v0.9.0) - 2025-01-14

No changes in this release.

## `augurs-forecaster` - [0.9.0](https://github.com/grafana/augurs/compare/augurs-forecaster-v0.8.1...augurs-forecaster-v0.9.0) - 2025-01-14

### Added
- allow ignoring NaNs in power transforms ([#234](https://github.com/grafana/augurs/pull/234))
- add NaN handling to MinMaxScaler and StandardScaler ([#227](https://github.com/grafana/augurs/pull/227))

## `augurs-mstl` - [0.9.0](https://github.com/grafana/augurs/compare/augurs-mstl-v0.8.1...augurs-mstl-v0.9.0) - 2025-01-14

No changes in this release.

## `augurs-outlier` - [0.9.0](https://github.com/grafana/augurs/compare/augurs-outlier-v0.8.1...augurs-outlier-v0.9.0) - 2025-01-14

No changes in this release.

## `augurs-prophet` - [0.9.0](https://github.com/grafana/augurs/compare/augurs-prophet-v0.8.1...augurs-prophet-v0.9.0) - 2025-01-14

No changes in this release.
