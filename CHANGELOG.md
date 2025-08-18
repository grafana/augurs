# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## `augurs-seasons` - [0.10.1](https://github.com/grafana/augurs/compare/augurs-seasons-v0.10.0...augurs-seasons-v0.10.1) - 2025-08-18

### Fixed
- fix clippy lints for Rust 1.88 ([#315](https://github.com/grafana/augurs/pull/315))

## `augurs-prophet` - [0.10.1](https://github.com/grafana/augurs/compare/augurs-prophet-v0.10.0...augurs-prophet-v0.10.1) - 2025-08-18

### Fixed
- fix clippy lints for Rust 1.88 ([#315](https://github.com/grafana/augurs/pull/315))
- *(prophet)* use holiday prior scale for regressors ([#305](https://github.com/grafana/augurs/pull/305))

## `augurs-outlier` - [0.10.1](https://github.com/grafana/augurs/compare/augurs-outlier-v0.10.0...augurs-outlier-v0.10.1) - 2025-08-18

### Other
- allow dead code in outlier detector test to serve as example

## `augurs-ets` - [0.10.1](https://github.com/grafana/augurs/compare/augurs-ets-v0.10.0...augurs-ets-v0.10.1) - 2025-08-18

### Fixed
- fix clippy lints for Rust 1.88 ([#315](https://github.com/grafana/augurs/pull/315))

## `augurs-dtw` - [0.10.1](https://github.com/grafana/augurs/compare/augurs-dtw-v0.10.0...augurs-dtw-v0.10.1) - 2025-08-18

### Fixed
- fix clippy lints for Rust 1.88 ([#315](https://github.com/grafana/augurs/pull/315))

## [0.10.0](https://github.com/grafana/augurs/compare/augurs-v0.9.0...augurs-v0.10.0) - 2025-05-19

### `augurs-seasons`

No changes in this release.

### `augurs-prophet`

- Add JS bindings for Prophet make_future_dataframe ([#257](https://github.com/grafana/augurs/pull/257), by @shenxiangzhuang)
- *(deps)* bump zipfile to 3.x ([#286](https://github.com/grafana/augurs/pull/286))
- *(deps)* update ureq requirement from 2.10.1 to 3.0.0 ([#245](https://github.com/grafana/augurs/pull/245))
- *(deps)* bump wasmtime and wasmtime-wasi to 32 ([#289](https://github.com/grafana/augurs/pull/289))

### `augurs-outlier`

#### Added
- add setters for parameters ([#253](https://github.com/grafana/augurs/pull/253), by @shenxiangzhuang)

### `augurs-ets`

No changes in this release.

### `augurs-mstl`

No changes in this release.

### `augurs-dtw`

No changes in this release.

### `augurs-clustering`

No changes in this release.

### `augurs-changepoint`

No changes in this release.

## [0.9.0](https://github.com/grafana/augurs/compare/augurs-v0.8.1...augurs-v0.9.0) - 2025-01-14

### `augurs-changepoint`

No changes in this release.

### `augurs-clustering`

#### Changed
- *(clustering)* [**breaking**] use new DbscanCluster type instead of isize ([#233](https://github.com/grafana/augurs/pull/233))

This changes the return type of `DbscanClusterer::fit` from `Vec<isize>` to `Vec<DbscanCluster>`, which is a more self-explanatory type.
The ID of the first cluster is now `1` (instead of `0`), and the IDs of the subsequent clusters are incremented by `1`.

### `augurs-core`

#### Added
- exposed `FloatIterExt` with helper methods for calculating summary statistics on iterators over floats ([#227](https://github.com/grafana/augurs/pull/227))

### `augurs-dtw`

No changes in this release.

### `augurs-ets`

No changes in this release.

### `augurs-forecaster`

#### Added
- allow ignoring NaNs in power transforms ([#234](https://github.com/grafana/augurs/pull/234))
- add NaN handling to MinMaxScaler and StandardScaler ([#227](https://github.com/grafana/augurs/pull/227))

### `augurs-mstl`

No changes in this release.

### `augurs-outlier`

No changes in this release.

### `augurs-prophet`

No changes in this release.
