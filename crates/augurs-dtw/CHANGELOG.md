# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.9.0](https://github.com/grafana/augurs/compare/augurs-dtw-v0.8.1...augurs-dtw-v0.9.0) - 2025-01-14

### Added

- *(forecaster)* add NaN handling to MinMaxScaler and StandardScaler (#227)

## [0.8.1](https://github.com/grafana/augurs/compare/augurs-dtw-v0.8.0...augurs-dtw-v0.8.1) - 2025-01-07

### Other

- update Cargo.toml dependencies

## [0.6.0](https://github.com/grafana/augurs/compare/augurs-dtw-v0.5.4...augurs-dtw-v0.6.0) - 2024-11-08

### Added

- [**breaking**] split JS package into separate crates ([#149](https://github.com/grafana/augurs/pull/149))

## [0.5.2](https://github.com/grafana/augurs/compare/augurs-dtw-v0.5.1...augurs-dtw-v0.5.2) - 2024-10-25

### Other

- add benchmark for Prophet ([#140](https://github.com/grafana/augurs/pull/140))

## [0.5.1](https://github.com/grafana/augurs/compare/augurs-dtw-v0.5.0...augurs-dtw-v0.5.1) - 2024-10-24

### Other

- define lints in Cargo.toml instead of each crate's lib.rs ([#138](https://github.com/grafana/augurs/pull/138))

## [0.5.0](https://github.com/grafana/augurs/compare/augurs-dtw-v0.5.0...augurs-dtw-v0.4.3) - 2024-10-18

No changes to the Rust crate; this version bump is due to breaking changes in the
Javascript package.

## [0.4.0](https://github.com/grafana/augurs/compare/augurs-dtw-v0.3.1...augurs-dtw-v0.4.0) - 2024-10-16

### Added

- add 'augurs' convenience crate, re-exporting other crates ([#117](https://github.com/grafana/augurs/pull/117))
- derive Clone for Dtw ([#114](https://github.com/grafana/augurs/pull/114))
- parallel DTW calculations in augurs-js ([#111](https://github.com/grafana/augurs/pull/111))
- add `augurs-dtw` crate with dynamic time warping implementation ([#98](https://github.com/grafana/augurs/pull/98))

### Other
- Add `augurs-dtw` crate
