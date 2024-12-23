# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.8.0](https://github.com/grafana/augurs/compare/augurs-prophet-v0.7.0...augurs-prophet-v0.8.0) - 2024-12-23

### Added

- add Forecaster wrapper for Prophet (#191)

### Fixed

- *(docs)* fix incorrect link to chrono
- add explicit link to chrono method (#192)

### Other

- *(deps)* update wasmtime requirement from 27 to 28 (#216)
- Commit prophet-wasmstan.wasm to git ([#206](https://github.com/grafana/augurs/pull/206))
- *(deps)* update statrs requirement from 0.17.1 to 0.18.0 (#187)

## [0.7.0](https://github.com/grafana/augurs/compare/augurs-prophet-v0.6.3...augurs-prophet-v0.7.0) - 2024-11-25

### Breaking Changes

- Support sub-daily & non-UTC holidays ([#181](https://github.com/grafana/augurs/pull/181))
- Changed type of lower/upper windows from `i32` to `u32` ([#177](https://github.com/grafana/augurs/pull/177))
- *(deps)* update wasmtime requirement from 26 to 27 ([#183](https://github.com/grafana/augurs/pull/183))

### Fixed

- add a separate feature for each holiday's lower/upper windows ([#179](https://github.com/grafana/augurs/pull/179))

### Dependencies

- *(deps)* update thiserror requirement from 1.0.40 to 2.0.3 ([#164](https://github.com/grafana/augurs/pull/164))
- *(deps)* update wasmtime requirement from 26 to 27 ([#183](https://github.com/grafana/augurs/pull/183))

## [0.6.3](https://github.com/grafana/augurs/compare/augurs-prophet-v0.6.2...augurs-prophet-v0.6.3) - 2024-11-20

### Fixed

- correctly set holiday feature to 1 for all holiday dates ([#175](https://github.com/grafana/augurs/pull/175))

### Other

- Fix broken intra-doc link in wasmstan module

## [0.6.2](https://github.com/grafana/augurs/compare/augurs-prophet-v0.6.1...augurs-prophet-v0.6.2) - 2024-11-10

### Fixed

- use OUT_DIR instead of CARGO_MANIFEST_DIR

### Other

- Add Prophet WASM example ([#152](https://github.com/grafana/augurs/pull/152))

## [0.6.1](https://github.com/grafana/augurs/compare/augurs-prophet-v0.6.0...augurs-prophet-v0.6.1) - 2024-11-09

### Fixed

- fix build script to work in docs.rs builds ([#150](https://github.com/grafana/augurs/pull/150))

## [0.6.0](https://github.com/grafana/augurs/compare/augurs-prophet-v0.5.4...augurs-prophet-v0.6.0) - 2024-11-08

### Added

- add wasmtime based optimizer for dependency-free Rust impl

### Other

- Improve error handling in wasmstan optimizer

## [0.5.3](https://github.com/grafana/augurs/compare/augurs-prophet-v0.5.2...augurs-prophet-v0.5.3) - 2024-10-25

### Fixed

- correctly pass `cap_scaled` to `piecewise_logistic` ([#142](https://github.com/grafana/augurs/pull/142))

### Other

- Remove commented code

## [0.5.2](https://github.com/grafana/augurs/compare/augurs-prophet-v0.5.1...augurs-prophet-v0.5.2) - 2024-10-25

### Other

- add benchmark for Prophet ([#140](https://github.com/grafana/augurs/pull/140))

## [0.5.1](https://github.com/grafana/augurs/compare/augurs-prophet-v0.5.0...augurs-prophet-v0.5.1) - 2024-10-24

### Fixed

- do matrix multiplication for feature calculation properly

### Other

- define lints in Cargo.toml instead of each crate's lib.rs ([#138](https://github.com/grafana/augurs/pull/138))

## [0.5.0](https://github.com/grafana/augurs/compare/augurs-prophet-v0.5.0...augurs-prophet-v0.4.3) - 2024-10-18

No changes to the Rust crate; this version bump is due to breaking changes in the
Javascript package.

## [0.4.3](https://github.com/grafana/augurs/compare/augurs-prophet-v0.4.2...augurs-prophet-v0.4.3) - 2024-10-16

### Other

- fix docs.rs builds better this time

## [0.4.2](https://github.com/grafana/augurs/compare/augurs-prophet-v0.4.1...augurs-prophet-v0.4.2) - 2024-10-16

### Fixed

- *(docs)* handle build failures in docs.rs gracefully

### Other

- Fix sad clippy

## [0.4.1](https://github.com/grafana/augurs/compare/augurs-prophet-v0.4.0...augurs-prophet-v0.4.1) - 2024-10-16

### Fixed

- fix several issues with JS bindings ([#131](https://github.com/grafana/augurs/pull/131))

## [0.4.0](https://github.com/grafana/augurs/compare/augurs-prophet-v0.3.1...augurs-prophet-v0.4.0) - 2024-10-16

### Added

- add Prophet functionality to augurs-js ([#125](https://github.com/grafana/augurs/pull/125))
- capture stdout/stderr from cmdstan and emit tracing events ([#124](https://github.com/grafana/augurs/pull/124))
- add cmdstan-based optimizer for augurs-prophet ([#121](https://github.com/grafana/augurs/pull/121))

### Other

- Add test for parsing output of cmdstan
- Fix up build/.gitignore in augurs-prophet
- Improve augurs-prophet readme slightly
- Add Prophet algorithm in `augurs-prophet` crate ([#118](https://github.com/grafana/augurs/pull/118))
