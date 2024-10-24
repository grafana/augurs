# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
