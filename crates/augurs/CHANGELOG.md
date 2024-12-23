# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.8.0](https://github.com/grafana/augurs/compare/augurs-v0.7.0...augurs-v0.8.0) - 2024-12-23

### Added

- [**breaking**] switch `transform` to a trait (#213)

## [0.6.0](https://github.com/grafana/augurs/compare/augurs-v0.5.4...augurs-v0.6.0) - 2024-11-08

### Added

- add wasmtime based optimizer for dependency-free Rust impl

## [0.5.2](https://github.com/grafana/augurs/compare/augurs-v0.5.1...augurs-v0.5.2) - 2024-10-25

### Other

- add benchmark for Prophet ([#140](https://github.com/grafana/augurs/pull/140))

## [0.5.1](https://github.com/grafana/augurs/compare/augurs-v0.5.0...augurs-v0.5.1) - 2024-10-24

### Other

- define lints in Cargo.toml instead of each crate's lib.rs ([#138](https://github.com/grafana/augurs/pull/138))

## [0.5.0](https://github.com/grafana/augurs/compare/augurs-v0.5.0...augurs-v0.4.3) - 2024-10-18

No changes to the Rust crate; this version bump is due to breaking changes in the
Javascript package.

## [0.4.3](https://github.com/grafana/augurs/compare/augurs-v0.4.1...augurs-v0.4.3) - 2024-10-16

### Fixed

- fixed docs.rs builds

## [0.4.1](https://github.com/grafana/augurs/compare/augurs-v0.4.0...augurs-v0.4.1) - 2024-10-16

### Fixed

- specify all-features=true for docs.rs

## [0.4.0](https://github.com/grafana/augurs/compare/augurs-v0.3.1...augurs-v0.4.0) - 2024-10-16

### Added

- add cmdstan-based optimizer for augurs-prophet ([#121](https://github.com/grafana/augurs/pull/121))
- add 'augurs' convenience crate, re-exporting other crates ([#117](https://github.com/grafana/augurs/pull/117))

### Other

- Add Prophet algorithm in `augurs-prophet` crate ([#118](https://github.com/grafana/augurs/pull/118))
