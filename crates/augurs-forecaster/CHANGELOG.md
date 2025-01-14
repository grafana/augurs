# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.9.0](https://github.com/grafana/augurs/compare/augurs-forecaster-v0.8.1...augurs-forecaster-v0.9.0) - 2025-01-14

### Added

- allow ignoring NaNs in power transforms (#234)
- *(forecaster)* add NaN handling to MinMaxScaler and StandardScaler (#227)

## [0.8.1](https://github.com/grafana/augurs/compare/augurs-forecaster-v0.8.0...augurs-forecaster-v0.8.1) - 2025-01-07

### Other

- update Cargo.toml dependencies

## [0.8.0](https://github.com/grafana/augurs/compare/augurs-forecaster-v0.7.0...augurs-forecaster-v0.8.0) - 2024-12-23

This release includes some major, breaking changes to the `augurs-forecaster` crate. See the [migration guide](https://docs.augu.rs/migrating.html#from-07-to-08) for more information on how to upgrade.

### Added

- [**breaking**] switch `transform` to a trait (#213)
- allow creating a Box-Cox or Yeo-Johnson transform with either lambda or data (#212)
- add standard scaler transform (#204)
- add 'transforms' JS crate and include in augurs JS bindings (#195)

### Fixed

- use box_cox instead of boxcox (#203)
- make Transform enum non-exhaustive (#194)

### Other

- restructure transforms into modules (#210)
- precalculate offset and scale factor for min-max scale transformer (#196)
- Add power transformation logic to forecaster transforms ([#185](https://github.com/grafana/augurs/pull/185))

## [0.7.0](https://github.com/grafana/augurs/compare/augurs-forecaster-v0.6.3...augurs-forecaster-v0.7.0) - 2024-11-25

This release includes some major, breaking changes to how holidays are handled in Prophet. See the [migration guide](https://docs.augu.rs/migrating.html#from-06-to-07) for more information on how to upgrade.

### Other

- update Cargo.toml dependencies

## [0.5.1](https://github.com/grafana/augurs/compare/augurs-forecaster-v0.5.0...augurs-forecaster-v0.5.1) - 2024-10-24

### Other

- define lints in Cargo.toml instead of each crate's lib.rs ([#138](https://github.com/grafana/augurs/pull/138))

## [0.5.0](https://github.com/grafana/augurs/compare/augurs-forecaster-v0.5.0...augurs-forecaster-v0.4.3) - 2024-10-18

No changes to the Rust crate; this version bump is due to breaking changes in the
Javascript package.

## [0.4.0](https://github.com/grafana/augurs/compare/augurs-forecaster-v0.3.1...augurs-forecaster-v0.4.0) - 2024-10-16

### Added

- add 'augurs' convenience crate, re-exporting other crates ([#117](https://github.com/grafana/augurs/pull/117))

### Fixed

- fix invalid lifetime warning on nightly ([#113](https://github.com/grafana/augurs/pull/113))

## [0.3.1](https://github.com/grafana/augurs/compare/augurs-forecaster-v0.3.0...augurs-forecaster-v0.3.1) - 2024-07-30

No notable changes in this release.

## [0.3.0](https://github.com/grafana/augurs/compare/augurs-forecaster-v0.2.0...augurs-forecaster-v0.3.0) - 2024-07-30

### Other
- Remove unsupported .github/workflows/bencher subdirectory and old benchmark workflow ([#90](https://github.com/grafana/augurs/pull/90))

## [0.2.0](https://github.com/grafana/augurs/compare/augurs-forecaster-v0.1.2...augurs-forecaster-v0.2.0) - 2024-06-05

### Added
- [**breaking**] add transformations and high-level forecasting API ([#65](https://github.com/grafana/augurs/pull/65))

### Other
- Add empty CHANGELOGs

