# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.5.0](https://github.com/grafana/augurs/compare/augurs-mstl-v0.5.0...augurs-mstl-v0.4.3) - 2024-10-18

No changes to the Rust crate; this version bump is due to breaking changes in the
Javascript package.

## [0.4.0](https://github.com/grafana/augurs/compare/augurs-mstl-v0.3.1...augurs-mstl-v0.4.0) - 2024-10-16

### Added

- add 'augurs' convenience crate, re-exporting other crates ([#117](https://github.com/grafana/augurs/pull/117))
- add `augurs-dtw` crate with dynamic time warping implementation ([#98](https://github.com/grafana/augurs/pull/98))

## [0.3.1](https://github.com/grafana/augurs/compare/augurs-mstl-v0.3.0...augurs-mstl-v0.3.1) - 2024-07-30

No notable changes in this release.

## [0.3.0](https://github.com/grafana/augurs/compare/augurs-mstl-v0.2.0...augurs-mstl-v0.3.0) - 2024-07-30

### Other
- Remove unsupported .github/workflows/bencher subdirectory and old benchmark workflow ([#90](https://github.com/grafana/augurs/pull/90))

## [0.2.0](https://github.com/grafana/augurs/compare/augurs-mstl-v0.1.2...augurs-mstl-v0.2.0) - 2024-06-05

### Added
- [**breaking**] add transformations and high-level forecasting API ([#65](https://github.com/grafana/augurs/pull/65))

### Other
- use clone_from instead of assigning result of clone ([#73](https://github.com/grafana/augurs/pull/73))

## [0.1.2](https://github.com/grafana/augurs/compare/augurs-mstl-v0.1.1...augurs-mstl-v0.1.2) - 2024-02-20

### Added
- *(mstl)* add interpolation iterator adaptor ([#63](https://github.com/grafana/augurs/pull/63))

## [0.1.1](https://github.com/grafana/augurs/compare/augurs-mstl-v0.1.0...augurs-mstl-v0.1.1) - 2024-02-15

### Other
- Add license files to repo root and symlinks in crate directories ([#43](https://github.com/grafana/augurs/pull/43))
- Add repository to sub-crate Cargo.tomls ([#42](https://github.com/grafana/augurs/pull/42))

## [0.1.0-alpha.0](https://github.com/grafana/augurs/releases/tag/augurs-mstl-v0.1.0-alpha.0) - 2023-09-08

### Other
- Add workspace metadata and use in all the subpackages ([#33](https://github.com/grafana/augurs/pull/33))
- (cargo-release) version 0.1.0-alpha.1
- Use -alpha.0 suffix in crate versions
- Update some comments
- Accept &[f64] in MSTL::fit, rather than owned Vec ([#17](https://github.com/grafana/augurs/pull/17))
- Add iai benchmarks for benchmarking in CI ([#9](https://github.com/grafana/augurs/pull/9))
- Use workspace dependencies for some more shared dependencies
- Don't hardcode 95% CI ppf in NaiveTrend prediction intervals
- Short circuit where horizon == 0 in predict methods
- Initial commit
