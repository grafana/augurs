# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
