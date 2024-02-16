# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.1](https://github.com/grafana/augurs/compare/augurs-ets-v0.1.0...augurs-ets-v0.1.1) - 2024-02-15

### Other
- fix clippy lint for unneeded vec macro ([#53](https://github.com/grafana/augurs/pull/53))
- Add license files to repo root and symlinks in crate directories ([#43](https://github.com/grafana/augurs/pull/43))
- Add repository to sub-crate Cargo.tomls ([#42](https://github.com/grafana/augurs/pull/42))

## [0.1.0-alpha.0](https://github.com/grafana/augurs/releases/tag/augurs-ets-v0.1.0-alpha.0) - 2023-09-08

### Other
- Add workspace metadata and use in all the subpackages ([#33](https://github.com/grafana/augurs/pull/33))
- (cargo-release) version 0.1.0-alpha.1
- Use -alpha.0 suffix in crate versions
- Add some more comments and debug assertions
- Update some comments
- Bump all versions to latest ([#26](https://github.com/grafana/augurs/pull/26))
- Update itertools requirement from 0.10.5 to 0.11.0 ([#25](https://github.com/grafana/augurs/pull/25))
- Only bother calculating AMSE when it's required ([#24](https://github.com/grafana/augurs/pull/24))
- Make slight changes to inner loop to avoid as much unsafe indexing ([#23](https://github.com/grafana/augurs/pull/23))
- Add __repr__ for pyaugurs structs
- Add iai benchmarks for benchmarking in CI ([#9](https://github.com/grafana/augurs/pull/9))
- Use workspace dependencies for some more shared dependencies
- Don't hardcode 95% CI ppf in NaiveTrend prediction intervals
- Modify compute_sigma_h to be more functional
- Short circuit where horizon == 0 in predict methods
- Initial commit
