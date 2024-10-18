# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.5.0] - 2024-10-18

This is a large release with some breaking changes and enhancements. In particular, it
ensures the APIs are more ergonomic and the code is more idiomatic Javascript.
It also adds tests for the Javascript APIs.

## [0.1.0-alpha.0](https://github.com/grafana/augurs/releases/tag/augurs-js-v0.1.0-alpha.0) - 2023-09-08

### Other
- Add workspace metadata and use in all the subpackages ([#33](https://github.com/grafana/augurs/pull/33))
- (cargo-release) version 0.1.0-alpha.1
- Use -alpha.0 suffix in crate versions
- Bump all versions to latest ([#26](https://github.com/grafana/augurs/pull/26))
- Add explicit getrandom dep with js feature in augurs-js
- Accept &[f64] in MSTL::fit, rather than owned Vec ([#17](https://github.com/grafana/augurs/pull/17))
- Initial commit
