# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.5.1](https://github.com/grafana/augurs/compare/augurs-clustering-v0.5.0...augurs-clustering-v0.5.1) - 2024-10-24

### Other

- define lints in Cargo.toml instead of each crate's lib.rs ([#138](https://github.com/grafana/augurs/pull/138))

## [0.5.0](https://github.com/grafana/augurs/compare/augurs-clustering-v0.5.0...augurs-clustering-v0.4.3) - 2024-10-18

No changes to the Rust crate; this version bump is due to breaking changes in the
Javascript package.

## [0.4.0](https://github.com/grafana/augurs/compare/augurs-clustering-v0.3.1...augurs-clustering-v0.4.0) - 2024-10-16

### Added

- add 'augurs' convenience crate, re-exporting other crates ([#117](https://github.com/grafana/augurs/pull/117))
- add augurs-clustering crate with DBSCAN algorithm ([#100](https://github.com/grafana/augurs/pull/100))

### Other
- Add `augurs-clustering` crate
