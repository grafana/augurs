# augurs - a time series framework for Rust

[![Python](https://github.com/grafana/augurs/actions/workflows/python.yml/badge.svg)](https://github.com/grafana/augurs/actions/workflows/python.yml)
[![Rust](https://github.com/grafana/augurs/actions/workflows/rust.yml/badge.svg)](https://github.com/grafana/augurs/actions/workflows/rust.yml)
[![docs.rs](https://docs.rs/augurs-core/badge.svg)](https://docs.rs/augurs-core)
[![crates.io](https://img.shields.io/crates/v/augurs-core.svg)](https://crates.io/crates/augurs-core)

This repository contains `augurs`, a time series framework built in Rust.
It aims to provide some useful primitives for working with time series,
as well as the main functionality: heavily optimized forecasting models
based on existing R or Python implementations.

As well as the core Rust library, augurs will provide bindings to other
languages such as Python and Javascript (via WASM).

*Status*: please note that this repository is very much in progress.
APIs are subject to change, and functionality may not be fully implemented.

## Crate descriptions

| Name                     | Purpose                                                              | Status                                                               |
| ------------------------ | -------------------------------------------------------------------- | -------------------------------------------------------------------- |
| [`augurs-changepoint`][] | Changepoint detection for time series                                | alpha - API is flexible right now                                    |
| [`augurs-clustering`][]  | Time series clustering algorithms
| [`augurs-core`][]        | Common structs and traits                                            | alpha - API is flexible right now                                    |
| [`augurs-dtw`][]         | Dynamic Time Warping (DTW)                                        | alpha - API is flexible right now                                    |
| [`augurs-ets`][]         | Automatic exponential smoothing models                               | alpha - non-seasonal models working and tested against statsforecast |
| [`augurs-mstl`][]        | Multiple Seasonal Trend Decomposition using LOESS (MSTL)             | beta - working and tested against R                                  |
| [`augurs-outlier`][]     | Outlier detection for time series                                    | alpha - API is flexible right now                                    |
| [`augurs-seasons`][]     | Seasonality detection using periodograms                             | alpha - working and tested against Python in limited scenarios       |
| [`augurs-testing`][]     | Testing data and, eventually, evaluation harness for implementations | alpha - just data right now                                          |
| [`augurs-js`][]          | WASM bindings to augurs                                              | alpha - untested, should work though                                 |
| [`pyaugurs`][]           | Python bindings to augurs                                            | alpha - untested, should work though                                 |

## Releasing

Releases are made using `release-plz`: a PR should be automatically created for each release, and merging will perform the release and publish automatically.

### Releasing the `augurs` Python library

The first exception to the `release-plz` flow is the `augurs` Python library, which is only released when a new tag beginning with `pyaugurs` is pushed. This must be done manually for now (ideally soon after the `release-plz` PR is merged).

E.g.:

```
git tag pyaugurs-v0.3.0 -m "Release pyaugurs v0.3.0"
git push --tags
```

### Releasing the `augurs` npm library

The `augurs` npm library must also be published manually. This can be done using `just publish-npm`; note you'll need to login with npm first.

```
npm login
# Log in online, etc...
just publish-npm
```

## License

Dual-licensed to be compatible with the Rust project.
Licensed under the Apache License, Version 2.0 `<http://www.apache.org/licenses/LICENSE-2.0>` or the MIT license `<http://opensource.org/licenses/MIT>`, at your option.

[`augurs-changepoint`]: https://crates.io/crates/augurs-changepoint
[`augurs-clustering`]: https://crates.io/crates/augurs-clustering
[`augurs-core`]: https://crates.io/crates/augurs-core
[`augurs-dtw`]: https://crates.io/crates/augurs-dtw
[`augurs-ets`]: https://crates.io/crates/augurs-ets
[`augurs-mstl`]: https://crates.io/crates/augurs-mstl
[`augurs-js`]: https://crates.io/crates/augurs-js
[`augurs-outlier`]: https://crates.io/crates/augurs-outlier
[`augurs-seasons`]: https://crates.io/crates/augurs-seasons
[`augurs-testing`]: https://crates.io/crates/augurs-testing
[`pyaugurs`]: https://crates.io/crates/pyaugurs
