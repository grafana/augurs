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

| Name                 | Purpose                                                              | Status                                                               |
| ---------------      | -------                                                              | ------                                                               |
| [`augurs-core`][]    | Common structs and traits                                            | alpha - API is flexible right now                                    |
| [`augurs-ets`][]     | Automatic exponential smoothing models                               | alpha - non-seasonal models working and tested against statsforecast |
| [`augurs-mstl`][]    | Multiple Seasonal Trend Decomposition using LOESS (MSTL)             | beta - working and tested against R                                  |
| [`augurs-testing`][] | Testing data and, eventually, evaluation harness for implementations | alpha - just data right now                                          |
| [`augurs-js`][]      | WASM bindings to augurs                                              | alpha - untested, should work though                                 |
| [`pyaugurs`][]       | Python bindings to augurs                                            | alpha - untested, should work though                                 |

## Releasing

Releases are made using `release-plz`: a PR should be automatically created for each release, and merging will perform the release and publish automatically.

## License

Dual-licensed to be compatible with the Rust project.
Licensed under the Apache License, Version 2.0 `<http://www.apache.org/licenses/LICENSE-2.0>` or the MIT license `<http://opensource.org/licenses/MIT>`, at your option.

[`augurs-core`]: https://crates.io/crates/augurs-core
[`augurs-ets`]: https://crates.io/crates/augurs-ets
[`augurs-mstl`]: https://crates.io/crates/augurs-mstl
[`augurs-js`]: https://crates.io/crates/augurs-js
[`augurs-testing`]: https://crates.io/crates/augurs-testing
[`pyaugurs`]: https://crates.io/crates/pyaugurs
