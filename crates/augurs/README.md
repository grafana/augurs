# augurs - a time series toolkit for Rust

[![Python](https://github.com/grafana/augurs/actions/workflows/python.yml/badge.svg)](https://github.com/grafana/augurs/actions/workflows/python.yml)
[![Rust](https://github.com/grafana/augurs/actions/workflows/rust.yml/badge.svg)](https://github.com/grafana/augurs/actions/workflows/rust.yml)
[![docs.rs](https://docs.rs/augurs-core/badge.svg)](https://docs.rs/augurs-core)
[![crates.io](https://img.shields.io/crates/v/augurs-core.svg)](https://crates.io/crates/augurs-core)

`augurs` is a time series toolkit built in Rust. It contains functionality for
outlier detection, clustering, seasonality detection, changepoint detection
and more.

If you're looking for the Python package, see [augurs on PyPI], which provides Python
bindings for `augurs`' functionality. Similarly, if you're looking for the
JavaScript package, see the [augurs npm package], which provides Javascript bindings
using WASM.

This crate can be used to access the functionality of all other crates in the
`augurs` ecosystem, as it re-exports them under a single namespace. The following
feature flags can be enabled to include only the functionality you need:

- `changepoint`: changepoint detection
- `clustering`: clustering algorithms
- `dtw`: dynamic time warping
- `ets`: exponential smoothing models
- `forecaster`: forecasting
- `mstl`: multiple seasonal trend decomposition
- `mstl-ets`: use ETS models as the trend model for MSTL
- `outlier`: outlier detection
- `seasons`: seasonality detection

Alternatively, use the `full` feature flag to enable all of the above.

## Getting started

First, add augurs to your project:

```sh
cargo add augurs --features full
```

Then import the pieces you need. For example, to use MSTL to forecast the next 10 values
of a time series using an ETS model for the trend component:

```rust
use augurs::{
    ets::AutoETS,
    mstl::MSTLModel,
    prelude::*,
};

let data = &[
    1.0, 1.2, 1.4, 1.5, 1.4, 1.4, 1.2,
    1.5, 1.6, 2.0, 1.9, 1.8, 1.9, 2.0,
];;

let periods = vec![7];  // Daily data with weekly seasonality.
let trend_model = AutoETS::non_seasonal().into_trend_model();  // Or, use a non-seasonal ETS model as the trend model.
let mstl = MSTLModel::new(periods, trend_model);
let fit = mstl.fit(data).unwrap();
let forecasts = fit.predict(10, 0.95).unwrap();
```

See the [examples](https://github.com/grafana/augurs/tree/main/crates/augurs/examples) for more usage examples.

## License

Dual-licensed to be compatible with the Rust project.
Licensed under the Apache License, Version 2.0 `<http://www.apache.org/licenses/LICENSE-2.0>` or the MIT license `<http://opensource.org/licenses/MIT>`, at your option.

[augurs on PyPI]: https://pypi.org/project/augurs/
[augurs npm package]: https://www.npmjs.com/package/@bsull/augurs
