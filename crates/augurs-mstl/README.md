# Multiple Seasonal-Trend decomposition with LOESS (MSTL)

Fast, effective forecasting of time series exhibiting multiple seasonality and trend.

## Introduction

The MSTL algorithm, introduced in [this paper][mstl-paper],
provides a way of applying Seasonal-Trend decomposition to
multiple seasonalities. This allows effective modelling of
time series with multiple complex components.

As well as the MSTL algorithm this crate also provides the
[`MSTLModel`] struct, which is capable of running the MSTL
algorithm over some time series data, then modelling the
final trend component using a given trend forecaster.
It can then recombine the forecasted trend with the
decomposed seasonalities to generate in-sample and
out-of-sample forecasts.

The latter use case is the main entrypoint of this crate.

## Usage

```rust
use augurs_mstl::MSTLModel;

# fn main() -> Result<(), Box<dyn std::error::Error>> {
// Input data must be a `&[f64]` for the MSTL algorithm.
let y = &[1.5, 3.0, 2.5, 4.2, 2.7, 1.9, 1.0, 1.2, 0.8];

// Define the number of seasonal observations per period.
// In this example we have hourly data and both daily and
// weekly seasonality.
let periods = vec![3, 4];
// Create an MSTL model using a naive trend forecaster.
// Note: in real life you may want to use a different
// trend forecaster - see below.
let mstl = MSTLModel::naive(periods);
// Fit the model. Note this consumes `mstl` and returns
// a fitted version.
let fit = mstl.fit(y)?;

// Obtain in-sample and out-of-sample predictions, along
// with prediction intervals.
let in_sample = fit.predict_in_sample(0.95)?;
let out_of_sample = fit.predict(10, 0.95)?;
# Ok(())
# }
```

### Using alternative trend models

The `MSTLModel` is generic over the trend model used. As long
as the model passed implements the `TrendModel` trait from this
crate, it can be used to model the trend after decomposition.
For example, the `AutoETS` struct from the `ets` crate can be
used instead. First, add the `augurs_ets` crate to your `Cargo.toml`
with the `mstl` feature enabled:

```toml
[dependencies]
augurs_ets = { version = "*", features = ["mstl"] }
```

```rust,compile_fail
use augurs_ets::AutoETS;
use augurs_mstl::MSTLModel;

let y = vec![1.5, 3.0, 2.5, 4.2, 2.7, 1.9, 1.0, 1.2, 0.8];

let periods = vec![24, 24 * 7];
let trend_model = AutoETS::non_seasonal();
let mstl = MSTLModel::new(periods, trend_model);
let fit = mstl.fit(y)?;

let in_sample = fit.predict_in_sample(0.95)?;
let out_of_sample = fit.predict(10, 0.95)?;
```

(Note that the above example doesn't compile for this crate due to a circular
dependency, but would work in a separate crate!)

### Implementing a trend model

To use your own trend model, you'll need a struct that implements
the `TrendModel` trait. See below for an example of a model
that predicts a constant value for all time points in the horizon.

```rust
use std::borrow::Cow;

use augurs_core::{Forecast, ForecastIntervals};
use augurs_mstl::TrendModel;

#[derive(Debug)]
struct ConstantTrendModel {
    // The constant value to predict.
    constant: f64,
    // The number of values in the training data.
    y_len: usize,
}

impl TrendModel for ConstantTrendModel {
    fn name(&self) -> Cow<'_, str> {
        Cow::Borrowed("Constant")
    }

    fn fit(
        &mut self,
        y: &[f64],
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync + 'static>> {
        // Your algorithm should do whatever it needs to do to fit the model.
        // You have access to the data through the `y` slice, and a mutable
        // reference to `self` so you can store the results of the fitting
        // process.
        // Here we just store the number of elements in the training data.
        self.y_len = y.len();
        Ok(())
    }

    fn predict(
        &self,
        horizon: usize,
        level: Option<f64>,
    ) -> Result<Forecast, Box<dyn std::error::Error + Send + Sync + 'static>> {
        Ok(Forecast {
            point: vec![self.constant; horizon],
            intervals: level.map(|level| {
                let lower = vec![self.constant; horizon];
                let upper = vec![self.constant; horizon];
                ForecastIntervals {
                    level,
                    lower,
                    upper,
                }
            }),
        })
    }

    fn predict_in_sample(
        &self,
        level: Option<f64>,
    ) -> Result<Forecast, Box<dyn std::error::Error + Send + Sync + 'static>> {
        Ok(Forecast {
            point: vec![self.constant; self.y_len],
            intervals: level.map(|level| {
                let lower = vec![self.constant; self.y_len];
                let upper = vec![self.constant; self.y_len];
                ForecastIntervals {
                    level,
                    lower,
                    upper,
                }
            }),
        })
    }
}
```

## Credits

This implementation is based heavily on both the [R implementation][r-impl] and the [statsforecast implementation][statsforecast-impl].
It also makes heavy use of the [stlrs][] crate.

[r-impl]: https://pkg.robjhyndman.com/forecast/reference/mstl.html
[statsforecast-impl]: https://nixtla.github.io/statsforecast/models.html#mstl

## References

[Bandara, Kasun & Hyndman, Rob & Bergmeir, Christoph. (2021). “MSTL: A Seasonal-Trend Decomposition Algorithm for Time Series with Multiple Seasonal Patterns”.][mstl-paper]

[mstl-paper]: https://arxiv.org/abs/2107.13462

## License

Dual-licensed to be compatible with the Rust project.
Licensed under the Apache License, Version 2.0 `<http://www.apache.org/licenses/LICENSE-2.0>` or the MIT license `<http://opensource.org/licenses/MIT>`, at your option.
