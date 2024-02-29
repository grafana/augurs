#![doc = include_str!("../README.md")]
#![warn(
    missing_docs,
    missing_debug_implementations,
    rust_2018_idioms,
    unreachable_pub
)]

/// Common traits and types for time series forecasting models.
pub mod prelude {
    pub use super::{Fit, Predict};
    pub use crate::forecast::{Forecast, ForecastIntervals};
}

mod forecast;
pub mod interpolate;
mod traits;

use std::convert::Infallible;

pub use forecast::{Forecast, ForecastIntervals};
pub use traits::{Fit, Predict};

/// An error produced by a time series forecasting model.
pub trait ModelError: std::error::Error + Sync + Send + 'static {}

impl std::error::Error for Box<dyn ModelError> {}
impl ModelError for Infallible {}
