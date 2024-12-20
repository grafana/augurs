#![doc = include_str!("../README.md")]

/// Common traits and types for time series forecasting models.
pub mod prelude {
    pub use super::{Fit, Predict};
    pub use crate::forecast::{Forecast, ForecastIntervals};
}

mod distance;
mod forecast;
mod traits;

use std::convert::Infallible;

pub use distance::DistanceMatrix;
pub use forecast::{Forecast, ForecastIntervals};
pub use traits::{Fit, Predict};

/// An error produced by a time series forecasting model.
pub trait ModelError: std::error::Error + Sync + Send + 'static {}

impl std::error::Error for Box<dyn ModelError> {}
impl ModelError for Infallible {}
