#![doc = include_str!("../README.md")]
#![warn(
    missing_docs,
    missing_debug_implementations,
    rust_2018_idioms,
    unreachable_pub
)]

#[cfg(feature = "changepoint")]
pub use augurs_changepoint as changepoint;
#[cfg(feature = "clustering")]
pub use augurs_clustering as clustering;
#[cfg(feature = "dtw")]
pub use augurs_dtw as dtw;
#[cfg(feature = "ets")]
pub use augurs_ets as ets;
#[cfg(feature = "forecaster")]
pub use augurs_forecaster as forecaster;
#[cfg(feature = "mstl")]
pub use augurs_mstl as mstl;
#[cfg(feature = "outlier")]
pub use augurs_outlier as outlier;
#[cfg(feature = "seasons")]
pub use augurs_seasons as seasons;

pub use augurs_core::{
    prelude, DistanceMatrix, Fit, Forecast, ForecastIntervals, ModelError, Predict,
};
