#![doc = include_str!("../README.md")]
#![warn(
    missing_docs,
    missing_debug_implementations,
    rust_2018_idioms,
    unreachable_pub
)]

#[doc(inline)]
#[cfg(feature = "changepoint")]
pub use augurs_changepoint as changepoint;
#[doc(inline)]
#[cfg(feature = "clustering")]
pub use augurs_clustering as clustering;
#[doc(inline)]
#[cfg(feature = "dtw")]
pub use augurs_dtw as dtw;
#[doc(inline)]
#[cfg(feature = "ets")]
pub use augurs_ets as ets;
#[doc(inline)]
#[cfg(feature = "forecaster")]
pub use augurs_forecaster as forecaster;
#[doc(inline)]
#[cfg(feature = "mstl")]
pub use augurs_mstl as mstl;
#[doc(inline)]
#[cfg(feature = "outlier")]
pub use augurs_outlier as outlier;
#[doc(inline)]
#[cfg(feature = "seasons")]
pub use augurs_seasons as seasons;

pub use augurs_core::{
    prelude, DistanceMatrix, Fit, Forecast, ForecastIntervals, ModelError, Predict,
};
