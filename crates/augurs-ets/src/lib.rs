//! Exponential smoothing models.
//!
//! This crate provides exponential smoothing models for time series forecasting.
//! The models are implemented in Rust and are based on the [statsforecast][] Python package.
//!
//! **Important**: This crate is still in development and the API is subject to change.
//! Seasonal models are not yet implemented, and some model types have not been tested.
//!
//! # Example
//!
//! ```
//! use augurs_ets::AutoETS;
//!
//! let data: Vec<_> = (0..10).map(|x| x as f64).collect();
//! let mut search = AutoETS::new(1, "ZZN")
//!     .expect("ZZN is a valid model search specification string");
//! let model = search.fit(&data).expect("fit should succeed");
//! let forecast = model.predict(5, 0.95);
//! assert_eq!(forecast.point.len(), 5);
//! assert_eq!(forecast.point, vec![10.0, 11.0, 12.0, 13.0, 14.0]);
//! ```
//!
//! [statsforecast]: https://nixtla.github.io/statsforecast/models.html#autoets
#![warn(missing_docs)]

mod auto;
pub mod data;
mod ets;
pub mod model;
mod stat;
#[cfg(feature = "mstl")]
mod trend;

pub use auto::{AutoETS, AutoSpec};

#[cfg(test)]
// Assert that a is within (tol * 100)% of b.
#[macro_export]
macro_rules! assert_closeish {
    ($a:expr, $b:expr, $tol:expr) => {
        assert!(
            (($a - $b) / $a).abs() < $tol,
            "{} is not within {}% of {}",
            $a,
            $tol * 100.0,
            $b
        );
    };
}

/// Errors returned by this crate.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// An error occurred while parsing an error specification string.
    #[error("invalid error component string '{0}', must be one of 'A', 'M', 'Z'")]
    InvalidErrorComponentString(char),
    /// An error occurred while parsing a trend or seasonal specification string.
    #[error("invalid component string '{0}', must be one of 'N', 'A', 'M', 'Z'")]
    InvalidComponentString(char),
    /// An error occurred while parsing a model specification string.
    #[error("invalid model specification '{0}'")]
    InvalidModelSpec(String),

    /// The bounds of a parameter were inconsistent, i.e. the lower bound was
    /// greater than the upper bound.
    #[error("inconsistent parameter boundaries")]
    InconsistentBounds,
    /// One or more of the provided parameters was out of range.
    /// The definition of 'out of range' depends on the type of
    /// [`Bounds`][model::Bounds] used.
    #[error("parameters out of range")]
    ParamsOutOfRange,
    /// Not enough data was provided to fit a model.
    #[error("not enough data")]
    NotEnoughData,

    /// An error occurred solving a linear system while initializing state.
    #[error("least squares: {0}")]
    LeastSquares(&'static str),

    /// No suitable model was found.
    #[error("no model found")]
    NoModelFound,

    /// The model has not yet been fit.
    #[error("model not fit")]
    ModelNotFit,
}

type Result<T> = std::result::Result<T, Error>;

// Commented out because I haven't implemented seasonal models yet.
// fn fourier(y: &[f64], period: &[usize], K: &[usize]) -> DMatrix<f64> {
//     let times: Vec<_> = (1..y.len() + 1).collect();
//     let len_p = K.iter().fold(0, |sum, k| sum + k.min(&0));
//     let mut p = vec![f64::NAN; len_p];
//     let idx = 0;
//     for (j, p_) in period.iter().enumerate() {
//         let k = K[j];
//         if k > 0 {
//             for (i, x) in p[idx..(idx + k)].iter_mut().enumerate() {
//                 *x = i as f64 / *p_ as f64;
//             }
//         }
//     }
//     p.dedup();
//     // Determine columns where sinpi=0.
//     let k: Vec<bool> = zip(
//         p.iter().map(|x| 2.0 * x),
//         p.iter().map(|x| (2.0 * x).round()),
//     )
//     .map(|(a, b)| a - b > f64::MIN)
//     .collect();
//     let mut x = DMatrix::from_element(times.len(), 2 * p.len(), f64::NAN);
//     for (j, p_) in p.iter().enumerate() {
//         if k[j] {
//             x.column_mut(2 * j - 1)
//                 .iter_mut()
//                 .enumerate()
//                 .for_each(|(i, val)| {
//                     *val = (2.0 * p_ * (i + 1) as f64 * std::f64::consts::PI).sin()
//                 });
//         }
//         x.column_mut(2 * j)
//             .iter_mut()
//             .enumerate()
//             .for_each(|(i, val)| *val = (2.0 * p_ * (i + 1) as f64 * std::f64::consts::PI).cos());
//     }
//     let cols_to_delete: Vec<_> = x
//         .column_sum()
//         .iter()
//         .enumerate()
//         .filter_map(|(i, sum)| if sum.is_nan() { Some(i) } else { None })
//         .collect();
//     x.remove_columns_at(&cols_to_delete)
// }
