#![doc = include_str!("../README.md")]
#![warn(
    missing_docs,
    missing_debug_implementations,
    rust_2018_idioms,
    unreachable_pub
)]

/// Forecast intervals.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ForecastIntervals {
    /// The confidence level for the intervals.
    pub level: f64,
    /// The lower prediction intervals.
    pub lower: Vec<f64>,
    /// The upper prediction intervals.
    pub upper: Vec<f64>,
}

impl ForecastIntervals {
    /// Return empty forecast intervals.
    pub fn empty(level: f64) -> ForecastIntervals {
        Self {
            level,
            lower: Vec::new(),
            upper: Vec::new(),
        }
    }
}

/// A forecast containing point forecasts and, optionally, prediction intervals.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Forecast {
    /// The point forecasts.
    pub point: Vec<f64>,
    /// The forecast intervals, if requested and supported
    /// by the trend model.
    pub intervals: Option<ForecastIntervals>,
}
