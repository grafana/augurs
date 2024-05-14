#![doc = include_str!("../README.md")]
// Annoying, hopefully https://github.com/madonoharu/tsify/issues/42 will
// be resolved at some point.
#![allow(non_snake_case, clippy::empty_docs)]
#![warn(
    missing_docs,
    missing_debug_implementations,
    rust_2018_idioms,
    unreachable_pub
)]

use serde::Serialize;
use tsify::Tsify;
use wasm_bindgen::prelude::*;

mod changepoints;
pub mod ets;
pub mod mstl;
pub mod seasons;

/// Initialize the logger and panic hook.
///
/// This will be called automatically when the module is imported.
/// It sets the default tracing subscriber to `tracing-wasm`, and
/// sets WASM panics to print to the console with a helpful error
/// message.
#[wasm_bindgen(start)]
pub fn custom_init() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
    #[cfg(feature = "tracing-wasm")]
    tracing_wasm::try_set_as_global_default().ok();
}

// Wrapper types for the core types, so we can derive `Tsify` for them.
// This avoids having to worry about `tsify` in the `augurs-core` crate.

/// Forecast intervals.
#[derive(Clone, Debug, Serialize, Tsify)]
#[tsify(into_wasm_abi)]
pub struct ForecastIntervals {
    /// The confidence level for the intervals.
    pub level: f64,
    /// The lower prediction intervals.
    pub lower: Vec<f64>,
    /// The upper prediction intervals.
    pub upper: Vec<f64>,
}

impl From<augurs_core::ForecastIntervals> for ForecastIntervals {
    fn from(f: augurs_core::ForecastIntervals) -> Self {
        Self {
            level: f.level,
            lower: f.lower,
            upper: f.upper,
        }
    }
}

/// A forecast containing point forecasts and, optionally, prediction intervals.
#[derive(Clone, Debug, Serialize, Tsify)]
#[tsify(into_wasm_abi)]
pub struct Forecast {
    /// The point forecasts.
    pub point: Vec<f64>,
    /// The forecast intervals, if requested and supported
    /// by the trend model.
    pub intervals: Option<ForecastIntervals>,
}

impl From<augurs_core::Forecast> for Forecast {
    fn from(f: augurs_core::Forecast) -> Self {
        Self {
            point: f.point,
            intervals: f.intervals.map(Into::into),
        }
    }
}
