//! Javascript bindings for augurs seasonality detection.

use wasm_bindgen::prelude::*;

use augurs_seasons::{Detector, PeriodogramDetector};

/// Detect the seasonal periods in a time series.
#[wasm_bindgen]
pub fn seasonalities(y: &[f64]) -> Vec<usize> {
    PeriodogramDetector::builder().build(y).detect().collect()
}
