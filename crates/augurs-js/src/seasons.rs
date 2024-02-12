//! Javascript bindings for augurs seasonality detection.

use serde::Deserialize;
use tsify::Tsify;
use wasm_bindgen::prelude::*;

use augurs_seasons::{Detector, PeriodogramDetector};

/// Options for detecting seasonal periods.
#[derive(Debug, Default, Deserialize, Tsify)]
#[tsify(from_wasm_abi)]
pub struct SeasonalityOptions {
    /// The minimum period to consider when detecting seasonal periods.
    ///
    /// The default is 4.
    pub min_period: Option<u32>,

    /// The maximum period to consider when detecting seasonal periods.
    ///
    /// The default is the length of the data divided by 3, or 512, whichever is smaller.
    pub max_period: Option<u32>,

    /// The threshold for detecting peaks in the periodogram.
    ///
    /// The value will be clamped to the range 0.01 to 0.99.
    ///
    /// The default is 0.9.
    pub threshold: Option<f64>,
}

impl From<SeasonalityOptions> for PeriodogramDetector {
    fn from(options: SeasonalityOptions) -> Self {
        let mut builder = PeriodogramDetector::builder();
        if let Some(min_period) = options.min_period {
            builder = builder.min_period(min_period);
        }
        if let Some(max_period) = options.max_period {
            builder = builder.max_period(max_period);
        }
        if let Some(threshold) = options.threshold {
            builder = builder.threshold(threshold);
        }
        builder.build()
    }
}

/// Detect the seasonal periods in a time series.
#[wasm_bindgen]
pub fn seasonalities(y: &[f64], options: Option<SeasonalityOptions>) -> Vec<u32> {
    PeriodogramDetector::from(options.unwrap_or_default()).detect(y)
}
