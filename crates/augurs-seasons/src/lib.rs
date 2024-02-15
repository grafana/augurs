#![doc = include_str!("../README.md")]
#![warn(
    missing_docs,
    missing_debug_implementations,
    rust_2018_idioms,
    unreachable_pub
)]

mod periodogram;
#[cfg(test)]
mod test_data;

pub use periodogram::{
    Builder as PeriodogramDetectorBuilder, Detector as PeriodogramDetector, Periodogram,
};

/// A detector of periodic signals in a time series.
pub trait Detector {
    /// Detects the periods of a time series.
    fn detect(&self, data: &[f64]) -> Vec<u32>;
}
