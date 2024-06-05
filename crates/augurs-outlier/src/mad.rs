// TODO: add MAD implementation.
#![allow(dead_code, unused_variables)]
use crate::OutlierDetector;

/// Scale factor k to approximate standard deviation of a Normal distribution.
// See https://en.wikipedia.org/wiki/Median_absolute_deviation.
const MAD_K: f64 = 1.4826;

pub struct MADDetector {}

impl OutlierDetector for MADDetector {
    type PreprocessedData = Vec<Vec<f64>>;
    fn preprocess(&self, y: &[&[f64]]) -> Self::PreprocessedData {
        todo!()
    }
    fn detect(&self, y: &Self::PreprocessedData) -> crate::OutlierResult {
        todo!()
    }
}
