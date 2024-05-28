// TODO: add MAD implementation.
#![allow(dead_code, unused_variables)]
use crate::OutlierDetector;

/// Scale factor k to approximate standard deviation of a Normal distribution.
// See https://en.wikipedia.org/wiki/Median_absolute_deviation.
const MAD_K: f64 = 1.4826;

pub struct MADDetector {}

impl OutlierDetector for MADDetector {
    fn detect(&self, y: &[&[f64]]) -> crate::OutlierResult {
        todo!()
    }
}
