//! Testing utilities and data for the augurs time series framework.
//!
//! Eventually I'd like this to be a fully fledged testing harness to automatically
//! compare results between the augurs, Python and R implementations, but for now
//! it's just a place to put some data.
#![warn(
    missing_docs,
    missing_debug_implementations,
    rust_2018_idioms,
    unreachable_pub
)]

pub mod data;

pub use assert_approx_eq::assert_approx_eq;

/// Assert that two slices are approximately equal.
#[track_caller]
pub fn assert_all_close(actual: &[f64], expected: &[f64]) {
    for (actual, expected) in actual.iter().zip(expected) {
        if actual.is_nan() {
            assert!(expected.is_nan());
        } else {
            assert_approx_eq!(actual, expected, 1e-1);
        }
    }
}
