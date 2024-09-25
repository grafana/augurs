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
    assert_eq!(
        actual.len(),
        expected.len(),
        "slices have different lengths"
    );
    for (actual, expected) in actual.iter().zip(expected) {
        if actual.is_nan() {
            assert!(expected.is_nan());
        } else {
            assert_approx_eq!(actual, expected, 1e-1);
        }
    }
}

/// Assert that a is within (tol * 100)% of b.
#[macro_export]
macro_rules! assert_within_pct {
    ($a:expr, $b:expr, $tol:expr) => {
        if $a == 0.0 {
            assert!(
                ($b as f64).abs() < $tol,
                "{} is not within {}% of 0",
                $b,
                $tol * 100.0
            );
        } else {
            assert!(
                (($a - $b) / $a).abs() < $tol,
                "{} is not within {}% of {}",
                $a,
                $tol * 100.0,
                $b
            );
        }
    };
}
