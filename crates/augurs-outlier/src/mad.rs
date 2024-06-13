// TODO: add MAD implementation.
// #![allow(dead_code, unused_variables)]

use itertools::Itertools;
use roots::{find_root_brent, SimpleConvergency};
use rv::{
    dist::Beta,
    traits::{Cdf, ContinuousDistr},
};

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

#[derive(Debug)]
enum MADError {
    NoConvergence(roots::SearchError),
    InvalidParameters(rv::dist::BetaError),
    EmptyInput,
}

fn beta_hdi(alpha: f64, beta: f64, width: f64) -> Result<(f64, f64), MADError> {
    const EPS: f64 = 1e-9;
    if alpha < 1.0 + EPS && beta < 1.0 + EPS {
        // Degenerate case
        return Ok((0.0, 0.0));
    } else if alpha < 1.0 + EPS && beta > 1.0 {
        // Left border case
        return Ok((0.0, width));
    } else if alpha > 1.0 && beta < 1.0 + EPS {
        // Right border case
        return Ok((width, 0.0));
    }
    if width > 1.0 - EPS {
        return Ok((0.0, 1.0));
    }

    // Middle case
    let mode = (alpha - 1.0) / (alpha + beta - 2.0);
    let dist = Beta::new(alpha, beta).map_err(MADError::InvalidParameters)?;

    let lower = (mode - width).max(0.0);
    let upper = mode.min(1.0 - width);
    let mut convergency = SimpleConvergency {
        eps: f64::EPSILON,
        max_iter: 30,
    };
    let left = find_root_brent(
        lower,
        upper,
        |x| dist.pdf(&x) - dist.pdf(&(x + width)),
        &mut convergency,
    )
    .map_err(MADError::NoConvergence)?;
    let right = left + width;
    Ok((left, right))
}

fn thd_quantile(x: &[f64], q: f64, ignore_nan: bool, sort: bool) -> Result<f64, MADError> {
    let mut x = if ignore_nan {
        x.iter()
            .copied()
            .filter(|&v| !v.is_nan())
            .collect::<Vec<_>>()
    } else {
        x.to_vec()
    };
    if sort {
        x.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Greater));
    }

    let width = 1.0 / (x.len() as f64).sqrt();
    let n = x.len();
    if n == 0 {
        return Err(MADError::EmptyInput);
    } else if n == 1 {
        return Ok(x[0]);
    }

    let alpha = (n as f64 + 1.0) * q;
    let beta = (n as f64 + 1.0) * (1.0 - q);
    let dist = Beta::new(alpha, beta).map_err(MADError::InvalidParameters)?;
    let (hdi_left, hdi_right) = beta_hdi(alpha, beta, width)?;
    let hdi_cdf_left = dist.cdf(&hdi_left);
    let hdi_cdf_right = dist.cdf(&hdi_right);

    let left_index = (hdi_left * n as f64).floor() as usize;
    let right_index = (hdi_right * n as f64).ceil() as usize;
    let weights = (left_index..(right_index + 1))
        .map(|i| {
            let numerator = dist.cdf(&(i as f64 / n as f64)) - hdi_cdf_left;
            let denominator = hdi_cdf_right - hdi_cdf_left;
            (numerator / denominator).clamp(0.0, 1.0)
        })
        .tuple_windows()
        .map(|(a, b)| b - a);
    Ok(x[left_index..right_index]
        .iter()
        .zip(weights)
        .map(|(&x_i, w)| x_i * w)
        .sum())
}

fn thd_median(x: &[f64], ignore_nan: bool, sort: bool) -> Result<f64, MADError> {
    thd_quantile(x, 0.5, ignore_nan, sort)
}

fn thd_nanmedian(x: &[f64], sort: bool) -> Result<f64, MADError> {
    thd_quantile(x, 0.5, true, sort)
}

#[cfg(test)]
mod test {
    #[test]
    fn beta_hdi() {
        assert_eq!(
            super::beta_hdi(5.5, 5.5, 0.31622776601683794).unwrap(),
            (0.341886116991581, 0.658113883008419)
        );
    }

    struct TestCase {
        data: &'static [f64],
        expected: f64,
    }
    const TEST_CASES: &[TestCase] = &[
        TestCase {
            data: &[
                -0.565, -0.106, -0.095, 0.363, 0.404, 0.633, 1.371, 1.512, 2.018, 100_000.0,
            ],
            expected: 0.6268069427582939,
        },
        TestCase {
            data: &[-6.0, -5.0, -4.0, -16.0, -5.0, 15.0, -7.0, -8.0, -16.0],
            expected: -6.0,
        },
    ];
    const Q: f64 = 0.5;

    #[test]
    fn thd_quantile() {
        for tc in TEST_CASES {
            assert_eq!(
                super::thd_quantile(tc.data, Q, false, true).unwrap(),
                tc.expected
            );
        }
    }

    #[test]
    fn thd_median() {
        for tc in TEST_CASES {
            assert_eq!(
                super::thd_median(tc.data, false, true).unwrap(),
                tc.expected
            );
            assert_eq!(
                super::thd_quantile(tc.data, 0.5, false, true).unwrap(),
                tc.expected
            );
        }
        assert!(
            super::thd_median(&[f64::NAN, f64::NAN, f64::NAN, 1.0, 1.0], false, true)
                .unwrap()
                .is_nan()
        );
        assert!(super::thd_median(
            &[f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN],
            false,
            true
        )
        .unwrap()
        .is_nan());
    }

    #[test]
    fn thd_nanmedian() {
        for tc in TEST_CASES {
            assert_eq!(super::thd_nanmedian(tc.data, true).unwrap(), tc.expected);
            assert_eq!(
                super::thd_quantile(tc.data, 0.5, true, true).unwrap(),
                tc.expected
            );
        }
        assert_eq!(
            super::thd_nanmedian(&[f64::NAN, f64::NAN, f64::NAN, 1.0, 1.0], true).unwrap(),
            1.0
        );
        assert!(matches!(
            super::thd_nanmedian(&[f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN], true),
            Err(super::MADError::EmptyInput),
        ));
    }
}
