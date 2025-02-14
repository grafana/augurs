// TODO: add MAD implementation.
// #![allow(dead_code, unused_variables)]

use itertools::Itertools;
use roots::{find_root_brent, SimpleConvergency};
use rv::{
    dist::Beta,
    traits::{Cdf, ContinuousDistr},
};

use crate::{
    error::{DetectionError, PreprocessingError},
    Band, Error, OutlierDetector, OutlierOutput, Sensitivity, Series,
};

/// Scale factor k to approximate standard deviation of a Normal distribution.
// See https://en.wikipedia.org/wiki/Median_absolute_deviation.
const MAD_K: f64 = 1.4826;

#[derive(Debug, Clone, Copy)]
enum ThresholdOrSensitivity {
    /// A scale-invariant sensitivity parameter.
    ///
    /// This must be in (0, 1) and will be used to estimate a sensible
    /// threshold at detection-time.
    Sensitivity(Sensitivity),
    /// The threshold above which points are considered anomalous
    Threshold(f64),
}

impl ThresholdOrSensitivity {
    fn resolve_threshold(&self) -> f64 {
        match self {
            Self::Sensitivity(Sensitivity(sensitivity)) => {
                // Z-score at which individual datapoints are considered an outliers
                // higher sensitivity = lower threshold value (e.g. lower tolerance)
                const MAX_T: f64 = 7.941444487; // percentile = 0.9999999999999999
                const MIN_T: f64 = 0.841621234; // percentile = 0.80

                // use non-linear sensitivity scale, to be more sensitive at lower values
                // sensitivity = 0.5 -> threshold = 2.92 ~= percentile 0.998
                MAX_T - ((MAX_T - MIN_T) * sensitivity.sqrt())
            }
            Self::Threshold(threshold) => *threshold,
        }
    }
}

/// The precalculated medians to be used in MAD detection.
#[derive(Debug, Clone)]
pub struct Medians {
    lower: f64,
    global: f64,
    upper: f64,
}

/// A detector using the Median Absolute Deviation (MAD) to detect outliers.
#[derive(Debug, Clone)]
pub struct MADDetector {
    /// The maximum distance between points in a cluster.
    threshold_or_sensitivity: ThresholdOrSensitivity,

    /// The precalculated medians.
    ///
    /// If this is `None`, the medians will be calculated from the data.
    ///
    /// This can be provided to avoid recalculating the medians of the data,
    /// and to use a better estimate of the medians from a larger time range.
    medians: Option<Medians>,
}

impl MADDetector {
    /// Create a new MAD detector with the given threshold.
    pub fn with_threshold(threshold: f64) -> Self {
        Self {
            threshold_or_sensitivity: ThresholdOrSensitivity::Threshold(threshold),
            medians: None,
        }
    }

    /// Create a new MAD detector with the given sensitivity.
    ///
    /// At detection-time, a sensible value for `threshold` will be calculated
    /// using the scale of the data and the sensitivity value.
    pub fn with_sensitivity(sensitivity: f64) -> Result<Self, Error> {
        Ok(Self {
            threshold_or_sensitivity: ThresholdOrSensitivity::Sensitivity(sensitivity.try_into()?),
            medians: None,
        })
    }

    /// Set the precalculated medians.
    ///
    /// The medians can be calculated using [`MADDetector::calculate_double_medians`].
    pub fn set_medians(&mut self, medians: Medians) {
        self.medians = Some(medians);
    }

    /// Calculate the medians of the unprocessed data.
    ///
    /// This can be used to precalculate the medians of a larger set of data,
    /// for example.
    ///
    /// # Errors
    ///
    /// Returns an error if the data is empty, contains only NaNs, or if the
    /// lower or upper median is 0.0.
    pub fn calculate_double_medians(data: &[&[f64]]) -> Result<Medians, PreprocessingError> {
        let flattened = data
            .iter()
            .flat_map(|x| x.iter())
            .copied()
            .collect::<Vec<_>>();
        let global = thd_nanmedian(&flattened, true)?;
        let (mut lower_deviations, mut upper_deviations) = (Vec::new(), Vec::new());
        for row in data {
            for value in *row {
                if !value.is_finite() {
                    continue;
                }
                let deviation = value - global;
                match deviation {
                    // Explicitly handle the case where the global deviation is 0.0.
                    0.0 => {
                        upper_deviations.push(0.0);
                        lower_deviations.push(0.0);
                    }
                    _ if deviation > 0.0 => upper_deviations.push(deviation),
                    _ => lower_deviations.push(-deviation),
                }
            }
        }

        let lower = thd_median(&lower_deviations, false, true);
        let upper = thd_median(&upper_deviations, false, true);
        if let (Ok(lower), Ok(upper)) = (lower, upper) {
            if lower == 0.0 || upper == 0.0 {
                Err(PreprocessingError::from(MADError::DivideByZero))
            } else {
                Ok(Medians {
                    lower,
                    global,
                    upper,
                })
            }
        } else {
            Err(PreprocessingError::from(MADError::DivideByZero))
        }
    }

    fn calculate_mad(
        data: &[&[f64]],
        Medians {
            global,
            lower,
            upper,
        }: &Medians,
    ) -> Vec<Vec<f64>> {
        data.iter()
            .map(|row| {
                row.iter()
                    .map(|&value| {
                        if !value.is_finite() {
                            return f64::NAN;
                        }
                        let deviation = value - global;
                        let mut score = if deviation == 0.0 {
                            0.0
                        } else if deviation < 0.0 {
                            -deviation / (MAD_K * lower)
                        } else {
                            deviation / (MAD_K * upper)
                        };
                        if score.is_infinite() {
                            score = f64::NAN;
                        }
                        score
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
    }

    fn detect_impl(
        &self,
        y: &<Self as OutlierDetector>::PreprocessedData,
    ) -> Result<OutlierOutput, DetectionError> {
        let PreprocessedData {
            mad_scores,
            medians,
        } = y;
        let threshold = self.threshold_or_sensitivity.resolve_threshold();
        let upper_limit = medians.global + MAD_K * medians.upper * threshold;
        let lower_limit = medians.global - MAD_K * medians.lower * threshold;
        let n_series = mad_scores.len();
        let n_timestamps = mad_scores
            .first()
            .map(Vec::len)
            .ok_or(MADError::EmptyInput)?;

        // The normal band is constant across all timestamps.
        let normal_band = Some(Band {
            min: vec![lower_limit; n_timestamps],
            max: vec![upper_limit; n_timestamps],
        });

        // For each series, track the indices where it started/stopped being an outlier.
        let mut serieses = Series::preallocated(n_series, n_timestamps);
        for (series, scores) in serieses.iter_mut().zip(mad_scores.iter()) {
            series.scores.clone_from(scores);
            // Track whether the series is currently outlying.
            let mut current = false;
            for (i, score) in scores.iter().copied().enumerate() {
                if score > threshold {
                    series.is_outlier = true;
                    if !current {
                        series.outlier_intervals.add_start(i);
                    }
                    current = true;
                } else if current {
                    series.outlier_intervals.add_end(i);
                    current = false;
                }
            }
        }

        Ok(OutlierOutput::new(serieses, normal_band))
    }
}

/// The preprocessed data for the MAD detector.
///
/// This is produced by [`MADDetector::preprocess`] and consumed by
/// [`MADDetector::detect`].
#[derive(Debug, Clone)]
pub struct PreprocessedData {
    medians: Medians,
    mad_scores: Vec<Vec<f64>>,
}

impl OutlierDetector for MADDetector {
    type PreprocessedData = PreprocessedData;
    fn preprocess(y: &[&[f64]]) -> Result<Self::PreprocessedData, Error> {
        let medians = Self::calculate_double_medians(y)
            .map_err(|x| PreprocessingError::from(Box::new(x) as Box<dyn std::error::Error>))?;
        let mad_scores = Self::calculate_mad(y, &medians);
        Ok(PreprocessedData {
            medians,
            mad_scores,
        })
    }

    fn detect(&self, y: &Self::PreprocessedData) -> Result<OutlierOutput, Error> {
        Ok(self.detect_impl(y)?)
    }
}

#[derive(Debug, thiserror::Error)]
enum MADError {
    #[error("no convergence: {0}")]
    NoConvergence(roots::SearchError),
    #[error("invalid parameters: {0}")]
    InvalidParameters(rv::dist::BetaError),
    #[error("empty input")]
    EmptyInput,
    #[error("division by zero")]
    DivideByZero,
}

impl From<MADError> for PreprocessingError {
    fn from(e: MADError) -> Self {
        PreprocessingError::from(Box::new(e) as Box<dyn std::error::Error>)
    }
}

impl From<MADError> for DetectionError {
    fn from(e: MADError) -> Self {
        DetectionError::from(Box::new(e) as Box<dyn std::error::Error>)
    }
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
    use itertools::Itertools;
    use rv::prelude::*;

    use crate::{testing::flatten_intervals, MADDetector, OutlierDetector};

    use super::Medians;

    #[test]
    fn beta_hdi() {
        assert_eq!(
            super::beta_hdi(5.5, 5.5, 0.31622776601683794).unwrap(),
            (0.341886116991581, 0.658113883008419)
        );
    }

    struct THDTestCase {
        data: &'static [f64],
        expected: f64,
    }
    const THD_TEST_CASES: &[THDTestCase] = &[
        THDTestCase {
            data: &[
                -0.565, -0.106, -0.095, 0.363, 0.404, 0.633, 1.371, 1.512, 2.018, 100_000.0,
            ],
            expected: 0.6268069427582939,
        },
        THDTestCase {
            data: &[-6.0, -5.0, -4.0, -16.0, -5.0, 15.0, -7.0, -8.0, -16.0],
            expected: -6.0,
        },
    ];
    const Q: f64 = 0.5;

    #[test]
    fn thd_quantile() {
        for tc in THD_TEST_CASES {
            assert_eq!(
                super::thd_quantile(tc.data, Q, false, true).unwrap(),
                tc.expected
            );
        }
    }

    #[test]
    fn thd_median() {
        for tc in THD_TEST_CASES {
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
        for tc in THD_TEST_CASES {
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

    #[derive(Debug, Clone)]
    struct Expected {
        outliers: &'static [f64],
        intervals: &'static [usize],
    }

    #[derive(Debug, Clone)]
    struct MADTestCase<'a> {
        name: &'static str,
        data: &'a [f64],
        expected: Result<Expected, &'static str>,
        precalculated_medians: Option<Medians>,
    }

    const BASE_SAMPLE: &[f64] = &[
        -2002., -2001., -2000., 9., 47., 50., 71., 78., 79., 97., 98., 117., 123., 136., 138.,
        143., 145., 167., 185., 202., 216., 217., 229., 235., 242., 257., 297., 300., 315., 344.,
        347., 347., 360., 362., 368., 387., 400., 428., 455., 468., 484., 493., 523., 557., 574.,
        586., 605., 617., 618., 634., 641., 646., 649., 674., 678., 689., 699., 703., 709., 714.,
        740., 795., 798., 839., 880., 938., 941., 983., 1014., 1021., 1022., 1165., 1183., 1195.,
        1250., 1254., 1288., 1292., 1326., 1362., 1363., 1421., 1549., 1585., 1605., 1629., 1694.,
        1695., 1719., 1799., 1827., 1828., 1862., 1991., 2140., 2186., 2255., 2266., 2295., 2321.,
        2419., 2919., 3612., 6000., 6001., 6002.,
    ];

    fn gen_multi_modal_data() -> Vec<f64> {
        let mut rng = rand::thread_rng();
        let lower = rv::dist::Uniform::new_unchecked(0., 100.).sample(100, &mut rng);
        let upper = rv::dist::Uniform::new_unchecked(1000., 1100.).sample(100, &mut rng);
        lower.into_iter().interleave(upper).collect()
    }

    const MAD_TEST_CASES: &[MADTestCase<'_>] = &[
        MADTestCase {
            name: "normal",
            data: BASE_SAMPLE,
            expected: Ok(Expected {
                outliers: &[-2002., -2001., -2000., 6000., 6001., 6002.],
                intervals: &[0, 3, 103],
            }),
            precalculated_medians: None,
        },
        MADTestCase {
            name: "precalculated medians",
            data: BASE_SAMPLE,
            expected: Ok(Expected {
                outliers: &[-2002., -2001., -2000.],
                intervals: &[0, 3, 103],
            }),
            precalculated_medians: Some(Medians {
                lower: 378.,  // default: 378.8531095384087
                global: 663., // default: 663.0906684124299
                upper: 6000., // default: 706.54026614077
            }),
        },
        MADTestCase {
            name: "all same so no outliers [throws]",
            data: &[2., 2., 2., 2., 2., 2., 2., 2., 2.],
            expected: Err("division by zero"),
            precalculated_medians: None,
        },
        MADTestCase {
            data: &[-6., -5., -4., -16., -5., 15., -7., -8., -16.],
            name: "mixed positive and negative",
            expected: Ok(Expected {
                outliers: &[15.],
                intervals: &[5, 6],
            }),
            precalculated_medians: None,
        },
        MADTestCase {
            name: "zero majority [throws]",
            data: &[0., 0., 0., 0., 0., 0., 11., 12., 0., 11.],
            expected: Err("division by zero"),
            precalculated_medians: None,
        },
        MADTestCase {
            name: "only zero [throws]",
            data: &[0., 0., 0., 0., 0., 0., 0., 0., 0.],
            expected: Err("division by zero"),
            precalculated_medians: None,
        },
        MADTestCase {
            name: "the -2 likely outlying here",
            data: &[-2., f64::NAN, 21., 22., 23., f64::NAN, f64::NAN, 21., 24.],
            expected: Ok(Expected {
                outliers: &[-2.],
                intervals: &[0, 1],
            }),
            precalculated_medians: None,
        },
        MADTestCase {
            name: "mostly 3s mixed with nans, no outliers [throws]",
            data: &[3., f64::NAN, 3., 3., 3., f64::NAN, f64::NAN, 3., 4.],
            expected: Err("division by zero"),
            precalculated_medians: None,
        },
        MADTestCase {
            name: "just checking floats are ok",
            data: &[
                31.6, 33.12, 33.84, 38.234, 12.83, 15.23, 33.23, 32.85, 24.72,
            ],
            expected: Ok(Expected {
                outliers: &[38.234],
                intervals: &[3, 4],
            }),
            precalculated_medians: None,
        },
        MADTestCase {
            name: "all nans returns an error",
            data: &[
                f64::NAN,
                f64::NAN,
                f64::NAN,
                f64::NAN,
                f64::NAN,
                f64::NAN,
                f64::NAN,
                f64::NAN,
                f64::NAN,
            ],
            // Note: this differs from some implementations which would return
            // an empty anomaly list.
            expected: Err("empty input"),
            precalculated_medians: None,
        },
        MADTestCase {
            name: "single very large outlier",
            data: &[
                -0.565, -0.106, -0.095, 0.363, 0.404, 0.633, 1.371, 1.512, 2.018, 100_000.,
            ],
            expected: Ok(Expected {
                outliers: &[100_000.],
                intervals: &[9],
            }),
            precalculated_medians: None,
        },
        MADTestCase {
            name: "zero global median with outliers",
            data: &[
                -1000., -2., -1., -0.1, 0., 0., 0., 0., 0., 0.1, 1., 2., 1000.,
            ],
            expected: Ok(Expected {
                outliers: &[-1000., -2., -1., 1., 2., 1000.],
                intervals: &[0, 3, 10],
            }),
            precalculated_medians: None,
        },
    ];

    fn test_calculate_mad(tc: &MADTestCase<'_>) {
        let mad = MADDetector::with_sensitivity(0.5).unwrap();
        let result = tc
            .precalculated_medians
            .clone()
            .map(Ok)
            .unwrap_or_else(|| MADDetector::calculate_double_medians(&[tc.data]))
            .map(|medians| MADDetector::calculate_mad(&[tc.data], &medians));
        match &tc.expected {
            Ok(Expected { outliers, .. }) => {
                assert!(
                    result.is_ok(),
                    "case {} failed, got {}",
                    tc.name,
                    result.unwrap_err()
                );
                let scores = result.unwrap();
                let got_outliers = tc
                    .data
                    .iter()
                    .enumerate()
                    .filter_map(|(i, x)| if scores[0][i] > 3.0 { Some(x) } else { None })
                    .copied()
                    .collect_vec();
                assert_eq!(outliers, &got_outliers, "case {} failed", tc.name);
            }
            Err(exp) => {
                assert!(result.is_err(), "case {} failed", tc.name);
                assert_eq!(
                    &result.unwrap_err().to_string(),
                    exp,
                    "case {} failed",
                    tc.name
                );
            }
        }
    }

    #[test]
    fn calculate_mad() {
        for case in MAD_TEST_CASES {
            test_calculate_mad(case);
        }
    }

    #[test]
    fn calculate_mad_missing_one() {
        let mut data = BASE_SAMPLE.to_vec();
        data[0] = f64::NAN;
        let tc = MADTestCase {
            name: "missing one",
            data: &data,
            expected: Ok(Expected {
                outliers: &[-2001., -2000., 6000., 6001., 6002.],
                intervals: &[],
            }),
            precalculated_medians: None,
        };
        test_calculate_mad(&tc)
    }

    #[test]
    fn calculate_mad_multimodal() {
        let tc = MADTestCase {
            name: "multimodal data",
            data: &gen_multi_modal_data(),
            expected: Ok(Expected {
                outliers: &[],
                intervals: &[],
            }),
            precalculated_medians: None,
        };
        test_calculate_mad(&tc)
    }

    #[test]
    fn run() {
        for tc in MAD_TEST_CASES {
            let sensitivity = 0.5;
            let mad = MADDetector::with_sensitivity(sensitivity).unwrap();
            let result = MADDetector::preprocess(&[tc.data])
                .and_then(|preprocessed| mad.detect(&preprocessed));
            match &tc.expected {
                Ok(Expected { intervals, .. }) => {
                    assert!(
                        result.is_ok(),
                        "case {} failed, got {}",
                        tc.name,
                        result.unwrap_err()
                    );
                    let output = result.unwrap();
                    assert_eq!(output.series_results.len(), 1, "case {} failed", tc.name);
                    let got_intervals =
                        flatten_intervals(&output.series_results[0].outlier_intervals.intervals);
                    assert_eq!(intervals, &got_intervals, "case {} failed", tc.name);
                }
                Err(exp) => {
                    assert!(result.is_err(), "case {} failed", tc.name);
                    assert!(
                        &result.unwrap_err().to_string().contains(exp),
                        "case {} failed",
                        tc.name
                    );
                }
            }
        }
    }
}
