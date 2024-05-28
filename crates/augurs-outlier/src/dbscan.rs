use std::iter;

use tinyvec::TinyVec;
use tracing::instrument;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::{Band, OutlierDetector, OutlierResult, Sensitivity, SensitivityError, Series};

/// The epsilon or sensitivity parameter for the DBSCAN algorithm.
#[derive(Debug, Clone)]
enum EpsilonOrSensitivity {
    /// A scale-invariant sensitivity parameter.
    ///
    /// This must be in (0, 1) and will be used to estimate a sensible
    /// value of epsilon based on the data at detection-time.
    Sensitivity(Sensitivity),
    /// The maximum distance between points in a cluster.
    Epsilon(f64),
}

impl EpsilonOrSensitivity {
    fn resolve_epsilon(&self, data: &Data) -> f64 {
        match self {
            Self::Sensitivity(Sensitivity(sensitivity)) => {
                const SENSITIVITY_PADDING: f64 = 1.1;
                const MIN_SENSITIVITY: f64 = 1.1754943508222875e-38;
                let data_span = data.span();
                // trim sensitivity to avoid epsilon being 0
                // why 1e-3? any lower (e.g. 1e-6), the epsilon is so small that everything is an outlier
                let trimmed = sensitivity.min(1.0 - 1e-3);
                ((1.0 - trimmed) * data_span * SENSITIVITY_PADDING).max(MIN_SENSITIVITY)
            }
            Self::Epsilon(epsilon) => epsilon.max(f64::MIN_POSITIVE),
        }
    }
}

/// A detector for outliers using a 1 dimensional DBSCAN algorithm.
///
/// It detects outliers for each timestamp by sorting the values at that timestamp, and
/// coming up with a cluster of values that are close to each other (using the one
/// parameter, `epsilon`, to determine closeness). If the cluster is at least half
/// the size of the total number of values, then the cluster is considered
/// normal, and the rest are outliers.
#[derive(Debug, Clone)]
pub struct DBSCANDetector {
    /// The maximum distance between points in a cluster.
    epsilon_or_sensitivity: EpsilonOrSensitivity,

    parallelize: bool,
}

impl OutlierDetector for DBSCANDetector {
    fn detect(&self, y: &[&[f64]]) -> OutlierResult {
        let data = Data::from_row_major(y);
        self.run(data)
    }

    fn detect_transposed(&self, y: &[&[f64]]) -> OutlierResult {
        let data = Data::from_column_major(y);
        self.run(data)
    }
}

impl DBSCANDetector {
    /// Create a new DBSCAN detector with the given epsilon.
    pub fn with_epsilon(epsilon: f64) -> Self {
        Self {
            epsilon_or_sensitivity: EpsilonOrSensitivity::Epsilon(epsilon),
            parallelize: false,
        }
    }

    /// Create a new DBSCAN detector with the given sensitivity.
    ///
    /// At detection-time, a sensible value for `epsilon` will be calculated
    /// using the scale of the data and the sensitivity value.
    pub fn with_sensitivity(sensitivity: f64) -> Result<Self, SensitivityError> {
        Ok(Self {
            epsilon_or_sensitivity: EpsilonOrSensitivity::Sensitivity(Sensitivity::try_from(
                sensitivity,
            )?),
            parallelize: false,
        })
    }

    /// Parallelize the DBSCAN algorithm.
    pub fn parallelize(&mut self, parallelize: bool) {
        self.parallelize = parallelize;
    }

    fn run(&self, data: Data) -> OutlierResult {
        let epsilon = self.epsilon_or_sensitivity.resolve_epsilon(&data);
        let n_timestamps = data.sorted.len();
        let mut serieses: Vec<_> = iter::repeat_with(|| {
            let mut s = Series::with_capacity(n_timestamps);
            s.scores.resize(n_timestamps, 0.0);
            s
        })
        .take(data.sorted[0].indices.len())
        .collect();
        let mut normal_band = Band::new(n_timestamps);

        // TODO: come up with an educated guess for capacity.
        let mut outliers_so_far = TinyVec::with_capacity(data.sorted.len());

        // Run DBSCANs in parallel using Rayon if specified.
        #[cfg(feature = "parallel")]
        let dbscans: Vec<_> = if self.parallelize {
            data.sorted
                .par_iter()
                .map(|ts_data| Self::dbscan_1d(ts_data, epsilon))
                .collect()
        } else {
            data.sorted
                .iter()
                .map(|ts_data| Self::dbscan_1d(ts_data, epsilon))
                .collect()
        };
        #[cfg(not(feature = "parallel"))]
        let dbscans: Vec<_> = data
            .sorted
            .iter()
            .map(|ts_data| Self::dbscan_1d(ts_data, epsilon))
            .collect();

        for (i, dbscan) in dbscans.into_iter().enumerate() {
            let DBScan1DResults {
                cluster_min,
                cluster_max,
                outlier_indices,
            } = dbscan;

            // Construct the normal band, if found.
            if let Some((min, max)) = cluster_min.zip(cluster_max) {
                normal_band.min[i] = min - epsilon / 2.0;
                normal_band.max[i] = max + epsilon / 2.0;
            }

            // Mark the outlier series and fill in any positive scores.
            outlier_indices.iter().for_each(|Index(idx)| {
                let series = &mut serieses[*idx];
                series.is_outlier = true;
                series.scores[i] = 1.0;
            });

            // For each series that has outliers, find the intervals where they are outliers.
            if !outlier_indices.is_empty() {
                // Compare with outliers so far.

                // What was in previous outliers, but now not
                let stopped_being_outlier = outliers_so_far
                    .iter()
                    .filter(|x| !outlier_indices.contains(x));
                for stopped_index in stopped_being_outlier {
                    serieses[stopped_index.into_inner()]
                        .outlier_intervals
                        .add_end(i);
                }

                // What has started being outlier
                let started_being_outlier = outlier_indices
                    .iter()
                    .filter(|x| !outliers_so_far.contains(x));
                for started_index in started_being_outlier {
                    serieses[started_index.into_inner()]
                        .outlier_intervals
                        .add_start(i);
                }

                outliers_so_far = outlier_indices;
            } else {
                // all series considered normal at this timestamp, so take all outliers_so_far entries,
                // mark them as stopped and empty it
                for stopped_index in &outliers_so_far {
                    serieses[stopped_index.into_inner()]
                        .outlier_intervals
                        .add_end(i);
                }
                outliers_so_far.clear();
            }
        }
        OutlierResult::new(serieses, normal_band)
    }

    // Following impl inspired by https://github.com/d-chambers/dbscan1d
    //
    // Main idea: as the array is sorted, compare the distance between each neighbour. If
    // distance less than epsilon, is candidate for cluster. Try next neighbour to see if
    // it close enough to join cluster. If cluster size grows to half of all points, that
    // is the main/final cluster. If not, the points are outliers.
    fn dbscan_1d(sorted: &SortedData, eps: f64) -> DBScan1DResults {
        let SortedData {
            sorted,
            indices: sort_indices,
        } = sorted;
        // if <=2 series, can return quickly as no anomaly
        if sorted.len() <= 2 {
            return DBScan1DResults {
                cluster_min: None,
                cluster_max: None,
                outlier_indices: TinyVec::new(),
            };
        }

        // Below DBSCAN impl relies on the fact we mandate the cluster contains at least 50% of all values
        let min_cluster_size = sorted.len() / 2 + 1;

        let mut this_cluster_bottom = None;
        let mut this_cluster_top;
        let mut this_cluster_size = 1;
        let mut in_cluster = false;
        let (mut cluster_min, mut cluster_max) = (None, None);
        let mut outlier_indices = TinyVec::with_capacity(sorted.len());

        // Ideally we'd use `array_windows` here but it's still unstable so
        // we're stuck with `windows`. This means we need to manually add
        // some assertions to help the compiler to optimize the code,
        // and there's an unfortunate `unreachable!` call which should
        // never be hit.
        for window in sorted.windows(2) {
            assert_eq!(window.len(), 2);
            let &[a, b] = &window else { unreachable!() };
            if (b - a).abs() <= eps {
                if !in_cluster {
                    this_cluster_bottom = Some(*a);
                }
                in_cluster = true;
                this_cluster_top = *b;
                this_cluster_size += 1;

                if this_cluster_size >= min_cluster_size {
                    cluster_min = this_cluster_bottom;
                    cluster_max = Some(this_cluster_top);
                }
            } else {
                this_cluster_size = 1;
                in_cluster = false;
            }
        }

        if let Some((cluster_bottom, cluster_top)) = cluster_min.zip(cluster_max) {
            for (i, val) in sorted.iter().enumerate() {
                let original_index = sort_indices[i];
                if *val < cluster_bottom || *val > cluster_top {
                    outlier_indices.push(original_index);
                }
            }
        } else {
            // everything is an outlier
            outlier_indices = TinyVec::from(sort_indices.as_slice());
        }

        DBScan1DResults {
            cluster_min,
            cluster_max,
            outlier_indices,
        }
    }
}

pub struct DBScan1DResults {
    cluster_min: Option<f64>,
    cluster_max: Option<f64>,
    outlier_indices: TinyVec<[Index; 24]>,
}

/// Newtype wrapper to ensure that we use the correct type when converting from
/// sorted data to original indexes.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
#[repr(transparent)]
struct Index(usize);

impl Index {
    fn into_inner(self) -> usize {
        self.0
    }
}

/// Preprocessed data for the DBSCAN algorithm.
#[derive(Debug)]
struct Data {
    sorted: Vec<SortedData>,
}

impl Data {
    /// Create a `Data` struct from row-major data.
    #[instrument(skip(data))]
    fn from_row_major(data: &[&[f64]]) -> Self {
        let n_columns = data.len();
        let n_timestamps = data[0].len();
        // First transpose the data.
        let mut transposed = vec![vec![f64::NAN; n_columns]; n_timestamps];
        data.iter().enumerate().for_each(|(i, chunk)| {
            chunk.iter().enumerate().for_each(|(j, value)| {
                transposed[j][i] = *value;
            })
        });
        // Check that the transposition worked.
        debug_assert_eq!(transposed.len(), n_timestamps);
        #[cfg(debug_assertions)]
        transposed.iter().for_each(|row| {
            debug_assert_eq!(row.len(), n_columns);
        });
        transposed.iter().for_each(|row| {
            debug_assert_eq!(row.len(), n_columns);
        });
        // Then sort values at each timestamp.
        let sorted = transposed.into_iter().map(SortedData::new).collect();
        Self { sorted }
    }

    /// Create a `Data` struct from column-major data.
    fn from_column_major(data: &[&[f64]]) -> Self {
        let sorted = data.iter().map(|x| SortedData::new(x.to_vec())).collect();
        Self { sorted }
    }

    /// Calculate the span of the data: the difference between the highest and lowest values.
    fn span(&self) -> f64 {
        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;
        for ts in &self.sorted {
            if let Some(low) = ts.sorted.first() {
                min = min.min(*low);
            }
            if let Some(high) = ts.sorted.last() {
                max = max.max(*high);
            }
        }
        (max - min).abs().max(0.1)
    }
}

/// A sorted list of values at a single timestamp, with the original indices of the values.
#[derive(Debug)]
struct SortedData {
    /// The values at the timestamp, sorted.
    sorted: Vec<f64>,
    /// The original indices of the sorted values.
    indices: Vec<Index>,
}

impl SortedData {
    #[instrument(skip(vals))]
    fn new(vals: Vec<f64>) -> Self {
        let mut vals_with_idx: Vec<_> = vals
            .iter()
            .enumerate()
            .filter_map(|(i, val)| (!val.is_nan()).then_some((Index(i), val)))
            .collect();
        vals_with_idx.sort_unstable_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap());
        let (indices, sorted) = vals_with_idx.into_iter().unzip();
        Self { sorted, indices }
    }
}

#[cfg(test)]
mod tests {
    use crate::{OutlierDetector, OutlierResult};

    use super::DBSCANDetector;

    const UNDEFINED: f64 = f64::NAN;
    // Transposed dataset for testing DBSCAN.
    // There are 13 timestamps and 9 series: each inner
    // array contains the values for the all series at that timestamp.
    pub static DBSCAN_DATASET: &[&[f64]] = &[
        // all in cluster if eps<=1
        &[1., 2., 3., 4., 5., 6., 7., 8., 9.],
        // all anomalous unless eps>=3
        &[0., 3., 7., 11., 17., 24., 33., 40., 51.],
        // all same so all in cluster
        &[2., 2., 2., 2., 2., 2., 2., 2., 2.],
        // cluster of size 6 if eps <= 2
        &[0., 1., 3., 4., 5., 6., 9., 10., 15.],
        // cluster of size 6 again if eps>=1., just ensuring sign & order are irrelevant
        &[-6., -5., -4., -16., -5., 15., -7., -8., -16.],
        // 2 equally sized clusters of size 4, neither large enough to count, all anomalous
        &[1., 2., 3., 4., 8., 12., 13., 14., 15.],
        // the -2 likely outlying here
        &[
            -2., UNDEFINED, 21., 22., 23., UNDEFINED, UNDEFINED, 21., 24.,
        ],
        // cluster of 3s most likely
        &[3., UNDEFINED, 3., 3., 3., UNDEFINED, UNDEFINED, 3., 4.],
        // just checking floats are ok
        &[
            31.6, 33.12, 33.84, 38.234, 12.83, 15.23, 33.23, 32.85, 24.72,
        ],
        // nans are always non-anomalous
        &[
            UNDEFINED, UNDEFINED, UNDEFINED, UNDEFINED, UNDEFINED, UNDEFINED, UNDEFINED, UNDEFINED,
            UNDEFINED,
        ],
        // cluster of 3s valid as only 4 valid points here, but the 1 will be anomalous if eps < 2
        &[
            3., UNDEFINED, 3., 1., 3., UNDEFINED, UNDEFINED, UNDEFINED, UNDEFINED,
        ],
        // cluster only appears if eps>=3
        &[1., 4., 7., 10., 13., 16., 19., 22., 25.],
        // only anomalies appear if epsilon around 0.1
        &[
            1.41, 1.103, UNDEFINED, 1.037, 1.44, 0.892, 1.233, 1.092, 1.185,
        ],
    ];

    struct TestCase {
        eps: f64,
        expected: [&'static [bool]; 13],
    }

    const ALL_FALSE: &[bool] = &[false; 9];
    const ALL_TRUE: &[bool] = &[true; 9];

    static CASES: &[TestCase] = &[
        TestCase {
            eps: 1.0,
            expected: [
                ALL_FALSE,
                ALL_TRUE,
                ALL_FALSE,
                ALL_TRUE,
                &[false, false, false, true, false, true, false, false, true],
                ALL_TRUE,
                &[true, false, false, false, false, false, false, false, false],
                ALL_FALSE,
                ALL_TRUE,
                ALL_FALSE,
                &[false, false, false, true, false, false, false, false, false],
                ALL_TRUE,
                ALL_FALSE,
            ],
        },
        TestCase {
            eps: 2.,
            expected: [
                ALL_FALSE,
                ALL_TRUE,
                ALL_FALSE,
                &[false, false, false, false, false, false, true, true, true],
                &[false, false, false, true, false, true, false, false, true],
                ALL_TRUE,
                &[true, false, false, false, false, false, false, false, false],
                ALL_FALSE,
                &[false, false, false, true, true, true, false, false, true],
                ALL_FALSE,
                ALL_FALSE,
                ALL_TRUE,
                ALL_FALSE,
            ],
        },
        TestCase {
            eps: 0.5,
            expected: [
                ALL_TRUE,
                ALL_TRUE,
                ALL_FALSE,
                ALL_TRUE,
                ALL_TRUE,
                ALL_TRUE,
                &[true, false, true, true, true, false, false, true, true],
                &[false, false, false, false, false, false, false, false, true],
                ALL_TRUE,
                ALL_FALSE,
                &[false, false, false, true, false, false, false, false, false],
                ALL_TRUE,
                ALL_FALSE,
            ],
        },
        TestCase {
            eps: 3.0,
            expected: [
                ALL_FALSE,
                ALL_TRUE,
                ALL_FALSE,
                &[false, false, false, false, false, false, false, false, true],
                &[false, false, false, true, false, true, false, false, true],
                ALL_TRUE,
                &[true, false, false, false, false, false, false, false, false],
                ALL_FALSE,
                &[false, false, false, true, true, true, false, false, true],
                ALL_FALSE,
                ALL_FALSE,
                ALL_FALSE,
                ALL_FALSE,
            ],
        },
        TestCase {
            eps: 0.1,
            expected: [
                ALL_TRUE,
                ALL_TRUE,
                ALL_FALSE,
                ALL_TRUE,
                ALL_TRUE,
                ALL_TRUE,
                &[true, false, true, true, true, false, false, true, true],
                &[false, false, false, false, false, false, false, false, true],
                ALL_TRUE,
                ALL_FALSE,
                &[false, false, false, true, false, false, false, false, false],
                ALL_TRUE,
                &[true, false, false, false, true, true, false, false, false],
            ],
        },
    ];

    fn outlier_intervals_to_boolean_table(results: &OutlierResult) -> Vec<Vec<bool>> {
        // Start by prepopulating a [n_timestamps x n_series] matrix of false values.
        let series_count = results.series_results.len();
        let timestamp_count = DBSCAN_DATASET.len();
        let mut matrix = vec![vec![false; series_count]; timestamp_count];

        // For each series, iterate over the outlier intervals, marking the points in each intervals as outliers
        // in the matrix.
        for (j, series) in results.series_results.iter().enumerate() {
            let mut outlier_state = false;
            let mut outlier_indices = series.outlier_intervals.indices.iter();
            let mut next_idx = outlier_indices.next();
            for (i, item) in matrix.iter_mut().enumerate() {
                if next_idx.map_or(false, |next_idx| i >= *next_idx) {
                    outlier_state = !outlier_state;
                    next_idx = outlier_indices.next();
                }
                item[j] = outlier_state;
            }
        }
        matrix
    }

    fn outlier_scores_to_boolean_table(results: &OutlierResult) -> Vec<Vec<bool>> {
        // Start by prepopulating a [n_timestamps x n_series] matrix of false values.
        let series_count = results.series_results.len();
        let timestamp_count = DBSCAN_DATASET.len();
        let mut matrix = vec![vec![false; series_count]; timestamp_count];

        // For each series, iterate over the outlier intervals, marking the points in each intervals as outliers
        // in the matrix.
        for (j, series) in results.series_results.iter().enumerate() {
            for (i, item) in matrix.iter_mut().enumerate() {
                item[j] = series.scores[i] > 0.0;
            }
        }
        matrix
    }

    #[test]
    fn test_synthetic() {
        for TestCase { eps, expected } in CASES {
            let dbscan = DBSCANDetector::with_epsilon(*eps);
            let results = dbscan.detect_transposed(DBSCAN_DATASET);
            let table = outlier_intervals_to_boolean_table(&results);
            let scores = outlier_scores_to_boolean_table(&results);
            for (i, row) in table.iter().enumerate() {
                assert_eq!(
                    row, expected[i],
                    "unexpected result for epsilon {eps}, series {i}"
                );
            }
            for (i, row) in scores.iter().enumerate() {
                assert_eq!(
                    row, expected[i],
                    "unexpected result for epsilon {eps}, series {i}"
                );
            }
        }
    }

    #[test]
    fn test_realistic() {
        let data = crate::testing::SERIES;

        let dbscan = DBSCANDetector::with_sensitivity(0.8).unwrap();
        let results = dbscan.detect(data);
        assert!(!results.outlying_series.contains(&0));
        assert!(!results.outlying_series.contains(&1));
        assert!(results.outlying_series.contains(&2));

        assert!(results.series_results[0]
            .outlier_intervals
            .indices
            .is_empty());
        assert!(results.series_results[1]
            .outlier_intervals
            .indices
            .is_empty());
        assert_eq!(results.series_results[2].outlier_intervals.indices[0], 40);
        assert_eq!(results.series_results[2].outlier_intervals.indices[1], 42);
        assert_eq!(results.series_results[2].outlier_intervals.indices[2], 140);
        assert_eq!(results.series_results[2].outlier_intervals.indices[3], 142);
        assert_eq!(results.series_results[2].outlier_intervals.indices[4], 240);
        assert_eq!(results.series_results[2].outlier_intervals.indices[5], 242);
    }
}
