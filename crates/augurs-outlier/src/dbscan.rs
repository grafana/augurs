use std::ops::Range;

use tracing::instrument;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::{Band, Error, OutlierDetector, OutlierOutput, Sensitivity, Series};

/// The epsilon or sensitivity parameter for the DBSCAN algorithm.
#[derive(Debug, Clone, Copy)]
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
pub struct DbscanDetector {
    /// The maximum distance between points in a cluster.
    epsilon_or_sensitivity: EpsilonOrSensitivity,

    parallelize: bool,
}

impl OutlierDetector for DbscanDetector {
    type PreprocessedData = Data;
    fn preprocess(&self, y: &[&[f64]]) -> Result<Self::PreprocessedData, Error> {
        Data::try_from_row_major(y)
    }

    fn detect(&self, y: &Self::PreprocessedData) -> Result<OutlierOutput, Error> {
        Ok(self.run(y))
    }
}

impl DbscanDetector {
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
    pub fn with_sensitivity(sensitivity: f64) -> Result<Self, Error> {
        let sensitivity = Sensitivity::try_from(sensitivity)?;
        Ok(Self {
            epsilon_or_sensitivity: EpsilonOrSensitivity::Sensitivity(sensitivity),
            parallelize: false,
        })
    }

    /// Set epsilon for the DBSCAN algorithm.
    ///
    /// # Example
    ///
    /// ```rust
    /// use augurs::outlier::DbscanDetector;
    ///
    /// let mut dbscan = DbscanDetector::with_epsilon(0.1);
    /// dbscan.set_epsilon(0.2);
    /// ```
    pub fn set_epsilon(&mut self, epsilon: f64) {
        self.epsilon_or_sensitivity = EpsilonOrSensitivity::Epsilon(epsilon);
    }

    /// Set sensitivity for the DBSCAN algorithm.
    ///
    /// # Example
    ///
    /// ```rust
    /// use augurs::outlier::DbscanDetector;
    ///
    /// let mut dbscan = DbscanDetector::with_sensitivity(0.1).expect("sensitivity is between 0.0 and 1.0");
    /// dbscan.set_sensitivity(0.2).expect("sensitivity is between 0.0 and 1.0");
    /// ```
    pub fn set_sensitivity(&mut self, sensitivity: f64) -> Result<(), Error> {
        self.epsilon_or_sensitivity =
            EpsilonOrSensitivity::Sensitivity(Sensitivity::try_from(sensitivity)?);
        Ok(())
    }

    /// Parallelize the DBSCAN algorithm.
    ///
    /// This requires the `parallel` feature to be enabled, otherwise it will be ignored.
    pub fn parallelize(mut self, parallelize: bool) -> Self {
        self.parallelize = parallelize;
        self
    }

    fn run(&self, data: &Data) -> OutlierOutput {
        let epsilon = self.epsilon_or_sensitivity.resolve_epsilon(data);
        let n_timestamps = data.n_timestamps();
        let mut serieses = Series::preallocated(data.n_series, n_timestamps);
        let mut normal_band = None;

        // Run DBSCANs in parallel using Rayon if specified.
        #[cfg(feature = "parallel")]
        let dbscans: Vec<_> = if self.parallelize {
            (0..n_timestamps)
                .into_par_iter()
                .map(|i| Self::dbscan_1d(data.timestamp(i), epsilon))
                .collect()
        } else {
            (0..n_timestamps)
                .map(|i| Self::dbscan_1d(data.timestamp(i), epsilon))
                .collect()
        };
        #[cfg(not(feature = "parallel"))]
        let dbscans: Vec<_> = (0..n_timestamps)
            .map(|i| Self::dbscan_1d(data.timestamp(i), epsilon))
            .collect();

        // The series that currently have an open outlier interval. Whether a
        // given series is one of them is answered by
        // `OutlierIntervals::is_open`, so this list only needs to name *which*
        // series to check, rather than duplicating that flag itself.
        //
        // Keeping this list (rather than searching the previous timestamp's
        // outliers) is what keeps the bookkeeping below `O(n_outliers)` per
        // timestamp rather than `O(n_outliers ^ 2)`.
        let mut open_series: Vec<usize> = Vec::new();
        // Scratch flags marking the series that are outlying at this timestamp.
        // Always fully reset before the end of each iteration.
        let mut outlying_now = vec![false; data.n_series];

        for (i, dbscan) in dbscans.into_iter().enumerate() {
            let DBScan1DResults {
                cluster_min,
                cluster_max,
                cluster,
            } = dbscan;

            // The values at each timestamp are sorted, so the outliers are exactly
            // those that sort outside the cluster: a prefix and a suffix. When no
            // cluster was found `cluster` is empty and everything is an outlier.
            let indices = data.timestamp(i).indices;
            let (below, above) = (&indices[..cluster.start], &indices[cluster.end..]);

            // Construct the normal band, if found.
            if let Some((min, max)) = cluster_min.zip(cluster_max) {
                let band = normal_band.get_or_insert_with(|| Band::new(n_timestamps));
                band.min[i] = min - epsilon / 2.0;
                band.max[i] = max + epsilon / 2.0;
            }

            // Mark the outlier series and fill in any positive scores.
            for &Index(idx) in below.iter().chain(above) {
                let idx = idx as usize;
                let series = &mut serieses[idx];
                series.is_outlier = true;
                series.scores[i] = 1.0;
                outlying_now[idx] = true;
            }

            // Close the interval of every series that has stopped being an
            // outlier, dropping it from the open list in the same pass rather
            // than closing and then re-scanning the list to filter it.
            open_series.retain(|&idx| {
                if outlying_now[idx] {
                    true
                } else {
                    serieses[idx].outlier_intervals.add_end(i);
                    false
                }
            });

            // Open an interval for every series that has started being an
            // outlier, resetting its scratch flag ready for the next
            // timestamp in the same pass since nothing else reads it in
            // between. Only the flags we set need clearing, which is what
            // keeps this loop proportional to the number of outliers rather
            // than the number of series.
            for &Index(idx) in below.iter().chain(above) {
                let idx = idx as usize;
                if !serieses[idx].outlier_intervals.is_open() {
                    serieses[idx].outlier_intervals.add_start(i);
                    open_series.push(idx);
                }
                outlying_now[idx] = false;
            }
        }
        OutlierOutput::new(serieses, normal_band)
    }

    // Following impl inspired by https://github.com/d-chambers/dbscan1d
    //
    // Main idea: as the array is sorted, a cluster is just a run of consecutive
    // values where every neighbouring pair is within epsilon of each other. We
    // mandate that the cluster contains more than half of all values, which has a
    // useful consequence: at most one run can ever be large enough, and any run
    // that large must span the middle of the sorted values.
    //
    // (A run of length `L > n / 2` starting at `s` and ending at `e` satisfies
    // `s <= n - L < n / 2` and `e >= L - 1 >= n / 2`, so it always contains index
    // `n / 2`. Two such runs cannot both exist, as they are disjoint and would need
    // more than `n` values between them.)
    //
    // So instead of scanning every gap and then rescanning to collect the outliers,
    // start from the middle value and walk outwards for as long as the gap to the
    // next value is within epsilon. That both finds the only candidate cluster and
    // gives us its bounds directly, and it can stop early when there is no cluster.
    fn dbscan_1d(timestamp: Timestamp<'_>, eps: f64) -> DBScan1DResults {
        let values = timestamp.values;
        let n = values.len();
        // if <=2 series, can return quickly as no anomaly
        if n <= 2 {
            return DBScan1DResults::all_normal(n);
        }

        let min_cluster_size = min_majority_cluster_size(n);
        // The "must span the middle" argument above relies on the cluster holding a
        // strict majority of the values. If that ever changes, this shortcut is no
        // longer valid and the full scan has to come back. This is also checked
        // unconditionally (not just in debug builds) by
        // `min_majority_cluster_size_is_always_a_strict_majority` below.
        debug_assert!(
            min_cluster_size * 2 > n,
            "dbscan_1d assumes a cluster holds a strict majority of values"
        );

        // Note there's no need for `abs` when comparing gaps: `values` is sorted
        // ascending, so every gap is already non-negative.
        let middle = n / 2;
        let mut start = middle;
        while start > 0 && values[start] - values[start - 1] <= eps {
            start -= 1;
        }
        let mut end = middle;
        while end + 1 < n && values[end + 1] - values[end] <= eps {
            end += 1;
        }

        if end - start + 1 >= min_cluster_size {
            DBScan1DResults {
                cluster_min: Some(values[start]),
                cluster_max: Some(values[end]),
                cluster: start..end + 1,
            }
        } else {
            // No run is large enough, so everything is an outlier.
            DBScan1DResults::all_outlying()
        }
    }
}

/// The minimum size a cluster of `n` sorted values must reach to hold a
/// strict majority of them.
///
/// `dbscan_1d`'s "walk outwards from the middle" shortcut is only valid
/// because a cluster of at least this size is guaranteed to contain the
/// middle index (see the comment on `dbscan_1d`). That guarantee is checked
/// unconditionally by `min_majority_cluster_size_is_always_a_strict_majority`
/// below, so that changing this formula in a way that breaks it fails a test
/// in every build, not just debug ones.
fn min_majority_cluster_size(n: usize) -> usize {
    n / 2 + 1
}

pub(crate) struct DBScan1DResults {
    cluster_min: Option<f64>,
    cluster_max: Option<f64>,
    /// The range of the timestamp's sorted values that falls inside the cluster.
    ///
    /// Everything outside this range is an outlier, so an empty range means every
    /// value at this timestamp is an outlier.
    cluster: Range<usize>,
}

impl DBScan1DResults {
    /// There were too few values at this timestamp to call any of them outlying.
    ///
    /// Note this is *not* the same as [`Self::all_outlying`]: there is no cluster
    /// band either way, but here every value is treated as normal.
    fn all_normal(n: usize) -> Self {
        Self {
            cluster_min: None,
            cluster_max: None,
            cluster: 0..n,
        }
    }

    /// No run of values was large enough to form a cluster, so all of them are
    /// outlying.
    fn all_outlying() -> Self {
        Self {
            cluster_min: None,
            cluster_max: None,
            cluster: 0..0,
        }
    }
}

/// Newtype wrapper to ensure that we use the correct type when converting from
/// sorted data to original indexes.
///
/// A `u32` is plenty: it bounds the number of input series, not the number of
/// timestamps, and halves the size of the index array relative to a `usize`.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
#[repr(transparent)]
struct Index(u32);

/// The largest number of input series that [`Data::build`] can index.
const MAX_SERIES: usize = u32::MAX as usize;

/// Check that `n_series` fits in the `u32` that [`Index`] narrows series
/// indices to.
fn validate_series_count(n_series: usize) -> Result<(), Error> {
    if n_series > MAX_SERIES {
        return Err(Error::Preprocessing(
            Box::<dyn std::error::Error>::from(format!(
                "too many series: got {n_series}, but at most {MAX_SERIES} are supported"
            ))
            .into(),
        ));
    }
    Ok(())
}

/// Push `(value, series)` onto `scratch` unless `value` is `NaN`.
///
/// `series` is narrowed to the `u32` that [`Index`] stores; callers must have
/// already checked it fits via [`validate_series_count`].
fn push_if_present(scratch: &mut Vec<(f64, Index)>, value: f64, series: usize) {
    if !value.is_nan() {
        scratch.push((value, Index(series as u32)));
    }
}

/// Preprocessed data for the DBSCAN algorithm.
///
/// This holds the input values transposed, so that the values of every series at
/// a given timestamp are contiguous, and sorted ascending within each timestamp.
///
/// `NaN` inputs are dropped during preprocessing, which is why the number of
/// values at a timestamp can be smaller than `n_series`. Rather than reserve a
/// fixed `n_series`-wide slot per timestamp regardless of how many values
/// survive that filtering, every timestamp's surviving values are packed back
/// to back in one flat allocation, with `offsets[i]` recording where
/// timestamp `i`'s slice starts.
#[derive(Debug)]
pub struct Data {
    /// The values at each timestamp, sorted ascending, packed contiguously.
    values: Vec<f64>,
    /// The original series index of each entry in `values`.
    indices: Vec<Index>,
    /// The start offset of each timestamp's slice into `values`/`indices`.
    offsets: Vec<usize>,
    /// The number of non-`NaN` values at each timestamp.
    counts: Vec<u32>,
    n_series: usize,
}

/// A borrowed view of the values of every series at a single timestamp.
#[derive(Debug, Clone, Copy)]
struct Timestamp<'a> {
    /// The non-`NaN` values at this timestamp, sorted ascending.
    values: &'a [f64],
    /// The original series index of each entry in `values`.
    indices: &'a [Index],
}

impl Data {
    /// Create a `Data` struct from row-major data.
    ///
    /// # Panics
    ///
    /// Panics if `data` is empty or if all rows do not have the same length.
    /// Use [`try_from_row_major`](Self::try_from_row_major) for a fallible version.
    #[instrument(skip(data))]
    pub fn from_row_major(data: &[&[f64]]) -> Self {
        Self::try_from_row_major(data).expect("all rows in data must have the same length")
    }

    /// Try to create a `Data` struct from row-major data.
    ///
    /// Returns an error if `data` is empty or if all rows do not have the same length.
    #[instrument(skip(data))]
    pub fn try_from_row_major(data: &[&[f64]]) -> Result<Self, Error> {
        if data.is_empty() {
            return Err(Error::Preprocessing(
                Box::<dyn std::error::Error>::from("data must not be empty").into(),
            ));
        }

        let n_series = data.len();
        let n_timestamps = data[0].len();
        validate_series_count(n_series)?;

        // Validate that all rows have the same length (skip first row since it's our reference).
        for (i, row) in data.iter().enumerate().skip(1) {
            if row.len() != n_timestamps {
                return Err(Error::Preprocessing(
                    Box::<dyn std::error::Error>::from(format!(
                        "all rows must have the same length: row 0 has {} elements, but row {} has {} elements",
                        n_timestamps, i, row.len()
                    ))
                    .into(),
                ));
            }
        }

        // Transpose and sort in one pass. Gathering column `i` walks down the rows,
        // which looks strided but reads the same handful of cache lines for a run of
        // consecutive timestamps, so there's no need for a separate transpose buffer.
        Ok(Self::build(n_series, n_timestamps, |i, scratch| {
            for (j, row) in data.iter().enumerate() {
                push_if_present(scratch, row[i], j);
            }
        }))
    }

    /// Create a `Data` struct from column-major data.
    ///
    /// # Panics
    ///
    /// Panics if `data` is empty or if all columns do not have the same length.
    /// Use [`try_from_column_major`](Self::try_from_column_major) for a fallible version.
    #[instrument(skip(data))]
    pub fn from_column_major(data: &[&[f64]]) -> Self {
        Self::try_from_column_major(data).expect("all columns in data must have the same length")
    }

    /// Try to create a `Data` struct from column-major data.
    ///
    /// Returns an error if `data` is empty or if all columns do not have the same length.
    #[instrument(skip(data))]
    pub fn try_from_column_major(data: &[&[f64]]) -> Result<Self, Error> {
        if data.is_empty() {
            return Err(Error::Preprocessing(
                Box::<dyn std::error::Error>::from("data must not be empty").into(),
            ));
        }

        let n_series = data[0].len();
        validate_series_count(n_series)?;

        // Validate that all columns have the same length (skip first column since it's our reference).
        for (i, col) in data.iter().enumerate().skip(1) {
            if col.len() != n_series {
                return Err(Error::Preprocessing(
                    Box::<dyn std::error::Error>::from(format!(
                        "all columns must have the same length: column 0 has {} elements, but column {} has {} elements",
                        n_series, i, col.len()
                    ))
                    .into(),
                ));
            }
        }

        // The data is already grouped by timestamp, so gathering is just a copy.
        Ok(Self::build(n_series, data.len(), |i, scratch| {
            for (j, &value) in data[i].iter().enumerate() {
                push_if_present(scratch, value, j);
            }
        }))
    }

    /// Transpose and sort the input into the flat layout described on [`Data`].
    ///
    /// `gather` is called once per timestamp and should push the `(value, series)`
    /// pair of every non-`NaN` value at that timestamp onto the scratch buffer.
    #[instrument(skip(gather))]
    fn build(
        n_series: usize,
        n_timestamps: usize,
        mut gather: impl FnMut(usize, &mut Vec<(f64, Index)>),
    ) -> Self {
        let mut counts = Vec::with_capacity(n_timestamps);
        let mut offsets = Vec::with_capacity(n_timestamps);
        // Grown by exactly as many values as survive `NaN`-filtering, rather
        // than pre-sized (and zero-initialized) for the dense `n_series *
        // n_timestamps` upper bound: with sparse data most of that grid would
        // never be read back through `timestamp`, since it only ever slices
        // `offsets[i]..offsets[i] + counts[i]`.
        let mut values = Vec::new();
        let mut indices = Vec::new();
        // Reused across timestamps so that sorting doesn't allocate per timestamp.
        let mut scratch: Vec<(f64, Index)> = Vec::with_capacity(n_series);

        for i in 0..n_timestamps {
            scratch.clear();
            gather(i, &mut scratch);
            // Sorting `(f64, Index)` pairs by the value keeps the value inline, so
            // each comparison is a load from the pair rather than a pointer chase
            // through a separate array. `total_cmp` is total, so no `unwrap` is
            // needed; `NaN`s have already been filtered out by `gather`, and the
            // only case where it disagrees with `partial_cmp` is the relative order
            // of `-0.0` and `0.0`, which is arbitrary under an unstable sort anyway.
            scratch.sort_unstable_by(|(a, _), (b, _)| a.total_cmp(b));

            offsets.push(values.len());
            counts.push(scratch.len() as u32);
            values.extend(scratch.iter().map(|&(value, _)| value));
            indices.extend(scratch.iter().map(|&(_, index)| index));
        }

        Self {
            values,
            indices,
            offsets,
            counts,
            n_series,
        }
    }

    /// The number of timestamps in the data.
    fn n_timestamps(&self) -> usize {
        self.counts.len()
    }

    /// The sorted values of every series at timestamp `i`.
    fn timestamp(&self, i: usize) -> Timestamp<'_> {
        let base = self.offsets[i];
        let range = base..base + self.counts[i] as usize;
        Timestamp {
            values: &self.values[range.clone()],
            indices: &self.indices[range],
        }
    }

    /// Calculate the span of the data: the difference between the highest and lowest values.
    fn span(&self) -> f64 {
        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;
        for i in 0..self.n_timestamps() {
            // Values are sorted, so the extremes of each timestamp are its ends.
            let values = self.timestamp(i).values;
            if let Some(low) = values.first() {
                min = min.min(*low);
            }
            if let Some(high) = values.last() {
                max = max.max(*high);
            }
        }
        (max - min).abs().max(0.1)
    }
}

#[cfg(test)]
mod tests {
    use crate::{testing::flatten_intervals, OutlierDetector, OutlierOutput};

    use super::*;

    const UNDEFINED: f64 = f64::NAN;
    // Transposed dataset for testing DBSCAN.
    // There are 13 timestamps and 9 series: each inner
    // array contains the values for the all series at that timestamp.
    static DBSCAN_DATASET: &[&[f64]] = &[
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

    fn outlier_intervals_to_boolean_table(results: &OutlierOutput) -> Vec<Vec<bool>> {
        // Start by prepopulating a [n_timestamps x n_series] matrix of false values.
        let series_count = results.series_results.len();
        let timestamp_count = DBSCAN_DATASET.len();
        let mut matrix = vec![vec![false; series_count]; timestamp_count];

        // For each series, iterate over the outlier intervals, marking the points in each intervals as outliers
        // in the matrix.
        for (j, series) in results.series_results.iter().enumerate() {
            let mut outlier_state = false;
            let outlier_indices = flatten_intervals(&series.outlier_intervals.intervals);
            let mut iter = outlier_indices.iter();
            let mut next_idx = iter.next();
            for (i, item) in matrix.iter_mut().enumerate() {
                if next_idx.is_some_and(|next_idx| i >= *next_idx) {
                    outlier_state = !outlier_state;
                    next_idx = iter.next();
                }
                item[j] = outlier_state;
            }
        }
        matrix
    }

    fn outlier_scores_to_boolean_table(results: &OutlierOutput) -> Vec<Vec<bool>> {
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
    fn test_tiny() {
        let data: &[&[f64]] = &[
            &[1.0, 2.0, 1.5, 2.3],
            &[1.9, 2.2, 1.2, 2.4],
            &[1.5, 2.1, 6.4, 8.5],
        ];
        let detector =
            DbscanDetector::with_sensitivity(0.5).expect("sensitivity is between 0.0 and 1.0");
        let processed = detector.preprocess(data).unwrap();
        let outliers = detector.detect(&processed).unwrap();

        assert_eq!(outliers.outlying_series.len(), 1);
        assert!(outliers.outlying_series.contains(&2));
        assert!(outliers.series_results[2].is_outlier);
        assert_eq!(outliers.series_results[2].scores, vec![0.0, 0.0, 1.0, 1.0]);
        assert!(outliers.cluster_band.is_some());
    }

    #[test]
    fn test_synthetic() {
        for TestCase { eps, expected } in CASES {
            let dbscan = DbscanDetector::with_epsilon(*eps);
            let data = Data::from_column_major(DBSCAN_DATASET);
            let results = dbscan.detect(&data).unwrap();
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
        let dbscan = DbscanDetector::with_sensitivity(0.8).unwrap();
        let data = dbscan.preprocess(crate::testing::SERIES).unwrap();
        let results = dbscan.detect(&data).unwrap();
        assert!(!results.outlying_series.contains(&0));
        assert!(!results.outlying_series.contains(&1));
        assert!(results.outlying_series.contains(&2));

        assert!(results.series_results[0]
            .outlier_intervals
            .intervals
            .is_empty());
        assert!(results.series_results[1]
            .outlier_intervals
            .intervals
            .is_empty());
        let indices = flatten_intervals(&results.series_results[2].outlier_intervals.intervals);
        assert_eq!(indices[0], 40);
        assert_eq!(indices[1], 42);
        assert_eq!(indices[2], 140);
        assert_eq!(indices[3], 142);
        assert_eq!(indices[4], 240);
        assert_eq!(indices[5], 242);
        assert!(results.cluster_band.is_some());
    }

    // Test that the DBSCAN detector can handle missing data at the start of a series.
    // This is a regression test for a bug where the DBSCAN detector would panic when
    // the first value in a series was missing, because we used the number of values
    // at timestamp 0 _after_ omitting NANs to determine the number of series.
    #[test]
    fn test_missing_data_at_start() {
        let data: &[&[f64]] = &[
            &[f64::NAN, 2.0, 1.5, 2.3],
            &[1.9, 2.2, f64::NAN, 2.4],
            &[1.5, 2.1, 6.4, 8.5],
        ];
        let dbscan = DbscanDetector::with_epsilon(1.0);
        let processed = dbscan.preprocess(data).unwrap();
        // Should not panic.
        let results = dbscan.detect(&processed).unwrap();

        assert!(!results.outlying_series.contains(&0));
        assert!(!results.outlying_series.contains(&1));
        assert!(results.outlying_series.contains(&2));
        assert!(results.cluster_band.is_some());
    }

    // Having too few values at a timestamp and finding no cluster at a timestamp both
    // mean there is no cluster band, but they mean opposite things about the values:
    // with fewer than three values nothing can be called outlying, whereas if no run
    // of values is big enough to be a cluster then everything is outlying. It's easy
    // to conflate the two, so pin the distinction down.
    #[test]
    fn test_too_few_values_is_not_the_same_as_no_cluster() {
        // Two series, far enough apart that no cluster could form: neither is an
        // outlier, because with two values there's nothing to be an outlier of.
        let data: &[&[f64]] = &[&[1.0, 2.0], &[100.0, 200.0]];
        let dbscan = DbscanDetector::with_epsilon(1.0);
        let results = dbscan.detect(&dbscan.preprocess(data).unwrap()).unwrap();
        assert!(results.outlying_series.is_empty());
        assert!(results.cluster_band.is_none());
        for series in &results.series_results {
            assert_eq!(series.scores, vec![0.0, 0.0]);
            assert!(series.outlier_intervals.intervals.is_empty());
        }

        // Three series spread out so that no cluster can reach the required majority:
        // now every series is an outlier at every timestamp.
        let data: &[&[f64]] = &[&[1.0, 2.0], &[100.0, 200.0], &[10_000.0, 20_000.0]];
        let results = dbscan.detect(&dbscan.preprocess(data).unwrap()).unwrap();
        assert_eq!(results.outlying_series.len(), 3);
        assert!(results.cluster_band.is_none());
        for series in &results.series_results {
            assert_eq!(series.scores, vec![1.0, 1.0]);
        }
    }

    // NaNs are dropped during preprocessing, so a timestamp can drop below three
    // values even when there are plenty of series. That has to be judged on the
    // number of values actually present, not the number of series.
    #[test]
    fn test_nans_reduce_values_below_cluster_threshold() {
        let data: &[&[f64]] = &[
            &[1.0, 1.0],
            &[100.0, UNDEFINED],
            &[10_000.0, UNDEFINED],
            &[20_000.0, UNDEFINED],
        ];
        let dbscan = DbscanDetector::with_epsilon(1.0);
        let results = dbscan.detect(&dbscan.preprocess(data).unwrap()).unwrap();

        // Timestamp 0 has four spread-out values and so no cluster: all outlying.
        // Timestamp 1 has a single value, which can't be outlying.
        assert_eq!(results.series_results[0].scores, vec![1.0, 0.0]);
        for series in &results.series_results[1..] {
            assert_eq!(series.scores, vec![1.0, 0.0]);
        }
        // Every series was outlying at timestamp 0, and stopped at timestamp 1.
        for series in &results.series_results {
            assert_eq!(series.outlier_intervals.intervals.len(), 1);
            assert_eq!(series.outlier_intervals.intervals[0].start, 0);
            assert_eq!(series.outlier_intervals.intervals[0].end, Some(1));
        }
    }

    // Column-major data can describe timestamps that contain no series at all, which
    // makes the flat value storage zero-width. That must not panic.
    #[test]
    fn test_column_major_no_series() {
        let data: &[&[f64]] = &[&[], &[]];
        let dbscan = DbscanDetector::with_epsilon(1.0);
        let results = dbscan.detect(&Data::from_column_major(data)).unwrap();
        assert!(results.series_results.is_empty());
        assert!(results.outlying_series.is_empty());
        assert!(results.cluster_band.is_none());
    }

    // Row-major data can likewise describe series with no timestamps.
    #[test]
    fn test_row_major_no_timestamps() {
        let data: &[&[f64]] = &[&[], &[]];
        let dbscan = DbscanDetector::with_epsilon(1.0);
        let results = dbscan.detect(&Data::from_row_major(data)).unwrap();
        assert_eq!(results.series_results.len(), 2);
        for series in &results.series_results {
            assert!(series.scores.is_empty());
            assert!(!series.is_outlier);
        }
        assert!(results.cluster_band.is_none());
    }

    #[test]
    fn test_no_cluster_band_small_eps() {
        let data: &[&[f64]] = &[
            &[1.0, 2.0, 3.0, 4.0],
            &[4.0, 5.0, 6.0, 7.0],
            &[7.0, 8.0, 9.0, 10.0],
        ];
        let dbscan = DbscanDetector::with_epsilon(1.0);
        let processed = dbscan.preprocess(data).unwrap();
        let results = dbscan.detect(&processed).unwrap();
        assert!(results.cluster_band.is_none());
    }

    #[test]
    fn test_no_cluster_band_two_series() {
        let data: &[&[f64]] = &[&[1.0, 2.0, 3.0, 4.0], &[4.0, 5.0, 6.0, 7.0]];
        let dbscan = DbscanDetector::with_epsilon(4.0);
        let processed = dbscan.preprocess(data).unwrap();
        let results = dbscan.detect(&processed).unwrap();
        assert!(results.cluster_band.is_none());
    }

    #[test]
    fn test_try_from_row_major_empty_data() {
        let data: &[&[f64]] = &[];
        let result = Data::try_from_row_major(data);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("data must not be empty"));
    }

    #[test]
    fn test_try_from_row_major_non_rectangular_data() {
        let data: &[&[f64]] = &[&[1.0, 2.0, 3.0], &[4.0, 5.0], &[7.0, 8.0, 9.0]];
        let result = Data::try_from_row_major(data);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("all rows must have the same length"));
        assert!(err.contains("row 0 has 3 elements"));
        assert!(err.contains("row 1 has 2 elements"));
    }

    #[test]
    #[should_panic(expected = "all rows in data must have the same length")]
    fn test_from_row_major_panics_on_non_rectangular_data() {
        let data: &[&[f64]] = &[&[1.0, 2.0, 3.0], &[4.0, 5.0]];
        Data::from_row_major(data);
    }

    #[test]
    fn test_from_row_major_rectangular_data_succeeds() {
        let data: &[&[f64]] = &[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], &[7.0, 8.0, 9.0]];
        let result = Data::try_from_row_major(data);
        assert!(result.is_ok());
        let data_struct = result.unwrap();
        assert_eq!(data_struct.n_series, 3);
    }

    #[test]
    fn test_try_from_column_major_empty_data() {
        let data: &[&[f64]] = &[];
        let result = Data::try_from_column_major(data);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("data must not be empty"));
    }

    #[test]
    fn test_try_from_column_major_non_rectangular_data() {
        let data: &[&[f64]] = &[&[1.0, 2.0, 3.0], &[4.0, 5.0], &[7.0, 8.0, 9.0]];
        let result = Data::try_from_column_major(data);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("all columns must have the same length"));
        assert!(err.contains("column 0 has 3 elements"));
        assert!(err.contains("column 1 has 2 elements"));
    }

    #[test]
    #[should_panic(expected = "all columns in data must have the same length")]
    fn test_from_column_major_panics_on_non_rectangular_data() {
        let data: &[&[f64]] = &[&[1.0, 2.0, 3.0], &[4.0, 5.0]];
        Data::from_column_major(data);
    }

    #[test]
    fn test_from_column_major_rectangular_data_succeeds() {
        let data: &[&[f64]] = &[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], &[7.0, 8.0, 9.0]];
        let result = Data::try_from_column_major(data);
        assert!(result.is_ok());
        let data_struct = result.unwrap();
        assert_eq!(data_struct.n_series, 3);
    }

    // `dbscan_1d`'s "walk outwards from the middle" shortcut is only valid
    // because a cluster of `min_majority_cluster_size(n)` values is guaranteed
    // to hold a strict majority of them (and so must contain the middle
    // index). This checks that property directly with a regular `assert!`,
    // which (unlike the `debug_assert!` in `dbscan_1d`) still runs in release
    // builds, so a future change to the formula that breaks the guarantee is
    // always caught.
    #[test]
    fn min_majority_cluster_size_is_always_a_strict_majority() {
        for n in 0..10_000 {
            let min_cluster_size = min_majority_cluster_size(n);
            assert!(
                min_cluster_size * 2 > n,
                "min_majority_cluster_size({n}) = {min_cluster_size} is not a strict majority"
            );
        }
    }

    #[test]
    fn validate_series_count_accepts_up_to_u32_max() {
        assert!(validate_series_count(0).is_ok());
        assert!(validate_series_count(1).is_ok());
        assert!(validate_series_count(MAX_SERIES).is_ok());
    }

    #[test]
    fn validate_series_count_rejects_more_than_u32_max() {
        let err = validate_series_count(MAX_SERIES + 1).unwrap_err();
        assert!(err.to_string().contains("too many series"));
    }

    #[test]
    fn push_if_present_drops_nan_and_narrows_index() {
        let mut scratch = Vec::new();
        push_if_present(&mut scratch, 1.0, 0);
        push_if_present(&mut scratch, f64::NAN, 1);
        push_if_present(&mut scratch, 2.0, 2);
        assert_eq!(scratch, vec![(1.0, Index(0)), (2.0, Index(2))]);
    }

    // `Data::build` used to allocate and zero-initialize a dense `n_series *
    // n_timestamps` grid regardless of how many values actually survived
    // `NaN`-filtering. With sparse data this test would previously have
    // observed `values.len() == n_series * n_timestamps`; now it should be
    // sized to exactly the values kept.
    #[test]
    fn test_build_does_not_over_allocate_for_sparse_data() {
        let n_series = 100;
        let n_timestamps = 5;
        let mut data: Vec<Vec<f64>> = vec![vec![UNDEFINED; n_timestamps]; n_series];
        // Exactly one non-NaN value per timestamp.
        for (t, row) in data.iter_mut().enumerate().take(n_timestamps) {
            row[t] = t as f64;
        }
        let rows: Vec<&[f64]> = data.iter().map(|r| r.as_slice()).collect();
        let result = Data::try_from_row_major(&rows).unwrap();

        assert_eq!(result.values.len(), n_timestamps);
        assert_eq!(result.indices.len(), n_timestamps);
        assert!(result.values.len() < n_series * n_timestamps);
    }

    // The interval-open bookkeeping in `run` used to track a separate
    // `interval_open` flag per series alongside `OutlierIntervals`'s own
    // notion of being open, and closed + filtered `open_series` in two
    // separate passes. This test exercises a series that repeatedly starts
    // and stops being an outlier, to guard against that refactor losing or
    // duplicating an interval.
    #[test]
    fn test_flickering_outlier_intervals() {
        // Series 0 alternates between matching the rest of the cluster and
        // being far away from it, at every other timestamp.
        let data: &[&[f64]] = &[
            &[0.0, 100.0, 0.0, 100.0, 0.0, 100.0],
            &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ];
        let dbscan = DbscanDetector::with_epsilon(1.0);
        let processed = dbscan.preprocess(data).unwrap();
        let results = dbscan.detect(&processed).unwrap();

        assert_eq!(
            results.series_results[0].scores,
            vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
        );
        let intervals = &results.series_results[0].outlier_intervals.intervals;
        assert_eq!(intervals.len(), 3);
        // The first two intervals close as soon as the series stops being an
        // outlier; the last one is still open when the data ends.
        for interval in &intervals[..2] {
            assert_eq!(interval.end, interval.start.checked_add(1));
        }
        assert_eq!(intervals[2].end, None);
        for series in &results.series_results[1..] {
            assert!(series.outlier_intervals.intervals.is_empty());
        }
    }
}
