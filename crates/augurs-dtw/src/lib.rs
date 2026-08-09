#![doc = include_str!("../README.md")]

use augurs_core::DistanceMatrix;

#[cfg(feature = "parallel")]
use rayon::prelude::*;
#[cfg(feature = "parallel")]
use tracing::debug;

/// A trait for defining a distance function.
///
/// This trait is used to define the distance function used in the DTW algorithm.
/// Examples of distance functions are Euclidean distance and Manhattan distance.
pub trait Distance {
    /// Compute the distance between two points.
    ///
    /// E.g. Euclidean distance or Manhattan distance.
    ///
    /// This function is called in a hot loop, so should be annotated with
    /// `#[inline]` for performance (this was confirmed by benchmarks of the
    /// Euclidean distance).
    fn distance(&self, a: f64, b: f64) -> f64;

    /// Transform the final distance.
    ///
    /// This is offered as an optimization to avoid performing certain operations
    /// on every distance calculation. For example, when using the Euclidean distance,
    /// the square root can be applied only once to the final distance, rather than
    /// on every distance calculation, which can be expensive.
    ///
    /// The default function is the identity function, but this should be overridden
    /// if the distance function implementation requires a transformation.
    #[inline]
    fn transform_result(&self, dist: f64) -> f64 {
        dist
    }
}

/// Euclidean distance, also known as L2 distance.
///
/// This is the default distance function used in DTW.
///
/// It is defined as the square root of the sum of the squared difference
/// between two points.
#[derive(Debug)]
pub struct Euclidean;

impl Distance for Euclidean {
    #[inline]
    fn distance(&self, a: f64, b: f64) -> f64 {
        // Note: we don't take the square root here to avoid
        // the extra computation. The final result will be
        // transformed by the `transform_result` method.
        (a - b).powi(2)
    }

    #[inline]
    fn transform_result(&self, dist: f64) -> f64 {
        dist.sqrt()
    }
}

/// Manhattan distance, also known as L1 distance.
///
/// It is defined as the sum of the absolute difference between
/// two points.
#[derive(Debug)]
pub struct Manhattan;

impl Distance for Manhattan {
    #[inline]
    fn distance(&self, a: f64, b: f64) -> f64 {
        f64::abs(a - b)
    }
}

impl Distance for Box<dyn Distance> {
    #[inline]
    fn distance(&self, a: f64, b: f64) -> f64 {
        self.as_ref().distance(a, b)
    }

    #[inline]
    fn transform_result(&self, dist: f64) -> f64 {
        self.as_ref().transform_result(dist)
    }
}

/// Dynamic Time Warping (DTW) algorithm.
///
/// Dynamic time warping is used to compare two sequences that may vary in time or speed.
///
/// This implementation has built-in support for both Euclidean and Manhattan distance,
/// and can be extended to support other distance functions by implementing the [`Distance`]
/// trait and using the [`Dtw::new`] constructor.
///
/// The algorithm is based on the code from the [UCR Suite][ucr-suite].
///
/// # Example
///
/// ```
/// use augurs_dtw::Dtw;
///
/// let a = &[0.0, 1.0, 2.0];
/// let b = &[3.0, 4.0, 5.0];
/// let dist = Dtw::euclidean().distance(a, b);
/// assert_eq!(dist, 5.0990195135927845);
/// ```
///
/// [ucr-suite]: https://www.cs.ucr.edu/~eamonn/UCRsuite.html
#[derive(Debug, Clone)]
pub struct Dtw<T: Distance + Send + Sync> {
    // The Sakoe-Chiba warping window.
    window: Option<usize>,
    // The distance function.
    distance_fn: T,
    // Whether to parallelize the computation of the distance matrix.
    // Note that the `parallel` feature must be enabled for this to work,
    // otherwise this flag is ignored.
    parallelize: bool,
    // The maximum distance, used for early stopping.
    max_distance: Option<f64>,

    // Lower bound for early abandoning.
    lower_bound: Option<f64>,
    // Upper bound for early abandoning.
    upper_bound: Option<f64>,
}

impl Dtw<Euclidean> {
    /// Create a new DTW instance using Euclidean distance.
    #[must_use]
    pub fn euclidean() -> Self {
        Self::new(Euclidean)
    }
}

impl Default for Dtw<Euclidean> {
    fn default() -> Self {
        Self::euclidean()
    }
}

impl Dtw<Manhattan> {
    /// Create a new DTW instance using Manhattan distance.
    #[must_use]
    pub fn manhattan() -> Self {
        Self::new(Manhattan)
    }
}

impl<T: Distance + Send + Sync> Dtw<T> {
    /// Create a new DTW instance with a custom distance function.
    #[must_use]
    pub fn new(distance_fn: T) -> Self {
        Self {
            distance_fn,
            window: None,
            parallelize: false,
            max_distance: None,
            lower_bound: None,
            upper_bound: None,
        }
    }
}

impl<T: Distance + Send + Sync> Dtw<T> {
    /// Get the size of the Sakoe-Chiba warping window, if set.
    pub fn window(&self) -> Option<usize> {
        self.window
    }

    /// Set the size of the Sakoe-Chiba warping window, `w`.
    ///
    /// Using a window limits shifts up to this amount away from the diagonal.
    #[must_use]
    pub fn with_window(mut self, window: usize) -> Self {
        self.window = Some(window);
        self
    }

    /// Set the maximum distance for early stopping.
    ///
    /// During the `dtw` calculation, if the cumulative distance
    /// between two series exceeds this value, the computation will stop
    /// and return `max_distance`.
    #[must_use]
    pub fn with_max_distance(mut self, max_distance: f64) -> Self {
        self.max_distance = Some(max_distance);
        self
    }

    /// Set the lower bound limit for early abandoning.
    ///
    /// If the lower bound distance between two series exceeds this value, the computation will stop
    /// and return `lower_bound`.
    ///
    /// Multiple lower bounds are calculated and cascaded in order of
    /// cheapest to most expensive to compute:
    ///
    /// - `LB_Kim`, a constant time lower bound.
    /// - `LB_Keogh`, a linear time lower bound.
    ///
    /// Generally it is a good idea to set this to the same as
    /// the `max_distance` parameter, as this can speed up the computation
    /// by avoiding unnecessary calculations.
    #[must_use]
    pub fn with_lower_bound(mut self, lower_bound: f64) -> Self {
        self.lower_bound = Some(lower_bound);
        self
    }

    /// Set the upper bound limit for early abandoning.
    ///
    /// If the upper bound on the distance between two series is less than `upper_bound`,
    /// the computation will stop and return `upper_bound`.
    ///
    /// This can be used to speed up the computation by avoiding unnecessary calculations.
    /// For example, if the distance matrix is only used for clustering with DBSCAN,
    /// we only care if the distance is <= epsilon, so we can stop early if the upper bound
    /// is < epsilon.
    #[must_use]
    pub fn with_upper_bound(mut self, upper_bound: f64) -> Self {
        self.upper_bound = Some(upper_bound);
        self
    }

    /// Set whether to use parallel computation for the distance matrix computation.
    ///
    /// This does not affect the computation of individual distances using [`Dtw::distance`],
    /// only the computation of the distance matrix using [`Dtw::distance_matrix`].
    pub fn parallelize(mut self, parallelize: bool) -> Self {
        self.parallelize = parallelize;
        self
    }

    /// Compute the distance between two sequences under Dynamic Time Warping.
    ///
    /// # Example
    /// ```
    /// use augurs_dtw::Dtw;
    ///
    /// let a: &[f64] = &[0.0, 1.0, 2.0];
    /// let b: &[f64] = &[3.0, 4.0, 5.0];
    /// let dist = Dtw::euclidean().distance(a, b);
    /// assert_eq!(dist, 5.0990195135927845);
    /// ```
    ///
    /// # Example with `max_distance`.
    /// ```
    /// use augurs_dtw::Dtw;
    ///
    /// let a: &[f64] = &[0.0, 1.0, 2.0];
    /// let b: &[f64] = &[3.0, 4.0, 5.0];
    /// let dist = Dtw::euclidean().with_max_distance(2.0).distance(a, b);
    /// assert_eq!(dist, 2.0);
    /// let dist = Dtw::euclidean().with_max_distance(5.0).distance(a, b);
    /// assert_eq!(dist, 5.0);
    /// let dist = Dtw::euclidean().with_max_distance(6.0).distance(a, b);
    /// assert_eq!(dist, 5.0990195135927845);
    /// ```
    pub fn distance(&self, s: &[f64], t: &[f64]) -> f64 {
        if s.is_empty() || t.is_empty() {
            return f64::INFINITY;
        }

        // Choose the longest sequence as the outer iterator.
        let (outer_iter, inner_iter) = if s.len() >= t.len() {
            (s.iter(), t.iter())
        } else {
            (t.iter(), s.iter())
        };

        // The algorithm is based on the code from the UCR Suite:
        // https://www.cs.ucr.edu/~eamonn/UCRsuite.html
        let m = outer_iter.len();
        let n = inner_iter.len();
        let max_window = self
            .window
            .map_or_else(|| m.max(n), |w| w.max(n.abs_diff(m)));
        let max_distance_transformed = self.max_distance.map(|d| self.distance_fn.distance(d, 0.0));
        let max_k = 2 * max_window - 1;
        let (mut cost, mut prev_cost) = (
            vec![f64::INFINITY; 2 * max_window + 1],
            vec![f64::INFINITY; 2 * max_window + 1],
        );
        let (mut x, mut y, mut z);

        let mut k = 0;

        // `cost_k_minus_1` is the cost of the cell to the left of the current one,
        // i.e. `cost[k - 1]`. On the first row there is no cell to the left, and
        // seeding it with 0.0 makes the very first cell evaluate to nothing but its
        // own distance -- which is exactly the DTW base case `D[0][0] = d(0, 0)`.
        // Every later row is seeded with infinity instead, at the bottom of the loop.
        //
        // Seeding it out here rather than special-casing `i == 0 && j == 0` inside
        // the loop keeps two loop-invariant branches out of the innermost loop. The
        // first is that check itself, which is true exactly once in the whole
        // function yet was evaluated for every cell. The second is the `k == 0`
        // check it made necessary: `k` only ever increases, so `k == 0` implies
        // we're on the first inner iteration of a row with `i >= max_window`, and
        // there `cost_k_minus_1` still holds the infinity seeded below.
        let mut cost_k_minus_1: f64 = 0.0;

        for (i, t_i) in outer_iter.copied().enumerate() {
            k = max_window.saturating_sub(i);

            let lower_bound = i.saturating_sub(max_window);
            let upper_bound = usize::min(n - 1, i + max_window);
            let mut min_cost = f64::INFINITY;

            for (s_j, c) in inner_iter
                .clone()
                .skip(lower_bound)
                .take(upper_bound - lower_bound + 1)
                .copied()
                .zip(&mut cost[k..])
            {
                // SAFETY: prev_cost has length 2 * max_window + 1,
                // and k is always in the range 0..=2 * max_window
                // since we start at k = max_window and increment it
                // by 1 each iteration, and the inner loop runs
                // `max_window` times.
                z = unsafe { *prev_cost.get_unchecked(k) };
                y = cost_k_minus_1;
                let min = if k > max_k {
                    y.min(z)
                } else {
                    // SAFETY: prev_cost has length 2 * max_window + 1,
                    // and k is always in the range 0..=(2 * max_window - 1)
                    // since we start at k = max_window and bound it using
                    // `max_k`, which is `2 * max_window - 1`.
                    x = unsafe { *prev_cost.get_unchecked(k + 1) };
                    x.min(y.min(z))
                };

                let dist = self.distance_fn.distance(s_j, t_i);
                *c = dist + min;
                min_cost = min_cost.min(*c);
                k += 1;
                cost_k_minus_1 = *c;
            }

            if max_distance_transformed.is_some_and(|d| min_cost >= d) {
                return self.max_distance.unwrap();
            }
            (prev_cost, cost) = (cost, prev_cost);
            // Seed the next row: unlike the first row, every later one starts at a
            // cell whose left neighbour is outside the warping band.
            cost_k_minus_1 = f64::INFINITY;
        }
        k = k.saturating_sub(1);

        // If the final cost exceeds the `max_distance` in the final loop then
        // we won't catch it. Check here whether it
        let final_cost = prev_cost[k];
        self.distance_fn.transform_result(
            max_distance_transformed
                .map(|d| final_cost.min(d))
                .unwrap_or(final_cost),
        )
    }

    /// Compute the distance between two sequences under Dynamic Time Warping with early stopping.
    ///
    /// If `lower_bound` is `None`, this just calls [`Dtw::distance`].
    ///
    /// Otherwise, it cascades through various lower and upper bounds on the distance,
    /// stopping early if any of the lower bounds exceed `lower_bound` or if the upper
    /// bound is less than `upper_bound`.
    /// Only if none of these are true is the actual DTW distance computed.
    ///
    /// This is useful in many cases:
    /// - when clustering with DBSCAN we only care if the distance is <= epsilon,
    ///   since that is the only condition for two series to be considered neighbours.
    ///   Therefore if we know that the lower bound of the distance is already > epsilon,
    ///   we can stop early. Similarly, if we know that the upper bound is < epsilon,
    ///   we can also stop early.
    fn distance_with_early_stopping(&self, s: &[f64], t: &[f64]) -> f64 {
        if s.is_empty() && t.is_empty() {
            return 0.0;
        }
        if s.is_empty() || t.is_empty() {
            return f64::INFINITY;
        }
        if self.lower_bound.and(self.upper_bound).is_none() {
            return self.distance(s, t);
        }
        if let Some(lower_bound) = self.lower_bound {
            // LB_kim is constant time so try that first.
            if self.lb_kim(s, t, lower_bound) >= lower_bound {
                return lower_bound;
            }
            // // LB_keogh is linear time.
            // if lb_keogh(s, t) >= max_distance {
            //     return max_distance;
            // }
        };
        if let Some(upper_bound) = self.upper_bound {
            let ub = self.ub_diag(s, t, upper_bound);
            if ub < upper_bound {
                return ub;
            }
        }
        self.distance(s, t)
    }

    fn lb_kim(&self, s: &[f64], t: &[f64], max_dist: f64) -> f64 {
        if s.len() < 2 || t.len() < 2 {
            return 0.0;
        }
        // First check the first points at the front and back.
        let (s_0, t_0, s_last, t_last) = (s[0], t[0], s[s.len() - 1], t[t.len() - 1]);
        let mut sum =
            self.distance_fn.distance(s_0, t_0) + self.distance_fn.distance(s_last, t_last);
        if sum >= max_dist {
            return sum;
        }
        // Next check the second point at the front.
        let (s_1, t_1) = (s[1], t[1]);
        let mut d = self
            .distance_fn
            .distance(s_0, t_1)
            .min(self.distance_fn.distance(s_1, t_0))
            .min(self.distance_fn.distance(s_1, t_1));
        sum += d;
        if sum >= max_dist || s.len() < 3 || t.len() < 3 {
            return sum;
        }
        // Next check the second point at the back.
        let (s_last_1, t_last_1) = (s[s.len() - 2], t[t.len() - 2]);
        d = self
            .distance_fn
            .distance(s_last, t_last_1)
            .min(self.distance_fn.distance(s_last_1, t_last))
            .min(self.distance_fn.distance(s_last_1, t_last_1));
        sum += d;
        // TODO: add some more checks.
        sum
    }

    fn ub_diag(&self, s: &[f64], t: &[f64], max_dist: f64) -> f64 {
        let mut sum = 0.0;
        for (s_i, t_i) in s.iter().copied().zip(t.iter().copied()) {
            let d = self.distance_fn.distance(s_i, t_i);
            sum += d;
            if sum >= max_dist {
                return sum;
            }
        }
        sum
    }

    /// Compute the distance matrix between all pairs of series.
    ///
    /// The series do not all have to be the same length.
    ///
    /// If the `parallel` feature is enabled _and_ the `Dtw` instance has
    /// been configured to use parallelism using [`Dtw::parallelize`],
    /// the calculation is done in parallel.
    ///
    /// # Example
    /// ```
    /// use augurs_dtw::Dtw;
    /// let dtw = Dtw::euclidean();
    /// let series: &[&[f64]] = &[
    ///     &[0.0_f64, 1.0, 2.0],
    ///     &[3.0_f64, 4.0, 5.0],
    ///     &[6.0_f64, 7.0, 8.0],
    /// ];
    /// let dists = dtw.distance_matrix(&series);
    /// assert_eq!(dists[0], vec![0.0, 5.0990195135927845, 10.392304845413264]);
    /// assert_eq!(dists[(0, 1)], 5.0990195135927845);
    /// ```
    pub fn distance_matrix(&self, series: &[&[f64]]) -> DistanceMatrix {
        #[cfg(feature = "parallel")]
        let matrix = if self.parallelize {
            let n = series.len();
            let mut matrix = Vec::with_capacity(n);
            debug!("Calculating distance matrix in parallel");
            series
                .par_iter()
                .map(|s| {
                    series
                        .iter()
                        .map(|t| self.distance_with_early_stopping(s, t))
                        .collect()
                })
                .collect_into_vec(&mut matrix);
            matrix
        } else {
            series
                .iter()
                .map(|s| {
                    series
                        .iter()
                        .map(|t| self.distance_with_early_stopping(s, t))
                        .collect()
                })
                .collect()
        };

        #[cfg(not(feature = "parallel"))]
        let matrix = series
            .iter()
            .map(|s| {
                series
                    .iter()
                    .map(|t| self.distance_with_early_stopping(s, t))
                    .collect()
            })
            .collect();
        DistanceMatrix::try_from_square(matrix).unwrap()
    }
}

#[cfg(test)]
mod test {
    use crate::Dtw;

    #[test]
    fn euclidean() {
        let dtw = Dtw::euclidean();
        let result = dtw.distance(&[0.0, 1.0, 2.0], &[3.0, 4.0, 5.0]);
        assert_eq!(result, 5.0990195135927845);

        let dtw = Dtw::euclidean().with_window(2);
        let result = dtw.distance(&[0.0, 1.0, 2.0], &[3.0, 4.0, 5.0]);
        assert_eq!(result, 5.0990195135927845);

        let dtw = Dtw::euclidean().with_window(2);
        let result = dtw.distance(&[0.0, 1.0, 2.0], &[6.0, 7.0, 8.0]);
        assert_eq!(result, 10.392304845413264);
    }

    #[test]
    fn identical() {
        let dtw = Dtw::euclidean();
        let result = dtw.distance(&[0.0, 1.0, 2.0], &[0.0, 1.0, 2.0]);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn empty() {
        let dtw = Dtw::euclidean();
        let result = dtw.distance(&[], &[0.0, 1.0, 2.0]);
        assert_eq!(result, f64::INFINITY);

        let dtw = Dtw::euclidean();
        let result = dtw.distance(&[], &[3.0, 4.0, 5.0]);
        assert_eq!(result, f64::INFINITY);

        let dtw = Dtw::euclidean();
        let result = dtw.distance(&[3.0, 4.0, 5.0], &[]);
        assert_eq!(result, f64::INFINITY);
    }

    #[test]
    fn max_distance() {
        let a: &[f64] = &[0.0, 1.0, 2.0];
        let b: &[f64] = &[3.0, 4.0, 5.0];
        let dist = Dtw::euclidean().with_max_distance(2.0).distance(a, b);
        assert_eq!(dist, 2.0);
        let dist = Dtw::euclidean().with_max_distance(5.0).distance(a, b);
        assert_eq!(dist, 5.0);
        let dist = Dtw::euclidean().with_max_distance(6.0).distance(a, b);
        assert_eq!(dist, 5.0990195135927845);
    }

    #[test]
    fn different_lengths() {
        let dtw = Dtw::euclidean();
        let result = dtw.distance(&[0.0, 1.0, 2.0], &[3.0, 4.0, 5.0, 6.0]);
        assert_eq!(result, 6.48074069840786);

        let dtw = Dtw::euclidean();
        let result = dtw.distance(&[0.0, 1.0, 2.0, 3.0], &[3.0, 4.0, 5.0]);
        assert_eq!(result, 4.358898943540674);
    }

    #[test]
    fn manhattan() {
        let dtw = Dtw::manhattan().with_window(50);
        let result = dtw.distance(
            &[0., 0., 1., 2., 1., 0., 1., 0., 0.],
            &[0., 1., 2., 0., 0., 0., 0., 0., 0.],
        );
        assert_eq!(result, 2.0);
    }

    #[test]
    fn distance_matrix() {
        let dtw = Dtw::euclidean();
        let series: &[&[f64]] = &[&[0.0, 1.0, 2.0], &[3.0, 4.0, 5.0], &[6.0, 7.0, 8.0]];
        let dists = dtw.distance_matrix(series);
        assert_eq!(dists.shape(), (3, 3));
        assert_eq!(dists[0], vec![0.0, 5.0990195135927845, 10.392304845413264]);

        // Test with different length series.
        let dtw = Dtw::euclidean();
        let series: &[&[f64]] = &[&[0.0, 1.0, 2.0], &[3.0], &[6.0, 7.0]];
        let dists = dtw.distance_matrix(series);
        assert_eq!(dists.shape(), (3, 3));
        assert_eq!(dists[0], vec![0.0, 3.7416573867739413, 9.273618495495704]);
    }

    #[test]
    fn distance_matrix_odd_lengths() {
        // Test with an empty series.
        let dtw = Dtw::euclidean();
        let series: &[&[f64]] = &[&[0.0, 1.0, 2.0], &[3.0, 4.0, 5.0], &[6.0, 7.0, 8.0], &[]];
        let dists = dtw.distance_matrix(series);
        assert_eq!(
            dists.into_inner(),
            vec![
                vec![0.0, 5.0990195135927845, 10.392304845413264, f64::INFINITY],
                vec![5.0990195135927845, 0.0, 5.0990195135927845, f64::INFINITY],
                vec![10.392304845413264, 5.0990195135927845, 0.0, f64::INFINITY],
                vec![f64::INFINITY, f64::INFINITY, f64::INFINITY, 0.0],
            ]
        );

        // Test with different length series.
        let dtw = Dtw::euclidean();
        let series: &[&[f64]] = &[&[0.0, 1.0, 2.0], &[3.0], &[6.0, 7.0]];
        let dists = dtw.distance_matrix(series);
        assert_eq!(dists.shape(), (3, 3));
        assert_eq!(
            dists.into_inner(),
            vec![
                vec![0.0, 3.7416573867739413, 9.273618495495704],
                vec![3.7416573867739413, 0.0, 5.0],
                vec![9.273618495495704, 5.0, 0.0],
            ],
        );
    }

    /// Textbook full-matrix DTW, used as an independent oracle for the banded
    /// implementation in [`Dtw::distance`].
    ///
    /// This is deliberately the naive `O(m * n)` formulation with no windowing,
    /// no early stopping and no `transform_result` trickery, so that it shares no
    /// code (and therefore no bugs) with the real implementation.
    fn reference_euclidean(s: &[f64], t: &[f64]) -> f64 {
        let (m, n) = (s.len(), t.len());
        let mut dp = vec![vec![f64::INFINITY; n + 1]; m + 1];
        dp[0][0] = 0.0;
        for i in 1..=m {
            for j in 1..=n {
                let cost = (s[i - 1] - t[j - 1]).powi(2);
                dp[i][j] = cost + dp[i - 1][j - 1].min(dp[i - 1][j]).min(dp[i][j - 1]);
            }
        }
        dp[m][n].sqrt()
    }

    /// Check the banded implementation against a full-matrix reference over a
    /// deterministic spread of series lengths and shapes.
    ///
    /// Without a window the band covers the whole matrix, so the two must agree.
    #[test]
    fn matches_reference_implementation() {
        // Simple deterministic LCG: avoids a dev-dependency on `rand` while still
        // covering far more shapes than hand-written cases would.
        let mut state = 0x00C0_FFEE_u64;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 11) as f64) / ((1u64 << 53) as f64)
        };

        for len_s in 1..12 {
            for len_t in 1..12 {
                let s: Vec<f64> = (0..len_s).map(|_| next() * 20.0 - 10.0).collect();
                let t: Vec<f64> = (0..len_t).map(|_| next() * 20.0 - 10.0).collect();
                let expected = reference_euclidean(&s, &t);
                let actual = Dtw::euclidean().distance(&s, &t);
                assert!(
                    (actual - expected).abs() <= 1e-9 * expected.abs().max(1.0),
                    "len_s={len_s} len_t={len_t}: expected {expected}, got {actual}\ns={s:?}\nt={t:?}"
                );
            }
        }
    }

    /// A window at least as wide as the longer series must give the same answer as
    /// no window at all, since the band then covers the entire cost matrix.
    #[test]
    fn wide_window_matches_unwindowed() {
        let s: &[f64] = &[0.0, 3.0, -2.0, 5.0, 1.0, 1.0, -4.0];
        let t: &[f64] = &[1.0, 2.0, 2.0, -1.0, 6.0];
        let unwindowed = Dtw::euclidean().distance(s, t);
        for window in [s.len(), s.len() + 1, s.len() * 2, 100] {
            let windowed = Dtw::euclidean().with_window(window).distance(s, t);
            assert_eq!(
                windowed, unwindowed,
                "window={window} should not constrain the path"
            );
        }
    }

    #[test]
    fn distance_matrix_odd_lengths_window_max_distance() {
        // Test with short series with max_distance set.
        let dtw = Dtw::euclidean().with_window(10).with_max_distance(10.0);
        let series: &[&[f64]] = &[&[0.0], &[1.0, 2.0], &[3.0, 4.0, 5.0], &[]];
        let dists = dtw.distance_matrix(series);
        assert_eq!(
            dists.into_inner(),
            vec![
                vec![0.0, 2.23606797749979, 7.0710678118654755, f64::INFINITY],
                vec![2.23606797749979, 0.0, 4.123105625617661, f64::INFINITY],
                vec![7.0710678118654755, 4.123105625617661, 0.0, f64::INFINITY],
                vec![f64::INFINITY, f64::INFINITY, f64::INFINITY, 0.0]
            ]
        );
    }
}
