#![doc = include_str!("../README.md")]

use std::ops::Index;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// A trait for defining a distance function.
///
/// This trait is used to define the distance function used in the DTW algorithm.
/// Examples of distance functions are Euclidean distance and Manhattan distance.
pub trait Distance {
    /// Compute the distance between two points.
    ///
    /// E.g. Euclidean distance or Manhattan distance.
    fn distance(&self, a: f64, b: f64) -> f64;

    /// Transform the final distance.
    ///
    /// E.g. Square root for Euclidean distance, or identity
    /// for Manhattan distance.
    fn transform_result(&self, dist: f64) -> f64;
}

/// Euclidean distance, also known as L2 distance.
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
#[derive(Debug)]
pub struct Manhattan;

impl Distance for Manhattan {
    #[inline]
    fn distance(&self, a: f64, b: f64) -> f64 {
        f64::abs(a - b)
    }

    #[inline]
    fn transform_result(&self, dist: f64) -> f64 {
        dist
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
#[derive(Debug)]
pub struct Dtw<T: Distance + Send + Sync> {
    window: Option<usize>,
    distance_fn: T,
    parallelize: bool,

    max_distance: Option<f64>,
}

impl Dtw<Euclidean> {
    /// Create a new DTW instance using Euclidean distance.
    #[must_use]
    pub fn euclidean() -> Self {
        Self {
            window: None,
            distance_fn: Euclidean,
            parallelize: false,
            max_distance: None,
        }
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
        Self {
            distance_fn: Manhattan,
            window: None,
            parallelize: false,
            max_distance: None,
        }
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
    /// If the distance between two series exceeds this value, the computation will stop
    /// and return `max_distance`.
    #[must_use]
    pub fn with_max_distance(mut self, max_distance: f64) -> Self {
        self.max_distance = Some(max_distance);
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
        let mut max_cost = 0.0_f64;
        let (mut cost, mut prev_cost) = (
            vec![f64::INFINITY; 2 * max_window + 1],
            vec![f64::INFINITY; 2 * max_window + 1],
        );
        let (mut x, mut y, mut z);

        let mut k = 0;

        for (i, t_i) in outer_iter.copied().enumerate() {
            k = max_window.saturating_sub(i);

            let lower_bound = i.saturating_sub(max_window);
            let upper_bound = usize::min(n - 1, i + max_window);
            let mut min_cost = f64::INFINITY;

            let mut cost_k_minus_1 = f64::INFINITY;
            for ((j, s_j), c) in inner_iter
                .clone()
                .skip(lower_bound)
                .take(upper_bound - lower_bound + 1)
                .copied()
                .enumerate()
                .zip(&mut cost[k..])
            {
                if i == 0 && j == 0 {
                    *c = self.distance_fn.distance(s_j, t_i);
                    cost_k_minus_1 = *c;
                    min_cost = min_cost.max(*c);
                    k += 1;
                    continue;
                }
                // SAFETY: prev_cost has length 2 * max_window + 1,
                // and k is always in the range 0..=2 * max_window
                // since we start at k = max_window and increment it
                // by 1 each iteration, and the inner loop runs
                // `max_window` times.
                z = unsafe { *prev_cost.get_unchecked(k) };
                if k == 0 {
                    y = f64::INFINITY;
                } else {
                    y = cost_k_minus_1;
                }
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

            max_cost = max_cost.max(min_cost);
            if max_distance_transformed.map_or(false, |d| max_cost >= d) {
                return self.max_distance.unwrap();
            }
            (prev_cost, cost) = (cost, prev_cost);
        }
        k = k.saturating_sub(1);

        self.distance_fn.transform_result(prev_cost[k])
    }

    /// Compute the distance between two sequences under Dynamic Time Warping with early stopping.
    ///
    /// If `max_distance` is `None`, this just calls [`Dtw::distance`].
    ///
    /// Otherwise, it cascades through various lower bounds on the distance,
    /// stopping early if any of the lower bounds exceed the `max_distance`.
    /// Only if none of the lower bounds exceed the `max_distance` is the full
    /// DTW distance computed.
    ///
    /// This is useful in many cases:
    /// - when clustering with DBSCAN we only care if the distance is <= epsilon,
    ///   since that is the only condition for two series to be considered neighbours.
    ///   Therefore if we know that the lower bound of the distance is already > epsilon,
    ///   we can stop early. Similarly, if we know that the upper bound is < epsilon,
    ///   we can also stop early.
    fn distance_with_early_stopping(&self, s: &[f64], t: &[f64]) -> f64 {
        let Some(max_distance) = self.max_distance else {
            return self.distance(s, t);
        };
        // LB_kim is constant time so try that first.
        if self.lb_kim(s, t, max_distance) >= max_distance {
            return max_distance;
        }
        // // LB_keogh is linear time.
        // if lb_keogh(s, t) >= max_distance {
        //     return max_distance;
        // }
        if self.ub_diag(s, t, max_distance) < max_distance {
            return max_distance;
        }
        self.distance(s, t)
    }

    fn lb_kim(&self, s: &[f64], t: &[f64], max_dist: f64) -> f64 {
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
        if sum >= max_dist {
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
    /// been configured
    /// of the the calculation is done in parallel.
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

#[derive(Debug)]
pub enum Error {
    InvalidDistanceMatrix,
}

/// A matrix representing the distances between all pairs of series.
///
/// This is produced by [`Dtw::distance_matrix`].
#[derive(Debug)]
pub struct DistanceMatrix {
    matrix: Vec<Vec<f64>>,
}

impl DistanceMatrix {
    pub fn try_from_square(matrix: Vec<Vec<f64>>) -> Result<Self, Error> {
        if matrix.iter().all(|x| x.len() == matrix.len()) {
            Ok(Self { matrix })
        } else {
            Err(Error::InvalidDistanceMatrix)
        }
    }

    /// Consumes the `DistanceMatrix` and returns the inner matrix.
    pub fn into_inner(self) -> Vec<Vec<f64>> {
        self.matrix
    }

    /// Returns an iterator over the rows of the matrix.
    pub fn iter(&self) -> DistanceMatrixIter<'_> {
        DistanceMatrixIter {
            iter: self.matrix.iter(),
        }
    }

    /// Returns the shape of the matrix.
    ///
    /// The first element is the number of rows and the second element
    /// is the number of columns.
    ///
    /// The matrix is square, so the number of rows is equal to the number of columns
    /// and the number of input series.
    pub fn shape(&self) -> (usize, usize) {
        (self.matrix.len(), self.matrix.len())
    }
}

impl Index<usize> for DistanceMatrix {
    type Output = [f64];
    fn index(&self, index: usize) -> &Self::Output {
        &self.matrix[index]
    }
}

impl Index<(usize, usize)> for DistanceMatrix {
    type Output = f64;
    fn index(&self, (i, j): (usize, usize)) -> &Self::Output {
        &self.matrix[i][j]
    }
}

impl IntoIterator for DistanceMatrix {
    type Item = Vec<f64>;
    type IntoIter = std::vec::IntoIter<Self::Item>;
    fn into_iter(self) -> Self::IntoIter {
        self.matrix.into_iter()
    }
}

/// An iterator over the rows of a `DistanceMatrix`.
pub struct DistanceMatrixIter<'a> {
    iter: std::slice::Iter<'a, Vec<f64>>,
}

impl<'a> Iterator for DistanceMatrixIter<'a> {
    type Item = &'a Vec<f64>;
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
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
}
