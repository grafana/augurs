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
}

impl Dtw<Euclidean> {
    /// Create a new DTW instance using Euclidean distance.
    #[must_use]
    pub fn euclidean() -> Self {
        Self {
            window: None,
            distance_fn: Euclidean,
            parallelize: false,
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

    /// Set whether to use parallel computation for the distance matrix computation.
    ///
    /// This does not affect the computation of individual distances using [`Dtw::distance`],
    /// only the computation of the distance matrix using [`Dtw::distance_matrix`].
    pub fn parallelize(mut self, parallelize: bool) -> Self {
        self.parallelize = parallelize;
        self
    }

    /// Compute the distance between two sequences under Dynamic Time Warping.
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
        let max_k = 2 * max_window - 1;
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
                    k += 1;
                    continue;
                }
                z = unsafe { *prev_cost.get_unchecked(k) };
                if k == 0 {
                    y = f64::INFINITY;
                } else {
                    y = cost_k_minus_1;
                }
                let min = if k > max_k {
                    y.min(z)
                } else {
                    x = unsafe { *prev_cost.get_unchecked(k + 1) };
                    x.min(y.min(z))
                };

                let dist = self.distance_fn.distance(s_j, t_i);
                *c = dist + min;
                k += 1;
                cost_k_minus_1 = *c;
            }

            (prev_cost, cost) = (cost, prev_cost);
        }
        k = k.saturating_sub(1);

        self.distance_fn.transform_result(prev_cost[k])
    }

    /// Compute the distance matrix between all pairs of series.
    ///
    /// The series do not all have to be the same length.
    ///
    /// If the `parallel` feature is enabled _and_ the `Dtw` instance has
    /// been configured
    /// of the the calculation is done in parallel.
    pub fn distance_matrix(&self, series: &[&[f64]]) -> DistanceMatrix {
        let n = series.len();

        #[cfg(feature = "parallel")]
        let matrix = if self.parallelize {
            let mut matrix = Vec::with_capacity(n);
            series
                .par_iter()
                .enumerate()
                .map(|(i, s)| {
                    series
                        .iter()
                        // Only calculate the upper diagonal.
                        // The matrix is symmetric, so we can just
                        // copy the values into the lower diagonal.
                        // This is done in the `DistanceMatrix::from_upper_diag` method.
                        .skip(i + 1)
                        .map(|t| self.distance(s, t))
                        .collect()
                })
                .collect_into_vec(&mut matrix);
            matrix
        } else {
            series
                .iter()
                .enumerate()
                .map(|(i, s)| {
                    series
                        .iter()
                        // Only calculate the upper diagonal.
                        // The matrix is symmetric, so we can just
                        // copy the values into the lower diagonal.
                        // This is done in the `DistanceMatrix::from_upper_diag` method.
                        .skip(i + 1)
                        .map(|t| self.distance(s, t))
                        .collect()
                })
                .collect()
        };

        #[cfg(not(feature = "parallel"))]
        let matrix = series
            .iter()
            .enumerate()
            .map(|(i, s)| {
                series
                    .iter()
                    // Only calculate the upper diagonal.
                    // The matrix is symmetric, so we can just
                    // copy the values into the lower diagonal.
                    // This is done in the `DistanceMatrix::from_upper_diag` method.
                    .skip(i + 1)
                    .map(|t| self.distance(s, t))
                    .collect()
            })
            .collect();
        DistanceMatrix::from_upper_diag(matrix)
    }
}

/// A matrix representing the distances between all pairs of series.
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
#[derive(Debug)]
pub struct DistanceMatrix {
    matrix: Vec<Vec<f64>>,
}

impl DistanceMatrix {
    fn from_upper_diag(upper_diag: Vec<Vec<f64>>) -> Self {
        let matrix = upper_diag
            .into_iter()
            .enumerate()
            .map(|(i, x)| std::iter::repeat(0.0).take(i + 1).chain(x).collect())
            .collect();
        Self { matrix }
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
