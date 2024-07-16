#![doc = include_str!("../README.md")]

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
pub struct Dtw<T: Distance> {
    window: Option<usize>,
    distance_fn: T,
}

impl Dtw<Euclidean> {
    /// Create a new DTW instance using Euclidean distance.
    #[must_use]
    pub fn euclidean() -> Self {
        Dtw {
            window: None,
            distance_fn: Euclidean,
        }
    }
}

impl Dtw<Manhattan> {
    /// Create a new DTW instance using Manhattan distance.
    #[must_use]
    pub fn manhattan() -> Self {
        Dtw {
            window: None,
            distance_fn: Manhattan,
        }
    }
}

impl<T: Distance> Dtw<T> {
    /// Create a new DTW instance with a custom distance function.
    #[must_use]
    pub fn new(distance_fn: T) -> Self {
        Dtw {
            window: None,
            distance_fn,
        }
    }
}

impl<T: Distance> Dtw<T> {
    /// Get the size of the Sakoe-Chiba warping window, if set.
    pub fn window(&self) -> Option<usize> {
        self.window
    }

    /// Set the size of the Sakoe-Chiba warping window, `w`.
    ///
    /// Using a window limits shifts up to this amount away from the diagonal.
    pub fn with_window(mut self, window: usize) -> Self {
        self.window = Some(window);
        self
    }

    /// Compute the distance between two sequences under Dynamic Time Warping.
    pub fn distance(&self, s: &[f64], t: &[f64]) -> f64 {
        let m = s.len();
        let n = t.len();
        let max_window = self
            .window
            .map(|w| w.max(n.abs_diff(m)))
            .unwrap_or(m.max(n));
        let max_k = 2 * max_window - 1;
        let (mut cost, mut prev_cost) = (
            vec![f64::INFINITY; 2 * max_window + 1],
            vec![f64::INFINITY; 2 * max_window + 1],
        );
        let (mut x, mut y, mut z);

        let mut k = 0;

        for (i, t_i) in t.iter().copied().enumerate() {
            k = max_window.saturating_sub(i);

            let lower_bound = i.saturating_sub(max_window);
            let upper_bound = usize::min(n - 1, i + max_window);

            let mut cost_k_minus_1 = f64::INFINITY;
            for ((j, s_j), c) in s
                .iter()
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
                let min;
                if k > max_k {
                    min = y.min(z);
                } else {
                    x = unsafe { *prev_cost.get_unchecked(k + 1) };
                    min = x.min(y.min(z));
                }

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
}

impl Default for Dtw<Euclidean> {
    fn default() -> Self {
        Dtw::euclidean()
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
}
