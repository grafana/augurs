use num_traits::Float;

pub(crate) trait FloatIterExt<T: Float>: Iterator<Item = T> {
    /// Returns the minimum of all elements in the iterator, handling NaN values.
    ///
    /// If `ignore_nans` is true, NaN values will be ignored and
    /// not included in the minimum.
    /// Otherwise, the minimum will be NaN if any element is NaN.
    fn nanmin(self, ignore_nans: bool) -> T
    where
        Self: Sized,
    {
        self.reduce(|acc, x| {
            if ignore_nans && x.is_nan() {
                acc
            } else if x.is_nan() || acc.is_nan() {
                T::nan()
            } else {
                acc.min(x)
            }
        })
        .unwrap_or_else(T::nan)
    }

    /// Returns the maximum of all elements in the iterator, handling NaN values.
    ///
    /// If `ignore_nans` is true, NaN values will be ignored and
    /// not included in the maximum.
    /// Otherwise, the maximum will be NaN if any element is NaN.
    fn nanmax(self, ignore_nans: bool) -> T
    where
        Self: Sized,
    {
        self.reduce(|acc, x| {
            if ignore_nans && x.is_nan() {
                acc
            } else if x.is_nan() || acc.is_nan() {
                T::nan()
            } else {
                acc.max(x)
            }
        })
        .unwrap_or_else(T::nan)
    }

    /// Returns the mean of all elements in the iterator, handling NaN values.
    ///
    /// If `ignore_nans` is true, NaN values will be ignored and
    /// not included in the mean.
    /// Otherwise, the mean will be NaN if any element is NaN.
    fn nanmean(self, ignore_nans: bool) -> T
    where
        Self: Sized,
    {
        let (n, sum) = self.fold((0, T::zero()), |(n, sum), x| {
            if ignore_nans && x.is_nan() {
                (n, sum)
            } else if x.is_nan() || sum.is_nan() {
                (n, T::nan())
            } else {
                (n + 1, sum + x)
            }
        });
        if n == 0 {
            T::nan()
        } else if sum.is_nan() {
            sum
        } else {
            sum / T::from(n).unwrap()
        }
    }
}

impl<T: Float, I: Iterator<Item = T>> FloatIterExt<T> for I {}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn empty() {
        let x: &[f64] = &[];
        assert!(x.iter().copied().nanmin(true).is_nan());
        assert!(x.iter().copied().nanmax(true).is_nan());
    }

    #[test]
    fn no_nans() {
        let x: &[f64] = &[-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
        assert_eq!(x.iter().copied().nanmin(true), -3.0);
        assert_eq!(x.iter().copied().nanmax(true), 3.0);
        assert_eq!(x.iter().copied().nanmin(false), -3.0);
        assert_eq!(x.iter().copied().nanmax(false), 3.0);
    }

    #[test]
    fn nans() {
        let x: &[f64] = &[-3.0, -2.0, -1.0, f64::NAN, 1.0, 2.0, 3.0];
        assert_eq!(x.iter().copied().nanmin(true), -3.0);
        assert_eq!(x.iter().copied().nanmax(true), 3.0);

        assert!(x.iter().copied().nanmin(false).is_nan());
        assert!(x.iter().copied().nanmax(false).is_nan());
    }
}
