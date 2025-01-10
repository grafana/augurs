use std::cmp::Ordering;

use num_traits::{Float, FromPrimitive};

/// The result of a call to `nanminmax`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NanMinMaxResult<T> {
    /// The iterator contains multiple distinct float; the minimum and maximum are returned.
    MinMax(T, T),
    /// The iterator contains exactly one distict float, after optionally ignoring NaNs.
    OneElement(T),
    /// The iterator was empty, or was empty after ignoring NaNs.
    NoElements,
    /// The iterator contains at least one NaN value, and NaNs were not ignored.
    ///
    /// This is unreachable if `nanminmax` was called with `ignore_nans: true`.
    NaN,
}

/// Helper trait for calculating summary statistics on floating point iterators with alternative NaN handling.
///
/// This is intended to be similar to numpy's `nanmean`, `nanmin`, `nanmax` etc.
pub trait FloatIterExt<T: Float + FromPrimitive>: Iterator<Item = T> {
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

    /// Returns the minimum and maximum of all elements in the iterator,
    /// handling NaN values.
    ///
    /// If `ignore_nans` is true, NaN values will be ignored and
    /// not included in the minimum or maximum.
    /// Otherwise, the minimum and maximum will be NaN if any element is NaN.
    ///
    /// The return value is a [`NanMinMaxResult`], which is similar to
    /// [`itertools::MinMaxResult`](https://docs.rs/itertools/latest/itertools/enum.MinMaxResult.html)
    /// and provides more granular information on the result.
    ///
    /// # Examples
    ///
    /// ## Simple usage, ignoring NaNs
    ///
    /// ```
    /// use augurs_core::{FloatIterExt, NanMinMaxResult};
    ///
    /// let x = [1.0, 2.0, 3.0, f64::NAN, 5.0];
    /// let min_max = x.iter().copied().nanminmax(true);
    /// assert_eq!(min_max, NanMinMaxResult::MinMax(1.0, 5.0));
    /// ```
    ///
    /// ## Simple usage, including NaNs
    ///
    /// ```
    /// use augurs_core::{FloatIterExt, NanMinMaxResult};
    ///
    /// let x = [1.0, 2.0, 3.0, f64::NAN, 5.0];
    /// let min_max = x.iter().copied().nanminmax(false);
    /// assert_eq!(min_max, NanMinMaxResult::NaN);
    /// ```
    ///
    /// ## Only NaNs
    ///
    /// ```
    /// use augurs_core::{FloatIterExt, NanMinMaxResult};
    ///
    /// let x = [f64::NAN, f64::NAN, f64::NAN];
    /// let min_max = x.iter().copied().nanminmax(true);
    /// assert_eq!(min_max, NanMinMaxResult::NoElements);
    ///
    /// let min_max = x.iter().copied().nanminmax(false);
    /// assert_eq!(min_max, NanMinMaxResult::NaN);
    /// ```
    ///
    /// ## Empty iterator
    ///
    /// ```
    /// use augurs_core::{FloatIterExt, NanMinMaxResult};
    ///
    /// let x: [f64; 0] = [];
    /// let min_max = x.iter().copied().nanminmax(true);
    /// assert_eq!(min_max, NanMinMaxResult::NoElements);
    ///
    /// let min_max = x.iter().copied().nanminmax(false);
    /// assert_eq!(min_max, NanMinMaxResult::NoElements);
    /// ```
    ///
    /// ## Only one distinct element
    ///
    /// ```
    /// use augurs_core::{FloatIterExt, NanMinMaxResult};
    ///
    /// let x = [1.0, f64::NAN, 1.0];
    /// let min_max = x.iter().copied().nanminmax(true);
    /// assert_eq!(min_max, NanMinMaxResult::OneElement(1.0));
    ///
    /// let min_max = x.iter().copied().nanminmax(false);
    /// assert_eq!(min_max, NanMinMaxResult::NaN);
    /// ```
    fn nanminmax(self, ignore_nans: bool) -> NanMinMaxResult<T>
    where
        Self: Sized,
    {
        let mut acc = NanMinMaxResult::NoElements;
        for x in self {
            let is_nan = x.is_nan();
            if is_nan && !ignore_nans {
                return NanMinMaxResult::NaN;
            }
            if is_nan {
                continue;
            }
            // From here on, we're ignoring NaNs.
            acc = match acc {
                NanMinMaxResult::NoElements => NanMinMaxResult::OneElement(x),
                NanMinMaxResult::OneElement(one) => match one.partial_cmp(&x).unwrap() {
                    Ordering::Equal => acc,
                    Ordering::Less => NanMinMaxResult::MinMax(one, x),
                    Ordering::Greater => NanMinMaxResult::MinMax(x, one),
                },
                NanMinMaxResult::MinMax(min, max) => {
                    NanMinMaxResult::MinMax(min.min(x), max.max(x))
                }
                // This case is unreachable because we return early for NaN values when ignore_nans is false
                NanMinMaxResult::NaN => {
                    unreachable!("NaN case should have been handled by early return")
                }
            };
        }
        acc
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
            sum / T::from_usize(n).unwrap_or_else(|| T::nan())
        }
    }
}

impl<T: Float + FromPrimitive, I: Iterator<Item = T>> FloatIterExt<T> for I {}

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

    #[test]
    fn nanmean() {
        let x: &[f64] = &[-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
        assert_eq!(x.iter().copied().nanmean(true), 0.0);

        let y: &[f64] = &[-3.0, -2.0, -1.0, f64::NAN, 1.0, 2.0, 3.0];
        assert_eq!(y.iter().copied().nanmean(true), 0.0);
        assert!(y.iter().copied().nanmean(false).is_nan());

        let z: &[f64] = &[f64::NAN, f64::NAN];
        assert!(z.iter().copied().nanmean(true).is_nan());
    }

    #[test]
    fn nanminmax() {
        let x: &[f64] = &[-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
        assert_eq!(
            x.iter().copied().nanminmax(true),
            NanMinMaxResult::MinMax(-3.0, 3.0)
        );
        assert_eq!(
            x.iter().copied().nanminmax(false),
            NanMinMaxResult::MinMax(-3.0, 3.0)
        );

        let y: &[f64] = &[-3.0, -2.0, -1.0, f64::NAN, 1.0, 2.0, 3.0];
        assert_eq!(
            y.iter().copied().nanminmax(true),
            NanMinMaxResult::MinMax(-3.0, 3.0)
        );
        assert_eq!(y.iter().copied().nanminmax(false), NanMinMaxResult::NaN);

        let z: &[f64] = &[f64::NAN, f64::NAN];
        assert_eq!(
            z.iter().copied().nanminmax(true),
            NanMinMaxResult::NoElements
        );
        assert_eq!(z.iter().copied().nanminmax(false), NanMinMaxResult::NaN);

        let e: &[f64] = &[];
        assert_eq!(
            e.iter().copied().nanminmax(true),
            NanMinMaxResult::NoElements
        );
        assert_eq!(
            e.iter().copied().nanminmax(false),
            NanMinMaxResult::NoElements
        );

        let o: &[f64] = &[1.0, f64::NAN, 1.0];
        assert_eq!(
            o.iter().copied().nanminmax(true),
            NanMinMaxResult::OneElement(1.0),
        );
        assert_eq!(o.iter().copied().nanminmax(false), NanMinMaxResult::NaN);
    }
}
