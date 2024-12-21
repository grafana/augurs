/*!
Contains an interpolation iterator adapter.

The adapter can be used to fill in missing values in a time series
using interpolation, similar to the `interpolate` method on a
`Series` in the `pandas` or `polars` libraries.
*/

use std::{
    collections::VecDeque,
    iter::repeat_with,
    ops::{Add, Div, Mul, Sub},
};

use super::{Error, Transform};

/// A type that can be used to interpolate between values.
pub trait Interpolater {
    /// Interpolate between two values.
    ///
    /// The `low` and `high` values are the start and end of the range to interpolate,
    /// and `n` is the number of values to interpolate.
    ///
    /// The return value is an iterator that yields `n` values between `low` and `high`.
    /// It should return a half-open range, i.e. it should include `low` but not `high`.
    /// It should return exactly `n` values, so if `n` is `1` it should return an iterator
    /// that yields only `low`, and if `n` is `0` it should return an empty iterator.
    fn interpolate<T: Interpolatable>(&self, low: T, high: T, n: usize) -> impl Iterator<Item = T>;
}

/// A linear interpolater.
///
/// This interpolater uses linear interpolation to fill in missing values in a time series.
///
/// # Example
///
/// ```
/// use augurs_forecaster::transforms::interpolate::*;
/// let got = LinearInterpolator::default().interpolate(1.0, 2.0, 4).collect::<Vec<_>>();
/// assert_eq!(got, vec![1.0, 1.25, 1.5, 1.75]);
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct LinearInterpolator {
    _priv: (),
}

impl LinearInterpolator {
    /// Create a new `LinearInterpolator`.
    pub fn new() -> Self {
        Self::default()
    }
}

impl Interpolater for LinearInterpolator {
    fn interpolate<T: Interpolatable>(&self, low: T, high: T, n: usize) -> impl Iterator<Item = T> {
        let diff = high - low;
        let step = diff / (T::from_usize(n));
        (0..n).map(move |i| low + T::from_usize(i) * step)
    }
}

impl Transform for LinearInterpolator {
    fn transform(&mut self, data: &mut [f64]) -> Result<(), Error> {
        let interpolated: Vec<_> = data.iter().copied().interpolate(*self).collect();
        data.copy_from_slice(&interpolated);
        Ok(())
    }

    fn inverse_transform(&self, _data: &mut [f64]) -> Result<(), Error> {
        Ok(())
    }
}

/// An iterator that interpolates between NaN values in the input.
///
/// This iterator is used to fill in missing values in a time series by
/// linearly interpolating between the nearest defined values.
/// The iterator will yield the same number of values as the input, but
/// with any NaN values replaced by interpolated values.
///
/// If the first or last value in the input is NaN, the iterator will
/// yield NaN values at the start or end of the output, respectively.
///
/// # Example
/// ```
/// use augurs_forecaster::transforms::interpolate::*;
/// let x = vec![1.0, f32::NAN, f32::NAN, f32::NAN, 2.0];
/// let interp: Vec<_> = x.into_iter().interpolate(LinearInterpolator::default()).collect();
/// assert_eq!(interp, vec![1.0, 1.25, 1.5, 1.75, 2.0]);
/// ```
#[derive(Debug, Clone)]
pub struct Interpolate<T: Iterator, I> {
    inner: T,
    low: T::Item,
    high: Option<T::Item>,
    buf: VecDeque<T::Item>,
    interpolator: I,
}

impl<T, I> Iterator for Interpolate<T, I>
where
    T: Iterator,
    T::Item: Interpolatable,
    I: Interpolater,
{
    type Item = T::Item;

    fn next(&mut self) -> Option<Self::Item> {
        // If we have values in the buffer, use them first.
        if !self.buf.is_empty() {
            return self.buf.pop_front();
        }

        // If we have a high value from the previous iteration, use it and
        // reset the high value to None, so that we don't use it again.
        if let Some(high) = self.high.take() {
            self.low = high;
            return Some(high);
        }

        let next = self.inner.next();
        match next {
            Some(x) if x.is_nan() => {
                // Count the number of NaNs we see, starting with this one (`x`).
                let mut n: usize = 1;
                for h in self.inner.by_ref() {
                    if h.is_nan() {
                        n += 1;
                        continue;
                    }
                    // h is not NaN.
                    self.high = Some(h);
                    break;
                }

                if self.low.is_nan() {
                    // We've seen NaNs at the start.
                    self.buf = repeat_with(Self::Item::nan).take(n - 1).collect();
                    return Some(self.low);
                }

                if let Some(high) = self.high {
                    // Here we've seen NaNs in between some defined values, so we
                    // can interpolate.
                    let mut iter = self
                        .interpolator
                        // We need to interpolate `n + 1` values, because `n` doesn't
                        // include the last non-NaN value which `interpolate` expects
                        // to be included.
                        .interpolate(self.low, high, n + 1)
                        // Limit the number of values we yield to `n` since we know we need
                        // that many NaNs but can't ensure that downstream implementors of
                        // `Interpolater` respect that.
                        .take(n + 1)
                        // Skip the first value, which is the low value we've already seen.
                        .skip(1);
                    let first = iter.next();
                    self.buf = iter.collect();
                    first
                } else {
                    // We've seen NaNs at the end. Fill the buffer with NaNs to be
                    // used by any subsequent calls to `next`.
                    self.buf = repeat_with(Self::Item::nan).take(n - 1).collect();
                    Some(T::Item::nan())
                }
            }
            Some(x) => {
                // We've seen a defined value, so we can store it as the low value
                // for the next iteration and yield it.
                self.low = x;
                Some(x)
            }
            // We've reached the end of the input.
            None => None,
        }
    }
}

/// An extension trait for iterators that adds the `interpolation` method.
pub trait InterpolateExt: Iterator {
    /// Interpolate between NaN values in the input.
    ///
    /// Returns an iterator that yields the same number of values as the input,
    /// but with any NaN values replaced by linearly interpolated values.
    ///
    /// If the first or last value in the input is NaN, the iterator will
    /// yield NaN values at the start or end of the output, respectively.
    ///
    /// # Example
    /// ```
    /// use augurs_forecaster::transforms::interpolate::*;
    /// let x = vec![1.0, f32::NAN, f32::NAN, f32::NAN, 2.0];
    /// let interp: Vec<_> = x.into_iter().interpolate(LinearInterpolator::default()).collect();
    /// assert_eq!(interp, vec![1.0, 1.25, 1.5, 1.75, 2.0]);
    /// ```
    fn interpolate<I>(self, method: I) -> Interpolate<Self, I>
    where
        Self: Sized,
        Self::Item: Interpolatable + Sized,
        I: Interpolater,
    {
        Interpolate {
            inner: self,
            low: Self::Item::nan(),
            high: None,
            buf: VecDeque::new(),
            interpolator: method,
        }
    }
}

impl<T> InterpolateExt for T where T: Iterator {}

/// A trait for types that can be interpolated.
///
/// This is used to abstract over various types that can be interpolated.
/// It is implemented for `f32` and `f64`, but can be implemented for more
/// types if necessary.
pub trait Interpolatable:
    Add<Self, Output = Self>
    + Div<Self, Output = Self>
    + Mul<Self, Output = Self>
    + Sub<Self, Output = Self>
    + Copy
    + Default
    + Sized
{
    /// Return a NaN value of the type.
    fn nan() -> Self;

    /// Check if the value is NaN.
    fn is_nan(&self) -> bool;

    /// Convert a `usize` to the type.
    fn from_usize(x: usize) -> Self;
}

impl Interpolatable for f32 {
    fn nan() -> Self {
        f32::NAN
    }
    fn is_nan(&self) -> bool {
        f32::is_nan(*self)
    }
    fn from_usize(x: usize) -> Self {
        x as f32
    }
}

impl Interpolatable for f64 {
    fn nan() -> Self {
        f64::NAN
    }
    fn is_nan(&self) -> bool {
        f64::is_nan(*self)
    }
    fn from_usize(x: usize) -> Self {
        x as f64
    }
}

#[cfg(test)]
mod test {
    use super::*;

    fn assert_approx_eq(a: f32, b: f32) -> bool {
        if a.is_nan() && b.is_nan() {
            return true;
        }
        (a - b).abs() < f32::EPSILON
    }

    fn assert_all_approx_eq(a: &[f32], b: &[f32]) {
        if a.len() != b.len() {
            assert_eq!(a, b);
        }
        for (ai, bi) in a.iter().zip(b) {
            if !assert_approx_eq(*ai, *bi) {
                assert_eq!(a, b);
            }
        }
    }

    #[test]
    fn linear_interpreter() {
        let got = LinearInterpolator::default()
            .interpolate(1.0, 2.0, 4)
            .collect::<Vec<_>>();
        assert_eq!(got, vec![1.0, 1.25, 1.5, 1.75]);
    }

    #[test]
    fn all_nan() {
        let x = vec![f32::NAN, f32::NAN, f32::NAN];
        let interp: Vec<_> = x
            .clone()
            .into_iter()
            .interpolate(LinearInterpolator::default())
            .collect();
        assert_all_approx_eq(&interp, &x);
    }

    #[test]
    fn empty() {
        let x: Vec<f32> = vec![];
        let interp: Vec<_> = x
            .clone()
            .into_iter()
            .interpolate(LinearInterpolator::default())
            .collect();
        assert_all_approx_eq(&interp, &x);
    }

    #[test]
    fn all_defined() {
        let x = vec![1.0, 2.0, 3.0];
        let interp: Vec<_> = x
            .clone()
            .into_iter()
            .interpolate(LinearInterpolator::default())
            .collect();
        assert_all_approx_eq(&interp, &x);
    }

    #[test]
    fn nans_in_middle() {
        let x = vec![1.0, f32::NAN, f32::NAN, f32::NAN, 2.0];
        let interp: Vec<_> = x
            .clone()
            .into_iter()
            .interpolate(LinearInterpolator::default())
            .collect();
        assert_all_approx_eq(&interp, &[1.0, 1.25, 1.5, 1.75, 2.0]);
    }

    #[test]
    fn nans_at_start() {
        let x = vec![f32::NAN, f32::NAN, 1.0, f32::NAN, f32::NAN, f32::NAN, 2.0];
        let interp: Vec<_> = x
            .clone()
            .into_iter()
            .interpolate(LinearInterpolator::default())
            .collect();
        assert_all_approx_eq(&interp, &[f32::NAN, f32::NAN, 1.0, 1.25, 1.5, 1.75, 2.0]);
    }

    #[test]
    fn nans_at_end() {
        let x = vec![1.0, f32::NAN, f32::NAN, f32::NAN, 2.0, f32::NAN, f32::NAN];
        let interp: Vec<_> = x
            .clone()
            .into_iter()
            .interpolate(LinearInterpolator::default())
            .collect();
        assert_all_approx_eq(&interp, &[1.0, 1.25, 1.5, 1.75, 2.0, f32::NAN, f32::NAN]);
    }

    #[test]
    fn one_nan() {
        let x = vec![0.0, 1.0, f32::NAN, 2.0, 3.0];
        let interp: Vec<_> = x
            .clone()
            .into_iter()
            .interpolate(LinearInterpolator::default())
            .collect();
        assert_all_approx_eq(&interp, &[0.0, 1.0, 1.5, 2.0, 3.0]);
    }

    #[test]
    fn one_value() {
        let x = vec![1.0];
        let interp: Vec<_> = x
            .clone()
            .into_iter()
            .interpolate(LinearInterpolator::default())
            .collect();
        assert_all_approx_eq(&interp, &x);
    }

    #[test]
    fn one_value_amongst_nans() {
        let x = vec![f32::NAN, f32::NAN, 1.0, f32::NAN, f32::NAN];
        let interp: Vec<_> = x
            .clone()
            .into_iter()
            .interpolate(LinearInterpolator::default())
            .collect();
        assert_all_approx_eq(&interp, &x);
    }

    #[test]
    fn one_value_before_nans() {
        let x = vec![1.0, f32::NAN, f32::NAN, f32::NAN, f32::NAN];
        let interp: Vec<_> = x
            .clone()
            .into_iter()
            .interpolate(LinearInterpolator::default())
            .collect();
        assert_all_approx_eq(&interp, &x);
    }

    #[test]
    fn one_value_after_nans() {
        let x = vec![f32::NAN, f32::NAN, f32::NAN, f32::NAN, 1.0];
        let interp: Vec<_> = x
            .clone()
            .into_iter()
            .interpolate(LinearInterpolator::default())
            .collect();
        assert_all_approx_eq(&interp, &x);
    }

    #[test]
    fn everything() {
        let x = vec![
            f32::NAN,
            f32::NAN,
            1.0,
            f32::NAN,
            f32::NAN,
            f32::NAN,
            2.0,
            f32::NAN,
            f32::NAN,
        ];
        let interp: Vec<_> = x
            .clone()
            .into_iter()
            .interpolate(LinearInterpolator::default())
            .collect();
        assert_all_approx_eq(
            &interp,
            &[
                f32::NAN,
                f32::NAN,
                1.0,
                1.25,
                1.5,
                1.75,
                2.0,
                f32::NAN,
                f32::NAN,
            ],
        );
    }
}
