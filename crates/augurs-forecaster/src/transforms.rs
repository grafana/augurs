/*!
Data transformations.
*/

mod power;
mod scale;

use argmin::core::Error;
use augurs_core::{
    interpolate::{InterpolateExt, LinearInterpolator},
    Forecast,
};

use power::{
    optimize_box_cox_lambda, optimize_yeo_johnson_lambda, BoxCoxExt, InverseBoxCoxExt,
    InverseYeoJohnsonExt, YeoJohnsonExt,
};
use scale::{InverseMinMaxScaleExt, InverseStandardScaleExt, MinMaxScaleExt, StandardScaleExt};
pub use scale::{MinMaxScaleParams, StandardScaleParams};

/// Transforms and Transform implementations.
///
/// The `Transforms` struct is a collection of `Transform` instances that can be applied to a time series.
/// The `Transform` enum represents a single transformation that can be applied to a time series.
#[derive(Debug, Default)]
pub(crate) struct Transforms(Vec<Transform>);

impl Transforms {
    /// create a new `Transforms` instance with the given transforms.
    pub(crate) fn new(transforms: Vec<Transform>) -> Self {
        Self(transforms)
    }

    /// Apply the transformations to the given time series.
    pub(crate) fn transform<'a, T>(&'a self, input: T) -> Box<dyn Iterator<Item = f64> + 'a>
    where
        T: Iterator<Item = f64> + 'a,
    {
        self.0
            .iter()
            .fold(Box::new(input) as _, |y, t| t.transform(y))
    }

    /// Apply the inverse transformations to the given forecast.
    pub(crate) fn inverse_transform(&self, forecast: Forecast) -> Forecast {
        self.0
            .iter()
            .rev()
            .fold(forecast, |f, t| t.inverse_transform_forecast(f))
    }
}

/// A transformation that can be applied to a time series.
#[derive(Debug)]
#[non_exhaustive]
pub enum Transform {
    /// Linear interpolation.
    ///
    /// This can be used to fill in missing values in a time series
    /// by interpolating between the nearest non-missing values.
    LinearInterpolator,
    /// Min-max scaling.
    MinMaxScaler(MinMaxScaleParams),
    /// Standard scaling.
    StandardScaler(StandardScaleParams),
    /// Logit transform.
    Logit,
    /// Log transform.
    Log,
    /// Box-Cox transform.
    BoxCox {
        /// The lambda parameter for the Box-Cox transformation.
        /// If lambda == 0, the transformation is equivalent to the natural logarithm.
        /// Otherwise, the transformation is (x^lambda - 1) / lambda.
        lambda: f64,
    },
    /// Yeo-Johnson transform.
    YeoJohnson {
        /// The lambda parameter for the Yeo-Johnson transformation.
        /// If lambda == 0, the transformation is equivalent to the natural logarithm.
        /// Otherwise, the transformation is ((x + 1)^lambda - 1) / lambda.
        lambda: f64,
    },
}

impl Transform {
    /// Create a new linear interpolator.
    ///
    /// This interpolator uses linear interpolation to fill in missing values.
    pub fn linear_interpolator() -> Self {
        Self::LinearInterpolator
    }

    /// Create a new min-max scaler.
    ///
    /// This scaler scales each item to the range [0, 1].
    ///
    /// Because transforms operate on iterators, the data min and max must be passed for now.
    /// This also allows for the possibility of using different min and max values; for example,
    /// if you know that the true possible min and max of your data differ from the sample.
    pub fn min_max_scaler(min_max_params: MinMaxScaleParams) -> Self {
        Self::MinMaxScaler(min_max_params)
    }

    /// Create a new standard scaler.
    ///
    /// This scaler standardizes features by removing the mean and scaling to unit variance.
    ///
    /// The standard score of a sample x is calculated as:
    ///
    /// ```text
    /// z = (x - u) / s
    /// ```
    ///
    /// where u is the mean and s is the standard deviation in the provided
    /// `StandardScaleParams`.
    pub fn standard_scaler(scale_params: StandardScaleParams) -> Self {
        Self::StandardScaler(scale_params)
    }

    /// Create a new logit transform.
    ///
    /// This transform applies the logit function to each item.
    pub fn logit() -> Self {
        Self::Logit
    }

    /// Create a new log transform.
    ///
    /// This transform applies the natural logarithm to each item.
    pub fn log() -> Self {
        Self::Log
    }

    /// Create a new Box-Cox transform.
    ///
    /// This transform applies the Box-Cox transformation to each item.
    ///
    /// The Box-Cox transformation is defined as:
    ///
    /// - if lambda == 0: x.ln()
    /// - otherwise: (x^lambda - 1) / lambda
    pub fn box_cox(lambda: f64) -> Self {
        Self::BoxCox { lambda }
    }

    /// Create a new Yeo-Johnson transform.
    ///
    /// This transform applies the Yeo-Johnson transformation to each item.
    ///
    /// The Yeo-Johnson transformation is a generalization of the Box-Cox transformation that
    /// supports negative values. It is defined as:
    ///
    /// - if lambda != 0 and x >= 0: ((x + 1)^lambda - 1) / lambda
    /// - if lambda == 0 and x >= 0: (x + 1).ln()
    /// - if lambda != 2 and x < 0:  ((-x + 1)^2 - 1) / 2
    /// - if lambda == 2 and x < 0:  (-x + 1).ln()
    pub fn yeo_johnson(lambda: f64) -> Self {
        Self::YeoJohnson { lambda }
    }

    /// Create a power transform that optimizes the lambda parameter.
    ///
    /// # Algorithm Selection
    ///
    /// - If all values are positive: Uses Box-Cox transformation
    /// - If any values are negative or zero: Uses Yeo-Johnson transformation
    ///
    /// # Returns
    ///
    /// Returns `Result<Self, Error>` to handle optimization failures gracefully
    pub fn power_transform(data: &[f64]) -> Result<Self, Error> {
        if data.iter().all(|&x| x > 0.0) {
            optimize_box_cox_lambda(data).map(|lambda| Self::BoxCox { lambda })
        } else {
            optimize_yeo_johnson_lambda(data).map(|lambda| Self::YeoJohnson { lambda })
        }
    }

    /// Apply the transformation to the given time series.
    ///
    /// # Returns
    ///
    /// A boxed iterator over the transformed values.
    ///
    /// # Example
    ///
    /// ```
    /// use augurs_forecaster::transforms::Transform;
    ///
    /// let data = vec![1.0, 2.0, 3.0];
    /// let transform = Transform::log();
    /// let transformed: Vec<_> = transform.transform(data.into_iter()).collect();
    /// ```
    pub fn transform<'a, T>(&'a self, input: T) -> Box<dyn Iterator<Item = f64> + 'a>
    where
        T: Iterator<Item = f64> + 'a,
    {
        match self {
            Self::LinearInterpolator => Box::new(input.interpolate(LinearInterpolator::default())),
            Self::MinMaxScaler(params) => Box::new(input.min_max_scale(params)),
            Self::StandardScaler(params) => Box::new(input.standard_scale(params)),
            Self::Logit => Box::new(input.logit()),
            Self::Log => Box::new(input.log()),
            Self::BoxCox { lambda } => Box::new(input.box_cox(*lambda)),
            Self::YeoJohnson { lambda } => Box::new(input.yeo_johnson(*lambda)),
        }
    }

    /// Apply the inverse transformation to the given time series.
    ///
    /// # Returns
    ///
    /// A boxed iterator over the inverse transformed values.
    ///
    /// # Example
    ///
    /// ```
    /// use augurs_forecaster::transforms::Transform;
    ///
    /// let data = vec![1.0, 2.0, 3.0];
    /// let transform = Transform::log();
    /// let transformed: Vec<_> = transform.inverse_transform(data.into_iter()).collect();
    /// ```
    pub fn inverse_transform<'a, T>(&'a self, input: T) -> Box<dyn Iterator<Item = f64> + 'a>
    where
        T: Iterator<Item = f64> + 'a,
    {
        match self {
            Self::LinearInterpolator => Box::new(input),
            Self::MinMaxScaler(params) => Box::new(input.inverse_min_max_scale(params)),
            Self::StandardScaler(params) => Box::new(input.inverse_standard_scale(params)),
            Self::Logit => Box::new(input.logistic()),
            Self::Log => Box::new(input.exp()),
            Self::BoxCox { lambda } => Box::new(input.inverse_box_cox(*lambda)),
            Self::YeoJohnson { lambda } => Box::new(input.inverse_yeo_johnson(*lambda)),
        }
    }

    /// Apply the inverse transformations to the given forecast.
    pub fn inverse_transform_forecast(&self, mut f: Forecast) -> Forecast {
        f.point = self.inverse_transform(f.point.into_iter()).collect();
        if let Some(mut intervals) = f.intervals.take() {
            intervals.lower = self
                .inverse_transform(intervals.lower.into_iter())
                .collect();
            intervals.upper = self
                .inverse_transform(intervals.upper.into_iter())
                .collect();
            f.intervals = Some(intervals);
        }
        f
    }
}

// Actual implementations of the transforms.
// These may be moved to a separate module or crate in the future.

// Logit and logistic functions.

/// Returns the logistic function of the given value.
fn logistic(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Returns the logit function of the given value.
fn logit(x: f64) -> f64 {
    (x / (1.0 - x)).ln()
}

/// An iterator adapter that applies the logit function to each item.
#[derive(Clone, Debug)]
struct Logit<T> {
    inner: T,
}

impl<T> Iterator for Logit<T>
where
    T: Iterator<Item = f64>,
{
    type Item = f64;
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(logit)
    }
}

trait LogitExt: Iterator<Item = f64> {
    fn logit(self) -> Logit<Self>
    where
        Self: Sized,
    {
        Logit { inner: self }
    }
}

impl<T> LogitExt for T where T: Iterator<Item = f64> {}

/// An iterator adapter that applies the logistic function to each item.
#[derive(Clone, Debug)]
struct Logistic<T> {
    inner: T,
}

impl<T> Iterator for Logistic<T>
where
    T: Iterator<Item = f64>,
{
    type Item = f64;
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(logistic)
    }
}

trait LogisticExt: Iterator<Item = f64> {
    fn logistic(self) -> Logistic<Self>
    where
        Self: Sized,
    {
        Logistic { inner: self }
    }
}

impl<T> LogisticExt for T where T: Iterator<Item = f64> {}

/// An iterator adapter that applies the log function to each item.
#[derive(Clone, Debug)]
struct Log<T> {
    inner: T,
}

impl<T> Iterator for Log<T>
where
    T: Iterator<Item = f64>,
{
    type Item = f64;
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(f64::ln)
    }
}

trait LogExt: Iterator<Item = f64> {
    fn log(self) -> Log<Self>
    where
        Self: Sized,
    {
        Log { inner: self }
    }
}

impl<T> LogExt for T where T: Iterator<Item = f64> {}

/// An iterator adapter that applies the exponential function to each item.
#[derive(Clone, Debug)]
struct Exp<T> {
    inner: T,
}

impl<T> Iterator for Exp<T>
where
    T: Iterator<Item = f64>,
{
    type Item = f64;
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(f64::exp)
    }
}

trait ExpExt: Iterator<Item = f64> {
    fn exp(self) -> Exp<Self>
    where
        Self: Sized,
    {
        Exp { inner: self }
    }
}

impl<T> ExpExt for T where T: Iterator<Item = f64> {}

#[cfg(test)]
mod test {
    use augurs_testing::assert_approx_eq;

    use super::*;

    #[test]
    fn test_logistic() {
        let x = 0.0;
        let expected = 0.5;
        let actual = logistic(x);
        assert_approx_eq!(expected, actual);
        let x = 1.0;
        let expected = 1.0 / (1.0 + (-1.0_f64).exp());
        let actual = logistic(x);
        assert_approx_eq!(expected, actual);
        let x = -1.0;
        let expected = 1.0 / (1.0 + 1.0_f64.exp());
        let actual = logistic(x);
        assert_approx_eq!(expected, actual);
    }

    #[test]
    fn test_logit() {
        let x = 0.5;
        let expected = 0.0;
        let actual = logit(x);
        assert_eq!(expected, actual);
        let x = 0.75;
        let expected = (0.75_f64 / (1.0 - 0.75)).ln();
        let actual = logit(x);
        assert_eq!(expected, actual);
        let x = 0.25;
        let expected = (0.25_f64 / (1.0 - 0.25)).ln();
        let actual = logit(x);
        assert_eq!(expected, actual);
    }

    #[test]
    fn logistic_transform() {
        let data = vec![0.0, 1.0, -1.0];
        let expected = vec![
            0.5_f64,
            1.0 / (1.0 + (-1.0_f64).exp()),
            1.0 / (1.0 + 1.0_f64.exp()),
        ];
        let actual: Vec<_> = data.into_iter().logistic().collect();
        assert_eq!(expected, actual);
    }

    #[test]
    fn logit_transform() {
        let data = vec![0.5, 0.75, 0.25];
        let expected = vec![
            0.0_f64,
            (0.75_f64 / (1.0 - 0.75)).ln(),
            (0.25_f64 / (1.0 - 0.25)).ln(),
        ];
        let actual: Vec<_> = data.into_iter().logit().collect();
        assert_eq!(expected, actual);
    }

    #[test]
    fn log_transform() {
        let data = vec![1.0, 2.0, 3.0];
        let expected = vec![0.0_f64, 2.0_f64.ln(), 3.0_f64.ln()];
        let actual: Vec<_> = data.into_iter().log().collect();
        assert_eq!(expected, actual);
    }
}
