/*!
Data transformations.
*/

use argmin::core::Error;
use augurs_core::{
    interpolate::{InterpolateExt, LinearInterpolator},
    Forecast,
};

use crate::power_transforms::{optimize_box_cox_lambda, optimize_yeo_johnson_lambda};

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

/// Parameters for the min-max scaler.
///
/// The target range is [0, 1] by default. Use [`MinMaxScaleParams::with_scaled_range`]
/// to set a custom range.
#[derive(Debug, Clone)]
pub struct MinMaxScaleParams {
    data_min: f64,
    data_max: f64,
    scaled_min: f64,
    scaled_max: f64,
}

impl MinMaxScaleParams {
    /// Create a new `MinMaxScaleParams` with the given data min and max.
    ///
    /// The scaled range is set to [0, 1] by default.
    pub fn new(data_min: f64, data_max: f64) -> Self {
        Self {
            data_min,
            data_max,
            scaled_min: 0.0 + f64::EPSILON,
            scaled_max: 1.0 - f64::EPSILON,
        }
    }

    /// Set the scaled range for the transformation.
    pub fn with_scaled_range(mut self, min: f64, max: f64) -> Self {
        self.scaled_min = min;
        self.scaled_max = max;
        self
    }

    /// Create a new `MinMaxScaleParams` from the given data.
    pub fn from_data<T>(data: T) -> Self
    where
        T: Iterator<Item = f64>,
    {
        let (min, max) = data.fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), x| {
            (min.min(x), max.max(x))
        });
        Self::new(min, max)
    }
}

/// Iterator adapter that scales each item to the range [0, 1].
#[derive(Debug, Clone)]
struct MinMaxScale<T> {
    inner: T,
    scale_factor: f64,
    offset: f64,
}

impl<T> Iterator for MinMaxScale<T>
where
    T: Iterator<Item = f64>,
{
    type Item = f64;
    fn next(&mut self) -> Option<Self::Item> {
        let Self {
            scale_factor,
            offset,
            inner,
            ..
        } = self;
        inner.next().map(|x| *offset + (x * *scale_factor))
    }
}

trait MinMaxScaleExt: Iterator<Item = f64> {
    fn min_max_scale(self, params: &MinMaxScaleParams) -> MinMaxScale<Self>
    where
        Self: Sized,
    {
        let scale_factor =
            (params.scaled_max - params.scaled_min) / (params.data_max - params.data_min);
        let offset = params.scaled_min - (params.data_min * scale_factor);
        MinMaxScale {
            inner: self,
            scale_factor,
            offset,
        }
    }
}

impl<T> MinMaxScaleExt for T where T: Iterator<Item = f64> {}

struct InverseMinMaxScale<T> {
    inner: T,
    scale_factor: f64,
    offset: f64,
}

impl<T> Iterator for InverseMinMaxScale<T>
where
    T: Iterator<Item = f64>,
{
    type Item = f64;
    fn next(&mut self) -> Option<Self::Item> {
        let Self {
            inner,
            scale_factor,
            offset,
            ..
        } = self;
        inner.next().map(|x| *offset + (x * *scale_factor))
    }
}

trait InverseMinMaxScaleExt: Iterator<Item = f64> {
    fn inverse_min_max_scale(self, params: &MinMaxScaleParams) -> InverseMinMaxScale<Self>
    where
        Self: Sized,
    {
        let scale_factor =
            (params.data_max - params.data_min) / (params.scaled_max - params.scaled_min);
        let offset = params.data_min - (params.scaled_min * scale_factor);
        InverseMinMaxScale {
            inner: self,
            scale_factor,
            offset,
        }
    }
}

impl<T> InverseMinMaxScaleExt for T where T: Iterator<Item = f64> {}

/// Parameters for the standard scaler.
#[derive(Debug, Clone)]
pub struct StandardScaleParams {
    /// The mean of the data.
    pub mean: f64,
    /// The standard deviation of the data.
    pub std_dev: f64,
}

impl StandardScaleParams {
    /// Create a new `StandardScaleParams` with the given mean and standard deviation.
    pub fn new(mean: f64, std_dev: f64) -> Self {
        Self { mean, std_dev }
    }

    /// Create a new `StandardScaleParams` from the given data.
    ///
    /// Note: this uses Welford's online algorithm to compute mean and variance in a single pass,
    /// since we only have an iterator. The standard deviation is calculated using the
    /// biased estimator, for parity with the [scikit-learn implementation][sklearn].
    ///
    /// [sklearn]: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
    pub fn from_data<T>(data: T) -> Self
    where
        T: Iterator<Item = f64>,
    {
        // Use Welford's online algorithm to compute mean and variance in a single pass,
        // since we only have an iterator.
        let mut count = 0_u64;
        let mut mean = 0.0;
        let mut m2 = 0.0;

        for x in data {
            count += 1;
            let delta = x - mean;
            mean += delta / count as f64;
            let delta2 = x - mean;
            m2 += delta * delta2;
        }

        // Handle empty iterator case
        if count == 0 {
            return Self::new(0.0, 1.0);
        }

        // Calculate standard deviation
        let std_dev = (m2 / count as f64).sqrt();

        Self { mean, std_dev }
    }
}

/// Iterator adapter that scales each item using the given mean and standard deviation,
/// so that (assuming the adapter was created using the same data), the output items
/// have zero mean and unit standard deviation.
#[derive(Debug, Clone)]
struct StandardScale<T> {
    inner: T,
    mean: f64,
    std_dev: f64,
}

impl<T> Iterator for StandardScale<T>
where
    T: Iterator<Item = f64>,
{
    type Item = f64;
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|x| (x - self.mean) / self.std_dev)
    }
}

trait StandardScaleExt: Iterator<Item = f64> {
    fn standard_scale(self, params: &StandardScaleParams) -> StandardScale<Self>
    where
        Self: Sized,
    {
        StandardScale {
            inner: self,
            mean: params.mean,
            std_dev: params.std_dev,
        }
    }
}

impl<T> StandardScaleExt for T where T: Iterator<Item = f64> {}

/// Iterator adapter that applies the inverse standard scaling transformation.
#[derive(Debug, Clone)]
struct InverseStandardScale<T> {
    inner: T,
    mean: f64,
    std_dev: f64,
}

impl<T> Iterator for InverseStandardScale<T>
where
    T: Iterator<Item = f64>,
{
    type Item = f64;
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|x| (x * self.std_dev) + self.mean)
    }
}

trait InverseStandardScaleExt: Iterator<Item = f64> {
    fn inverse_standard_scale(self, params: &StandardScaleParams) -> InverseStandardScale<Self>
    where
        Self: Sized,
    {
        InverseStandardScale {
            inner: self,
            mean: params.mean,
            std_dev: params.std_dev,
        }
    }
}

impl<T> InverseStandardScaleExt for T where T: Iterator<Item = f64> {}

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

/// Returns the Box-Cox transformation of the given value.
/// Assumes x > 0.
pub fn box_cox(x: f64, lambda: f64) -> Result<f64, &'static str> {
    if x <= 0.0 {
        return Err("x must be greater than 0");
    }
    if lambda == 0.0 {
        Ok(x.ln())
    } else {
        Ok((x.powf(lambda) - 1.0) / lambda)
    }
}

/// Returns the Yeo-Johnson transformation of the given value.
pub fn yeo_johnson(x: f64, lambda: f64) -> Result<f64, &'static str> {
    if x.is_nan() || lambda.is_nan() {
        return Err("Input values must be valid numbers.");
    }

    if x >= 0.0 {
        if lambda == 0.0 {
            Ok((x + 1.0).ln())
        } else {
            Ok(((x + 1.0).powf(lambda) - 1.0) / lambda)
        }
    } else if lambda == 2.0 {
        Ok(-(-x + 1.0).ln())
    } else {
        Ok(-((-x + 1.0).powf(2.0 - lambda) - 1.0) / (2.0 - lambda))
    }
}

/// An iterator adapter that applies the Box-Cox transformation to each item.
#[derive(Clone, Debug)]
struct BoxCox<T> {
    inner: T,
    lambda: f64,
}

impl<T> Iterator for BoxCox<T>
where
    T: Iterator<Item = f64>,
{
    type Item = f64;
    fn next(&mut self) -> Option<Self::Item> {
        self.inner
            .next()
            .map(|x| box_cox(x, self.lambda).unwrap_or(f64::NAN))
    }
}

trait BoxCoxExt: Iterator<Item = f64> {
    fn box_cox(self, lambda: f64) -> BoxCox<Self>
    where
        Self: Sized,
    {
        BoxCox {
            inner: self,
            lambda,
        }
    }
}

impl<T> BoxCoxExt for T where T: Iterator<Item = f64> {}

/// Returns the inverse Box-Cox transformation of the given value.
fn inverse_box_cox(y: f64, lambda: f64) -> Result<f64, &'static str> {
    if lambda == 0.0 {
        Ok(y.exp())
    } else {
        let value = y * lambda + 1.0;
        if value <= 0.0 {
            Err("Invalid domain for inverse Box-Cox transformation")
        } else {
            Ok(value.powf(1.0 / lambda))
        }
    }
}

/// An iterator adapter that applies the inverse Box-Cox transformation to each item.
#[derive(Clone, Debug)]
struct InverseBoxCox<T> {
    inner: T,
    lambda: f64,
}

impl<T> Iterator for InverseBoxCox<T>
where
    T: Iterator<Item = f64>,
{
    type Item = f64;
    fn next(&mut self) -> Option<Self::Item> {
        self.inner
            .next()
            .map(|y| inverse_box_cox(y, self.lambda).unwrap_or(f64::NAN))
    }
}

trait InverseBoxCoxExt: Iterator<Item = f64> {
    fn inverse_box_cox(self, lambda: f64) -> InverseBoxCox<Self>
    where
        Self: Sized,
    {
        InverseBoxCox {
            inner: self,
            lambda,
        }
    }
}

impl<T> InverseBoxCoxExt for T where T: Iterator<Item = f64> {}

/// An iterator adapter that applies the Yeo-Johnson transformation to each item.
#[derive(Clone, Debug)]
struct YeoJohnson<T> {
    inner: T,
    lambda: f64,
}

impl<T> Iterator for YeoJohnson<T>
where
    T: Iterator<Item = f64>,
{
    type Item = f64;
    fn next(&mut self) -> Option<Self::Item> {
        self.inner
            .next()
            .map(|x| yeo_johnson(x, self.lambda).unwrap_or(f64::NAN))
    }
}

trait YeoJohnsonExt: Iterator<Item = f64> {
    fn yeo_johnson(self, lambda: f64) -> YeoJohnson<Self>
    where
        Self: Sized,
    {
        YeoJohnson {
            inner: self,
            lambda,
        }
    }
}

impl<T> YeoJohnsonExt for T where T: Iterator<Item = f64> {}

/// Returns the inverse Yeo-Johnson transformation of the given value.
fn inverse_yeo_johnson(y: f64, lambda: f64) -> f64 {
    const EPSILON: f64 = 1e-6;

    if y >= 0.0 && lambda.abs() < EPSILON {
        // For lambda close to 0 (positive values)
        (y.exp()) - 1.0
    } else if y >= 0.0 {
        // For positive values (lambda not close to 0)
        (y * lambda + 1.0).powf(1.0 / lambda) - 1.0
    } else if (lambda - 2.0).abs() < EPSILON {
        // For lambda close to 2 (negative values)
        -(-y.exp() - 1.0)
    } else {
        // For negative values (lambda not close to 2)
        -((-((2.0 - lambda) * y) + 1.0).powf(1.0 / (2.0 - lambda)) - 1.0)
    }
}

/// An iterator adapter that applies the inverse Yeo-Johnson transformation to each item.
#[derive(Clone, Debug)]
struct InverseYeoJohnson<T> {
    inner: T,
    lambda: f64,
}

impl<T> Iterator for InverseYeoJohnson<T>
where
    T: Iterator<Item = f64>,
{
    type Item = f64;
    fn next(&mut self) -> Option<Self::Item> {
        self.inner
            .next()
            .map(|y| inverse_yeo_johnson(y, self.lambda))
    }
}

trait InverseYeoJohnsonExt: Iterator<Item = f64> {
    fn inverse_yeo_johnson(self, lambda: f64) -> InverseYeoJohnson<Self>
    where
        Self: Sized,
    {
        InverseYeoJohnson {
            inner: self,
            lambda,
        }
    }
}

impl<T> InverseYeoJohnsonExt for T where T: Iterator<Item = f64> {}

#[cfg(test)]
mod test {
    use augurs_testing::{assert_all_close, assert_approx_eq};

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

    #[test]
    fn min_max_scale() {
        let data = vec![1.0, 2.0, 3.0];
        let min = 1.0;
        let max = 3.0;
        let expected = vec![0.0, 0.5, 1.0];
        let actual: Vec<_> = data
            .into_iter()
            .min_max_scale(&MinMaxScaleParams::new(min, max))
            .collect();
        assert_all_close(&expected, &actual);
    }

    #[test]
    fn min_max_scale_custom() {
        let data = vec![1.0, 2.0, 3.0];
        let min = 1.0;
        let max = 3.0;
        let expected = vec![0.0, 5.0, 10.0];
        let actual: Vec<_> = data
            .into_iter()
            .min_max_scale(&MinMaxScaleParams::new(min, max).with_scaled_range(0.0, 10.0))
            .collect();
        assert_all_close(&expected, &actual);
    }

    #[test]
    fn inverse_min_max_scale() {
        let data = vec![0.0, 0.5, 1.0];
        let min = 1.0;
        let max = 3.0;
        let expected = vec![1.0, 2.0, 3.0];
        let actual: Vec<_> = data
            .into_iter()
            .inverse_min_max_scale(&MinMaxScaleParams::new(min, max))
            .collect();
        assert_all_close(&expected, &actual);
    }

    #[test]
    fn inverse_min_max_scale_custom() {
        let data = vec![0.0, 5.0, 10.0];
        let min = 1.0;
        let max = 3.0;
        let expected = vec![1.0, 2.0, 3.0];
        let actual: Vec<_> = data
            .into_iter()
            .inverse_min_max_scale(&MinMaxScaleParams::new(min, max).with_scaled_range(0.0, 10.0))
            .collect();
        assert_all_close(&expected, &actual);
    }

    #[test]
    fn standard_scale() {
        let data = vec![1.0, 2.0, 3.0];
        let params = StandardScaleParams::new(2.0, 1.0); // mean=2, std=1
        let expected = vec![-1.0, 0.0, 1.0];
        let actual: Vec<_> = data.into_iter().standard_scale(&params).collect();
        assert_all_close(&expected, &actual);
    }

    #[test]
    fn inverse_standard_scale() {
        let data = vec![-1.0, 0.0, 1.0];
        let params = StandardScaleParams::new(2.0, 1.0); // mean=2, std=1
        let expected = vec![1.0, 2.0, 3.0];
        let actual: Vec<_> = data.into_iter().inverse_standard_scale(&params).collect();
        assert_all_close(&expected, &actual);
    }

    #[test]
    fn standard_scale_params_from_data() {
        // Test case 1: Simple sequence
        let data = vec![1.0, 2.0, 3.0];
        let params = StandardScaleParams::from_data(data.into_iter());
        assert_approx_eq!(params.mean, 2.0);
        assert_approx_eq!(params.std_dev, 0.816496580927726);

        // Test case 2: More complex data
        let data = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let params = StandardScaleParams::from_data(data.into_iter());
        assert_approx_eq!(params.mean, 5.0);
        assert_approx_eq!(params.std_dev, 2.0);

        // Test case 3: Empty iterator should return default values
        let data: Vec<f64> = vec![];
        let params = StandardScaleParams::from_data(data.into_iter());
        assert_approx_eq!(params.mean, 0.0);
        assert_approx_eq!(params.std_dev, 1.0);

        // Test case 4: Single value
        let data = vec![42.0];
        let params = StandardScaleParams::from_data(data.into_iter());
        assert_approx_eq!(params.mean, 42.0);
        assert_approx_eq!(params.std_dev, 0.0); // technically undefined, but we return 0
    }

    #[test]
    fn min_max_scale_params_from_data() {
        let data = [1.0, 2.0, f64::NAN, 3.0];
        let params = MinMaxScaleParams::from_data(data.iter().copied());
        assert_approx_eq!(params.data_min, 1.0);
        assert_approx_eq!(params.data_max, 3.0);
        assert_approx_eq!(params.scaled_min, 0.0);
        assert_approx_eq!(params.scaled_max, 1.0);
    }

    #[test]
    fn box_cox_test() {
        let data = vec![1.0, 2.0, 3.0];
        let lambda = 0.5;
        let expected = vec![0.0, 0.8284271247461903, 1.4641016151377544];
        let actual: Vec<_> = data.into_iter().box_cox(lambda).collect();
        assert_all_close(&expected, &actual);
    }

    #[test]
    fn inverse_box_cox_test() {
        let data = vec![0.0, 0.5_f64.ln(), 1.0_f64.ln()];
        let lambda = 0.5;
        let expected = vec![1.0, 0.426966072919605, 1.0];
        let actual: Vec<_> = data.into_iter().inverse_box_cox(lambda).collect();
        assert_all_close(&expected, &actual);
    }

    #[test]
    fn yeo_johnson_test() {
        let data = vec![-1.0, 0.0, 1.0];
        let lambda = 0.5;
        let expected = vec![-1.2189514164974602, 0.0, 0.8284271247461903];
        let actual: Vec<_> = data.into_iter().yeo_johnson(lambda).collect();
        assert_all_close(&expected, &actual);
    }

    #[test]
    fn inverse_yeo_johnson_test() {
        let data = vec![-1.2189514164974602, 0.0, 0.8284271247461903];
        let lambda = 0.5;
        let expected = vec![-1.0, 0.0, 1.0];
        let actual: Vec<_> = data.into_iter().inverse_yeo_johnson(lambda).collect();
        assert_all_close(&expected, &actual);
    }
}
