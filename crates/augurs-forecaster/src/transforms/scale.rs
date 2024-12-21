//! Scalers, including min-max and standard scalers.

use core::f64;

use itertools::{Itertools, MinMaxResult};

use super::{Error, Transform};

/// Helper struct holding the min and max for use in a `MinMaxScaler`.
#[derive(Debug, Clone, Copy)]
struct MinMax {
    min: f64,
    max: f64,
}

impl MinMax {
    fn zero_one() -> Self {
        Self {
            min: 0.0 + f64::EPSILON,
            max: 1.0 - f64::EPSILON,
        }
    }
}

/// Parameters for the min-max scaler.
///
/// Will be created by the `MinMaxScaler` when it is fit to the data,
/// or when it is supplied with a custom data range.
///
/// We store the scale factor and offset to avoid having to
/// recalculating them every time the transform is applied.
///
/// We store the input scale as well so we can recalculate the
/// scale factor and offset if the user changes the output scale.
#[derive(Debug, Clone)]
struct FittedMinMaxScalerParams {
    input_scale: MinMax,
    scale_factor: f64,
    offset: f64,
}

impl FittedMinMaxScalerParams {
    fn new(input_scale: MinMax, output_scale: MinMax) -> Self {
        let scale_factor =
            (output_scale.max - output_scale.min) / (input_scale.max - input_scale.min);
        Self {
            input_scale,
            scale_factor,
            offset: output_scale.min - (input_scale.min * scale_factor),
        }
    }
}

/// A transformer that scales each item to a custom range, defaulting to [0, 1].
#[derive(Debug, Clone)]
pub struct MinMaxScaler {
    output_scale: MinMax,
    // The parameters learned from the data and used to transform it.
    // Not known until the transform method is called.
    params: Option<FittedMinMaxScalerParams>,
}

impl Default for MinMaxScaler {
    fn default() -> Self {
        Self::new()
    }
}

impl MinMaxScaler {
    /// Create a new `MinMaxScaler` with the default output range of [0, 1].
    pub fn new() -> Self {
        Self {
            output_scale: MinMax::zero_one(),
            params: None,
        }
    }

    /// Set the output range for the transformation.
    pub fn with_scaled_range(mut self, min: f64, max: f64) -> Self {
        self.output_scale = MinMax { min, max };
        self.params.iter_mut().for_each(|p| {
            let input_scale = p.input_scale;
            *p = FittedMinMaxScalerParams::new(input_scale, self.output_scale);
        });
        self
    }

    /// Manually set the input range for the transformation.
    ///
    /// This is useful if you know the input range in advance and want to avoid
    /// the overhead of fitting the scaler to the data during the initial transform,
    /// and instead want to set the input range manually.
    ///
    /// Note that this will override any previously set (or learned) parameters.
    pub fn with_data_range(mut self, min: f64, max: f64) -> Self {
        let data_range = MinMax { min, max };
        self.params = Some(FittedMinMaxScalerParams::new(data_range, self.output_scale));
        self
    }

    fn fit(&self, data: &[f64]) -> Result<FittedMinMaxScalerParams, Error> {
        match data
            .iter()
            .copied()
            .minmax_by(|a, b| a.partial_cmp(b).unwrap())
        {
            e @ MinMaxResult::NoElements | e @ MinMaxResult::OneElement(_) => Err(e.into()),
            MinMaxResult::MinMax(min, max) => Ok(FittedMinMaxScalerParams::new(
                MinMax { min, max },
                self.output_scale,
            )),
        }
    }
}

impl Transform for MinMaxScaler {
    fn transform(&mut self, data: &mut [f64]) -> Result<(), Error> {
        let params = match &mut self.params {
            Some(p) => p,
            None => self.params.get_or_insert(self.fit(data)?),
        };
        data.iter_mut()
            .for_each(|x| *x = *x * params.scale_factor + params.offset);
        Ok(())
    }

    fn inverse_transform(&self, data: &mut [f64]) -> Result<(), Error> {
        let params = self.params.as_ref().ok_or(Error::NotFitted)?;
        data.iter_mut()
            .for_each(|x| *x = (*x - params.offset) / params.scale_factor);
        Ok(())
    }
}

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

/// A transformer that scales items to have zero mean and unit standard deviation.
///
/// The standard score of a sample `x` is calculated as:
///
/// ```text
/// z = (x - mean) / std_dev
/// ```
///
/// where `mean` is the mean and s is the standard deviation of the data first passed to
/// `transform` (or provided via `with_parameters`).
///
/// # Implementation
///
/// This transformer uses Welford's online algorithm to compute mean and variance in
/// one pass over the data. The standard deviation is calculated using the biased
/// estimator, for parity with the [scikit-learn implementation][sklearn].
///
/// [sklearn]: https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/preprocessing/_data.py#L128
///
/// # Example
///
/// ## Using the default constructor
///
/// ```
/// use augurs_forecaster::transforms::{StandardScaler, Transform};
///
/// let mut data = vec![1.0, 2.0, 3.0];
/// let mut scaler = StandardScaler::new();
/// scaler.transform(&mut data);
///
/// assert_eq!(data, vec![-1.224744871391589, 0.0, 1.224744871391589]);
/// ```
#[derive(Debug, Clone, Default)]
pub struct StandardScaler {
    params: Option<StandardScaleParams>,
}

impl StandardScaler {
    /// Create a new `StandardScaler`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the parameters for the scaler.
    ///
    /// This is useful if you know the mean and standard deviation in advance
    /// and want to avoid the overhead of fitting the scaler to the data
    /// during the initial transform, and instead want to set the parameters
    /// manually.
    pub fn with_parameters(mut self, params: StandardScaleParams) -> Self {
        self.params = Some(params);
        self
    }

    fn fit(&self, data: &[f64]) -> StandardScaleParams {
        StandardScaleParams::from_data(data.iter().copied())
    }
}

impl Transform for StandardScaler {
    fn transform(&mut self, data: &mut [f64]) -> Result<(), Error> {
        let params = match &mut self.params {
            Some(p) => p,
            None => self.params.get_or_insert(self.fit(data)),
        };
        data.iter_mut()
            .for_each(|x| *x = (*x - params.mean) / params.std_dev);
        Ok(())
    }

    fn inverse_transform(&self, data: &mut [f64]) -> Result<(), Error> {
        let params = self.params.as_ref().ok_or(Error::NotFitted)?;
        data.iter_mut()
            .for_each(|x| *x = (*x * params.std_dev) + params.mean);
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use augurs_testing::{assert_all_close, assert_approx_eq};

    use super::*;

    #[test]
    fn min_max_scale() {
        let mut data = vec![1.0, 2.0, 3.0];
        let expected = vec![0.0, 0.5, 1.0];
        let mut scaler = MinMaxScaler::new();
        scaler.transform(&mut data).unwrap();
        assert_all_close(&expected, &data);
    }

    #[test]
    fn min_max_scale_custom() {
        let mut data = vec![1.0, 2.0, 3.0];
        let expected = vec![0.0, 5.0, 10.0];
        let mut scaler = MinMaxScaler::new().with_scaled_range(0.0, 10.0);
        scaler.transform(&mut data).unwrap();
        assert_all_close(&expected, &data);
    }

    #[test]
    fn inverse_min_max_scale() {
        let mut data = vec![0.0, 0.5, 1.0];
        let expected = vec![1.0, 2.0, 3.0];
        let scaler = MinMaxScaler::new().with_data_range(1.0, 3.0);
        scaler.inverse_transform(&mut data).unwrap();
        assert_all_close(&expected, &data);
    }

    #[test]
    fn inverse_min_max_scale_custom() {
        let mut data = vec![0.0, 5.0, 10.0];
        let expected = vec![1.0, 2.0, 3.0];
        let scaler = MinMaxScaler::new()
            .with_scaled_range(0.0, 10.0)
            .with_data_range(1.0, 3.0);
        scaler.inverse_transform(&mut data).unwrap();
        assert_all_close(&expected, &data);
    }

    #[test]
    fn standard_scale() {
        let mut data = vec![1.0, 2.0, 3.0];
        // We use the biased estimator for standard deviation so the result is
        // not necessarily obvious.
        let expected = vec![-1.224744871391589, 0.0, 1.224744871391589];
        let mut scaler = StandardScaler::new(); // 2.0, 1.0); // mean=2, std=1
        scaler.transform(&mut data).unwrap();
        assert_all_close(&expected, &data);
    }

    #[test]
    fn standard_scale_custom() {
        let mut data = vec![1.0, 2.0, 3.0];
        let expected = vec![-1.0, 0.0, 1.0];
        let params = StandardScaleParams::new(2.0, 1.0); // mean=2, std=1
        let mut scaler = StandardScaler::new().with_parameters(params);
        scaler.transform(&mut data).unwrap();
        assert_all_close(&expected, &data);
    }

    #[test]
    fn inverse_standard_scale() {
        let mut data = vec![-1.0, 0.0, 1.0];
        let expected = vec![1.0, 2.0, 3.0];
        let params = StandardScaleParams::new(2.0, 1.0); // mean=2, std=1
        let scaler = StandardScaler::new().with_parameters(params);
        scaler.inverse_transform(&mut data).unwrap();
        assert_all_close(&expected, &data);
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
}
