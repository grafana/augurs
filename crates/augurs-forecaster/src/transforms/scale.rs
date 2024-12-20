//! Scalers, including min-max and standard scalers.

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
pub(crate) struct MinMaxScale<T> {
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

pub(crate) trait MinMaxScaleExt: Iterator<Item = f64> {
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

pub(crate) struct InverseMinMaxScale<T> {
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

pub(crate) trait InverseMinMaxScaleExt: Iterator<Item = f64> {
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
pub(crate) struct StandardScale<T> {
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

pub(crate) trait StandardScaleExt: Iterator<Item = f64> {
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
pub(crate) struct InverseStandardScale<T> {
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

pub(crate) trait InverseStandardScaleExt: Iterator<Item = f64> {
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

#[cfg(test)]
mod test {
    use augurs_testing::{assert_all_close, assert_approx_eq};

    use super::*;

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
}
