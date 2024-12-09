use augurs_core::{Fit, Forecast, Predict};

use crate::{Data, Error, Result, Transform, Transforms};

/// A high-level API to fit and predict time series forecasting models.
///
/// The `Forecaster` type allows you to combine a model with a set of
/// transformations and fit it to a time series, then use the fitted model to
/// make predictions. The predictions are back-transformed using the inverse of
/// the transformations applied to the input data.
#[derive(Debug)]
pub struct Forecaster<M: Fit> {
    model: M,
    fitted: Option<M::Fitted>,

    transforms: Transforms,
}

impl<M> Forecaster<M>
where
    M: Fit,
    M::Fitted: Predict,
{
    /// Create a new `Forecaster` with the given model.
    pub fn new(model: M) -> Self {
        Self {
            model,
            fitted: None,
            transforms: Transforms::default(),
        }
    }

    /// Set the transformations to be applied to the input data.
    pub fn with_transforms(mut self, transforms: Vec<Transform>) -> Self {
        self.transforms = Transforms::new(transforms);
        self
    }

    /// Fit the model to the given time series.
    pub fn fit<D: Data + Clone>(&mut self, y: D) -> Result<()> {
        let data: Vec<_> = self
            .transforms
            .transform(y.as_slice().iter().copied())
            .collect();
        self.fitted = Some(self.model.fit(&data).map_err(|e| Error::Fit {
            source: Box::new(e) as _,
        })?);
        Ok(())
    }

    fn fitted(&self) -> Result<&M::Fitted> {
        self.fitted.as_ref().ok_or(Error::ModelNotYetFit)
    }

    /// Predict the next `horizon` values, optionally including prediction
    /// intervals at the given level.
    pub fn predict(&self, horizon: usize, level: impl Into<Option<f64>>) -> Result<Forecast> {
        self.fitted()?
            .predict(horizon, level.into())
            .map_err(|e| Error::Predict {
                source: Box::new(e) as _,
            })
            .map(|f| self.transforms.inverse_transform(f))
    }

    /// Produce in-sample forecasts, optionally including prediction intervals
    /// at the given level.
    pub fn predict_in_sample(&self, level: impl Into<Option<f64>>) -> Result<Forecast> {
        self.fitted()?
            .predict_in_sample(level.into())
            .map_err(|e| Error::Predict {
                source: Box::new(e) as _,
            })
            .map(|f| self.transforms.inverse_transform(f))
    }
}

#[cfg(test)]
mod test {
    use itertools::{Itertools, MinMaxResult};

    use augurs::mstl::{MSTLModel, NaiveTrend};

    use crate::transforms::MinMaxScaleParams;

    use super::*;

    fn assert_approx_eq(a: f64, b: f64) -> bool {
        if a.is_nan() && b.is_nan() {
            return true;
        }
        (a - b).abs() < 0.001
    }

    fn assert_all_approx_eq(a: &[f64], b: &[f64]) {
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
    fn test_forecaster() {
        let data = &[1.0_f64, 2.0, 3.0, 4.0, 5.0];
        let MinMaxResult::MinMax(min, max) = data
            .iter()
            .copied()
            .minmax_by(|a, b| a.partial_cmp(b).unwrap())
        else {
            unreachable!();
        };
        let transforms = vec![
            Transform::linear_interpolator(),
            Transform::min_max_scaler(MinMaxScaleParams::new(min - 1e-3, max + 1e-3)),
            Transform::logit(),
        ];
        let model = MSTLModel::new(vec![2], NaiveTrend::new());
        let mut forecaster = Forecaster::new(model).with_transforms(transforms);
        forecaster.fit(data).unwrap();
        let forecasts = forecaster.predict(4, None).unwrap();
        assert_all_approx_eq(&forecasts.point, &[5.0, 5.0, 5.0, 5.0]);
    }

    #[test]
    fn test_forecaster_power_positive() {
        let data = &[1.0_f64, 2.0, 3.0, 4.0, 5.0];
        let transforms = vec![
            Transform::power_transform(data),
        ];
        let model = MSTLModel::new(vec![2], NaiveTrend::new());
        let mut forecaster = Forecaster::new(model).with_transforms(transforms);
        forecaster.fit(data).unwrap();
        let forecasts = forecaster.predict(4, None).unwrap();
        assert_all_approx_eq(&forecasts.point, &[5.084499064884572, 5.000000030329821, 5.084499064884572, 5.000000030329821]);
    }

    #[test]
    fn test_forecaster_power_non_positive() {
        let data = &[0.0, 2.0, 3.0, 4.0, 5.0];
        let transforms = vec![
            Transform::power_transform(data),
        ];
        let model = MSTLModel::new(vec![2], NaiveTrend::new());
        let mut forecaster = Forecaster::new(model).with_transforms(transforms);
        forecaster.fit(data).unwrap();
        let forecasts = forecaster.predict(4, None).unwrap();
        assert_all_approx_eq(&forecasts.point, &[6.205557727170964, 6.000000132803496, 6.205557727170964, 6.000000132803496]);
    }
}
