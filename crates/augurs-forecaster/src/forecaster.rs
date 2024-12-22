use augurs_core::{Fit, Forecast, Predict};

use crate::{Data, Error, Pipeline, Result, Transformer};

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

    pipeline: Pipeline,
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
            pipeline: Pipeline::default(),
        }
    }

    /// Set the transformations to be applied to the input data.
    pub fn with_transformers(mut self, transformers: Vec<Box<dyn Transformer>>) -> Self {
        self.pipeline = Pipeline::new(transformers);
        self
    }

    /// Fit the model to the given time series.
    pub fn fit<D: Data + Clone>(&mut self, y: D) -> Result<()> {
        let mut y = y.as_slice().to_vec();
        self.pipeline.fit_transform(&mut y)?;
        self.fitted = Some(self.model.fit(&y).map_err(|e| Error::Fit {
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
        let mut untransformed =
            self.fitted()?
                .predict(horizon, level.into())
                .map_err(|e| Error::Predict {
                    source: Box::new(e) as _,
                })?;
        self.pipeline
            .inverse_transform_forecast(&mut untransformed)?;
        Ok(untransformed)
    }

    /// Produce in-sample forecasts, optionally including prediction intervals
    /// at the given level.
    pub fn predict_in_sample(&self, level: impl Into<Option<f64>>) -> Result<Forecast> {
        let mut untransformed = self
            .fitted()?
            .predict_in_sample(level.into())
            .map_err(|e| Error::Predict {
                source: Box::new(e) as _,
            })?;
        self.pipeline
            .inverse_transform_forecast(&mut untransformed)?;
        Ok(untransformed)
    }
}

#[cfg(test)]
mod test {

    use augurs::mstl::{MSTLModel, NaiveTrend};
    use augurs_testing::assert_all_close;

    use crate::transforms::{BoxCox, LinearInterpolator, Logit, MinMaxScaler, YeoJohnson};

    use super::*;

    #[test]
    fn test_forecaster() {
        let data = &[1.0_f64, 2.0, 3.0, 4.0, 5.0];
        let transformers = vec![
            LinearInterpolator::new().boxed(),
            MinMaxScaler::new().boxed(),
            Logit::new().boxed(),
        ];
        let model = MSTLModel::new(vec![2], NaiveTrend::new());
        let mut forecaster = Forecaster::new(model).with_transformers(transformers);
        forecaster.fit(data).unwrap();
        let forecasts = forecaster.predict(4, None).unwrap();
        assert_all_close(&forecasts.point, &[5.0, 5.0, 5.0, 5.0]);
    }

    #[test]
    fn test_forecaster_power_positive() {
        let data = &[1.0_f64, 2.0, 3.0, 4.0, 5.0];
        let transformers = vec![BoxCox::new().boxed()];
        let model = MSTLModel::new(vec![2], NaiveTrend::new());
        let mut forecaster = Forecaster::new(model).with_transformers(transformers);
        forecaster.fit(data).unwrap();
        let forecasts = forecaster.predict(4, None).unwrap();
        assert_all_close(
            &forecasts.point,
            &[
                5.084499064884572,
                5.000000030329821,
                5.084499064884572,
                5.000000030329821,
            ],
        );
    }

    #[test]
    fn test_forecaster_power_non_positive() {
        let data = &[0.0, 2.0, 3.0, 4.0, 5.0];
        let transformers = vec![YeoJohnson::new().boxed()];
        let model = MSTLModel::new(vec![2], NaiveTrend::new());
        let mut forecaster = Forecaster::new(model).with_transformers(transformers);
        forecaster.fit(data).unwrap();
        let forecasts = forecaster.predict(4, None).unwrap();
        assert_all_close(
            &forecasts.point,
            &[
                5.205557727170964,
                5.000000132803496,
                5.205557727170964,
                5.000000132803496,
            ],
        );
    }
}
