//! [`Fit`] and [`Predict`] implementations for the Prophet algorithm.
use std::{cell::RefCell, num::NonZeroU32, sync::Arc};

use augurs_core::{Fit, ModelError, Predict};

use crate::{optimizer::OptimizeOpts, Error, IncludeHistory, Optimizer, Prophet, TrainingData};

impl ModelError for Error {}

/// A forecaster that uses the Prophet algorithm.
///
/// This is a wrapper around the [`Prophet`] struct that provides
/// a simpler API for fitting and predicting. Notably it implements
/// the [`Fit`] trait from `augurs_core`, so it can be
/// used with the `augurs` framework (e.g. with the `Forecaster` struct
/// in the `augurs::forecaster` module).
#[derive(Debug)]
pub struct ProphetForecaster {
    data: TrainingData,
    model: Prophet<Arc<dyn Optimizer>>,
    optimize_opts: OptimizeOpts,
}

impl ProphetForecaster {
    /// Create a new Prophet forecaster.
    ///
    /// # Parameters
    ///
    /// - `opts`: The options to use for fitting the model.
    ///           Note that `uncertainty_samples` will be set to 1000 if it is 0,
    ///           to facilitate generating prediction intervals.
    /// - `optimizer`: The optimizer to use for fitting the model.
    /// - `optimize_opts`: The options to use for optimizing the model.
    pub fn new<T: Optimizer + 'static>(
        mut model: Prophet<T>,
        data: TrainingData,
        optimize_opts: OptimizeOpts,
    ) -> Self {
        let opts = model.opts_mut();
        if opts.uncertainty_samples == 0 {
            opts.uncertainty_samples = 1000;
        }
        Self {
            data,
            model: model.into_dyn_optimizer(),
            optimize_opts,
        }
    }
}

impl Fit for ProphetForecaster {
    type Fitted = FittedProphetForecaster;
    type Error = Error;

    fn fit(&self, y: &[f64]) -> Result<Self::Fitted, Self::Error> {
        // Use the training data from `self`...
        let mut training_data = self.data.clone();
        // ...but replace the `y` column with whatever we're passed
        // (which may be a transformed version of `y`, if the user is
        // using `augurs_forecaster`).
        training_data.y = y.to_vec();
        let mut fitted_model = self.model.clone();
        fitted_model.fit(training_data, self.optimize_opts.clone())?;
        Ok(FittedProphetForecaster {
            model: RefCell::new(fitted_model),
            training_n: y.len(),
        })
    }
}

/// A fitted Prophet forecaster.
#[derive(Debug)]
pub struct FittedProphetForecaster {
    model: RefCell<Prophet<Arc<dyn Optimizer>>>,
    training_n: usize,
}

impl Predict for FittedProphetForecaster {
    type Error = Error;

    fn predict_in_sample_inplace(
        &self,
        level: Option<f64>,
        forecast: &mut augurs_core::Forecast,
    ) -> Result<(), Self::Error> {
        if let Some(level) = level {
            self.model
                .borrow_mut()
                .set_interval_width(level.try_into()?);
        }
        let predictions = self.model.borrow().predict(None)?;
        forecast.point = predictions.yhat.point;
        if let Some(intervals) = forecast.intervals.as_mut() {
            intervals.lower = predictions
                .yhat
                .lower
                // This `expect` is OK because we've set uncertainty_samples > 0 in the
                // `ProphetForecaster` constructor.
                .expect("uncertainty_samples should be > 0, this is a bug");
            intervals.upper = predictions
                .yhat
                .upper
                // This `expect` is OK because we've set uncertainty_samples > 0 in the
                // `ProphetForecaster` constructor.
                .expect("uncertainty_samples should be > 0, this is a bug");
        }
        Ok(())
    }

    fn predict_inplace(
        &self,
        horizon: usize,
        level: Option<f64>,
        forecast: &mut augurs_core::Forecast,
    ) -> Result<(), Self::Error> {
        let horizon = match NonZeroU32::try_from(horizon as u32) {
            Ok(h) => h,
            // If horizon is 0, short circuit without even trying to predict.
            Err(_) => return Ok(()),
        };
        if let Some(level) = level {
            self.model
                .borrow_mut()
                .set_interval_width(level.try_into()?);
        }
        let predictions = {
            let model = self.model.borrow();
            let prediction_data = model.make_future_dataframe(horizon, IncludeHistory::No)?;
            model.predict(prediction_data)?
        };
        forecast.point = predictions.yhat.point;
        if let Some(intervals) = forecast.intervals.as_mut() {
            intervals.lower = predictions
                .yhat
                .lower
                // This `expect` is OK because we've set uncertainty_samples > 0 in the
                // `ProphetForecaster` constructor.
                .expect("uncertainty_samples should be > 0");
            intervals.upper = predictions
                .yhat
                .upper
                // This `expect` is OK because we've set uncertainty_samples > 0 in the
                // `ProphetForecaster` constructor.
                .expect("uncertainty_samples should be > 0");
        }
        Ok(())
    }

    fn training_data_size(&self) -> usize {
        self.training_n
    }
}

#[cfg(all(test, feature = "wasmstan"))]
mod test {

    use augurs_core::{Fit, Predict};
    use augurs_testing::assert_all_close;

    use crate::{
        testdata::{daily_univariate_ts, train_test_splitn},
        wasmstan::WasmstanOptimizer,
        IncludeHistory, Prophet,
    };

    use super::ProphetForecaster;

    #[test]
    fn forecaster() {
        let test_days = 30;
        let (train, _) = train_test_splitn(daily_univariate_ts(), test_days);

        let model = Prophet::new(Default::default(), WasmstanOptimizer::new());
        let forecaster = ProphetForecaster::new(model, train.clone(), Default::default());
        let fitted = forecaster.fit(&train.y).unwrap();
        let forecast_predictions = fitted.predict(30, 0.95).unwrap();

        let mut prophet = Prophet::new(Default::default(), WasmstanOptimizer::new());
        prophet.fit(train, Default::default()).unwrap();
        let prediction_data = prophet
            .make_future_dataframe(30.try_into().unwrap(), IncludeHistory::No)
            .unwrap();
        let predictions = prophet.predict(prediction_data).unwrap();

        // We should get the same results back when using the Forecaster impl.
        assert_eq!(
            predictions.yhat.point.len(),
            forecast_predictions.point.len()
        );
        assert_all_close(&predictions.yhat.point, &forecast_predictions.point);
    }
}
