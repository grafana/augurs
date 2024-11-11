//! [`Fit`] and [`Predict`] implementations for the Prophet algorithm.
use std::{cell::RefCell, num::NonZeroU32, sync::Arc};

use augurs_core::{Fit, ModelError, Predict};

use crate::{
    optimizer::OptimizeOpts, Error, IncludeHistory, Optimizer, Prophet, ProphetOptions,
    TrainingData,
};

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
    opts: ProphetOptions,
    optimizer: Arc<dyn Optimizer>,
    optimize_opts: OptimizeOpts,
}

impl ProphetForecaster {
    /// Create a new Prophet forecaster.
    ///
    /// # Parameters
    ///
    /// - `opts`: The options to use for fitting the model.
    /// - `optimizer`: The optimizer to use for fitting the model.
    /// - `optimize_opts`: The options to use for optimizing the model.
    pub fn new(
        data: TrainingData,
        mut opts: ProphetOptions,
        optimizer: Arc<dyn Optimizer>,
        optimize_opts: OptimizeOpts,
    ) -> Self {
        if opts.uncertainty_samples == 0 {
            opts.uncertainty_samples = 1000;
        }
        Self {
            data,
            opts,
            optimizer,
            optimize_opts,
        }
    }
}

impl Fit for ProphetForecaster {
    type Fitted = FittedProphetForecaster;
    type Error = Error;

    fn fit(&self, y: &[f64]) -> Result<Self::Fitted, Self::Error> {
        let mut training_data = self.data.clone();
        training_data.y = y.to_vec();
        let mut model = Prophet::new(self.opts.clone(), self.optimizer.clone());
        model.fit(training_data, self.optimize_opts.clone())?;
        Ok(FittedProphetForecaster {
            model: RefCell::new(model),
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

    fn predict_inplace(
        &self,
        horizon: usize,
        level: Option<f64>,
        forecast: &mut augurs_core::Forecast,
    ) -> Result<(), Self::Error> {
        if horizon == 0 {
            return Ok(());
        }
        if let Some(level) = level {
            self.model
                .borrow_mut()
                .set_interval_width(level.try_into()?);
        }
        let predictions = {
            let model = self.model.borrow();
            let prediction_data = model.make_future_dataframe(
                NonZeroU32::try_from(horizon as u32).expect("horizon should be > 0"),
                IncludeHistory::No,
            )?;
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

#[cfg(test)]
mod test {
    use std::sync::Arc;

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

        let forecaster = ProphetForecaster::new(
            train.clone(),
            Default::default(),
            Arc::new(WasmstanOptimizer::new()),
            Default::default(),
        );
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
