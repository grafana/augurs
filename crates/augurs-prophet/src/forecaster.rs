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
        mut opts: ProphetOptions,
        optimizer: Arc<dyn Optimizer>,
        optimize_opts: OptimizeOpts,
    ) -> Self {
        if opts.uncertainty_samples == 0 {
            opts.uncertainty_samples = 1000;
        }
        Self {
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
        let ds = vec![];
        let training_data = TrainingData::new(ds, y.to_vec())?;
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
