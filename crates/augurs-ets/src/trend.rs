/*!
Implementations of [`augurs_mstl::TrendModel`] using the [`AutoETS`] model.

This module provides the [`AutoETSTrendModel`] type, which is a trend model
implementation that uses the [`AutoETS`] model to fit and predict the trend
component of the [`augurs_mstl::MSTLModel`] model.

This module is gated behind the `mstl` feature.
*/
use std::borrow::Cow;

use augurs_core::{Fit, Forecast, Predict};
use augurs_mstl::TrendModel;

use crate::{AutoETS, FittedAutoETS};

/// An MSTL-compatible trend model using the [`AutoETS`] model.
#[derive(Debug, Clone)]
pub struct AutoETSTrendModel {
    model: AutoETS,
    fitted: Option<FittedAutoETS>,
}

impl From<AutoETS> for AutoETSTrendModel {
    fn from(model: AutoETS) -> Self {
        Self {
            model,
            fitted: None,
        }
    }
}

impl TrendModel for AutoETSTrendModel {
    fn name(&self) -> Cow<'_, str> {
        Cow::Borrowed("AutoETS")
    }

    fn fit(&mut self, y: &[f64]) -> Result<(), Box<dyn std::error::Error + Send + Sync + 'static>> {
        match self.model.fit(y) {
            Ok(fit) => {
                self.fitted = Some(fit);
                Ok(())
            }
            Err(e) => Err(e.into()),
        }
    }

    fn predict_inplace(
        &self,
        horizon: usize,
        level: Option<f64>,
        forecast: &mut Forecast,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync + 'static>> {
        self.fitted
            .as_ref()
            .ok_or("Model not yet fit")?
            .predict_inplace(horizon, level, forecast)
            .map_err(|e| e.into())
    }

    fn predict_in_sample_inplace(
        &self,
        level: Option<f64>,
        forecast: &mut Forecast,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync + 'static>> {
        self.fitted
            .as_ref()
            .ok_or("Model not yet fit")?
            .predict_in_sample_inplace(level, forecast)
            .map_err(|e| e.into())
    }

    fn training_data_size(&self) -> Option<usize> {
        self.fitted.as_ref().map(|f| f.training_data_size())
    }
}

impl AutoETS {
    /// Create a new `AutoETSTrendModel` using the given `AutoETS` model.
    pub fn into_trend_model(self) -> AutoETSTrendModel {
        self.into()
    }
}
