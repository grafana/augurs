//! JavaScript bindings for the MSTL model.
use serde::Deserialize;
use tsify_next::Tsify;
use wasm_bindgen::prelude::*;

use augurs_ets::{trend::AutoETSTrendModel, AutoETS};
use augurs_forecaster::{Forecaster, Transform};
use augurs_mstl::{MSTLModel, TrendModel};

use crate::{Forecast, VecF64, VecUsize};

/// The type of trend forecaster to use.
#[derive(Debug, Clone, Copy, Deserialize, Tsify)]
#[serde(rename_all = "kebab-case")]
#[tsify(from_wasm_abi)]
#[non_exhaustive]
pub enum MSTLTrendModel {
    /// Use the `ETS` trend model.
    Ets,
}

/// A MSTL model.
#[derive(Debug)]
#[wasm_bindgen]
pub struct MSTL {
    forecaster: Forecaster<MSTLModel<Box<dyn TrendModel + Send + Sync>>>,
}

#[wasm_bindgen]
impl MSTL {
    /// Create a new MSTL model with the given periods using the given trend model.
    #[wasm_bindgen(constructor)]
    pub fn new(
        trend_forecaster: MSTLTrendModel,
        periods: VecUsize,
        options: Option<ETSOptions>,
    ) -> Result<MSTL, JsError> {
        match trend_forecaster {
            MSTLTrendModel::Ets => MSTL::ets(periods, options),
        }
    }

    /// Create a new MSTL model with the given periods using the `AutoETS` trend model.
    #[wasm_bindgen]
    pub fn ets(periods: VecUsize, options: Option<ETSOptions>) -> Result<MSTL, JsError> {
        let ets: Box<dyn TrendModel + Sync + Send> =
            Box::new(AutoETSTrendModel::from(AutoETS::non_seasonal()));
        let model = MSTLModel::new(periods.convert()?, ets);
        let forecaster =
            Forecaster::new(model).with_transforms(options.unwrap_or_default().into_transforms());
        Ok(MSTL { forecaster })
    }

    /// Fit the model to the given time series.
    #[wasm_bindgen]
    pub fn fit(&mut self, y: VecF64) -> Result<(), JsValue> {
        self.forecaster
            .fit(y.convert()?)
            .map_err(|e| e.to_string())?;
        Ok(())
    }

    /// Predict the next `horizon` values, optionally including prediction
    /// intervals at the given level.
    ///
    /// If provided, `level` must be a float between 0 and 1.
    #[wasm_bindgen]
    pub fn predict(&self, horizon: usize, level: Option<f64>) -> Result<Forecast, JsValue> {
        let forecasts = self.forecaster.predict(horizon, level);
        Ok(forecasts.map(Into::into).map_err(|e| e.to_string())?)
    }

    /// Produce in-sample forecasts, optionally including prediction
    /// intervals at the given level.
    ///
    /// If provided, `level` must be a float between 0 and 1.
    #[wasm_bindgen]
    pub fn predictInSample(&self, level: Option<f64>) -> Result<Forecast, JsValue> {
        let forecasts = self.forecaster.predict_in_sample(level);
        Ok(forecasts.map(Into::into).map_err(|e| e.to_string())?)
    }
}

/// Options for the ETS MSTL model.
#[derive(Debug, Default, Deserialize, Tsify)]
#[serde(rename_all = "camelCase")]
#[tsify(from_wasm_abi)]
pub struct ETSOptions {
    /// Whether to impute missing values.
    #[tsify(optional)]
    pub impute: Option<bool>,

    /// Whether to logit-transform the data before forecasting.
    ///
    /// If `true`, the training data will be transformed using the logit function.
    /// Forecasts will be back-transformed using the logistic function.
    #[tsify(optional)]
    pub logit_transform: Option<bool>,
}

impl ETSOptions {
    fn into_transforms(self) -> Vec<Transform> {
        let mut transforms = vec![];
        if self.impute.unwrap_or_default() {
            transforms.push(Transform::linear_interpolator());
        }
        if self.logit_transform.unwrap_or_default() {
            transforms.push(Transform::logit());
        }
        transforms
    }
}

/// Create a new MSTL model with the given periods using the `AutoETS` trend model.
///
/// @deprecated use `MSTL.ets` instead
#[wasm_bindgen]
#[deprecated(since = "0.4.2", note = "use `MSTL.ets` instead")]
#[allow(deprecated)]
pub fn ets(periods: VecUsize, options: Option<ETSOptions>) -> Result<MSTL, JsError> {
    MSTL::ets(periods, options)
}
