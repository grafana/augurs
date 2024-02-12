//! JavaScript bindings for the MSTL model.
use js_sys::Float64Array;
use wasm_bindgen::prelude::*;

use augurs_ets::AutoETS;
use augurs_mstl::{Fit, MSTLModel, TrendModel, Unfit};

use crate::Forecast;

#[derive(Debug)]
enum MSTLEnum<T> {
    Unfit(MSTLModel<T, Unfit>),
    Fit(MSTLModel<T, Fit>),
}

/// A MSTL model.
#[derive(Debug)]
#[wasm_bindgen]
pub struct MSTL {
    inner: Option<MSTLEnum<Box<dyn TrendModel + Sync + Send>>>,
}

#[wasm_bindgen]
impl MSTL {
    /// Fit the model to the given time series.
    #[wasm_bindgen]
    pub fn fit(&mut self, y: Float64Array) -> Result<(), JsValue> {
        self.inner = match std::mem::take(&mut self.inner) {
            Some(MSTLEnum::Unfit(inner)) => Some(MSTLEnum::Fit(
                inner.fit(&y.to_vec()).map_err(|e| e.to_string())?,
            )),
            x => x,
        };
        Ok(())
    }

    /// Predict the next `horizon` values, optionally including prediction
    /// intervals at the given level.
    ///
    /// If provided, `level` must be a float between 0 and 1.
    #[wasm_bindgen]
    pub fn predict(&self, horizon: usize, level: Option<f64>) -> Result<Forecast, JsValue> {
        match &self.inner {
            Some(MSTLEnum::Fit(inner)) => Ok(inner
                .predict(horizon, level)
                .map(Into::into)
                .map_err(|e| e.to_string())?),
            _ => Err(JsValue::from_str("model is not fit")),
        }
    }

    /// Produce in-sample forecasts, optionally including prediction
    /// intervals at the given level.
    ///
    /// If provided, `level` must be a float between 0 and 1.
    #[wasm_bindgen]
    pub fn predict_in_sample(&self, level: Option<f64>) -> Result<Forecast, JsValue> {
        match &self.inner {
            Some(MSTLEnum::Fit(inner)) => Ok(inner
                .predict_in_sample(level)
                .map(Into::into)
                .map_err(|e| e.to_string())?),
            _ => Err(JsValue::from_str("model is not fit")),
        }
    }
}

#[wasm_bindgen]
/// Create a new MSTL model with the given periods using the `AutoETS` trend model.
pub fn ets(periods: Vec<usize>) -> MSTL {
    let ets = AutoETS::non_seasonal();
    MSTL {
        inner: Some(MSTLEnum::Unfit(MSTLModel::new(periods, Box::new(ets)))),
    }
}
