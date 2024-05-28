//! JavaScript bindings for the AutoETS model.

use js_sys::Float64Array;
use wasm_bindgen::prelude::*;

use augurs_core::prelude::*;

use crate::Forecast;

/// Automatic ETS model selection.
#[derive(Debug)]
#[wasm_bindgen]
pub struct AutoETS {
    /// The inner model search instance.
    inner: augurs_ets::AutoETS,
    fitted: Option<augurs_ets::FittedAutoETS>,
}

#[wasm_bindgen]
impl AutoETS {
    /// Create a new `AutoETS` model search instance.
    ///
    /// # Errors
    ///
    /// If the `spec` string is invalid, this function returns an error.
    #[wasm_bindgen(constructor)]
    pub fn new(seasonLength: usize, spec: String) -> Result<AutoETS, JsValue> {
        let inner =
            augurs_ets::AutoETS::new(seasonLength, spec.as_str()).map_err(|e| e.to_string())?;
        Ok(Self {
            inner,
            fitted: None,
        })
    }

    /// Search for the best model, fitting it to the data.
    ///
    /// The model will be stored on the inner `AutoETS` instance, after which
    /// forecasts can be produced using its `predict` method.
    ///
    /// # Errors
    ///
    /// If no model can be found, or if any parameters are invalid, this function
    /// returns an error.
    #[wasm_bindgen]
    pub fn fit(&mut self, y: Float64Array) -> Result<(), JsValue> {
        self.fitted = Some(self.inner.fit(&y.to_vec()).map_err(|e| e.to_string())?);
        Ok(())
    }

    /// Predict the next `horizon` values using the best model, optionally including
    /// prediction intervals at the specified level.
    ///
    /// `level` should be a float between 0 and 1 representing the confidence level.
    ///
    /// # Errors
    ///
    /// This function will return an error if no model has been fit yet (using [`AutoETS::fit`]).
    #[wasm_bindgen]
    pub fn predict(&self, horizon: usize, level: Option<f64>) -> Result<Forecast, JsValue> {
        Ok(self
            .fitted
            .as_ref()
            .map(|x| x.predict(horizon, level))
            .ok_or("model not fit yet")?
            .map(Into::into)
            .map_err(|e| e.to_string())?)
    }
}
