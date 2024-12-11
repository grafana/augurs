//! JavaScript bindings for augurs transformations, such as power transforms, scaling, etc.

use serde::Deserialize;
use tsify_next::Tsify;
use wasm_bindgen::prelude::*;

use augurs_core_js::VecF64;
use augurs_forecaster::transforms::Transform;

/// A power transform.
///
/// This transform applies the power function to each item.
///
/// If all values are positive, it will use the Box-Cox transform.
/// If any values are negative or zero, it will use the Yeo-Johnson transform.
///
/// The optimal value of the `lambda` parameter is calculated from the data
/// using maximum likelihood estimation.
#[derive(Debug)]
#[wasm_bindgen]
pub struct PowerTransform {
    inner: Transform,
}

#[wasm_bindgen]
impl PowerTransform {
    /// Create a new power transform for the given data.
    #[wasm_bindgen(constructor)]
    pub fn new(opts: PowerTransformOptions) -> Result<PowerTransform, JsError> {
        Ok(PowerTransform {
            inner: Transform::power_transform(&opts.data)
                .map_err(|e| JsError::new(&e.to_string()))?,
        })
    }

    /// Transform the given data.
    #[wasm_bindgen]
    pub fn transform(&self, data: VecF64) -> Result<Vec<f64>, JsError> {
        Ok(self
            .inner
            .transform(data.convert()?.iter().copied())
            .collect())
    }

    /// Inverse transform the given data.
    #[wasm_bindgen(js_name = "inverseTransform")]
    pub fn inverse_transform(&self, data: VecF64) -> Result<Vec<f64>, JsError> {
        Ok(self
            .inner
            .inverse_transform(data.convert()?.iter().copied())
            .collect())
    }
}

/// Options for the power transform.
#[derive(Debug, Default, Deserialize, Tsify)]
#[serde(rename_all = "camelCase")]
#[tsify(from_wasm_abi)]
pub struct PowerTransformOptions {
    /// The data to transform. This is used to calculate the optimal value of 'lambda'.
    #[tsify(type = "number[] | Float64Array")]
    pub data: Vec<f64>,
}
