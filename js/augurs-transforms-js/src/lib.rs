//! JavaScript bindings for augurs transformations, such as power transforms, scaling, etc.

use serde::{Deserialize, Serialize};
use tsify_next::Tsify;
use wasm_bindgen::prelude::*;

use augurs_core_js::VecF64;
use augurs_forecaster::transforms::{StandardScaleParams, Transform};

/// The algorithm used by a power transform.
#[derive(Debug, Serialize, Tsify)]
#[serde(rename_all = "camelCase")]
#[tsify(into_wasm_abi)]
pub enum PowerTransformAlgorithm {
    /// The Box-Cox transform.
    BoxCox,
    /// The Yeo-Johnson transform.
    YeoJohnson,
}

/// A power transform.
///
/// This transform applies the power function to each item.
///
/// If all values are positive, it will use the Box-Cox transform.
/// If any values are negative or zero, it will use the Yeo-Johnson transform.
///
/// The optimal value of the `lambda` parameter is calculated from the data
/// using maximum likelihood estimation.
///
/// @experimental
#[derive(Debug)]
#[wasm_bindgen]
pub struct PowerTransform {
    inner: Transform,
    scale_params: Option<StandardScaleParams>,
}

#[wasm_bindgen]
impl PowerTransform {
    /// Create a new power transform for the given data.
    ///
    /// @experimental
    #[wasm_bindgen(constructor)]
    pub fn new(opts: PowerTransformOptions) -> Result<PowerTransform, JsError> {
        let (scale_params, inner) = if opts.standardize {
            let scale_params = StandardScaleParams::from_data(opts.data.iter().copied());
            let scaler = Transform::standard_scaler(scale_params.clone());
            let scaled: Vec<_> = scaler.transform(opts.data.iter().copied()).collect();
            (
                Some(scale_params),
                Transform::power_transform(&scaled).map_err(|e| JsError::new(&e.to_string()))?,
            )
        } else {
            (
                None,
                Transform::power_transform(&opts.data).map_err(|e| JsError::new(&e.to_string()))?,
            )
        };
        Ok(PowerTransform {
            inner,
            scale_params,
        })
    }

    /// Transform the given data.
    ///
    /// The transformed data is then scaled using a standard scaler (unless
    /// `standardize` was set to `false` in the constructor).
    ///
    /// @experimental
    #[wasm_bindgen]
    pub fn transform(&self, data: VecF64) -> Result<Vec<f64>, JsError> {
        let data = data.convert()?;
        if let Some(scale_params) = &self.scale_params {
            let scaler = Transform::standard_scaler(scale_params.clone());
            let scaled: Vec<_> = scaler.transform(data.iter().copied()).collect();
            Ok(self.inner.transform(scaled.iter().copied()).collect())
        } else {
            Ok(self.inner.transform(data.iter().copied()).collect())
        }
    }

    /// Inverse transform the given data.
    ///
    /// The data is first scaled back to the original scale using the standard scaler
    /// (unless `standardize` was set to `false` in the constructor), then the
    /// inverse power transform is applied.
    ///
    /// @experimental
    #[wasm_bindgen(js_name = "inverseTransform")]
    pub fn inverse_transform(&self, data: VecF64) -> Result<Vec<f64>, JsError> {
        let data = data.convert()?;
        if let Some(scale_params) = &self.scale_params {
            let scaler = Transform::standard_scaler(scale_params.clone());
            let inverse_transformed = self.inner.inverse_transform(data.iter().copied());
            Ok(scaler.inverse_transform(inverse_transformed).collect())
        } else {
            Ok(self.inner.inverse_transform(data.iter().copied()).collect())
        }
    }

    /// Get the algorithm used by the power transform.
    ///
    /// @experimental
    pub fn algorithm(&self) -> PowerTransformAlgorithm {
        match self.inner {
            Transform::BoxCox { .. } => PowerTransformAlgorithm::BoxCox,
            Transform::YeoJohnson { .. } => PowerTransformAlgorithm::YeoJohnson,
            _ => unreachable!(),
        }
    }

    /// Retrieve the `lambda` parameter used to transform the data.
    ///
    /// @experimental
    pub fn lambda(&self) -> f64 {
        match self.inner {
            Transform::BoxCox { lambda, .. } | Transform::YeoJohnson { lambda, .. } => lambda,
            _ => unreachable!(),
        }
    }
}

fn default_standardize() -> bool {
    true
}

/// Options for the power transform.
#[derive(Debug, Default, Deserialize, Tsify)]
#[serde(rename_all = "camelCase")]
#[tsify(from_wasm_abi)]
pub struct PowerTransformOptions {
    /// The data to transform. This is used to calculate the optimal value of 'lambda'.
    #[tsify(type = "number[] | Float64Array")]
    pub data: Vec<f64>,

    /// Whether to standardize the data after applying the power transform.
    ///
    /// This is generally recommended, and defaults to `true`.
    #[serde(default = "default_standardize")]
    #[tsify(optional)]
    pub standardize: bool,
}
