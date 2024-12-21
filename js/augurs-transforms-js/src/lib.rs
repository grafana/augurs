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
    inner: Option<Transform>,
    standardize: Standardize,
    scale_params: Option<StandardScaleParams>,
}

#[wasm_bindgen]
impl PowerTransform {
    /// Create a new power transform for the given data.
    ///
    /// @experimental
    #[wasm_bindgen(constructor)]
    pub fn new(opts: PowerTransformOptions) -> Result<PowerTransform, JsError> {
        Ok(PowerTransform {
            inner: None,
            standardize: opts.standardize.unwrap_or_default(),
            scale_params: None,
        })
    }

    /// Transform the given data.
    ///
    /// The data is also scaled either before or after being transformed as per the standardize
    /// option.
    ///
    /// @experimental
    #[wasm_bindgen]
    pub fn transform(&mut self, data: VecF64) -> Result<Vec<f64>, JsError> {
        let data = data.convert()?;
        Ok(match self.standardize {
            Standardize::None => {
                let transform =
                    Transform::power_transform(&data).map_err(|e| JsError::new(&e.to_string()))?;
                let result = transform.transform(data.iter().copied()).collect();
                self.inner = Some(transform);
                result
            }
            Standardize::Before => {
                let scale_params = StandardScaleParams::from_data(data.iter().copied());
                let scaler = Transform::standard_scaler(scale_params.clone());
                self.scale_params = Some(scale_params);
                let scaled: Vec<_> = scaler.transform(data.iter().copied()).collect();

                let transform = Transform::power_transform(&scaled)
                    .map_err(|e| JsError::new(&e.to_string()))?;
                let result = transform.transform(scaled.iter().copied()).collect();
                self.inner = Some(transform);

                result
            }
            Standardize::After => {
                let transform =
                    Transform::power_transform(&data).map_err(|e| JsError::new(&e.to_string()))?;

                let transformed: Vec<_> = transform.transform(data.iter().copied()).collect();
                self.inner = Some(transform);

                let scale_params = StandardScaleParams::from_data(transformed.iter().copied());
                let scaler = Transform::standard_scaler(scale_params.clone());
                self.scale_params = Some(scale_params);
                scaler.transform(transformed.iter().copied()).collect()
            }
        })
    }

    /// Inverse transform the given data.
    ///
    /// The data is also inversely scaled according to the standardize option. The ordering is
    /// opposite the order done in transform, i.e if transform scales first then transforms, then
    /// inverse_transform transforms then scales.
    ///
    /// @experimental
    #[wasm_bindgen(js_name = "inverseTransform")]
    pub fn inverse_transform(&self, data: VecF64) -> Result<Vec<f64>, JsError> {
        let data = data.convert()?;
        let transformer = self.inner.clone().unwrap();
        Ok(match (self.standardize, self.scale_params.clone()) {
            (Standardize::Before, Some(scale_params)) => {
                let inverse_transformed = transformer.inverse_transform(data.iter().copied());
                let inverse_scaler = Transform::standard_scaler(scale_params.clone());
                inverse_scaler
                    .inverse_transform(inverse_transformed)
                    .collect()
            }
            (Standardize::After, Some(scale_params)) => {
                let inverse_scaler = Transform::standard_scaler(scale_params.clone());
                let scaled = inverse_scaler.inverse_transform(data.iter().copied());
                transformer.inverse_transform(scaled).collect()
            }
            _ => transformer
                .inverse_transform(data.iter().copied())
                .collect(),
        })
    }

    /*
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
    */
}

/// When to standardize the data.
#[derive(Debug, Default, Clone, Copy, Eq, PartialEq, Serialize, Deserialize, Tsify)]
#[serde(rename_all = "camelCase")]
#[tsify(from_wasm_abi)]
pub enum Standardize {
    /// Only run a power transform, do not standardize the data.
    None,
    /// Standardize the data before running the power transform. This may provide better results for data
    /// with a non-zero floor.
    Before,
    /// Standardize the data after running the power transform. This matches the default in sklearn.
    #[default]
    After,
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
    /// This is generally recommended, and defaults to [`Standardize::After`] to match sklearn.
    #[tsify(optional)]
    pub standardize: Option<Standardize>,
}
