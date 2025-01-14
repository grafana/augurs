//! JavaScript bindings for augurs transformations, such as power transforms, scaling, etc.

use serde::Deserialize;
use tsify_next::Tsify;
use wasm_bindgen::prelude::*;

use augurs_core_js::VecF64;
use augurs_forecaster::transforms::{MinMaxScaler, StandardScaler, Transformer, YeoJohnson};

/// A transformation to be applied to the data.
///
/// @experimental
#[derive(Debug, Deserialize, Tsify)]
#[serde(rename_all = "camelCase", tag = "type")]
#[tsify(from_wasm_abi)]
pub enum Transform {
    /// Scale the data to the range [0, 1].
    MinMaxScaler,
    /// Standardize the data such that it has zero mean and unit variance.
    StandardScaler {
        /// Whether to ignore NaNs.
        #[serde(default, rename = "ignoreNaNs")]
        ignore_nans: bool,
    },
    /// The Yeo-Johnson transform.
    YeoJohnson {
        /// Whether to ignore NaNs.
        #[serde(default, rename = "ignoreNaNs")]
        ignore_nans: bool,
    },
}

impl Transform {
    fn into_transformer(self) -> Box<dyn Transformer> {
        match self {
            Transform::MinMaxScaler => Box::new(MinMaxScaler::new()),
            Transform::StandardScaler { ignore_nans } => {
                Box::new(StandardScaler::new().ignore_nans(ignore_nans))
            }
            Transform::YeoJohnson { ignore_nans } => {
                Box::new(YeoJohnson::new().ignore_nans(ignore_nans))
            }
        }
    }
}

/// A transformation pipeline.
///
/// A pipeline consists of a sequence of transformations that are applied to
/// the data in order. Use the `transform` method to apply the pipeline to
/// the data, and the `inverseTransform` method to reverse the transformations.
///
/// @experimental
#[derive(Debug)]
#[wasm_bindgen]
pub struct Pipeline {
    inner: augurs_forecaster::Pipeline,
}

#[wasm_bindgen]
impl Pipeline {
    /// Create a new pipeline with the given transforms.
    ///
    /// @experimental
    #[wasm_bindgen(constructor)]
    pub fn new(transforms: Vec<Transform>) -> Self {
        Self {
            inner: augurs_forecaster::Pipeline::new(
                transforms
                    .into_iter()
                    .map(|t| t.into_transformer())
                    .collect(),
            ),
        }
    }

    /// Fit the pipeline to the given data.
    ///
    /// Prefer calling `fitTransform` if possible, as it avoids needing
    /// to copy the data as many times.
    ///
    /// @experimental
    #[wasm_bindgen]
    pub fn fit(&mut self, data: VecF64) -> Result<(), JsError> {
        let data = data.convert()?;
        self.inner.fit(&data)?;
        Ok(())
    }

    /// Transform the given data using the pipeline.
    ///
    /// Prefer calling `fitTransform` if possible, as it avoids needing
    /// to copy the data as many times.
    ///
    /// @experimental
    #[wasm_bindgen]
    pub fn transform(&mut self, data: VecF64) -> Result<Vec<f64>, JsError> {
        let mut data = data.convert()?;
        self.inner.transform(&mut data)?;
        Ok(data)
    }

    /// Fit and transform the given data.
    ///
    /// @experimental
    #[wasm_bindgen(js_name = "fitTransform")]
    pub fn fit_transform(&mut self, data: VecF64) -> Result<Vec<f64>, JsError> {
        let mut data = data.convert()?;
        self.inner.fit_transform(&mut data)?;
        Ok(data)
    }

    /// Inverse transform the given data using the pipeline.
    ///
    /// @experimental
    #[wasm_bindgen(js_name = "inverseTransform")]
    pub fn inverse_transform(&mut self, data: VecF64) -> Result<Vec<f64>, JsError> {
        let mut data = data.convert()?;
        self.inner.inverse_transform(&mut data)?;
        Ok(data)
    }
}
