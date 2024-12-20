//! JavaScript bindings for augurs transformations, such as power transforms, scaling, etc.

// TODO: rewrite all of this. We can just expose a simple enum of available transforms
// and a `Pipeline` struct which is a simpler wrapper of `augurs_forecaster::Pipeline`.

use serde::Deserialize;
use tsify_next::Tsify;
use wasm_bindgen::prelude::*;

use augurs_core_js::VecF64;
use augurs_forecaster::transforms;

/// A transformation to be applied to the data.
///
/// @experimental
#[derive(Debug, Deserialize, Tsify)]
#[serde(rename_all = "camelCase")]
#[tsify(from_wasm_abi)]
pub enum Transform {
    /// Standardize the data such that it has zero mean and unit variance.
    StandardScaler,
    /// The Yeo-Johnson transform.
    YeoJohnson,
}

impl Transform {
    fn into_transform(self) -> Box<dyn augurs_forecaster::Transform> {
        match self {
            Transform::StandardScaler => Box::new(transforms::StandardScaler::new()),
            Transform::YeoJohnson => Box::new(transforms::YeoJohnson::new()),
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
                transforms.into_iter().map(|t| t.into_transform()).collect(),
            ),
        }
    }

    /// Transform the given data using the pipeline.
    ///
    /// @experimental
    #[wasm_bindgen]
    pub fn transform(&mut self, data: VecF64) -> Result<Vec<f64>, JsError> {
        let mut data = data.convert()?;
        self.inner.transform(&mut data)?;
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
