//! JavaScript bindings for augurs transformations, such as power transforms, scaling, etc.

// TODO: rewrite all of this. We can just expose a simple enum of available transforms
// and a `Pipeline` struct which is a simpler wrapper of `augurs_forecaster::Pipeline`.

// use std::cell::RefCell;

// use serde::{Deserialize, Serialize};
// use tsify_next::Tsify;
// use wasm_bindgen::prelude::*;

// use augurs_core_js::VecF64;
// use augurs_forecaster::transforms::{self, Transform};

// /// The Yeo-Johnson transform.
// ///
// /// This transform applies the Yeo-Johnson transformation to each item.
// ///
// /// The optimal value of the `lambda` parameter is calculated from the data
// /// using maximum likelihood estimation.
// ///
// /// @experimental
// #[derive(Debug)]
// #[wasm_bindgen]
// pub struct YeoJohnson {
//     inner: transforms::YeoJohnson,
// }

// #[wasm_bindgen]
// impl YeoJohnson {
//     /// Create a new power transform for the given data.
//     ///
//     /// @experimental
//     #[wasm_bindgen(constructor)]
//     pub fn new() -> Self {
//         Self {
//             inner: transforms::YeoJohnson::new(),
//         }
//     }

//     /// Transform the given data.
//     ///
//     /// The transformed data is then scaled using a standard scaler (unless
//     /// `standardize` was set to `false` in the constructor).
//     ///
//     /// @experimental
//     #[wasm_bindgen]
//     pub fn transform(&self, data: VecF64) -> Result<Vec<f64>, JsError> {
//         let data = data.convert()?;
//         self.inner.transform(&mut data)?;
//         if !self.standardize {
//             Ok(data)
//         } else {
//             let scale_params = StandardScaleParams::from_data(transformed.iter().copied());
//             let scaler = Transform::standard_scaler(scale_params.clone());
//             self.scale_params.replace(Some(scale_params));
//             Ok(scaler.transform(transformed.iter().copied()).collect())
//         }
//     }

//     /// Inverse transform the given data.
//     ///
//     /// The data is first scaled back to the original scale using the standard scaler
//     /// (unless `standardize` was set to `false` in the constructor), then the
//     /// inverse power transform is applied.
//     ///
//     /// @experimental
//     #[wasm_bindgen(js_name = "inverseTransform")]
//     pub fn inverse_transform(&self, data: VecF64) -> Result<Vec<f64>, JsError> {
//         match (self.standardize, self.scale_params.borrow().as_ref()) {
//             (true, Some(scale_params)) => {
//                 let inverse_scaler = Transform::standard_scaler(scale_params.clone());
//                 let data = data.convert()?;
//                 let scaled = inverse_scaler.inverse_transform(data.iter().copied());
//                 Ok(self.inner.inverse_transform(scaled).collect())
//             }
//             _ => Ok(self
//                 .inner
//                 .inverse_transform(data.convert()?.iter().copied())
//                 .collect()),
//         }
//     }

//     /// Get the algorithm used by the power transform.
//     ///
//     /// @experimental
//     pub fn algorithm(&self) -> PowerTransformAlgorithm {
//         match self.inner {
//             Transform::BoxCox { .. } => PowerTransformAlgorithm::BoxCox,
//             Transform::YeoJohnson { .. } => PowerTransformAlgorithm::YeoJohnson,
//             _ => unreachable!(),
//         }
//     }

//     /// Retrieve the `lambda` parameter used to transform the data.
//     ///
//     /// @experimental
//     pub fn lambda(&self) -> f64 {
//         match self.inner {
//             Transform::BoxCox { lambda, .. } | Transform::YeoJohnson { lambda, .. } => lambda,
//             _ => unreachable!(),
//         }
//     }
// }
