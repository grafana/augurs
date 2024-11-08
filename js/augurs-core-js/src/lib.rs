//! JS bindings for core augurs functionality.
use js_sys::Float64Array;
use serde::Serialize;
use tsify_next::Tsify;
use wasm_bindgen::prelude::*;

#[cfg(feature = "logging")]
pub mod logging;

/// Initialize the logger and panic hook.
///
/// This will be called automatically when the module is imported.
/// It sets the default tracing subscriber to `tracing-wasm`, and
/// sets WASM panics to print to the console with a helpful error
/// message.
#[wasm_bindgen(start)]
pub fn custom_init() {
    console_error_panic_hook::set_once();
}

// Wrapper types for the core types, so we can derive `Tsify` for them.
// This avoids having to worry about `tsify` in the `augurs-core` crate.

/// Forecast intervals.
#[derive(Clone, Debug, Serialize, Tsify)]
#[tsify(into_wasm_abi)]
pub struct ForecastIntervals {
    /// The confidence level for the intervals.
    pub level: f64,
    /// The lower prediction intervals.
    pub lower: Vec<f64>,
    /// The upper prediction intervals.
    pub upper: Vec<f64>,
}

impl From<augurs_core::ForecastIntervals> for ForecastIntervals {
    fn from(f: augurs_core::ForecastIntervals) -> Self {
        Self {
            level: f.level,
            lower: f.lower,
            upper: f.upper,
        }
    }
}

/// A forecast containing point forecasts and, optionally, prediction intervals.
#[derive(Clone, Debug, Serialize, Tsify)]
#[tsify(into_wasm_abi)]
pub struct Forecast {
    /// The point forecasts.
    pub point: Vec<f64>,
    /// The forecast intervals, if requested and supported
    /// by the trend model.
    pub intervals: Option<ForecastIntervals>,
}

impl From<augurs_core::Forecast> for Forecast {
    fn from(f: augurs_core::Forecast) -> Self {
        Self {
            point: f.point,
            intervals: f.intervals.map(Into::into),
        }
    }
}

// These custom types are needed to have the correct TypeScript types generated
// for functions which accept either `number[]` or typed arrays when called
// from Javascript.
// They should always be preferred over using `Vec<T>` directly in functions
// exported to Javascript, even if it is a bit of hassle to convert them.
// They can be converted using:
//
//     let y = y.convert()?;
#[wasm_bindgen]
extern "C" {
    /// Custom type for `Vec<u32>`.
    #[wasm_bindgen(typescript_type = "number[] | Uint32Array")]
    #[derive(Debug)]
    pub type VecU32;

    /// Custom type for `Vec<usize>`.
    #[wasm_bindgen(typescript_type = "number[] | Uint32Array")]
    #[derive(Debug)]
    pub type VecUsize;

    /// Custom type for `Vec<f64>`.
    #[wasm_bindgen(typescript_type = "number[] | Float64Array")]
    #[derive(Debug)]
    pub type VecF64;

    /// Custom type for `Vec<Vec<f64>>`.
    #[wasm_bindgen(typescript_type = "number[][] | Float64Array[]")]
    #[derive(Debug)]
    pub type VecVecF64;
}

impl VecUsize {
    /// Convert to a `Vec<usize>`.
    pub fn convert(self) -> Result<Vec<usize>, JsError> {
        serde_wasm_bindgen::from_value(self.into())
            .map_err(|_| JsError::new("TypeError: expected array of integers or Uint32Array"))
    }
}

impl VecF64 {
    /// Convert to a `Vec<f64>`.
    pub fn convert(self) -> Result<Vec<f64>, JsError> {
        serde_wasm_bindgen::from_value(self.into())
            .map_err(|_| JsError::new("TypeError: expected array of numbers or Float64Array"))
    }
}

impl VecVecF64 {
    /// Convert to a `Vec<Vec<f64>>`.
    pub fn convert(self) -> Result<Vec<Vec<f64>>, JsError> {
        serde_wasm_bindgen::from_value(self.into()).map_err(|_| {
            JsError::new("TypeError: expected array of number arrays or array of Float64Array")
        })
    }
}

/// A distance matrix.
///
/// This is intentionally opaque; it should only be passed back to `augurs` for further processing,
/// e.g. to calculate nearest neighbors or perform clustering.
#[derive(Debug)]
pub struct DistanceMatrix {
    inner: augurs_core::DistanceMatrix,
}

impl DistanceMatrix {
    /// Get the inner distance matrix.
    pub fn inner(&self) -> &augurs_core::DistanceMatrix {
        &self.inner
    }
}

impl DistanceMatrix {
    /// Create a new `DistanceMatrix` from a raw distance matrix.
    #[allow(non_snake_case)]
    pub fn new(distance_matrix: VecVecF64) -> Result<DistanceMatrix, JsError> {
        Ok(Self {
            inner: augurs_core::DistanceMatrix::try_from_square(distance_matrix.convert()?)?,
        })
    }

    /// Get the shape of the distance matrix.
    pub fn shape(&self) -> Vec<usize> {
        let (m, n) = self.inner.shape();
        vec![m, n]
    }

    /// Get the distance matrix as an array of arrays.
    pub fn to_array(&self) -> Vec<Float64Array> {
        self.inner
            .clone()
            .into_inner()
            .into_iter()
            .map(|x| {
                let arr = Float64Array::new_with_length(x.len() as u32);
                arr.copy_from(&x);
                arr
            })
            .collect()
    }
}

impl From<augurs_core::DistanceMatrix> for DistanceMatrix {
    fn from(inner: augurs_core::DistanceMatrix) -> Self {
        Self { inner }
    }
}

impl From<DistanceMatrix> for augurs_core::DistanceMatrix {
    fn from(matrix: DistanceMatrix) -> Self {
        Self::try_from_square(matrix.inner.into_inner()).unwrap()
    }
}
