use js_sys::Float64Array;
use serde::{Deserialize, Serialize};
use tsify::Tsify;
use wasm_bindgen::prelude::*;

use augurs_changepoint::{dist, GaussianDetector};

/// A changepoint detector.
#[derive(Debug, Default)]
#[wasm_bindgen]
pub struct ChangepointDetector {
    detector: GaussianDetector,
}

#[wasm_bindgen]
impl ChangepointDetector {
    #[wasm_bindgen]
    /// Create a new changepoint detector.
    pub fn gaussian(opts: Option<GaussianDetectorOpts>) -> Result<ChangepointDetector, JsValue> {
        let GaussianDetectorOpts { hazard, prior } = opts.unwrap_or_default();
        Ok(Self {
            detector: GaussianDetector::gaussian(
                hazard.unwrap_or(250.0),
                dist::NormalGamma::try_from(prior.unwrap_or_default())
                    .map_err(|e| e.to_string())?,
            ),
        })
    }

    #[wasm_bindgen]
    /// Detect changepoints in the given time series.
    pub fn detect_changepoints(&mut self, y: Float64Array) -> Changepoints {
        Changepoints {
            indices: self.detector.detect_changepoints(&y.to_vec()),
        }
    }
}

/// Parameters for the Normal Gamma prior.
/// Options for the ETS MSTL model.
#[derive(Debug, Deserialize, Serialize, Tsify)]
#[tsify(from_wasm_abi, into_wasm_abi)]
pub struct NormalGammaParameters {
    /// The prior mean.
    ///
    /// Defaults to 0.0.
    #[tsify(optional)]
    pub mu: Option<f64>,
    /// The relative precision of μ versus data.
    ///
    /// Defaults to 1.0.
    #[tsify(optional)]
    pub rho: Option<f64>,
    /// The mean of rho (the precision) is v/s.
    ///
    /// Defaults to 1.0.
    #[tsify(optional)]
    pub s: Option<f64>,
    /// The degrees of freedom of precision of rho.
    ///
    /// Defaults to 1.0.
    #[tsify(optional)]
    pub v: Option<f64>,
}

impl NormalGammaParameters {
    const DEFAULT_MU: f64 = 0.0;
    const DEFAULT_RHO: f64 = 1.0;
    const DEFAULT_S: f64 = 1.0;
    const DEFAULT_V: f64 = 1.0;
}

impl Default for NormalGammaParameters {
    fn default() -> Self {
        Self {
            mu: Some(Self::DEFAULT_MU),
            rho: Some(Self::DEFAULT_RHO),
            s: Some(Self::DEFAULT_S),
            v: Some(Self::DEFAULT_V),
        }
    }
}

impl TryFrom<NormalGammaParameters> for dist::NormalGamma {
    type Error = dist::NormalGammaError;

    fn try_from(params: NormalGammaParameters) -> Result<Self, Self::Error> {
        let NormalGammaParameters { mu, rho, s, v } = params;
        Self::new(
            mu.unwrap_or(NormalGammaParameters::DEFAULT_MU),
            rho.unwrap_or(NormalGammaParameters::DEFAULT_RHO),
            s.unwrap_or(NormalGammaParameters::DEFAULT_S),
            v.unwrap_or(NormalGammaParameters::DEFAULT_V),
        )
    }
}

/// Options for the Normal Gamma changepoint detector.
#[derive(Debug, Default, Deserialize, Tsify)]
#[tsify(from_wasm_abi)]
pub struct GaussianDetectorOpts {
    #[tsify(optional)]
    pub hazard: Option<f64>,

    #[tsify(optional)]
    pub prior: Option<NormalGammaParameters>,
}

/// Changepoints detected in a time series.
#[derive(Debug, Serialize, Tsify)]
#[tsify(into_wasm_abi)]
pub struct Changepoints {
    /// The indices of the most likely changepoints.
    indices: Vec<usize>,
}
