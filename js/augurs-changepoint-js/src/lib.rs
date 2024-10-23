use std::num::NonZeroUsize;

use serde::{Deserialize, Serialize};
use tsify_next::Tsify;
use wasm_bindgen::prelude::*;

use augurs_changepoint::{
    dist, ArgpcpDetector, BocpdDetector, DefaultArgpcpDetector, Detector, NormalGammaDetector,
};
use augurs_core_js::VecF64;

#[derive(Debug)]
enum EitherDetector {
    NormalGamma(NormalGammaDetector),
    Argpcp(DefaultArgpcpDetector),
}

impl EitherDetector {
    fn detect_changepoints(&mut self, y: &[f64]) -> Vec<usize> {
        match self {
            EitherDetector::NormalGamma(x) => x.detect_changepoints(y),
            EitherDetector::Argpcp(x) => x.detect_changepoints(y),
        }
    }
}

/// The type of changepoint detector to use.
#[derive(Debug, Clone, Copy, Deserialize, Tsify)]
#[serde(rename_all = "kebab-case")]
#[tsify(from_wasm_abi)]
pub enum ChangepointDetectorType {
    /// A Bayesian Online Changepoint Detector with a Normal Gamma prior.
    NormalGamma,
    /// An autoregressive Gaussian Process changepoint detector,
    /// with the default kernel and parameters.
    DefaultArgpcp,
}

/// A changepoint detector.
#[derive(Debug)]
#[wasm_bindgen]
pub struct ChangepointDetector {
    detector: EitherDetector,
}

const DEFAULT_HAZARD_LAMBDA: f64 = 250.0;

#[wasm_bindgen]
impl ChangepointDetector {
    #[wasm_bindgen(constructor)]
    #[allow(non_snake_case)]
    pub fn new(detectorType: ChangepointDetectorType) -> Result<ChangepointDetector, JsValue> {
        match detectorType {
            ChangepointDetectorType::NormalGamma => Self::normal_gamma(None),
            ChangepointDetectorType::DefaultArgpcp => Self::default_argpcp(None),
        }
    }

    /// Create a new Bayesian Online changepoint detector with a Normal Gamma prior.
    #[wasm_bindgen(js_name = "normalGamma")]
    pub fn normal_gamma(
        opts: Option<NormalGammaDetectorOptions>,
    ) -> Result<ChangepointDetector, JsValue> {
        let NormalGammaDetectorOptions {
            hazard_lambda,
            prior,
        } = opts.unwrap_or_default();
        Ok(Self {
            detector: EitherDetector::NormalGamma(BocpdDetector::normal_gamma(
                hazard_lambda.unwrap_or(DEFAULT_HAZARD_LAMBDA),
                dist::NormalGamma::try_from(prior.unwrap_or_default())
                    .map_err(|e| e.to_string())?,
            )),
        })
    }

    /// Create a new Autoregressive Gaussian Process changepoint detector
    /// with the default kernel and parameters.
    #[wasm_bindgen(js_name = "defaultArgpcp")]
    pub fn default_argpcp(
        opts: Option<DefaultArgpcpDetectorOptions>,
    ) -> Result<ChangepointDetector, JsValue> {
        let mut builder = ArgpcpDetector::builder();
        if let Some(opts) = opts {
            if let Some(cv) = opts.constant_value {
                builder = builder.constant_value(cv);
            }
            // TODO: fill in the rest of these opts.
        }
        Ok(Self {
            detector: EitherDetector::Argpcp(builder.build()),
        })
    }

    /// Detect changepoints in the given time series.
    #[wasm_bindgen(js_name = "detectChangepoints")]
    pub fn detect_changepoints(&mut self, y: VecF64) -> Result<Changepoints, JsError> {
        Ok(Changepoints {
            indices: self.detector.detect_changepoints(&y.convert()?),
        })
    }
}

/// Parameters for the Normal Gamma prior.
/// Options for the ETS MSTL model.
#[derive(Debug, Deserialize, Serialize, Tsify)]
#[serde(rename_all = "camelCase")]
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
#[serde(rename_all = "camelCase")]
#[tsify(from_wasm_abi)]
pub struct NormalGammaDetectorOptions {
    /// The hazard lambda.
    ///
    /// `1/hazard` is the probability of the next step being a changepoint.
    /// Therefore, the larger the value, the lower the prior probability
    /// is for the any point to be a change-point.
    /// Mean run-length is lambda - 1.
    ///
    /// Defaults to 250.0.
    #[tsify(optional)]
    pub hazard_lambda: Option<f64>,

    /// The prior for the Normal distribution.
    #[tsify(optional)]
    pub prior: Option<NormalGammaParameters>,
}

/// Options for the default Autoregressive Gaussian Process detector.
#[derive(Debug, Default, Deserialize, Tsify)]
#[serde(rename_all = "camelCase")]
#[tsify(from_wasm_abi)]
pub struct DefaultArgpcpDetectorOptions {
    /// The value of the constant kernel.
    #[tsify(optional)]
    pub constant_value: Option<f64>,
    /// The length scale of the RBF kernel.
    #[tsify(optional)]
    pub length_scale: Option<f64>,
    /// The noise level of the white kernel.
    #[tsify(optional)]
    pub noise_level: Option<f64>,
    /// The maximum autoregressive lag.
    #[tsify(type = "number", optional)]
    pub max_lag: Option<NonZeroUsize>,
    /// Scale Gamma distribution alpha parameter.
    #[tsify(optional)]
    pub alpha0: Option<f64>,
    /// Scale Gamma distribution beta parameter.
    #[tsify(optional)]
    pub beta0: Option<f64>,
    #[tsify(optional)]
    pub logistic_hazard_h: Option<f64>,
    #[tsify(optional)]
    pub logistic_hazard_a: Option<f64>,
    #[tsify(optional)]
    pub logistic_hazard_b: Option<f64>,
}

/// Changepoints detected in a time series.
#[derive(Debug, Serialize, Tsify)]
#[tsify(into_wasm_abi)]
pub struct Changepoints {
    /// The indices of the most likely changepoints.
    indices: Vec<usize>,
}
