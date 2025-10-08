//! Implementation of a Wasm component that can perform changepoint detection on time series.

use std::{fmt, num::NonZeroUsize};

use augurs_changepoint::{
    self, dist::NormalGamma, BocpdDetector, DefaultArgpcpDetector, DefaultArgpcpDetectorBuilder,
    Detector,
};

// Wrap the wit-bindgen macro in a module so we don't get warned about missing docs in the generated trait.
mod bindings {
    wit_bindgen::generate!({
        world: "changepoint",
        default_bindings_module: "bindings",
    });
}
use crate::bindings::{
    export,
    grafana::augurs::types::{Algorithm, ArgpcpParams, Input, NormalGammaParams, Output},
    Guest,
};

struct ChangepointWorld;
export!(ChangepointWorld);

impl Guest for ChangepointWorld {
    fn detect(input: Input) -> Result<Output, String> {
        detect(input).map_err(|e| e.to_string())
    }
}

/// An error type for the changepoint detector.
#[derive(Debug)]
pub enum ChangepointError {
    /// An invalid parameter was provided to the Normal Gamma distribution.
    NormalGammaError(augurs_changepoint::dist::NormalGammaError),
    /// An overflow occurred when converting an integer to a `NonZeroUsize`.
    TryFromIntError(std::num::TryFromIntError),
    /// An invalid parameter was provided to the max lag.
    InvalidMaxLag(u32),
}

impl From<std::num::TryFromIntError> for ChangepointError {
    fn from(value: std::num::TryFromIntError) -> Self {
        Self::TryFromIntError(value)
    }
}

impl From<augurs_changepoint::dist::NormalGammaError> for ChangepointError {
    fn from(value: augurs_changepoint::dist::NormalGammaError) -> Self {
        Self::NormalGammaError(value)
    }
}

impl fmt::Display for ChangepointError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NormalGammaError(e) => write!(f, "invalid Normal Gamma distribution: {}", e),
            Self::TryFromIntError(e) => write!(f, "overflow converting to usize: {}", e),
            Self::InvalidMaxLag(ml) => write!(f, "invalid max lag: {}", ml),
        }
    }
}

impl std::error::Error for ChangepointError {}

impl TryFrom<ArgpcpParams> for DefaultArgpcpDetectorBuilder {
    type Error = ChangepointError;

    fn try_from(params: ArgpcpParams) -> Result<Self, Self::Error> {
        let mut builder = DefaultArgpcpDetector::builder();
        if let Some(cv) = params.constant_value {
            builder = builder.constant_value(cv);
        }
        if let Some(ls) = params.length_scale {
            builder = builder.length_scale(ls);
        }
        if let Some(nl) = params.noise_level {
            builder = builder.noise_level(nl);
        }
        if let Some(ml) = params.max_lag {
            let ml = NonZeroUsize::new(ml.try_into().map_err(ChangepointError::TryFromIntError)?)
                .ok_or(ChangepointError::InvalidMaxLag(ml))?;
            builder = builder.max_lag(ml);
        }
        if let Some(a0) = params.alpha0 {
            builder = builder.alpha0(a0);
        }
        if let Some(b0) = params.beta0 {
            builder = builder.beta0(b0);
        }
        if let Some(h) = params.logistic_hazard.h {
            builder = builder.logistic_hazard_h(h);
        }
        if let Some(a) = params.logistic_hazard.a {
            builder = builder.logistic_hazard_a(a);
        }
        if let Some(b) = params.logistic_hazard.b {
            builder = builder.logistic_hazard_b(b);
        }
        Ok(builder)
    }
}

fn convert_normal_gamma(
    params: Option<NormalGammaParams>,
) -> Result<NormalGamma, ChangepointError> {
    Ok(NormalGamma::new(
        params.and_then(|p| p.mu).unwrap_or(0.0),
        params.and_then(|p| p.rho).unwrap_or(1.0),
        params.and_then(|p| p.s).unwrap_or(1.0),
        params.and_then(|p| p.v).unwrap_or(1.0),
    )?)
}

fn detect(input: Input) -> Result<Output, ChangepointError> {
    match input.algorithm {
        Algorithm::Argpcp(params) => Ok(DefaultArgpcpDetectorBuilder::try_from(params)?
            .build()
            .detect_changepoints(&input.data)
            .into_iter()
            .map(|i| i.try_into())
            .collect::<Result<_, _>>()?),
        Algorithm::Bocpd(params) => Ok(BocpdDetector::normal_gamma(
            params.hazard_lambda.unwrap_or(250.0),
            convert_normal_gamma(params.normal_gamma_params)?,
        )
        .detect_changepoints(&input.data)
        .into_iter()
        .map(|i| i.try_into())
        .collect::<Result<_, _>>()?),
    }
}
