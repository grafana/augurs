//! Features used by Prophet, such as seasonality, regressors and holidays.
use std::num::NonZeroU32;

use crate::{positive_float::PositiveFloat, TimestampSeconds};

/// The mode of a seasonality, regressor, or holiday.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum FeatureMode {
    /// Additive mode.
    #[default]
    Additive,
    /// Multiplicative mode.
    Multiplicative,
}

/// A holiday.
#[derive(Debug, Clone)]
pub struct Holiday {
    pub(crate) ds: Vec<TimestampSeconds>,
    pub(crate) lower_window: Option<Vec<i32>>,
    pub(crate) upper_window: Option<Vec<i32>>,
    pub(crate) prior_scale: Option<PositiveFloat>,
}

impl Holiday {
    /// Create a new holiday.
    pub fn new(ds: Vec<TimestampSeconds>) -> Self {
        Self {
            ds,
            lower_window: None,
            upper_window: None,
            prior_scale: None,
        }
    }

    /// Set the lower window for the holiday.
    pub fn with_lower_window(mut self, lower_window: Vec<i32>) -> Self {
        self.lower_window = Some(lower_window);
        self
    }

    /// Set the upper window for the holiday.
    pub fn with_upper_window(mut self, upper_window: Vec<i32>) -> Self {
        self.upper_window = Some(upper_window);
        self
    }

    /// Add a prior scale for the holiday.
    pub fn with_prior_scale(mut self, prior_scale: PositiveFloat) -> Self {
        self.prior_scale = Some(prior_scale);
        self
    }
}

/// Whether or not to standardize a regressor.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum Standardize {
    #[default]
    Auto,
    Yes,
    No,
}

impl From<bool> for Standardize {
    fn from(b: bool) -> Self {
        if b {
            Standardize::Yes
        } else {
            Standardize::No
        }
    }
}

/// An exogynous regressor.
///
/// By default, regressors inherit the `seasonality_prior_scale`
/// configured on the Prophet model as their prior scale.
#[derive(Debug, Clone, Default)]
pub struct Regressor {
    pub(crate) mode: FeatureMode,
    pub(crate) prior_scale: Option<PositiveFloat>,
    pub(crate) standardize: Standardize,
    pub(crate) mu: f64,
    pub(crate) std: f64,
}

impl Regressor {
    /// Create a new additive regressor.
    pub fn additive() -> Self {
        Self {
            mode: FeatureMode::Additive,
            ..Default::default()
        }
    }

    /// Create a new multiplicative regressor.
    pub fn multiplicative() -> Self {
        Self {
            mode: FeatureMode::Multiplicative,
            ..Default::default()
        }
    }

    /// Set the prior scale of this regressor.
    ///
    /// By default, regressors inherit the `seasonality_prior_scale`
    /// configured on the Prophet model as their prior scale.
    pub fn with_prior_scale(mut self, prior_scale: PositiveFloat) -> Self {
        self.prior_scale = Some(prior_scale);
        self
    }

    /// Set whether to standardize this regressor.
    pub fn with_standardize(mut self, standardize: Standardize) -> Self {
        self.standardize = standardize;
        self
    }
}

/// A seasonality to include in the model.
#[derive(Debug, Clone)]
pub struct Seasonality {
    pub(crate) period: PositiveFloat,
    pub(crate) fourier_order: NonZeroU32,
    pub(crate) prior_scale: PositiveFloat,
    pub(crate) mode: FeatureMode,
    pub(crate) condition_name: Option<String>,
}

// TODO: add constructors and methods to Seasonality.
