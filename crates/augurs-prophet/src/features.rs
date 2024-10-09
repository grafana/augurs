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

/// A holiday to be considered by the Prophet model.
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
    ///
    /// The lower window is the number of days before the holiday
    /// that it is observed. For example, if the holiday is on
    /// 2023-01-01 and the lower window is -1, then the holiday will
    /// _also_ be observed on 2022-12-31.
    pub fn with_lower_window(mut self, lower_window: Vec<i32>) -> Self {
        self.lower_window = Some(lower_window);
        self
    }

    /// Set the upper window for the holiday.
    ///
    /// The upper window is the number of days after the holiday
    /// that it is observed. For example, if the holiday is on
    /// 2023-01-01 and the upper window is 1, then the holiday will
    /// _also_ be observed on 2023-01-02.
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
    /// Automatically determine whether to standardize.
    ///
    /// Numeric regressors will be standardized while
    /// binary regressors will not.
    #[default]
    Auto,
    /// Standardize this regressor.
    Yes,
    /// Do not standardize this regressor.
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

/// Scales for a regressor.
///
/// This will be inserted into [`Scales::extra_regressors`]
/// if the regressor is standardized.
#[derive(Debug, Clone, Default)]
pub(crate) struct RegressorScale {
    /// Whether to standardize this regressor.
    ///
    /// This is a `bool` rather than a `Standardize`
    /// because we'll have decided whether to automatically
    /// standardize by the time this is constructed.
    pub(crate) standardize: bool,
    /// The mean of the regressor.
    pub(crate) mu: f64,
    /// The standard deviation of the regressor.
    pub(crate) std: f64,
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
    pub(crate) prior_scale: Option<PositiveFloat>,
    pub(crate) mode: Option<FeatureMode>,
    pub(crate) condition_name: Option<String>,
}

impl Seasonality {
    /// Create a new `Seasonality` with the given period and fourier order.
    ///
    /// By default, the prior scale and mode will be inherited from the
    /// Prophet model config, and the seasonality is assumed to be
    /// non-conditional.
    pub fn new(period: PositiveFloat, fourier_order: NonZeroU32) -> Self {
        Self {
            period,
            fourier_order,
            prior_scale: None,
            mode: None,
            condition_name: None,
        }
    }

    /// Set the prior scale of this seasonality.
    ///
    /// By default, seasonalities inherit the prior scale
    /// configured on the Prophet model; this allows the
    /// prior scale to be customised for each seasonality.
    pub fn with_prior_scale(mut self, prior_scale: PositiveFloat) -> Self {
        self.prior_scale = Some(prior_scale);
        self
    }

    /// Set the mode of this seasonality.
    ///
    /// By default, seasonalities inherit the mode
    /// configured on the Prophet model; this allows the
    /// mode to be customised for each seasonality.
    pub fn with_mode(mut self, mode: FeatureMode) -> Self {
        self.mode = Some(mode);
        self
    }

    /// Set this seasonality as conditional.
    ///
    /// A column with the provided condition name must be
    /// present in the data passed to Prophet otherwise
    /// training will fail. This can be added with
    /// [`TrainingData::with_seasonality_conditions`](crate::TrainingData::with_seasonality_conditions).
    pub fn with_condition(mut self, condition_name: String) -> Self {
        self.condition_name = Some(condition_name);
        self
    }
}
