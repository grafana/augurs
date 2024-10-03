//! Options to configure Prophet.
use std::{collections::HashMap, num::NonZeroU32};

use crate::{FeatureMode, Holiday, PositiveFloat, TrendIndicator};

/// The type of growth to use.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GrowthType {
    /// Linear growth (default).
    Linear,
    /// 0
    /// Logistic growth.
    Logistic,
    /// 1
    /// Flat growth.
    Flat,
}

impl From<GrowthType> for TrendIndicator {
    fn from(value: GrowthType) -> Self {
        match value {
            GrowthType::Linear => TrendIndicator::Linear,
            GrowthType::Logistic => TrendIndicator::Logistic,
            GrowthType::Flat => TrendIndicator::Flat,
        }
    }
}

/// Define whether to include a specific seasonality, and how it should be specified.
#[derive(Clone, Copy, Debug, Default)]
pub enum SeasonalityOption {
    /// Automatically determine whether to include this seasonality.
    #[default]
    Auto,
    /// Manually specify whether to include this seasonality.
    Manual(bool),
    /// Enable this seasonality and use the provided number of Fourier terms.
    Fourier(NonZeroU32),
}

/// How to scale the data prior to fitting the model.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Scaling {
    /// Use abs-max scaling (the default).
    AbsMax,
    /// Use min-max scaling.
    MinMax,
}

/// How to do parameter estimation.
#[derive(Clone, Debug, Copy)]
pub enum EstimationMode {
    /// Use MLE estimation.
    Mle,
    /// Use MAP estimation.
    Map,
    /// Do full Bayesian inference with the specified number of MCMC samples.
    Mcmc(u32),
}

// TODO: consider getting rid of this? It's a bit weird, but it might
// make it easier for users of the crate...
/// Optional version of Prophet's options, before applying any defaults.
#[derive(Default, Debug, Clone)]
pub struct OptProphetOptions {
    pub growth: Option<GrowthType>,
    pub changepoints: Option<Vec<u64>>,
    pub n_changepoints: Option<u32>,
    pub changepoint_range: Option<f64>,
    pub yearly_seasonality: Option<SeasonalityOption>,
    pub weekly_seasonality: Option<SeasonalityOption>,
    pub daily_seasonality: Option<SeasonalityOption>,
    pub seasonality_mode: Option<FeatureMode>,
    pub seasonality_prior_scale: Option<PositiveFloat>,
    pub changepoint_prior_scale: Option<PositiveFloat>,
    pub estimation: Option<EstimationMode>,
    pub interval_width: Option<f64>,
    pub uncertainty_samples: Option<u32>,
    pub scaling: Option<Scaling>,
    pub holidays: Option<HashMap<String, Holiday>>,
    pub holidays_prior_scale: Option<PositiveFloat>,
    pub holidays_mode: Option<FeatureMode>,
}

impl From<OptProphetOptions> for ProphetOptions {
    fn from(value: OptProphetOptions) -> Self {
        let defaults = ProphetOptions::default();
        ProphetOptions {
            growth: value.growth.unwrap_or(defaults.growth),
            changepoints: value.changepoints,
            n_changepoints: value.n_changepoints.unwrap_or(defaults.n_changepoints),
            changepoint_range: value
                .changepoint_range
                .unwrap_or(defaults.changepoint_range),
            yearly_seasonality: value
                .yearly_seasonality
                .unwrap_or(defaults.yearly_seasonality),
            weekly_seasonality: value
                .weekly_seasonality
                .unwrap_or(defaults.weekly_seasonality),
            daily_seasonality: value
                .daily_seasonality
                .unwrap_or(defaults.daily_seasonality),
            seasonality_mode: value.seasonality_mode.unwrap_or(defaults.seasonality_mode),
            seasonality_prior_scale: value
                .seasonality_prior_scale
                .unwrap_or(defaults.seasonality_prior_scale),
            changepoint_prior_scale: value
                .changepoint_prior_scale
                .unwrap_or(defaults.changepoint_prior_scale),
            estimation: value.estimation.unwrap_or(defaults.estimation),
            interval_width: value.interval_width.unwrap_or(defaults.interval_width),
            uncertainty_samples: value
                .uncertainty_samples
                .unwrap_or(defaults.uncertainty_samples),
            scaling: value.scaling.unwrap_or(defaults.scaling),
            holidays: value.holidays.unwrap_or(defaults.holidays),
            holidays_prior_scale: value
                .holidays_prior_scale
                .unwrap_or(defaults.holidays_prior_scale),
            holidays_mode: value.holidays_mode.unwrap_or(defaults.holidays_mode),
        }
    }
}

/// Options for Prophet, after applying defaults.
#[derive(Debug, Clone)]
pub struct ProphetOptions {
    pub growth: GrowthType,
    pub changepoints: Option<Vec<u64>>,
    pub n_changepoints: u32,
    pub changepoint_range: f64,
    pub yearly_seasonality: SeasonalityOption,
    pub weekly_seasonality: SeasonalityOption,
    pub daily_seasonality: SeasonalityOption,
    pub seasonality_mode: FeatureMode,
    pub seasonality_prior_scale: PositiveFloat,
    pub changepoint_prior_scale: PositiveFloat,
    pub estimation: EstimationMode,
    pub interval_width: f64,
    pub uncertainty_samples: u32,
    pub scaling: Scaling,
    pub holidays: HashMap<String, Holiday>,
    pub holidays_prior_scale: PositiveFloat,
    pub holidays_mode: FeatureMode,
}

impl Default for ProphetOptions {
    fn default() -> Self {
        Self {
            growth: GrowthType::Linear,
            changepoints: None,
            n_changepoints: 25,
            changepoint_range: 0.8,
            yearly_seasonality: SeasonalityOption::default(),
            weekly_seasonality: SeasonalityOption::default(),
            daily_seasonality: SeasonalityOption::default(),
            seasonality_mode: FeatureMode::Additive,
            seasonality_prior_scale: 10.0.try_into().unwrap(),
            changepoint_prior_scale: 0.05.try_into().unwrap(),
            estimation: EstimationMode::Mle,
            interval_width: 0.8,
            uncertainty_samples: 1000,
            scaling: Scaling::AbsMax,
            holidays: HashMap::new(),
            holidays_prior_scale: 100.0.try_into().unwrap(),
            holidays_mode: FeatureMode::Additive,
        }
    }
}
