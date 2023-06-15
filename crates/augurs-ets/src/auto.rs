//! Automated ETS model selection.
//!
//! This module contains the [`AutoETS`] type, which can be used to automatically
//! select the best ETS model for a given time series.
//!
//! The search specification is controlled by the [`AutoSpec`] type. As a
//! convenience, [`AutoSpec`] implements [`FromStr`], so it can be parsed from a
//! string using the same framework as R's `ets` function.
//!
//! # Example
//!
//! ```
//! use augurs_ets::{AutoETS, AutoSpec};
//!
//! // Create an `AutoETS` instance from a specification string.
//! // The `"ZZN"` specification means that the search should consider all
//! // models with additive or multiplicative error and trend components, and
//! // no seasonal component.
//! let mut auto = AutoETS::new(1, "ZZN").expect("ZZN is a valid specification");
//! let data = (1..10).map(|x| x as f64).collect::<Vec<_>>();
//! let model = auto.fit(&data).expect("fit succeeds");
//! assert_eq!(&model.model_type().to_string(), "AAN");
//! ```

use std::{
    fmt::{self, Write},
    str::FromStr,
};

use augurs_core::Forecast;

use crate::{
    model::{self, Model, OptimizationCriteria, Params, Unfit},
    Error, Result,
};

/// Error component search specification.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum ErrorSpec {
    /// Only consider additive error models.
    Additive,
    /// Only consider multiplicative error models.
    Multiplicative,
    /// Consider both additive and multiplicative error models.
    Auto,
}

impl ErrorSpec {
    /// Returns the error component candidates for this specification.
    fn candidates(&self) -> &[model::ErrorComponent] {
        match self {
            Self::Additive => &[model::ErrorComponent::Additive],
            Self::Multiplicative => &[model::ErrorComponent::Multiplicative],
            Self::Auto => &[
                model::ErrorComponent::Additive,
                model::ErrorComponent::Multiplicative,
            ],
        }
    }
}

impl fmt::Display for ErrorSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Additive => f.write_char('A'),
            Self::Multiplicative => f.write_char('M'),
            Self::Auto => f.write_char('Z'),
        }
    }
}

impl TryFrom<char> for ErrorSpec {
    type Error = Error;

    fn try_from(c: char) -> Result<Self> {
        match c {
            'A' => Ok(Self::Additive),
            'M' => Ok(Self::Multiplicative),
            'Z' => Ok(Self::Auto),
            _ => Err(Error::InvalidErrorComponentString(c)),
        }
    }
}

/// Trend and seasonal component search specification.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum ComponentSpec {
    /// Only consider models without this component.
    None,
    /// Only consider additive models.
    Additive,
    /// Only consider multiplicative models.
    Multiplicative,
    /// Consider both additive and multiplicative models.
    Auto,
}

impl ComponentSpec {
    /// Returns `true` if this specification is not `None`.
    fn is_specified(&self) -> bool {
        matches!(self, Self::Additive | Self::Multiplicative)
    }

    /// Returns the trend component candidates for this specification.
    fn trend_candidates(&self, auto_multiplicative: bool) -> &[model::TrendComponent] {
        match (self, auto_multiplicative) {
            (Self::None, _) => &[],
            (Self::Additive, _) => &[model::TrendComponent::Additive],
            (Self::Multiplicative, _) => &[model::TrendComponent::Multiplicative],
            (Self::Auto, false) => &[model::TrendComponent::None, model::TrendComponent::Additive],
            (Self::Auto, true) => &[
                model::TrendComponent::None,
                model::TrendComponent::Additive,
                model::TrendComponent::Multiplicative,
            ],
        }
    }

    /// Returns the seasonal component candidates for this specification.
    fn seasonal_candidates(&self, season_length: usize) -> Vec<model::SeasonalComponent> {
        match self {
            ComponentSpec::None => vec![model::SeasonalComponent::None],
            ComponentSpec::Additive => {
                vec![model::SeasonalComponent::Additive { season_length }]
            }
            ComponentSpec::Multiplicative => {
                vec![model::SeasonalComponent::Multiplicative { season_length }]
            }
            ComponentSpec::Auto => vec![
                model::SeasonalComponent::None,
                model::SeasonalComponent::Additive { season_length },
                model::SeasonalComponent::Multiplicative { season_length },
            ],
        }
    }
}

impl fmt::Display for ComponentSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::None => f.write_char('N'),
            Self::Additive => f.write_char('A'),
            Self::Multiplicative => f.write_char('M'),
            Self::Auto => f.write_char('Z'),
        }
    }
}

impl TryFrom<char> for ComponentSpec {
    type Error = Error;

    fn try_from(c: char) -> Result<Self> {
        match c {
            'N' => Ok(Self::None),
            'A' => Ok(Self::Additive),
            'M' => Ok(Self::Multiplicative),
            'Z' => Ok(Self::Auto),
            _ => Err(Error::InvalidComponentString(c)),
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
enum Damped {
    Auto,
    Fixed(bool),
}

impl Damped {
    fn candidates(&self) -> &[bool] {
        match self {
            Self::Auto => &[true, false],
            Self::Fixed(x) => std::slice::from_ref(x),
        }
    }
}

/// Auto model search specification.
#[derive(Debug, Clone, Copy)]
pub struct AutoSpec {
    /// The types of error components to consider.
    pub error: ErrorSpec,
    /// The types of trend components to consider.
    pub trend: ComponentSpec,
    /// The types of seasonal components to consider.
    pub seasonal: ComponentSpec,
}

impl fmt::Display for AutoSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.error.fmt(f)?;
        self.trend.fmt(f)?;
        self.seasonal.fmt(f)?;
        Ok(())
    }
}

impl FromStr for AutoSpec {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self> {
        if s.len() != 3 {
            return Err(Error::InvalidModelSpec(s.to_owned()));
        }
        let mut iter = s.chars();
        let spec = Self {
            error: ErrorSpec::try_from(iter.next().unwrap())?,
            trend: ComponentSpec::try_from(iter.next().unwrap())?,
            seasonal: ComponentSpec::try_from(iter.next().unwrap())?,
        };
        use ComponentSpec::*;
        match spec {
            Self {
                error: ErrorSpec::Additive,
                trend: _,
                seasonal: Multiplicative,
            }
            | Self {
                error: ErrorSpec::Additive,
                trend: Multiplicative,
                seasonal: _,
            }
            | Self {
                error: ErrorSpec::Multiplicative,
                trend: Multiplicative,
                seasonal: Multiplicative,
            } => Err(Error::InvalidModelSpec(s.to_owned())),
            other => Ok(other),
        }
    }
}

impl TryFrom<&str> for AutoSpec {
    type Error = Error;

    fn try_from(s: &str) -> Result<Self> {
        s.parse()
    }
}

/// Automatic ETS model selection.
#[derive(Debug, Clone)]
pub struct AutoETS {
    /// The model search specification.
    spec: AutoSpec,
    /// The number of observations per unit of time.
    season_length: usize,

    /// Explicit parameters to use when fitting the model.
    ///
    /// If any of these are `None` then they will be estimated
    /// as part of the model fitting procedure.
    params: Params,

    /// Whether to use a damped trend.
    ///
    /// Defaults to trying both damped and non-damped trends.
    damped: Damped,

    /// Whether to allow multiplicative trend during automatic model selection.
    ///
    /// Defaults to `false`.
    allow_multiplicative_trend: bool,

    /// Number of steps over which to calculate the average MSE.
    ///
    /// Will be constrained to the range `[1, 30]`.
    ///
    /// Defaults to `3`.
    nmse: usize,

    /// The optimization criterion to use.
    ///
    /// Defaults to [`OptimizationCriteria::Likelihood`].
    opt_crit: OptimizationCriteria,

    /// The maximum number of iterations to use during optimization.
    ///
    /// Defaults to `2_000`.
    max_iterations: usize,

    /// The model that was selected.
    model: Option<Model>,
}

impl AutoETS {
    /// Create a new `AutoETS` model with the given period and model search specification string.
    ///
    /// The specification string should be of the form `XXX` where the first character is the error
    /// component, the second is the trend component, and the third is the seasonal component. The
    /// possible values for each component are:
    ///
    /// - `N` for no component
    /// - `A` for additive
    /// - `M` for multiplicative
    /// - `Z` for automatic
    ///
    /// Using `Z` for any component will cause the model to try all possible values for that component.
    /// For example, `ZAZ` will try all possible error and seasonal components, but only additive
    /// trend components.
    ///
    /// # Errors
    ///
    /// An error will be returned if the specification string is not of the correct length or contains
    /// invalid characters.
    pub fn new(season_length: usize, spec: impl TryInto<AutoSpec, Error = Error>) -> Result<Self> {
        let spec = spec.try_into()?;
        Ok(Self::from_spec(season_length, spec))
    }

    /// Create a new `AutoETS` model with the given period and model search specification.
    pub fn from_spec(season_length: usize, spec: AutoSpec) -> Self {
        let params = Params {
            alpha: f64::NAN,
            beta: f64::NAN,
            gamma: f64::NAN,
            phi: f64::NAN,
        };
        Self {
            season_length,
            spec,
            params,
            damped: Damped::Auto,
            allow_multiplicative_trend: false,
            nmse: 3,
            opt_crit: OptimizationCriteria::Likelihood,
            max_iterations: 2_000,
            model: None,
        }
    }

    /// Get the season length of the model.
    pub fn season_length(&self) -> usize {
        self.season_length
    }

    /// Get the search specification.
    pub fn spec(&self) -> AutoSpec {
        self.spec
    }

    /// Create a new `AutoETS` model search without any seasonal components.
    ///
    /// Equivalent to `AutoETS::new(1, "ZZN")`.
    pub fn non_seasonal() -> Self {
        Self::new(1, "ZZN").unwrap()
    }

    /// Fix the search to consider only damped or undamped trend.
    pub fn damped(mut self, damped: bool) -> Result<Self> {
        if damped && self.spec.trend == ComponentSpec::None {
            return Err(Error::InvalidModelSpec(format!(
                "damped trend not allowed for model spec '{}'",
                self.spec
            )));
        }
        self.damped = Damped::Fixed(damped);
        Ok(self)
    }

    /// Set the value of `alpha` to use when fitting the model.
    ///
    /// See the docs for [`Params::alpha`] for more details on `alpha`.
    pub fn alpha(mut self, alpha: f64) -> Self {
        self.params.alpha = alpha;
        self
    }

    /// Set the value of `beta` to use when fitting the model.
    ///
    /// See the docs for [`Params::beta`] for more details on `beta`.
    pub fn beta(mut self, beta: f64) -> Self {
        self.params.beta = beta;
        self
    }

    /// Set the value of `gamma` to use when fitting the model.
    ///
    /// See the docs for [`Params::gamma`] for more details on `gamma`.
    pub fn gamma(mut self, gamma: f64) -> Self {
        self.params.gamma = gamma;
        self
    }

    /// Set the value of `phi` to use when fitting the model.
    ///
    /// See the docs for [`Params::phi`] for more details on `phi`.
    pub fn phi(mut self, phi: f64) -> Self {
        self.params.phi = phi;
        self
    }

    /// Include models with multiplicative trend during automatic model selection.
    ///
    /// By default, models with multiplicative trend are excluded from the search space.
    pub fn allow_multiplicative_trend(mut self, allow: bool) -> Self {
        self.allow_multiplicative_trend = allow;
        self
    }

    /// Check whether a model combination is valid.
    ///
    /// Note that we currently enforce the 'restricted' mode of R's `ets` package
    /// which disallows models with infinite variance.
    fn valid_combination(
        &self,
        error: model::ErrorComponent,
        trend: model::TrendComponent,
        seasonal: model::SeasonalComponent,
        damped: bool,
        data_positive: bool,
    ) -> bool {
        use model::{ErrorComponent as EC, SeasonalComponent as SC, TrendComponent as TC};
        match (error, trend, seasonal, damped) {
            // Can't have no trend and damped trend.
            (_, TC::None, _, true) => false,
            // Restricted mode disallows additive error with multiplicative trend and seasonality.
            (EC::Additive, TC::Multiplicative, SC::Multiplicative { .. }, _) => false,
            // Restricted mode disallows multiplicative error with multiplicative trend and additive seasonality;
            (EC::Multiplicative, TC::Multiplicative, SC::Additive { .. }, _) => false,
            (EC::Multiplicative, _, _, _) if !data_positive => false,
            (_, _, SC::Multiplicative { .. }, _) if !data_positive => false,
            (
                _,
                _,
                SC::Additive { season_length: 1 } | SC::Multiplicative { season_length: 1 },
                _,
            ) => false,
            _ => true,
        }
    }

    /// Return an iterator over all model combinations.
    ///
    /// Note that this does not check that the model combinations are valid;
    /// some knowledge of the data is required for that.
    fn candidates(
        &self,
    ) -> impl Iterator<
        Item = (
            &model::ErrorComponent,
            &model::TrendComponent,
            model::SeasonalComponent,
            &bool,
        ),
    > {
        let error_candidates = self.spec.error.candidates();
        let trend_candidates = self
            .spec
            .trend
            .trend_candidates(self.allow_multiplicative_trend);
        let season_candidates = self.spec.seasonal.seasonal_candidates(self.season_length);
        let damped_candidates = self.damped.candidates();

        itertools::iproduct!(
            error_candidates,
            trend_candidates,
            season_candidates,
            damped_candidates
        )
    }

    /// Search for the best model, fitting it to the data.
    ///
    /// The model is stored on the `AutoETS` struct and can be retrieved with
    /// the `model` method. It is also returned by this function.
    ///
    /// # Errors
    ///
    /// If no model can be found, or if any parameters are invalid, this function
    /// returns an error.
    pub fn fit(&mut self, y: &[f64]) -> Result<&Model> {
        let data_positive = y.iter().fold(f64::INFINITY, |a, &b| a.min(b)) > 0.0;
        if self.spec.error == ErrorSpec::Multiplicative && !data_positive {
            return Err(Error::InvalidModelSpec(format!(
                "multiplicative error not allowed for model spec '{}' with non-positive data",
                self.spec
            )));
        }

        let n = y.len();
        let mut npars = 2; // alpha + l0
        if self.spec.trend.is_specified() {
            npars += 2; // beta + b0
        }
        if self.spec.seasonal.is_specified() {
            npars += 2; // gamma + s
        }
        if n <= npars + 4 {
            return Err(Error::NotEnoughData);
        }

        self.model = self
            .candidates()
            .filter_map(|(&error, &trend, season, &damped)| {
                if self.valid_combination(error, trend, season, damped, data_positive) {
                    let model = Unfit::new(model::ModelType {
                        error,
                        trend,
                        season,
                    })
                    .damped(damped)
                    .params(self.params.clone())
                    .nmse(self.nmse)
                    .opt_crit(self.opt_crit)
                    .max_iterations(self.max_iterations)
                    .fit(y)
                    .ok()?;
                    if model.aicc().is_nan() {
                        None
                    } else {
                        Some(model)
                    }
                } else {
                    None
                }
            })
            .min_by(|a, b| {
                a.aicc()
                    .partial_cmp(&b.aicc())
                    .expect("NaNs have already been filtered from the iterator")
            });
        self.model.as_ref().ok_or(Error::NoModelFound)
    }

    /// Predict the next `horizon` values using the best model, optionally including
    /// prediction intervals at the specified level.
    ///
    /// `level` should be a float between 0 and 1 representing the confidence level.
    ///
    /// # Errors
    ///
    /// This function will return an error if no model has been fit yet (using [`AutoETS::fit`]).
    pub fn predict(&self, h: usize, level: impl Into<Option<f64>>) -> Result<Forecast> {
        Ok(self
            .model
            .as_ref()
            .ok_or(Error::ModelNotFit)?
            .predict(h, level))
    }

    /// Return the in-sample predictions using the best model, optionally including
    /// prediction intervals at the specified level.
    ///
    /// `level` should be a float between 0 and 1 representing the confidence level.`
    ///
    /// # Errors
    ///
    /// This function will return an error if no model has been fit yet (using [`AutoETS::fit`]).
    pub fn predict_in_sample(&self, level: impl Into<Option<f64>>) -> Result<Forecast> {
        Ok(self
            .model
            .as_ref()
            .ok_or(Error::ModelNotFit)?
            .predict_in_sample(level))
    }
}

#[cfg(test)]
mod test {

    use super::{AutoETS, AutoSpec};
    use crate::{
        assert_closeish,
        data::AIR_PASSENGERS,
        model::{ErrorComponent, SeasonalComponent, TrendComponent},
        Error,
    };

    #[test]
    fn spec_from_str() {
        let cases = [
            "NNN", "NAN", "NAM", "NAZ", "NMN", "NMA", "NMM", "NMZ", "ANN", "AAN", "AAM", "AAZ",
            "AMN", "AMA", "AMM", "AMZ", "MNN", "MAN", "MAM", "MAZ", "MMN", "MMA", "MMM", "MMZ",
            "ZNN", "ZAN", "ZAM", "ZAZ", "ZMN", "ZMA", "ZMM", "ZMZ",
        ];
        for case in cases {
            let spec: Result<AutoSpec, Error> = case.try_into();
            let (error, rest) = case.split_at(1);
            let (trend, seasonal) = rest.split_at(1);
            match (error, trend, seasonal) {
                ("N", _, _) => {
                    assert!(
                        matches!(spec, Err(Error::InvalidErrorComponentString(_))),
                        "{:?}, case {}",
                        spec,
                        case
                    );
                }
                ("A", "M", _) | ("A", _, "M") | ("M", "M", "M") => {
                    assert!(
                        matches!(spec, Err(Error::InvalidModelSpec(_))),
                        "{:?}, case {}",
                        spec,
                        case
                    );
                }
                _ => {
                    assert!(spec.is_ok());
                }
            }
        }
    }

    #[test]
    fn air_passengers_fit() {
        let mut auto = AutoETS::new(1, "ZZN").unwrap();
        let model = auto.fit(&AIR_PASSENGERS).expect("fit failed");
        assert_eq!(model.model_type().error, ErrorComponent::Multiplicative);
        assert_eq!(model.model_type().trend, TrendComponent::Additive);
        assert_eq!(model.model_type().season, SeasonalComponent::None);
        assert_closeish!(model.log_likelihood(), -831.4883541595792, 0.01);
        assert_closeish!(model.aic(), 1672.9767083191584, 0.01);
    }
}
