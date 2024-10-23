use std::num::TryFromIntError;

use crate::bindings::exports::augurs::prophet::prophet::{
    EstimationMode, GrowthType, Holiday, Prediction, PredictionData, Predictions, ProphetOpts,
    Scaling, SeasonalityMode, SeasonalityOption, TrainingData,
};

#[derive(Debug, thiserror::Error)]
#[error("Conversion error for field \"{field}\": {source}")]
pub struct FieldConversionError {
    field: &'static str,
    #[source]
    source: Box<dyn std::error::Error>,
}

impl FieldConversionError {
    fn new(field: &'static str, source: impl std::error::Error + 'static) -> Self {
        Self {
            field,
            source: Box::new(source),
        }
    }
}

fn convert_field<T, E: std::error::Error + 'static, U: TryFrom<T, Error = E>>(
    field: &'static str,
    value: Option<T>,
) -> Result<Option<U>, FieldConversionError> {
    value
        .map(TryInto::try_into)
        .transpose()
        .map_err(|e| FieldConversionError::new(field, e))
}

impl From<EstimationMode> for augurs_prophet::EstimationMode {
    fn from(value: EstimationMode) -> Self {
        match value {
            EstimationMode::Map => augurs_prophet::EstimationMode::Map,
            EstimationMode::Mle => augurs_prophet::EstimationMode::Mle,
            // TODO: Uncomment once augurs_prophet supports MCMC
            // EstimationMode::Mcmc(value) => augurs_prophet::EstimationMode::Mcmc(value),
        }
    }
}

impl From<GrowthType> for augurs_prophet::GrowthType {
    fn from(value: GrowthType) -> Self {
        match value {
            GrowthType::Linear => augurs_prophet::GrowthType::Linear,
            GrowthType::Logistic => augurs_prophet::GrowthType::Logistic,
            GrowthType::Flat => augurs_prophet::GrowthType::Flat,
        }
    }
}

impl TryFrom<SeasonalityOption> for augurs_prophet::SeasonalityOption {
    type Error = TryFromIntError;
    fn try_from(value: SeasonalityOption) -> Result<Self, Self::Error> {
        Ok(match value {
            SeasonalityOption::Auto => augurs_prophet::SeasonalityOption::Auto,
            SeasonalityOption::Manual(value) => augurs_prophet::SeasonalityOption::Manual(value),
            SeasonalityOption::Fourier(value) => {
                augurs_prophet::SeasonalityOption::Fourier(value.try_into()?)
            }
        })
    }
}

impl From<SeasonalityMode> for augurs_prophet::FeatureMode {
    fn from(value: SeasonalityMode) -> Self {
        match value {
            SeasonalityMode::Additive => augurs_prophet::FeatureMode::Additive,
            SeasonalityMode::Multiplicative => augurs_prophet::FeatureMode::Multiplicative,
        }
    }
}

impl From<Scaling> for augurs_prophet::Scaling {
    fn from(value: Scaling) -> Self {
        match value {
            Scaling::AbsMax => augurs_prophet::Scaling::AbsMax,
            Scaling::MinMax => augurs_prophet::Scaling::MinMax,
        }
    }
}

impl TryFrom<Holiday> for (String, augurs_prophet::Holiday) {
    type Error = FieldConversionError;

    fn try_from(value: Holiday) -> Result<Self, Self::Error> {
        let name = value.name;
        let mut holiday = augurs_prophet::Holiday::new(value.ds);
        if let Some(lower_window) = value.lower_window {
            holiday =
                holiday
                    .with_lower_window(lower_window)
                    .map_err(|e| FieldConversionError {
                        field: "lower_window",
                        source: Box::new(e),
                    })?;
        }
        if let Some(upper_window) = value.upper_window {
            holiday =
                holiday
                    .with_upper_window(upper_window)
                    .map_err(|e| FieldConversionError {
                        field: "upper_window",
                        source: Box::new(e),
                    })?;
        }
        if let Some(prior_scale) = value.prior_scale {
            holiday = holiday.with_prior_scale(prior_scale.try_into().map_err(|e| {
                FieldConversionError {
                    field: "prior_scale",
                    source: Box::new(e),
                }
            })?);
        }
        Ok((name, holiday))
    }
}

impl TryFrom<ProphetOpts> for augurs_prophet::ProphetOptions {
    type Error = FieldConversionError;
    fn try_from(opts: ProphetOpts) -> Result<Self, Self::Error> {
        Ok(augurs_prophet::OptProphetOptions {
            growth: opts.growth.map(Into::into),
            changepoints: opts.changepoints,
            n_changepoints: opts.n_changepoints,
            changepoint_range: opts
                .changepoint_range
                .map(TryInto::try_into)
                .transpose()
                .map_err(|e| FieldConversionError::new("changepoint_range", e))?,
            yearly_seasonality: convert_field("yearly_seasonality", opts.yearly_seasonality)?,
            weekly_seasonality: convert_field("weekly_seasonality", opts.yearly_seasonality)?,
            daily_seasonality: convert_field("daily_seasonality", opts.yearly_seasonality)?,
            seasonality_mode: opts.seasonality_mode.map(Into::into),
            seasonality_prior_scale: convert_field(
                "seasonality_prior_scale",
                opts.seasonality_prior_scale,
            )?,
            changepoint_prior_scale: convert_field(
                "changepoint_prior_scale",
                opts.changepoint_prior_scale,
            )?,
            estimation: opts.estimation.map(Into::into),
            interval_width: convert_field("interval_width", opts.interval_width)?,
            uncertainty_samples: opts.uncertainty_samples,
            scaling: opts.scaling.map(Into::into),
            holidays: opts
                .holidays
                .map(|v| v.into_iter().map(TryInto::try_into).collect())
                .transpose()
                .map_err(|e| FieldConversionError::new("holidays", e))?,
            holidays_prior_scale: convert_field("holidays_prior_scale", opts.holidays_prior_scale)?,
            holidays_mode: opts.holidays_mode.map(Into::into),
        }
        .into())
    }
}

impl TryFrom<TrainingData> for augurs_prophet::TrainingData {
    type Error = augurs_prophet::Error;
    fn try_from(value: TrainingData) -> Result<Self, Self::Error> {
        let mut data = augurs_prophet::TrainingData::new(value.ds, value.y)?;
        if let Some(cap) = value.cap {
            data = data.with_cap(cap)?;
        }
        if let Some(floor) = value.floor {
            data = data.with_floor(floor)?;
        }
        if let Some(seasonality_conditions) = value.seasonality_conditions {
            data = data.with_seasonality_conditions(
                seasonality_conditions
                    .into_iter()
                    .map(|sc| (sc.name, sc.is_active))
                    .collect(),
            )?;
        }
        if let Some(regressors) = value.regressors {
            data =
                data.with_regressors(regressors.into_iter().map(|r| (r.name, r.values)).collect())?;
        }
        Ok(data)
    }
}

impl TryFrom<PredictionData> for augurs_prophet::PredictionData {
    type Error = augurs_prophet::Error;
    fn try_from(value: PredictionData) -> Result<Self, Self::Error> {
        let mut data = augurs_prophet::PredictionData::new(value.ds);
        if let Some(cap) = value.cap {
            data = data.with_cap(cap)?;
        }
        if let Some(floor) = value.floor {
            data = data.with_floor(floor)?;
        }
        if let Some(seasonality_conditions) = value.seasonality_conditions {
            data = data.with_seasonality_conditions(
                seasonality_conditions
                    .into_iter()
                    .map(|sc| (sc.name, sc.is_active))
                    .collect(),
            )?;
        }
        if let Some(regressors) = value.regressors {
            data =
                data.with_regressors(regressors.into_iter().map(|r| (r.name, r.values)).collect())?;
        }
        Ok(data)
    }
}

impl From<augurs_prophet::FeaturePrediction> for Prediction {
    fn from(value: augurs_prophet::FeaturePrediction) -> Self {
        Self {
            point: value.point,
            lower: value.lower,
            upper: value.upper,
        }
    }
}

impl From<augurs_prophet::Predictions> for Predictions {
    fn from(value: augurs_prophet::Predictions) -> Self {
        Self {
            ds: value.ds,
            yhat: value.yhat.into(),
            trend: value.trend.into(),
            cap: value.cap,
            floor: value.floor,
            additive_terms: value.additive.into(),
            multiplicative_terms: value.multiplicative.into(),
        }
    }
}
