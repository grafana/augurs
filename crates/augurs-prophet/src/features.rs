//! Features used by Prophet, such as seasonality, regressors and holidays.
use std::num::NonZeroU32;

use crate::{
    positive_float::PositiveFloat, prophet::prep::ONE_DAY_IN_SECONDS_INT, Error, TimestampSeconds,
};

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
    pub(crate) lower_window: Option<Vec<u32>>,
    pub(crate) upper_window: Option<Vec<u32>>,
    pub(crate) prior_scale: Option<PositiveFloat>,
    pub(crate) utc_offset: TimestampSeconds,
}

impl Holiday {
    /// Create a new holiday.
    pub fn new(ds: Vec<TimestampSeconds>) -> Self {
        Self {
            ds,
            lower_window: None,
            upper_window: None,
            prior_scale: None,
            utc_offset: 0,
        }
    }

    /// Set the lower window for the holiday.
    ///
    /// The lower window is the number of days before the holiday
    /// that it is observed. For example, if the holiday is on
    /// 2023-01-01 and the lower window is 1, then the holiday will
    /// _also_ be observed on 2022-12-31.
    pub fn with_lower_window(mut self, lower_window: Vec<u32>) -> Result<Self, Error> {
        if self.ds.len() != lower_window.len() {
            return Err(Error::MismatchedLengths {
                a_name: "ds".to_string(),
                a: self.ds.len(),
                b_name: "lower_window".to_string(),
                b: lower_window.len(),
            });
        }
        self.lower_window = Some(lower_window);
        Ok(self)
    }

    /// Set the upper window for the holiday.
    ///
    /// The upper window is the number of days after the holiday
    /// that it is observed. For example, if the holiday is on
    /// 2023-01-01 and the upper window is 1, then the holiday will
    /// _also_ be observed on 2023-01-02.
    pub fn with_upper_window(mut self, upper_window: Vec<u32>) -> Result<Self, Error> {
        if self.ds.len() != upper_window.len() {
            return Err(Error::MismatchedLengths {
                a_name: "ds".to_string(),
                a: self.ds.len(),
                b_name: "upper_window".to_string(),
                b: upper_window.len(),
            });
        }
        self.upper_window = Some(upper_window);
        Ok(self)
    }

    /// Add a prior scale for the holiday.
    pub fn with_prior_scale(mut self, prior_scale: PositiveFloat) -> Self {
        self.prior_scale = Some(prior_scale);
        self
    }

    /// Set the UTC offset for the holiday, in seconds.
    ///
    /// Timestamps of a holiday's occurrences are rounded down to the nearest day,
    /// but since we're using Unix timestamps rather than timezone-aware dates,
    /// holidays default to assuming the 'day' was for 24h from midnight UTC.
    ///
    /// If instead the holiday should be from midnight in a different timezone,
    /// use this method to set the offset from UTC of the desired timezone.
    ///
    /// Defaults to 0.
    pub fn with_utc_offset(mut self, utc_offset: TimestampSeconds) -> Self {
        self.utc_offset = utc_offset;
        self
    }

    /// Return the Unix timestamp of the given date, rounded down to the nearest day,
    /// adjusted by the holiday's UTC offset.
    pub(crate) fn floor_day(&self, ds: TimestampSeconds) -> TimestampSeconds {
        let remainder = (ds + self.utc_offset) % ONE_DAY_IN_SECONDS_INT;
        // Adjust the date to the holiday's UTC offset.
        ds - remainder
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
#[derive(Debug, Clone)]
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

impl Default for RegressorScale {
    fn default() -> Self {
        Self {
            standardize: false,
            mu: 0.0,
            std: 1.0,
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

#[cfg(test)]
mod test {
    use chrono::{FixedOffset, TimeZone, Utc};

    use crate::features::Holiday;

    #[test]
    fn holiday_floor_day_no_offset() {
        let holiday = Holiday::new(vec![]);
        let offset = Utc;
        let expected = offset
            .with_ymd_and_hms(2024, 11, 21, 0, 0, 0)
            .unwrap()
            .timestamp();
        assert_eq!(holiday.floor_day(expected), expected);
        assert_eq!(
            holiday.floor_day(
                offset
                    .with_ymd_and_hms(2024, 11, 21, 15, 3, 12)
                    .unwrap()
                    .timestamp()
            ),
            expected
        );
    }

    #[test]
    fn holiday_floor_day_positive_offset() {
        let offset = FixedOffset::east_opt(60 * 60 * 4).unwrap();
        let expected = offset
            .with_ymd_and_hms(2024, 11, 21, 0, 0, 0)
            .unwrap()
            .timestamp();

        let holiday = Holiday::new(vec![]).with_utc_offset(offset.local_minus_utc() as i64);
        assert_eq!(holiday.floor_day(expected), expected);
        assert_eq!(
            holiday.floor_day(
                offset
                    .with_ymd_and_hms(2024, 11, 21, 15, 3, 12)
                    .unwrap()
                    .timestamp()
            ),
            expected
        );
    }

    #[test]
    fn holiday_floor_day_negative_offset() {
        let offset = FixedOffset::west_opt(60 * 60 * 3).unwrap();
        let expected = offset
            .with_ymd_and_hms(2024, 11, 21, 0, 0, 0)
            .unwrap()
            .timestamp();

        let holiday = Holiday::new(vec![]).with_utc_offset(offset.local_minus_utc() as i64);
        assert_eq!(holiday.floor_day(expected), expected);
        assert_eq!(
            holiday.floor_day(
                offset
                    .with_ymd_and_hms(2024, 11, 21, 15, 3, 12)
                    .unwrap()
                    .timestamp()
            ),
            expected
        );
    }

    #[test]
    fn holiday_floor_day_edge_cases() {
        // Test maximum valid offset (UTC+14)
        let max_offset = 14 * 60 * 60;
        let offset = FixedOffset::east_opt(max_offset).unwrap();
        let expected = offset
            .with_ymd_and_hms(2024, 11, 21, 0, 0, 0)
            .unwrap()
            .timestamp();
        let holiday_max = Holiday::new(vec![]).with_utc_offset(offset.local_minus_utc() as i64);
        assert_eq!(
            holiday_max.floor_day(
                offset
                    .with_ymd_and_hms(2024, 11, 21, 12, 0, 0)
                    .unwrap()
                    .timestamp()
            ),
            expected
        );

        // Test near day boundary
        let offset = FixedOffset::east_opt(60).unwrap();
        let expected = offset
            .with_ymd_and_hms(2024, 11, 21, 0, 0, 0)
            .unwrap()
            .timestamp();
        let holiday_near = Holiday::new(vec![]).with_utc_offset(offset.local_minus_utc() as i64);
        assert_eq!(
            holiday_near.floor_day(
                holiday_max.floor_day(
                    offset
                        .with_ymd_and_hms(2024, 11, 21, 23, 59, 59)
                        .unwrap()
                        .timestamp()
                ),
            ),
            expected
        );
    }
}
