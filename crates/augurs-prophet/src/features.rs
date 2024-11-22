//! Features used by Prophet, such as seasonality, regressors and holidays.
use std::num::NonZeroU32;

use crate::{
    positive_float::PositiveFloat, prophet::prep::ONE_DAY_IN_SECONDS_INT, TimestampSeconds,
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

/// An occurrence of a holiday.
///
/// Each occurrence has a start and end time represented as
/// a Unix timestamp. Holiday occurrences are therefore
/// timestamp-unaware and can therefore span multiple days
/// or even sub-daily periods.
///
/// This differs from the Python and R Prophet implementations,
/// which require all holidays to be day-long events. Some
/// convenience methods are provided to create day-long
/// occurrences: see [`HolidayOccurrence::for_day`] and
/// [`HolidayOccurrence::for_day_in_tz`].
///
/// The caller is responsible for ensuring that the start
/// and end time provided are in the correct timezone.
/// One way to do this is to use [`chrono::FixedOffset`][fo]
/// to create an offset representing the time zone,
/// [`FixedOffset::with_ymd_and_hms`][wyah] to create a
/// [`DateTime`][dt] in that time zone, then [`DateTime::timestamp`][ts]
/// to get the Unix timestamp.
///
/// [fo]: https://docs.rs/chrono/latest/chrono/struct.FixedOffset.html
/// [wyah]: https://docs.rs/chrono/latest/chrono/struct.FixedOffset.html#method.with_ymd_and_hms
/// [dt]: https://docs.rs/chrono/latest/chrono/struct.DateTime.html
/// [ts]: https://docs.rs/chrono/latest/chrono/struct.DateTime.html#method.timestamp
#[derive(Debug, Clone)]
pub struct HolidayOccurrence {
    pub(crate) start: TimestampSeconds,
    pub(crate) end: TimestampSeconds,
}

impl HolidayOccurrence {
    /// Create a new holiday occurrence with the given
    /// start and end timestamp.
    pub fn new(start: TimestampSeconds, end: TimestampSeconds) -> Self {
        Self { start, end }
    }

    /// Create a new holiday encompassing midnight on the day
    /// of the given timestamp to midnight on the following day,
    /// in UTC.
    ///
    /// This is a convenience method to reproduce the behaviour
    /// of the Python and R Prophet implementations, which require
    /// all holidays to be day-long events.
    ///
    /// Note that this will _not_ handle daylight saving time
    /// transitions correctly. To handle this correctly, use
    /// [`HolidayOccurrence::new`] with the correct start and
    /// end times, e.g. by calculating them using [`chrono`].
    ///
    /// [`chrono`]: https://docs.rs/chrono/latest/chrono
    pub fn for_day(day: TimestampSeconds) -> Self {
        Self::for_day_in_tz(day, 0)
    }

    /// Create a new holiday encompassing midnight on the day
    /// of the given timestamp to midnight on the following day,
    /// in a timezone represented by the `utc_offset`.
    ///
    /// This is a convenience method to reproduce the behaviour
    /// of the Python and R Prophet implementations, which require
    /// all holidays to be day-long events.
    ///
    /// Note that this will _not_ handle daylight saving time
    /// transitions correctly. To handle this correctly, use
    /// [`HolidayOccurrence::new`] with the correct start and
    /// end times, e.g. by calculating them using [`chrono`].
    ///
    /// [`chrono`]: https://docs.rs/chrono/latest/chrono
    pub fn for_day_in_tz(day: TimestampSeconds, utc_offset: i32) -> Self {
        let day = floor_day(day, utc_offset);
        Self {
            start: day,
            end: day + ONE_DAY_IN_SECONDS_INT,
        }
    }

    /// Check if the given timestamp is within this occurrence.
    pub(crate) fn contains(&self, ds: TimestampSeconds) -> bool {
        self.start <= ds && ds < self.end
    }
}

/// A holiday to be considered by the Prophet model.
#[derive(Debug, Clone)]
pub struct Holiday {
    pub(crate) occurrences: Vec<HolidayOccurrence>,
    pub(crate) prior_scale: Option<PositiveFloat>,
}

impl Holiday {
    /// Create a new holiday with the given occurrences.
    pub fn new(occurrences: Vec<HolidayOccurrence>) -> Self {
        Self {
            occurrences,
            prior_scale: None,
        }
    }

    /// Set the prior scale for the holiday.
    pub fn with_prior_scale(mut self, prior_scale: PositiveFloat) -> Self {
        self.prior_scale = Some(prior_scale);
        self
    }
}

fn floor_day(ds: TimestampSeconds, offset: i32) -> TimestampSeconds {
    let adjusted_ds = ds + offset as TimestampSeconds;
    let remainder =
        ((adjusted_ds % ONE_DAY_IN_SECONDS_INT) + ONE_DAY_IN_SECONDS_INT) % ONE_DAY_IN_SECONDS_INT;
    // Adjust the date to the holiday's UTC offset.
    ds - remainder
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

    use crate::features::floor_day;

    #[test]
    fn floor_day_no_offset() {
        let offset = Utc;
        let expected = offset
            .with_ymd_and_hms(2024, 11, 21, 0, 0, 0)
            .unwrap()
            .timestamp();
        assert_eq!(floor_day(expected, 0), expected);
        assert_eq!(
            floor_day(
                offset
                    .with_ymd_and_hms(2024, 11, 21, 15, 3, 12)
                    .unwrap()
                    .timestamp(),
                0
            ),
            expected
        );
    }

    #[test]
    fn floor_day_positive_offset() {
        let offset = FixedOffset::east_opt(60 * 60 * 4).unwrap();
        let expected = offset
            .with_ymd_and_hms(2024, 11, 21, 0, 0, 0)
            .unwrap()
            .timestamp();

        assert_eq!(floor_day(expected, offset.local_minus_utc()), expected);
        assert_eq!(
            floor_day(
                offset
                    .with_ymd_and_hms(2024, 11, 21, 15, 3, 12)
                    .unwrap()
                    .timestamp(),
                offset.local_minus_utc()
            ),
            expected
        );
    }

    #[test]
    fn floor_day_negative_offset() {
        let offset = FixedOffset::west_opt(60 * 60 * 3).unwrap();
        let expected = offset
            .with_ymd_and_hms(2024, 11, 21, 0, 0, 0)
            .unwrap()
            .timestamp();

        assert_eq!(floor_day(expected, offset.local_minus_utc()), expected);
        assert_eq!(
            floor_day(
                offset
                    .with_ymd_and_hms(2024, 11, 21, 15, 3, 12)
                    .unwrap()
                    .timestamp(),
                offset.local_minus_utc()
            ),
            expected
        );
    }

    #[test]
    fn floor_day_edge_cases() {
        // Test maximum valid offset (UTC+14)
        let max_offset = 14 * 60 * 60;
        let offset = FixedOffset::east_opt(max_offset).unwrap();
        let expected = offset
            .with_ymd_and_hms(2024, 11, 21, 0, 0, 0)
            .unwrap()
            .timestamp();
        assert_eq!(
            floor_day(
                offset
                    .with_ymd_and_hms(2024, 11, 21, 12, 0, 0)
                    .unwrap()
                    .timestamp(),
                offset.local_minus_utc()
            ),
            expected
        );

        // Test near day boundary
        let offset = FixedOffset::east_opt(60).unwrap();
        let expected = offset
            .with_ymd_and_hms(2024, 11, 21, 0, 0, 0)
            .unwrap()
            .timestamp();
        assert_eq!(
            floor_day(
                offset
                    .with_ymd_and_hms(2024, 11, 21, 23, 59, 59)
                    .unwrap()
                    .timestamp(),
                offset.local_minus_utc()
            ),
            expected
        );

        // Test when the day is before the epoch.
        let offset = FixedOffset::west_opt(3600).unwrap();
        let expected = offset
            .with_ymd_and_hms(1969, 1, 1, 0, 0, 0)
            .unwrap()
            .timestamp();
        assert_eq!(
            floor_day(
                offset
                    .with_ymd_and_hms(1969, 1, 1, 0, 30, 0)
                    .unwrap()
                    .timestamp(),
                offset.local_minus_utc()
            ),
            expected
        );
    }
}
