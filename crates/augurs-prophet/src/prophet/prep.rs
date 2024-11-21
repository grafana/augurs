use std::{
    collections::{HashMap, HashSet},
    num::NonZeroU32,
};

use itertools::{izip, Either, Itertools, MinMaxResult};

use crate::{
    features::RegressorScale,
    optimizer::{Data, InitialParams},
    util::FloatIterExt,
    Error, FeatureMode, GrowthType, Holiday, PositiveFloat, Prophet, ProphetOptions, Scaling,
    Seasonality, SeasonalityOption, Standardize, TimestampSeconds, TrainingData,
};

const ONE_YEAR_IN_SECONDS: f64 = 365.25 * 24.0 * 60.0 * 60.0;
const ONE_WEEK_IN_SECONDS: f64 = 7.0 * 24.0 * 60.0 * 60.0;
const ONE_DAY_IN_SECONDS: f64 = 24.0 * 60.0 * 60.0;
pub(crate) const ONE_DAY_IN_SECONDS_INT: i64 = 24 * 60 * 60;

#[derive(Debug, Clone, Default)]
pub(super) struct Scales {
    pub(super) logistic_floor: bool,
    pub(super) y_min: f64,
    pub(super) y_scale: f64,
    pub(super) start: TimestampSeconds,
    pub(super) t_scale: f64,
    pub(super) regressors: HashMap<String, RegressorScale>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(super) struct Modes {
    pub(super) additive: HashSet<ComponentName>,
    pub(super) multiplicative: HashSet<ComponentName>,
}

impl Modes {
    /// Convenience method for inserting a name into the appropriate set.
    fn insert(&mut self, mode: FeatureMode, name: ComponentName) {
        match mode {
            FeatureMode::Additive => self.additive.insert(name),
            FeatureMode::Multiplicative => self.multiplicative.insert(name),
        };
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(super) enum ComponentName {
    Seasonality(String),
    Regressor(String),
    Holiday(String),
    Holidays,
    AdditiveTerms,
    MultiplicativeTerms,
    RegressorsAdditive,
    RegressorsMultiplicative,
}

#[derive(Debug, Clone)]
pub(super) struct ComponentColumns {
    pub(super) additive: Vec<i32>,
    pub(super) multiplicative: Vec<i32>,
    pub(super) all_holidays: Vec<i32>,
    pub(super) regressors_additive: Vec<i32>,
    pub(super) regressors_multiplicative: Vec<i32>,
    pub(super) seasonalities: HashMap<String, Vec<i32>>,
    pub(super) holidays: HashMap<String, Vec<i32>>,
    pub(super) regressors: HashMap<String, Vec<i32>>,
}

impl ComponentColumns {
    /// Create a new component matrix with the given components.
    ///
    /// The components are given as a list of (column index, component name) pairs.
    fn new(components: &[(usize, ComponentName)]) -> Self {
        // How many columns are there?
        let n_columns = components.iter().map(|(i, _)| i).max().unwrap_or(&0) + 1;
        let mut cols = Self {
            additive: vec![0; n_columns],
            multiplicative: vec![0; n_columns],
            all_holidays: vec![0; n_columns],
            regressors_additive: vec![0; n_columns],
            regressors_multiplicative: vec![0; n_columns],
            seasonalities: HashMap::new(),
            holidays: HashMap::new(),
            regressors: HashMap::new(),
        };
        for (i, name) in components {
            let i = *i;
            match name {
                ComponentName::AdditiveTerms => {
                    cols.additive[i] = 1;
                    cols.multiplicative[i] = 0;
                }
                ComponentName::MultiplicativeTerms => {
                    cols.additive[i] = 0;
                    cols.multiplicative[i] = 1;
                }
                ComponentName::Holidays => cols.all_holidays[i] = 1,
                ComponentName::RegressorsAdditive => cols.regressors_additive[i] = 1,
                ComponentName::RegressorsMultiplicative => cols.regressors_multiplicative[i] = 1,
                ComponentName::Seasonality(name) => {
                    cols.seasonalities
                        .entry(name.to_string())
                        .or_insert(vec![0; n_columns])[i] = 1;
                }
                ComponentName::Regressor(name) => {
                    cols.regressors
                        .entry(name.to_string())
                        .or_insert(vec![0; n_columns])[i] = 1;
                }
                ComponentName::Holiday(name) => {
                    cols.holidays
                        .entry(name.to_string())
                        .or_insert(vec![0; n_columns])[i] = 1;
                }
            }
        }
        cols
    }
}

/// The name of a feature column in the `X` matrix passed to Stan.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(super) enum FeatureName {
    /// A seasonality feature.
    Seasonality {
        /// The name of the seasonality.
        name: String,
        /// The ID of the seasonality feature. Each seasonality
        /// has a number of features, and this is used to
        /// distinguish between them.
        _id: usize,
    },
    /// A regressor feature.
    Regressor(String),
    /// A holiday feature.
    Holiday {
        /// The name of the holiday.
        name: String,
        /// The offset from the holiday date, as permitted
        /// by the lower or upper window.
        _offset: i32,
    },
    Dummy,
}

/// A data frame of features to be used for fitting or predicting.
///
/// The data will be passed to Stan to be used as the `X` matrix.
#[derive(Debug)]
pub(super) struct FeaturesFrame {
    pub(super) names: Vec<FeatureName>,
    pub(super) data: Vec<Vec<f64>>,
}

impl FeaturesFrame {
    fn new() -> Self {
        Self {
            names: Vec::new(),
            data: Vec::new(),
        }
    }

    fn extend(
        &mut self,
        names: impl Iterator<Item = FeatureName>,
        new_features: impl Iterator<Item = Vec<f64>>,
    ) {
        self.names.extend(names);
        self.data.extend(new_features);
    }

    fn push(&mut self, name: FeatureName, column: Vec<f64>) {
        self.names.push(name);
        self.data.push(column);
    }

    fn is_empty(&self) -> bool {
        self.names.is_empty()
    }
}

/// Final features to be included in the Stan model.
#[derive(Debug)]
pub(super) struct Features {
    /// The actual feature data.
    pub(super) features: FeaturesFrame,
    /// The indicator columns for the various features.
    pub(super) component_columns: ComponentColumns,
    /// The prior scales for each of the features.
    pub(super) prior_scales: Vec<PositiveFloat>,
    /// The modes of the features.
    pub(super) modes: Modes,

    holiday_names: HashSet<String>,
}

impl<O> Prophet<O> {
    pub(super) fn preprocess(&mut self, mut data: TrainingData) -> Result<Preprocessed, Error> {
        let n = data.ds.len();
        if n != data.y.len() {
            return Err(Error::MismatchedLengths {
                a: n,
                a_name: "ds".to_string(),
                b: data.y.len(),
                b_name: "y".to_string(),
            });
        }
        if n < 2 {
            return Err(Error::NotEnoughData);
        }
        (data.ds, data.y) = data
            .ds
            .into_iter()
            .zip(data.y)
            .filter(|(_, y)| !y.is_nan())
            .unzip();

        let mut history_dates = data.ds.clone();
        history_dates.sort_unstable();

        let (history, scales) = self.setup_dataframe(data, None)?;
        self.scales = Some(scales);
        self.set_auto_seasonalities(&history)?;
        let Features {
            features,
            prior_scales,
            modes,
            component_columns,
            holiday_names,
        } = self.make_all_features(&history)?;
        self.component_modes = Some(modes);
        self.train_holiday_names = Some(holiday_names);
        self.train_component_columns = Some(component_columns.clone());

        let (changepoints, changepoints_t) = self.get_changepoints(&history.ds)?;
        self.changepoints = Some(changepoints);
        self.changepoints_t = Some(changepoints_t.clone());

        let cap = if self.opts.growth == GrowthType::Logistic {
            history.cap_scaled.clone().ok_or(Error::MissingCap)?
        } else {
            vec![0.0; n]
        };

        // Transpose X; we store it column-major but Stan expects it a contiguous
        // array in row-major format.
        // format.
        #[allow(non_snake_case)]
        let mut X = vec![0.0; features.data.len() * features.data[0].len()];
        for (i, row) in features.data.iter().enumerate() {
            for (j, val) in row.iter().enumerate() {
                X[i + features.data.len() * j] = *val;
            }
        }

        let data = Data {
            T: history
                .ds
                .len()
                .try_into()
                .map_err(|_| Error::TooManyDataPoints(n))?,
            S: changepoints_t.len() as i32,
            K: features.names.len() as i32,
            tau: self.opts.changepoint_prior_scale,
            trend_indicator: self.opts.growth.into(),
            y: history.y_scaled.clone(),
            t: history.t.clone(),
            t_change: changepoints_t,
            X,
            sigmas: prior_scales,
            s_a: component_columns.additive,
            s_m: component_columns.multiplicative,
            cap,
        };

        Ok(Preprocessed {
            data,
            history,
            history_dates,
        })
    }

    /// Prepare dataframe for fitting or predicting.
    pub(super) fn setup_dataframe(
        &self,
        TrainingData {
            n,
            mut ds,
            mut y,
            mut seasonality_conditions,
            mut x,
            floor,
            cap,
        }: TrainingData,
        scales: Option<Scales>,
    ) -> Result<(ProcessedData, Scales), Error> {
        if y.iter().any(|y| y.is_infinite()) {
            return Err(Error::InfiniteValue {
                column: "y".to_string(),
            });
        }
        for name in self.regressors.keys() {
            if !x.contains_key(name) {
                return Err(Error::MissingRegressor(name.clone()));
            }
            // No need to check lengths or inf, we do that in [`TrainingData::with_regressors`].
        }
        for Seasonality { condition_name, .. } in self.seasonalities.values() {
            if let Some(condition_name) = condition_name {
                if !seasonality_conditions.contains_key(condition_name) {
                    return Err(Error::MissingSeasonalityCondition(condition_name.clone()));
                }
                // No need to check lengths or inf, we do that in [`TrainingData::with_regressors`].
            }
        }

        // Sort everything by date.
        let mut sort_indices = (0..n).collect_vec();
        sort_indices.sort_by_key(|i| ds[*i]);
        ds.sort();
        // y isn't provided for predictions.
        if !y.is_empty() {
            y = sort_indices.iter().map(|i| y[*i]).collect();
        }
        for condition in seasonality_conditions.values_mut() {
            *condition = sort_indices.iter().map(|i| condition[*i]).collect();
        }
        for regressor in x.values_mut() {
            *regressor = sort_indices.iter().map(|i| regressor[*i]).collect();
        }

        let scales = scales
            .map(Ok)
            .unwrap_or_else(|| self.initialize_scales(&ds, &y, &x, &floor, &cap))?;

        let floor = if scales.logistic_floor {
            floor.ok_or(Error::MissingFloor)?
        } else if self.opts.scaling == Scaling::AbsMax {
            vec![0.0; ds.len()]
        } else {
            vec![scales.y_min; ds.len()]
        };
        let cap_scaled = if self.opts.growth == GrowthType::Logistic {
            let cap = cap.as_ref().ok_or(Error::MissingCap)?;
            let mut cap_scaled = Vec::with_capacity(ds.len());
            for (cap, floor) in cap.iter().zip(&floor) {
                let cs = (cap - floor) / scales.y_scale;
                if cs <= 0.0 {
                    return Err(Error::CapNotGreaterThanFloor);
                }
                cap_scaled.push(cs);
            }
            Some(cap_scaled)
        } else {
            None
        };

        let t = ds
            .iter()
            .map(|ds| (ds - scales.start) as f64 / scales.t_scale)
            .collect();

        let y_scaled = y
            .iter()
            .zip(&floor)
            .map(|(y, floor)| (y - floor) / scales.y_scale)
            .collect();

        for (name, regressor) in &scales.regressors {
            let col = x
                .get_mut(name)
                .ok_or_else(|| Error::MissingRegressor(name.clone()))?;
            for x in col {
                *x = (*x - regressor.mu) / regressor.std;
            }
        }

        let data = ProcessedData {
            ds,
            t,
            y_scaled,
            cap,
            cap_scaled,
            floor,
            regressors: x,
            seasonality_conditions,
        };
        Ok((data, scales))
    }

    fn initialize_scales(
        &self,
        ds: &[TimestampSeconds],
        y: &[f64],
        regressors: &HashMap<String, Vec<f64>>,
        floor: &Option<Vec<f64>>,
        cap: &Option<Vec<f64>>,
    ) -> Result<Scales, Error> {
        let mut scales = Scales::default();
        match floor.as_ref() {
            Some(floor) if self.opts.growth == GrowthType::Logistic => {
                scales.logistic_floor = true;
                match (self.opts.scaling, cap) {
                    (Scaling::AbsMax, _) => {
                        let MinMaxResult::MinMax(y_min, y_scale) = y
                            .iter()
                            .zip(floor)
                            .map(|(y, floor)| (y - floor).abs())
                            .minmax_by(|a, b| {
                                a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                            })
                        else {
                            return Err(Error::AbsMaxScalingFailed);
                        };
                        scales.y_min = y_min;
                        scales.y_scale = y_scale;
                    }
                    (Scaling::MinMax, Some(cap)) => {
                        scales.y_min = floor.iter().copied().nanmin(true);
                        scales.y_scale = cap.iter().copied().nanmax(true) - scales.y_min;
                    }
                    _ => {
                        return Err(Error::MissingCap);
                    }
                }
            }
            Some(_) | None => match self.opts.scaling {
                Scaling::AbsMax => {
                    scales.y_min = 0.0;
                    scales.y_scale = y.iter().map(|y| y.abs()).nanmax(true);
                }
                Scaling::MinMax => {
                    scales.y_min = y.iter().copied().nanmin(true);
                    scales.y_scale = y.iter().copied().nanmax(true) - scales.y_min;
                }
            },
        };
        if scales.y_scale == 0.0 {
            scales.y_scale = 1.0;
        }

        scales.start = *ds.first().ok_or(Error::NotEnoughData)?;
        scales.t_scale = (*ds.last().ok_or(Error::NotEnoughData)? - scales.start) as f64;

        for (name, regressor) in self.regressors.iter() {
            // Standardize if requested.
            let col = regressors
                .get(name)
                .ok_or(Error::MissingRegressor(name.clone()))?;
            // If there are 2 or fewer unique values, don't standardize.
            let mut vals = col.to_vec();
            vals.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            vals.dedup();
            let mut regressor_scale = RegressorScale::default();
            if vals.len() >= 2 {
                if regressor.standardize == Standardize::Auto {
                    regressor_scale.standardize =
                        !(vals.len() == 2 && vals[0] == 0.0 && vals[1] == 1.0);
                }
                if regressor_scale.standardize {
                    let mean = vals.iter().sum::<f64>() / vals.len() as f64;
                    let variance =
                        vals.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / vals.len() as f64;
                    let std = variance.sqrt();
                    regressor_scale.mu = mean;
                    regressor_scale.std = std;
                }
            }
            scales.regressors.insert(name.clone(), regressor_scale);
        }
        Ok(scales)
    }

    fn handle_seasonality_opt(
        &self,
        name: &str,
        opt: SeasonalityOption,
        auto_disable: bool,
        default_order: NonZeroU32,
    ) -> Option<NonZeroU32> {
        match opt {
            SeasonalityOption::Auto if self.seasonalities.contains_key(name) => None,
            SeasonalityOption::Auto if auto_disable => None,
            SeasonalityOption::Auto | SeasonalityOption::Manual(true) => Some(default_order),
            SeasonalityOption::Manual(false) => None,
            SeasonalityOption::Fourier(order) => Some(order),
        }
    }

    /// Setup seasonalities that were configured to be automatically determined.
    ///
    /// Turns on yearly seasonality if there is >=2 years of history.
    /// Turns on weekly seasonality if there is >=2 weeks of history, and the
    /// spacing between dates in the history is <7 days.
    /// Turns on daily seasonality if there is >=2 days of history, and the
    /// spacing between dates in the history is <1 day.
    fn set_auto_seasonalities(&mut self, history: &ProcessedData) -> Result<(), Error> {
        let first_date = history.ds.first().unwrap();
        let last_date = history.ds.last().unwrap();
        let min_diff = history
            .ds
            .iter()
            .zip(history.ds.iter().skip(1))
            .filter_map(|(a, b)| {
                let diff = b - a;
                if diff > 0 {
                    Some(diff)
                } else {
                    None
                }
            })
            .min()
            .ok_or(Error::NotEnoughData)?;
        let range = (last_date - first_date) as f64;
        let yearly_disable = range < 2.0 * ONE_YEAR_IN_SECONDS;
        if let Some(fourier_order) = self.handle_seasonality_opt(
            "yearly",
            self.opts.yearly_seasonality,
            yearly_disable,
            NonZeroU32::new(10).unwrap(),
        ) {
            self.add_seasonality(
                "yearly".to_string(),
                Seasonality::new(365.25.try_into().unwrap(), fourier_order),
            )?;
        }

        let weekly_disable =
            range < 2.0 * ONE_WEEK_IN_SECONDS || min_diff as f64 >= ONE_WEEK_IN_SECONDS;
        if let Some(fourier_order) = self.handle_seasonality_opt(
            "weekly",
            self.opts.weekly_seasonality,
            weekly_disable,
            3.try_into().unwrap(),
        ) {
            self.add_seasonality(
                "weekly".to_string(),
                Seasonality::new(7.0.try_into().unwrap(), fourier_order),
            )?;
        }

        let daily_disable =
            range < 2.0 * ONE_DAY_IN_SECONDS || min_diff as f64 >= ONE_DAY_IN_SECONDS;
        if let Some(fourier_order) = self.handle_seasonality_opt(
            "daily",
            self.opts.daily_seasonality,
            daily_disable,
            4.try_into().unwrap(),
        ) {
            self.add_seasonality(
                "daily".to_string(),
                Seasonality::new(1.0.try_into().unwrap(), fourier_order),
            )?;
        }

        Ok(())
    }

    /// Compute fourier series components with the specified period
    /// and order.
    ///
    /// Note: this computes the transpose of the function in the Python
    /// code for simplicity, since we need it in a columnar format anyway.
    pub(super) fn fourier_series(
        dates: &[TimestampSeconds],
        period: PositiveFloat,
        order: NonZeroU32,
    ) -> Vec<Vec<f64>> {
        let order = order.get() as usize;
        // Convert seconds to days.
        let t = dates.iter().copied().map(|ds| ds as f64 / 3600.0 / 24.0);
        // Convert to radians.
        let x_t = t.map(|x| x * std::f64::consts::PI * 2.0).collect_vec();
        // Preallocate space for the fourier components.
        let mut fourier_components = Vec::with_capacity(2 * order);
        for i in 0..order {
            let (f1, f2) = x_t
                .iter()
                .map(|x| {
                    let angle = x * (i as f64 + 1.0) / *period;
                    (angle.sin(), angle.cos())
                })
                .unzip();
            fourier_components.push(f1);
            fourier_components.push(f2);
        }
        fourier_components
    }

    fn make_seasonality_features(
        dates: &[TimestampSeconds],
        seasonality: &Seasonality,
        prefix: &str,
        history: &ProcessedData,
        features: &mut FeaturesFrame,
    ) -> Result<usize, Error> {
        let mut new_features =
            Self::fourier_series(dates, seasonality.period, seasonality.fourier_order);
        // Handle conditions by setting feature values to zero where the condition is false.
        if let Some(condition_name) = &seasonality.condition_name {
            let col = history
                .seasonality_conditions
                .get(condition_name)
                .ok_or_else(|| Error::MissingSeasonalityCondition(condition_name.clone()))?;
            let false_indices = col
                .iter()
                .enumerate()
                .filter_map(|(i, b)| if *b { None } else { Some(i) })
                .collect_vec();
            for feature in new_features.iter_mut() {
                for idx in false_indices.iter().copied() {
                    feature[idx] = 0.0;
                }
            }
        }
        let n = new_features.len();
        let names = (0..n).map(|i| FeatureName::Seasonality {
            name: prefix.to_string(),
            _id: i + 1,
        });
        features.extend(names, new_features.into_iter());
        Ok(n)
    }

    fn construct_holidays(
        &self,
        _ds: &[TimestampSeconds],
    ) -> Result<HashMap<String, Holiday>, Error> {
        let mut all_holidays = self.opts.holidays.clone();
        // TODO: handle country holidays.

        // If predicting, drop future holidays not previously seen in training data,
        // and add holidays that were seen in the training data but aren't in the
        // prediction data.
        if let Some(train_holidays) = &self.train_holiday_names {
            all_holidays.retain(|name, _| train_holidays.contains(name));
            let holidays_to_add = train_holidays
                .iter()
                .filter(|name| !all_holidays.contains_key(name.as_str()))
                .collect_vec();
            all_holidays.extend(
                holidays_to_add
                    .into_iter()
                    .map(|name| (name.clone(), Holiday::new(vec![]))),
            );
        }
        Ok(all_holidays)
    }

    /// Construct a frame of features representing holidays.
    fn make_holiday_features(
        &self,
        ds: &[TimestampSeconds],
        holidays: HashMap<String, Holiday>,
        features: &mut FeaturesFrame,
        prior_scales: &mut Vec<PositiveFloat>,
        modes: &mut Modes,
    ) -> HashSet<String> {
        let mut holiday_names = HashSet::with_capacity(holidays.len());
        for (name, holiday) in holidays {
            // Keep track of holiday columns here.
            // For each day surrounding the holiday (decided by the lower and upper windows),
            // plus the holiday itself, we want to create a new feature which is 0.0 for all
            // days except that day, and 1.0 for that day.
            let mut this_holiday_features: HashMap<FeatureName, Vec<f64>> = HashMap::new();

            // Default to a window of 0 days either side.
            let lower = holiday
                .lower_window
                .as_ref()
                .map(|x| {
                    Box::new(x.iter().copied().map(|x| x as i32)) as Box<dyn Iterator<Item = i32>>
                })
                .unwrap_or_else(|| Box::new(std::iter::repeat(0)));
            let upper = holiday
                .upper_window
                .as_ref()
                .map(|x| {
                    Box::new(x.iter().copied().map(|x| x as i32)) as Box<dyn Iterator<Item = i32>>
                })
                .unwrap_or_else(|| Box::new(std::iter::repeat(0)));

            for (dt, lower, upper) in izip!(&holiday.ds, lower, upper) {
                // Round down the original timestamps to the nearest day.
                let dt_date = holiday.floor_day(*dt);

                // Check each of the possible offsets allowed by the lower/upper windows.
                // We know that the lower window is always positive since it was originally
                // a u32, so we can use `-lower..upper` here.
                for offset in -lower..=upper {
                    let offset_seconds = offset as i64 * ONE_DAY_IN_SECONDS as i64;
                    let occurrence = dt_date + offset_seconds;
                    let col_name = FeatureName::Holiday {
                        name: name.clone(),
                        _offset: offset,
                    };
                    let col = this_holiday_features
                        .entry(col_name.clone())
                        .or_insert_with(|| vec![0.0; ds.len()]);

                    // Get the indices of the ds column that are 'on holiday'.
                    // Set the value of the holiday column to 1.0 for those dates.
                    for loc in ds.iter().positions(|&x| holiday.floor_day(x) == occurrence) {
                        col[loc] = 1.0;
                    }
                }
            }
            // Add the holiday column to the features frame, and add a corresponding
            // prior scale.
            for (col_name, col) in this_holiday_features.drain() {
                features.push(col_name, col);
                prior_scales.push(
                    holiday
                        .prior_scale
                        .unwrap_or(self.opts.holidays_prior_scale),
                );
            }
            holiday_names.insert(name.clone());
            modes.insert(
                self.opts
                    .holidays_mode
                    .unwrap_or(self.opts.seasonality_mode),
                ComponentName::Holiday(name.clone()),
            );
        }
        holiday_names
    }

    /// Make all features for the model.
    // This is called `make_all_seasonality_features` in the Python
    // implementation but it includes holidays and regressors too so
    // it's been renamed here for clarity.
    pub(super) fn make_all_features(&self, history: &ProcessedData) -> Result<Features, Error> {
        let mut features = FeaturesFrame::new();
        let mut prior_scales = Vec::with_capacity(self.seasonalities.len());
        let mut modes = Modes::default();

        // Add seasonality features.
        for (name, seasonality) in &self.seasonalities {
            let n_new = Self::make_seasonality_features(
                &history.ds,
                seasonality,
                name,
                history,
                &mut features,
            )?;
            let prior_scale = seasonality
                .prior_scale
                .unwrap_or(self.opts.seasonality_prior_scale);
            prior_scales.extend(std::iter::repeat(prior_scale).take(n_new));
            modes.insert(
                seasonality.mode.unwrap_or(self.opts.seasonality_mode),
                ComponentName::Seasonality(name.clone()),
            )
        }

        let holidays = self.construct_holidays(&history.ds)?;
        let mut holiday_names = HashSet::new();
        if !holidays.is_empty() {
            holiday_names = self.make_holiday_features(
                &history.ds,
                holidays,
                &mut features,
                &mut prior_scales,
                &mut modes,
            );
        }

        // Add regressors.
        for (name, regressor) in &self.regressors {
            let col = history
                .regressors
                .get(name)
                .ok_or_else(|| Error::MissingRegressor(name.clone()))?;
            features.push(FeatureName::Regressor(name.clone()), col.clone());
            prior_scales.push(
                regressor
                    .prior_scale
                    .unwrap_or(self.opts.seasonality_prior_scale),
            );
            modes.insert(regressor.mode, ComponentName::Regressor(name.clone()));
        }

        // If there are no features, add a dummy column to prevent an empty features matrix.
        if features.is_empty() {
            features.push(FeatureName::Dummy, vec![0.0; history.ds.len()]);
            prior_scales.push(PositiveFloat::one());
        }

        let component_columns =
            self.regressor_column_matrix(&features.names, &holiday_names, &mut modes);
        Ok(Features {
            features,
            prior_scales,
            component_columns,
            modes,
            holiday_names,
        })
    }

    /// Compute a matrix indicating which columns of the features matrix correspond
    /// to which seasonality/regressor components.
    pub(super) fn regressor_column_matrix(
        &self,
        feature_names: &[FeatureName],
        train_holiday_names: &HashSet<String>,
        modes: &mut Modes,
    ) -> ComponentColumns {
        // Start with a vec of (col idx, component name) pairs.
        let mut components = feature_names
            .iter()
            .filter_map(|x| match x {
                FeatureName::Seasonality { name, _id: _ } => {
                    Some(ComponentName::Seasonality(name.clone()))
                }
                FeatureName::Regressor(name) => Some(ComponentName::Regressor(name.clone())),
                FeatureName::Holiday { name, .. } => Some(ComponentName::Holiday(name.clone())),
                _ => None,
            })
            .enumerate()
            .collect();

        // Add total for holidays.
        if !train_holiday_names.is_empty() {
            let component_names = train_holiday_names
                .iter()
                .map(|name| ComponentName::Holiday(name.clone()))
                .collect();
            Self::add_group_component(&mut components, ComponentName::Holidays, &component_names);
        }

        // Add additive and multiplicative components, and regressors.
        let (additive_regressors, multiplicative_regressors) =
            self.regressors.iter().partition_map(|(name, reg)| {
                if reg.mode == FeatureMode::Additive {
                    Either::Left(ComponentName::Regressor(name.clone()))
                } else {
                    Either::Right(ComponentName::Regressor(name.clone()))
                }
            });
        Self::add_group_component(
            &mut components,
            ComponentName::AdditiveTerms,
            &modes.additive,
        );
        Self::add_group_component(
            &mut components,
            ComponentName::RegressorsAdditive,
            &additive_regressors,
        );
        Self::add_group_component(
            &mut components,
            ComponentName::MultiplicativeTerms,
            &modes.multiplicative,
        );
        Self::add_group_component(
            &mut components,
            ComponentName::RegressorsMultiplicative,
            &multiplicative_regressors,
        );
        // Add the names of the group components to the modes.
        modes.additive.insert(ComponentName::AdditiveTerms);
        modes.additive.insert(ComponentName::RegressorsAdditive);
        modes
            .multiplicative
            .insert(ComponentName::MultiplicativeTerms);
        modes
            .multiplicative
            .insert(ComponentName::RegressorsMultiplicative);

        // Add holidays.
        modes.insert(
            self.opts
                .holidays_mode
                .unwrap_or(self.opts.seasonality_mode),
            ComponentName::Holidays,
        );

        ComponentColumns::new(&components)
    }

    /// Add a component with the given name that contains all of the components
    /// in `group`.
    fn add_group_component(
        components: &mut Vec<(usize, ComponentName)>,
        name: ComponentName,
        names: &HashSet<ComponentName>,
    ) {
        let group_cols = components
            .iter()
            .filter_map(|(i, n)| names.contains(n).then_some(*i))
            .dedup()
            .collect_vec();
        components.extend(group_cols.into_iter().map(|i| (i, name.clone())));
    }

    /// Get the changepoints for the model.
    ///
    /// Returns a tuple of (changepoints, changepoints_t) where
    /// `changepoints` is a vector of dates and `changepoints_t` is a
    /// vector of times.
    fn get_changepoints(
        &self,
        ds: &[TimestampSeconds],
    ) -> Result<(Vec<TimestampSeconds>, Vec<f64>), Error> {
        let first_date = ds.first().ok_or(Error::NotEnoughData)?;
        let last_date = ds.last().ok_or(Error::NotEnoughData)?;
        let changepoints = if let Some(changepoints) = self.opts.changepoints.as_ref() {
            if !changepoints.is_empty() {
                let too_low = changepoints.iter().any(|x| x < first_date);
                let too_high = changepoints.iter().any(|x| x > last_date);
                if too_low || too_high {
                    return Err(Error::ChangepointsOutOfRange);
                }
            }
            changepoints.clone()
        } else {
            let hist_size = (ds.len() as f64 * *self.opts.changepoint_range).floor() as usize;
            let mut n_changepoints = self.opts.n_changepoints as usize;
            if n_changepoints + 1 > hist_size {
                n_changepoints = hist_size - 1;
            }
            if n_changepoints > 0 {
                // Place changepoints evenly through the first `changepoint_range` percent of the history.
                // let step = ((hist_size - 1) as f64 / (n_changepoints + 1) as f64).round();
                // let cp_indices = (0..(hist_size - 1)).step_by(step as usize);
                let cp_indices = (0..(n_changepoints + 1)).map(|i| {
                    let num_steps = n_changepoints as f64; // note: don't add one since we don't include
                                                           // the last point.
                    (i as f64 / num_steps * (hist_size - 1) as f64).round() as usize
                });
                cp_indices.map(|i| ds[i]).skip(1).collect()
            } else {
                vec![]
            }
        };

        let scales = self
            .scales
            .as_ref()
            .expect("Scales not initialized when setting changepoints; this is a bug");
        let changepoints_t = if changepoints.is_empty() {
            vec![0.0] // Dummy changepoint.
        } else {
            changepoints
                .iter()
                .map(|ds| (ds - scales.start) as f64 / scales.t_scale)
                .collect()
        };
        Ok((changepoints, changepoints_t))
    }
}

/// Historical data after preprocessing.
#[derive(Debug, Clone)]
pub(super) struct ProcessedData {
    pub(super) ds: Vec<TimestampSeconds>,
    pub(super) t: Vec<f64>,
    pub(super) y_scaled: Vec<f64>,
    pub(super) cap: Option<Vec<f64>>,
    pub(super) cap_scaled: Option<Vec<f64>>,
    pub(super) floor: Vec<f64>,
    pub(super) regressors: HashMap<String, Vec<f64>>,
    pub(super) seasonality_conditions: HashMap<String, Vec<bool>>,
}

/// Processed data used for fitting.
#[derive(Debug, Clone)]
pub(super) struct Preprocessed {
    pub(super) data: Data,
    pub(super) history: ProcessedData,
    pub(super) history_dates: Vec<TimestampSeconds>,
}

impl Preprocessed {
    /// Calculate the initial parameters for the Stan optimization.
    pub(super) fn calculate_initial_params(
        &self,
        opts: &ProphetOptions,
    ) -> Result<InitialParams, Error> {
        let i_minmax = match self.history.ds.iter().position_minmax() {
            MinMaxResult::NoElements => return Err(Error::NotEnoughData),
            MinMaxResult::OneElement(i) => {
                return Err(Error::TimestampsAreConstant(self.history.ds[i]))
            }
            MinMaxResult::MinMax(i0, i1) => (i0, i1),
        };
        let (k, m) = match opts.growth {
            GrowthType::Linear => self.linear_growth_init(i_minmax),
            GrowthType::Flat => self.flat_growth_init(),
            GrowthType::Logistic => self.logistic_growth_init(i_minmax)?,
        };

        Ok(InitialParams {
            k,
            m,
            delta: vec![0.0; self.data.t_change.len()],
            beta: vec![0.0; self.data.K as usize],
            sigma_obs: 1.0.try_into().unwrap(),
        })
    }

    /// Initialize linear growth.
    ///
    /// Provides a strong initialization for linear growth by calculating the
    /// growth and offset parameters that pass the function through the first
    /// and last points in the time series.
    ///
    /// The argument `(i0, i1)` is the range of indices in the history that
    /// correspond to the first and last points in the time series.
    ///
    /// Returns a tuple (k, m) with the rate (k) and offset (m) of the linear growth
    /// function.
    fn linear_growth_init(&self, (i0, i1): (usize, usize)) -> (f64, f64) {
        let ProcessedData { t, y_scaled, .. } = &self.history;
        let t_diff = t[i1] - t[i0];
        let k = (y_scaled[i1] - y_scaled[i0]) / t_diff;
        let m = y_scaled[i0] - k * t[i0];
        (k, m)
    }

    /// Initialize flat growth.
    ///
    /// Provides a strong initialization for flat growth. Sets the growth to 0
    /// and offset parameter as mean of history `y_scaled` values.
    ///
    /// Returns a tuple (k, m) with the rate (k) and offset (m) of the linear growth
    /// function.
    fn flat_growth_init(&self) -> (f64, f64) {
        let ProcessedData { y_scaled, .. } = &self.history;
        let k = 0.0;
        let m = y_scaled.iter().sum::<f64>() / y_scaled.len() as f64;
        (k, m)
    }

    /// Initialize logistic growth.
    ///
    /// Provides a strong initialization for logistic growth by calculating the
    /// growth and offset parameters that pass the function through the first
    /// and last points in the time series.
    ///
    /// The argument `(i0, i1)` is the range of indices in the history that
    /// correspond to the first and last points in the time series.
    ///
    /// Returns a tuple (k, m) with the rate (k) and offset (m) of the logistic growth
    /// function.
    fn logistic_growth_init(&self, (i0, i1): (usize, usize)) -> Result<(f64, f64), Error> {
        let ProcessedData {
            t,
            y_scaled,
            cap_scaled,
            ..
        } = &self.history;

        let cap_scaled = cap_scaled.as_ref().ok_or(Error::MissingCap)?;
        let t_diff = t[i1] - t[i0];

        // Force valid values, in case y > cap or y < 0
        let (c0, c1) = (cap_scaled[i0], cap_scaled[i1]);
        let y0 = f64::max(0.01 * c0, f64::min(0.99 * c0, y_scaled[i0]));
        let y1 = f64::max(0.01 * c1, f64::min(0.99 * c1, y_scaled[i1]));

        let mut r0 = c0 / y0;
        let r1 = c1 / y1;
        if (r0 - r1).abs() <= 0.01 {
            r0 *= 1.05;
        }

        let l0 = (r0 - 1.0).ln();
        let l1 = (r1 - 1.0).ln();

        // Initialize the offset.
        let m = l0 * t_diff / (l0 - l1);
        // And the rate
        let k = (l0 - l1) / t_diff;
        Ok((k, m))
    }
}

#[cfg(test)]
mod test {
    use crate::{
        optimizer::mock_optimizer::MockOptimizer,
        testdata::{daily_univariate_ts, train_test_split},
        util::FloatIterExt,
        ProphetOptions, Regressor, Standardize,
    };

    use super::*;
    use augurs_testing::assert_approx_eq;
    use chrono::{FixedOffset, NaiveDate, TimeZone, Utc};
    use pretty_assertions::assert_eq;

    macro_rules! concat_all {
        ($($x:expr),+ $(,)?) => {{
            let mut result = Vec::new();
            $(
                result.extend($x.iter().cloned());
            )+
            result
        }};
    }

    #[test]
    fn setup_dataframe() {
        let (data, _) = train_test_split(daily_univariate_ts(), 0.5);
        let prophet = Prophet::new(ProphetOptions::default(), MockOptimizer::new());
        let (history, _) = prophet.setup_dataframe(data, None).unwrap();

        assert_approx_eq!(history.t.iter().copied().nanmin(true), 0.0);
        assert_approx_eq!(history.t.iter().copied().nanmax(true), 1.0);
        assert_approx_eq!(history.y_scaled.iter().copied().nanmax(true), 1.0);
    }

    #[test]
    fn logistic_floor() {
        let (mut data, _) = train_test_split(daily_univariate_ts(), 0.5);
        let n = data.len();
        data = data
            .with_floor(vec![10.0; n])
            .unwrap()
            .with_cap(vec![80.0; n])
            .unwrap();
        let opts = ProphetOptions {
            growth: GrowthType::Logistic,
            ..ProphetOptions::default()
        };
        let mut prophet = Prophet::new(opts.clone(), MockOptimizer::new());
        prophet.fit(data.clone(), Default::default()).unwrap();
        assert!(prophet.scales.unwrap().logistic_floor);
        assert_approx_eq!(prophet.processed.unwrap().history.y_scaled[0], 1.0);

        data.y.iter_mut().for_each(|y| *y += 10.0);
        for f in data.floor.as_mut().unwrap() {
            *f += 10.0;
        }
        for c in data.cap.as_mut().unwrap() {
            *c += 10.0;
        }
        let mut prophet = Prophet::new(opts.clone(), MockOptimizer::new());
        prophet.fit(data, Default::default()).unwrap();
        assert_eq!(prophet.processed.unwrap().history.y_scaled[0], 1.0);
    }

    #[test]
    fn logistic_floor_minmax() {
        let (mut data, _) = train_test_split(daily_univariate_ts(), 0.5);
        let n = data.len();
        data = data
            .with_floor(vec![10.0; n])
            .unwrap()
            .with_cap(vec![80.0; n])
            .unwrap();
        let opts = ProphetOptions {
            growth: GrowthType::Logistic,
            scaling: Scaling::MinMax,
            ..ProphetOptions::default()
        };
        let mut prophet = Prophet::new(opts.clone(), MockOptimizer::new());
        prophet.fit(data.clone(), Default::default()).unwrap();
        assert!(prophet.scales.unwrap().logistic_floor);
        assert!(
            prophet
                .processed
                .as_ref()
                .unwrap()
                .history
                .y_scaled
                .iter()
                .copied()
                .nanmin(true)
                > 0.0
        );
        assert!(
            prophet
                .processed
                .unwrap()
                .history
                .y_scaled
                .iter()
                .copied()
                .nanmax(true)
                < 1.0
        );

        data.y.iter_mut().for_each(|y| *y += 10.0);
        for f in data.floor.as_mut().unwrap() {
            *f += 10.0;
        }
        for c in data.cap.as_mut().unwrap() {
            *c += 10.0;
        }
        let mut prophet = Prophet::new(opts.clone(), MockOptimizer::new());
        prophet.fit(data, Default::default()).unwrap();
        assert!(
            prophet
                .processed
                .as_ref()
                .unwrap()
                .history
                .y_scaled
                .iter()
                .copied()
                .nanmin(true)
                > 0.0
        );
        assert!(
            prophet
                .processed
                .unwrap()
                .history
                .y_scaled
                .iter()
                .copied()
                .nanmax(true)
                < 1.0
        );
    }

    #[test]
    fn make_holiday_features() {
        // Create some hourly data between 2024-01-01 and 2024-01-07.
        let start = Utc.with_ymd_and_hms(2024, 1, 1, 0, 0, 0).unwrap();
        let end = Utc.with_ymd_and_hms(2024, 1, 7, 0, 0, 0).unwrap();
        let ds = std::iter::successors(Some(start), |d| {
            d.checked_add_signed(chrono::Duration::hours(1))
        })
        .take_while(|d| *d < end)
        .map(|d| d.timestamp())
        .collect_vec();
        // Create two holidays: one in UTC on 2024-01-02 and 2024-01-04;
        // one in UTC-3 on the same dates.
        // The holidays may appear more than once since the data is hourly,
        // and this shouldn't affect the results.
        // Ignore windows for now.
        let non_utc_tz = FixedOffset::west_opt(3600 * 3).unwrap();
        let holidays: HashMap<String, Holiday> = [
            (
                "UTC holiday".to_string(),
                Holiday::new(vec![
                    Utc.with_ymd_and_hms(2024, 1, 2, 0, 0, 0)
                        .unwrap()
                        .timestamp(),
                    Utc.with_ymd_and_hms(2024, 1, 2, 12, 0, 0)
                        .unwrap()
                        .timestamp(),
                    Utc.with_ymd_and_hms(2024, 1, 4, 0, 0, 0)
                        .unwrap()
                        .timestamp(),
                ]),
            ),
            (
                "Non-UTC holiday".to_string(),
                Holiday::new(vec![
                    non_utc_tz
                        .with_ymd_and_hms(2024, 1, 2, 0, 0, 0)
                        .unwrap()
                        .timestamp(),
                    non_utc_tz
                        .with_ymd_and_hms(2024, 1, 2, 12, 0, 0)
                        .unwrap()
                        .timestamp(),
                    non_utc_tz
                        .with_ymd_and_hms(2024, 1, 4, 0, 0, 0)
                        .unwrap()
                        .timestamp(),
                ])
                .with_utc_offset(-3 * 3600),
            ),
            (
                "Non-UTC holiday with windows".to_string(),
                Holiday::new(vec![
                    non_utc_tz
                        .with_ymd_and_hms(2024, 1, 2, 0, 0, 0)
                        .unwrap()
                        .timestamp(),
                    non_utc_tz
                        .with_ymd_and_hms(2024, 1, 2, 12, 0, 0)
                        .unwrap()
                        .timestamp(),
                    non_utc_tz
                        .with_ymd_and_hms(2024, 1, 4, 0, 0, 0)
                        .unwrap()
                        .timestamp(),
                ])
                .with_lower_window(vec![1; 3])
                .unwrap()
                .with_upper_window(vec![1; 3])
                .unwrap()
                .with_utc_offset(-3 * 3600),
            ),
        ]
        .into();
        let opts = ProphetOptions {
            holidays: holidays.clone(),
            ..Default::default()
        };
        let prophet = Prophet::new(opts, MockOptimizer::new());
        let mut features_frame = FeaturesFrame::new();
        let mut prior_scales = Vec::new();
        let mut modes = Modes::default();

        let holiday_names = prophet.make_holiday_features(
            &ds,
            holidays,
            &mut features_frame,
            &mut prior_scales,
            &mut modes,
        );
        assert_eq!(
            holiday_names,
            HashSet::from([
                "UTC holiday".to_string(),
                "Non-UTC holiday".to_string(),
                "Non-UTC holiday with windows".to_string()
            ])
        );

        assert_eq!(features_frame.names.len(), 5);
        let utc_idx = features_frame
            .names
            .iter()
            .position(|x| matches!(x, FeatureName::Holiday { name, .. } if name == "UTC holiday"))
            .unwrap();
        assert_eq!(
            features_frame.data[utc_idx],
            concat_all!(
                &[0.0; 24], // 2024-01-01 - off holiday
                &[1.0; 24], // 2024-01-02 - on holiday
                &[0.0; 24], // 2024-01-03 - off holiday
                &[1.0; 24], // 2024-01-04 - on holiday
                &[0.0; 48], // 2024-01-05 and 2024-01-06 - off holiday
            ),
        );
        let non_utc_idx = features_frame
            .names
            .iter()
            .position(
                |x| matches!(x, FeatureName::Holiday { name, .. } if name == "Non-UTC holiday"),
            )
            .unwrap();
        assert_eq!(
            features_frame.data[non_utc_idx],
            concat_all!(
                &[0.0; 24], // 2024-01-01 - off holiday
                &[0.0; 3],  // first 3 hours of 2024-01-02 in UTC are off holiday
                &[1.0; 24], // rest of 2024-01-02 in UTC, and first 3 hours of the next day, are on holiday
                &[0.0; 24], // continue the cycle...
                &[1.0; 24],
                &[0.0; 21 + 24],
            ),
        );

        let non_utc_lower_window_idx = features_frame
            .names
            .iter()
            .position(
                |x| matches!(x, FeatureName::Holiday { name, _offset: -1 } if name == "Non-UTC holiday with windows"),
            )
            .unwrap();
        assert_eq!(
            features_frame.data[non_utc_lower_window_idx],
            concat_all!(
                &[0.0; 3],  // first 3 hours of 2024-01-01 in UTC - off holiday
                &[1.0; 24], // rest of 2024-01-01 and start of 2024-01-02 are on holiday
                &[0.0; 24], // continue the cycle
                &[1.0; 24],
                &[0.0; 21 + 48],
            ),
        );
    }

    #[test]
    fn regressor_column_matrix() {
        let holiday_dates = ["2012-10-09", "2013-10-09"]
            .iter()
            .map(|s| {
                s.parse::<NaiveDate>()
                    .unwrap()
                    .and_hms_opt(0, 0, 0)
                    .unwrap()
                    .and_utc()
                    .timestamp()
            })
            .collect();
        let opts = ProphetOptions {
            holidays: [(
                "bens-bday".to_string(),
                Holiday::new(holiday_dates)
                    .with_lower_window(vec![0, 0])
                    .unwrap()
                    .with_upper_window(vec![1, 1])
                    .unwrap(),
            )]
            .into(),
            ..Default::default()
        };
        let mut prophet = Prophet::new(opts, MockOptimizer::new());
        prophet
            .add_regressor(
                "binary_feature".to_string(),
                Regressor::additive().with_prior_scale(0.2.try_into().unwrap()),
            )
            .add_regressor(
                "numeric_feature".to_string(),
                Regressor::additive().with_prior_scale(0.5.try_into().unwrap()),
            )
            .add_regressor(
                "numeric_feature2".to_string(),
                Regressor::multiplicative().with_prior_scale(0.5.try_into().unwrap()),
            )
            .add_regressor(
                "binary_feature2".to_string(),
                Regressor::additive().with_standardize(Standardize::Yes),
            );
        let mut modes = Modes {
            additive: HashSet::from([
                ComponentName::Seasonality("weekly".to_string()),
                ComponentName::Regressor("binary_feature".to_string()),
                ComponentName::Regressor("numeric_feature".to_string()),
                ComponentName::Regressor("binary_feature2".to_string()),
                ComponentName::Holiday("bens-bday".to_string()),
            ]),
            multiplicative: HashSet::from([ComponentName::Regressor(
                "numeric_feature2".to_string(),
            )]),
        };
        let cols = prophet.regressor_column_matrix(
            &[
                FeatureName::Seasonality {
                    name: "weekly".to_string(),
                    _id: 1,
                },
                FeatureName::Seasonality {
                    name: "weekly".to_string(),
                    _id: 2,
                },
                FeatureName::Seasonality {
                    name: "weekly".to_string(),
                    _id: 3,
                },
                FeatureName::Seasonality {
                    name: "weekly".to_string(),
                    _id: 4,
                },
                FeatureName::Seasonality {
                    name: "weekly".to_string(),
                    _id: 5,
                },
                FeatureName::Seasonality {
                    name: "weekly".to_string(),
                    _id: 6,
                },
                FeatureName::Holiday {
                    name: "bens-bday".to_string(),
                    _offset: 0,
                },
                FeatureName::Holiday {
                    name: "bens-bday".to_string(),
                    _offset: 1,
                },
                FeatureName::Regressor("binary_feature".to_string()),
                FeatureName::Regressor("numeric_feature".to_string()),
                FeatureName::Regressor("numeric_feature2".to_string()),
                FeatureName::Regressor("binary_feature2".to_string()),
            ],
            &["bens-bday".to_string()].into_iter().collect(),
            &mut modes,
        );
        assert_eq!(cols.additive, vec![1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1]);
        assert_eq!(
            cols.multiplicative,
            vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
        );
        assert_eq!(cols.all_holidays, vec![0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]);
        assert_eq!(
            cols.regressors_additive,
            vec![0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1]
        );
        assert_eq!(
            cols.regressors_multiplicative,
            vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
        );
        assert_eq!(cols.seasonalities.len(), 1);
        assert_eq!(
            cols.seasonalities["weekly"],
            &[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
        );
        assert_eq!(cols.holidays.len(), 1);
        assert_eq!(
            cols.holidays["bens-bday"],
            &[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
        );
        assert_eq!(cols.regressors.len(), 4);
        assert_eq!(
            cols.regressors["binary_feature"],
            &[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
        );
        assert_eq!(
            cols.regressors["numeric_feature"],
            &[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
        );
        assert_eq!(
            cols.regressors["numeric_feature2"],
            &[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
        );
        assert_eq!(
            cols.regressors["binary_feature2"],
            &[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        );
        assert_eq!(
            modes,
            Modes {
                additive: HashSet::from([
                    ComponentName::Seasonality("weekly".to_string()),
                    ComponentName::Regressor("binary_feature".to_string()),
                    ComponentName::Regressor("numeric_feature".to_string()),
                    ComponentName::Regressor("binary_feature2".to_string()),
                    ComponentName::Holiday("bens-bday".to_string()),
                    ComponentName::Holidays,
                    ComponentName::RegressorsAdditive,
                    ComponentName::AdditiveTerms,
                ]),
                multiplicative: HashSet::from([
                    ComponentName::Regressor("numeric_feature2".to_string()),
                    ComponentName::RegressorsMultiplicative,
                    ComponentName::MultiplicativeTerms,
                ]),
            }
        );
    }

    #[test]
    fn add_group_component() {
        let mut components = vec![
            (0, ComponentName::Seasonality("weekly".to_string())),
            (1, ComponentName::Seasonality("weekly".to_string())),
            (2, ComponentName::Seasonality("weekly".to_string())),
            (3, ComponentName::Seasonality("weekly".to_string())),
            (4, ComponentName::Seasonality("weekly".to_string())),
            (5, ComponentName::Seasonality("weekly".to_string())),
            (6, ComponentName::Holiday("birthday".to_string())),
            (7, ComponentName::Holiday("birthday".to_string())),
        ];
        let names = HashSet::from([ComponentName::Holiday("birthday".to_string())]);
        Prophet::<()>::add_group_component(&mut components, ComponentName::Holidays, &names);
        assert_eq!(
            components,
            vec![
                (0, ComponentName::Seasonality("weekly".to_string())),
                (1, ComponentName::Seasonality("weekly".to_string())),
                (2, ComponentName::Seasonality("weekly".to_string())),
                (3, ComponentName::Seasonality("weekly".to_string())),
                (4, ComponentName::Seasonality("weekly".to_string())),
                (5, ComponentName::Seasonality("weekly".to_string())),
                (6, ComponentName::Holiday("birthday".to_string())),
                (7, ComponentName::Holiday("birthday".to_string())),
                (6, ComponentName::Holidays),
                (7, ComponentName::Holidays),
            ]
        );
    }

    #[test]
    fn test_component_columns() {
        let components = [
            (0, ComponentName::Seasonality("weekly".to_string())),
            (1, ComponentName::Seasonality("weekly".to_string())),
            (2, ComponentName::Seasonality("weekly".to_string())),
            (3, ComponentName::Seasonality("weekly".to_string())),
            (4, ComponentName::Seasonality("weekly".to_string())),
            (5, ComponentName::Seasonality("weekly".to_string())),
            (6, ComponentName::Holiday("birthday".to_string())),
            (7, ComponentName::Holiday("birthday".to_string())),
            (8, ComponentName::Regressor("binary_feature".to_string())),
            (9, ComponentName::Regressor("numeric_feature".to_string())),
            (10, ComponentName::Regressor("numeric_feature2".to_string())),
            (11, ComponentName::Regressor("binary_feature2".to_string())),
            (6, ComponentName::Holidays),
            (7, ComponentName::Holidays),
            (0, ComponentName::AdditiveTerms),
            (1, ComponentName::AdditiveTerms),
            (2, ComponentName::AdditiveTerms),
            (3, ComponentName::AdditiveTerms),
            (4, ComponentName::AdditiveTerms),
            (5, ComponentName::AdditiveTerms),
            (8, ComponentName::AdditiveTerms),
            (9, ComponentName::AdditiveTerms),
            (11, ComponentName::AdditiveTerms),
            (8, ComponentName::RegressorsAdditive),
            (9, ComponentName::RegressorsAdditive),
            (11, ComponentName::RegressorsAdditive),
            (6, ComponentName::MultiplicativeTerms),
            (7, ComponentName::MultiplicativeTerms),
            (10, ComponentName::MultiplicativeTerms),
            (10, ComponentName::RegressorsMultiplicative),
        ];
        let cols = ComponentColumns::new(&components);
        assert_eq!(cols.additive, vec![1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1]);
        assert_eq!(
            cols.multiplicative,
            vec![0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0]
        );
        assert_eq!(cols.all_holidays, vec![0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]);
        assert_eq!(
            cols.regressors_additive,
            vec![0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1]
        );
        assert_eq!(
            cols.regressors_multiplicative,
            vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
        );
        assert_eq!(cols.seasonalities.len(), 1);
        assert_eq!(cols.holidays.len(), 1);
        assert_eq!(cols.regressors.len(), 4);
        assert_eq!(
            cols.seasonalities["weekly"],
            &[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
        );
        assert_eq!(
            cols.holidays["birthday"],
            &[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
        );
        assert_eq!(
            cols.regressors["binary_feature"],
            &[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
        );
        assert_eq!(
            cols.regressors["numeric_feature"],
            &[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
        );
        assert_eq!(
            cols.regressors["numeric_feature2"],
            &[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
        );
        assert_eq!(
            cols.regressors["binary_feature2"],
            &[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        );
    }

    #[test]
    fn constant_regressor() {
        let data = daily_univariate_ts();
        let n = data.len();
        let data = data
            .with_regressors(
                HashMap::from([("constant_feature".to_string(), vec![0.0; n])])
                    .into_iter()
                    .collect(),
            )
            .unwrap();
        let mut prophet = Prophet::new(ProphetOptions::default(), MockOptimizer::new());
        prophet.add_regressor("constant_feature".to_string(), Regressor::additive());
        prophet.fit(data, Default::default()).unwrap();
        let reg_scales = &prophet.scales.unwrap().regressors["constant_feature"];
        assert_approx_eq!(reg_scales.mu, 0.0);
        assert_approx_eq!(reg_scales.std, 1.0);
    }
}
