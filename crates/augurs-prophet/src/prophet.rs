pub(crate) mod options;

use std::{
    collections::{HashMap, HashSet},
    num::NonZeroU32,
};

use itertools::{Either, Itertools, MinMaxResult};
use options::{GrowthType, ProphetOptions, Scaling, SeasonalityOption};
use tracing::instrument;

use crate::{
    optimizer::{Data, InitialParams, OptimizeOpts, OptimizedParams, Optimizer},
    Error, FeatureMode, Holiday, PositiveFloat, PredictionData, Regressor, Seasonality,
    Standardize, TimestampSeconds, TrainingData,
};

const NO_REGRESSORS_PLACEHOLDER: &str = "__no_regressors_zeros__";

#[derive(Debug, Default)]
struct Scales {
    logistic_floor: bool,
    y_min: f64,
    y_scale: f64,
    start: TimestampSeconds,
    t_scale: f64,
}

#[derive(Debug, Default, PartialEq, Eq)]
struct Modes {
    additive: HashSet<String>,
    multiplicative: HashSet<String>,
}

impl Modes {
    /// Convenience method for inserting a name into the appropriate set.
    fn insert(&mut self, mode: FeatureMode, name: String) {
        if mode == FeatureMode::Additive {
            self.additive.insert(name);
        } else {
            self.multiplicative.insert(name);
        }
    }
}

#[derive(Debug, Clone)]
struct ComponentColumns {
    additive: Vec<i32>,
    multiplicative: Vec<i32>,
    holidays: Vec<i32>,
    extra_regressors_additive: Vec<i32>,
    extra_regressors_multiplicative: Vec<i32>,
    custom: HashMap<String, Vec<i32>>,
}

impl ComponentColumns {
    /// Create a new component matrix with the given components.
    ///
    /// The components are given as a list of (column index, component name) pairs.
    fn new(components: &[(usize, String)]) -> Self {
        // How many columns are there?
        let n_columns = components.iter().map(|(i, _)| i).max().unwrap_or(&0) + 1;
        let mut cols = Self {
            additive: vec![0; n_columns],
            multiplicative: vec![0; n_columns],
            holidays: vec![0; n_columns],
            extra_regressors_additive: vec![0; n_columns],
            extra_regressors_multiplicative: vec![0; n_columns],
            custom: HashMap::new(),
        };
        for (i, name) in components {
            let i = *i;
            if name == "additive_terms" {
                cols.additive[i] = 1;
                cols.multiplicative[i] = 0;
            } else if name == "multiplicative_terms" {
                cols.additive[i] = 0;
                cols.multiplicative[i] = 1;
            } else if name == "holidays" {
                cols.holidays[i] = 1;
            } else if name == "extra_regressors_additive" {
                cols.extra_regressors_additive[i] = 1;
            } else if name == "extra_regressors_multiplicative" {
                cols.extra_regressors_multiplicative[i] = 1;
            } else if name != NO_REGRESSORS_PLACEHOLDER {
                // Don't add the placeholder column.
                cols.custom
                    .entry(name.to_string())
                    .or_insert(vec![0; n_columns])[i] = 1;
            }
        }
        cols
    }
}

/// The name of a feature column in the `X` matrix passed to Stan.
#[derive(Debug, Clone)]
enum FeatureName {
    /// A seasonality feature.
    Seasonality(String, usize),
    /// A regressor feature.
    Regressor(String),
    Dummy,
}

/// A data frame of features to be used for fitting or predicting.
///
/// The data will be passed to Stan to be used as the `X` matrix.
#[derive(Debug)]
struct FeaturesFrame {
    names: Vec<FeatureName>,
    data: Vec<Vec<f64>>,
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
struct Features {
    /// The actual feature data.
    features: FeaturesFrame,
    /// The indicator columns for the various features.
    component_columns: ComponentColumns,
    /// The prior scales for each of the features.
    prior_scales: Vec<PositiveFloat>,
    /// The modes of the features.
    modes: Modes,
}

/// The Prophet time series forecasting model.
///
/// # Example
///
/// ```
/// use augurs_prophet::{Prophet, TrainingData};
///
/// let data = TrainingData::new(
///    vec![0, 1, 2, 3, 4],
///    vec![0.5, 1.4, 2.6, 3.5, 4.4],
/// );
/// let optimizer = DummyOptimizer;
///
/// let model = Prophet::new(Default::default(), &opt)?;
/// model.fit(&data);
/// ```
#[derive(Debug)]
pub struct Prophet {
    /// Options to be used for fitting.
    opts: ProphetOptions,

    /// Extra regressors.
    extra_regressors: HashMap<String, Regressor>,

    /// Custom seasonalities.
    seasonalities: HashMap<String, Seasonality>,

    // TODO: move all of the below into a separate struct.
    // That way we minimize the number of fields in this struct
    // and the number of permutations of optional fields,
    // so it's harder to accidentally get into an invalid state.
    /// Scaling factors for the data.
    ///
    /// This is calculated during fitting, and is used to scale the data
    /// before fitting.
    scales: Option<Scales>,

    /// The changepoints for the model.
    changepoints: Option<Vec<TimestampSeconds>>,

    /// The time of the changepoints.
    changepoints_t: Option<Vec<f64>>,

    /// The modes of the components.
    component_modes: Option<Modes>,

    /// The component columns used for training.
    train_component_columns: Option<ComponentColumns>,

    /// The names of the holidays that were seen in the training data.
    train_holiday_names: Option<HashSet<String>>,

    /// The optimizer to use.
    optimizer: &'static dyn Optimizer,

    /// The initial parameters passed to optimization.
    init: Option<InitialParams>,

    /// The optimized model, if it has been fit.
    optimized: Option<OptimizedParams>,
}

impl Prophet {
    /// Create a new Prophet model with the given options and optimizer.
    ///
    /// # Example
    ///
    /// ```
    /// use augurs_prophet::Prophet;
    ///
    /// let optimizer = DummyOptimizer;
    /// let model = Prophet::new(Default::default(), &opt)?;
    /// ```
    pub fn new(opts: ProphetOptions, optimizer: &'static dyn Optimizer) -> Self {
        Self {
            opts,
            extra_regressors: HashMap::new(),
            seasonalities: HashMap::new(),
            scales: None,
            changepoints: None,
            changepoints_t: None,
            component_modes: None,
            train_component_columns: None,
            train_holiday_names: None,
            optimizer,
            init: None,
            optimized: None,
        }
    }

    /// Add a custom seasonality to the model.
    pub fn add_seasonality(&mut self, name: String, seasonality: Seasonality) -> Result<(), Error> {
        // TODO: validate name
        // let prior_scale = prior_scale.unwrap_or(self.opts.seasonality_prior_scale);
        // let mode = mode.unwrap_or(self.opts.seasonality_mode);
        self.seasonalities.insert(name, seasonality);
        Ok(())
    }

    /// Add a regressor to the model.
    pub fn add_regressor(&mut self, name: String, regressor: Regressor) {
        self.extra_regressors.insert(name, regressor);
    }

    fn preprocess(&mut self, mut data: TrainingData) -> Result<Preprocessed, Error> {
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
            ..
        } = self.make_all_features(&history)?;
        self.component_modes = Some(modes);
        self.train_component_columns = Some(component_columns.clone());

        let (changepoints, changepoints_t) = self.get_changepoints(&history.ds)?;
        self.changepoints = Some(changepoints);
        self.changepoints_t = Some(changepoints_t.clone());

        let cap = if self.opts.growth == GrowthType::Logistic {
            history.cap_scaled.clone().ok_or(Error::MissingCap)?
        } else {
            vec![0.0; n]
        };

        let data = Data {
            T: history
                .ds
                .len()
                .try_into()
                .map_err(|_| Error::TooManyDataPoints(n))?,
            S: changepoints_t.len() as i32,
            K: features.data.len() as i32,
            tau: self.opts.changepoint_prior_scale,
            trend_indicator: self.opts.growth.into(),
            y: history.y_scaled.clone(),
            t: history.t.clone(),
            t_change: changepoints_t,
            X: features.data,
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
    fn setup_dataframe(
        &mut self,
        TrainingData {
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
        for name in self.extra_regressors.keys() {
            let col = x
                .get(name)
                .ok_or_else(|| Error::MissingRegressor(name.clone()))?;
            if col.len() != ds.len() {
                return Err(Error::MismatchedLengths {
                    a: y.len(),
                    a_name: "y".to_string(),
                    b: col.len(),
                    b_name: name.clone(),
                });
            }
            if col.iter().any(|x| x.is_nan()) {
                return Err(Error::InfiniteValue {
                    column: name.clone(),
                });
            }
        }
        for Seasonality { condition_name, .. } in self.seasonalities.values() {
            if let Some(condition_name) = condition_name {
                let col = seasonality_conditions
                    .get(condition_name)
                    .ok_or_else(|| Error::MissingSeasonalityCondition(condition_name.clone()))?;
                let column_length = col.len();
                if column_length != ds.len() {
                    return Err(Error::MismatchedLengths {
                        a: ds.len(),
                        a_name: "ds".to_string(),
                        b: column_length,
                        b_name: condition_name.clone(),
                    });
                }
            }
        }

        // Sort everything by date.
        let mut sort_indices = (0..ds.len()).collect_vec();
        sort_indices.sort_unstable_by_key(|i| ds[*i]);
        ds.sort_unstable();
        y = sort_indices.iter().map(|i| y[*i]).collect();
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

        for (name, regressor) in &self.extra_regressors {
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
            y,
            y_scaled,
            cap,
            cap_scaled,
            floor,
            extra_regressors: x,
            seasonality_conditions,
        };
        Ok((data, scales))
    }

    fn initialize_scales(
        &mut self,
        ds: &[TimestampSeconds],
        y: &[f64],
        extra_regressors: &HashMap<String, Vec<f64>>,
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
                            return Err(Error::Scaling);
                        };
                        scales.y_min = y_min;
                        scales.y_scale = y_scale;
                    }
                    (Scaling::MinMax, Some(cap)) => {
                        scales.y_min = *floor
                            .iter()
                            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                            .ok_or(Error::Scaling)?;
                        scales.y_scale = cap
                            .iter()
                            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                            .ok_or(Error::Scaling)?
                            - scales.y_min;
                    }
                    _ => {
                        return Err(Error::MissingCap);
                    }
                }
            }
            Some(_) | None => match self.opts.scaling {
                Scaling::AbsMax => {
                    scales.y_min = 0.0;
                    scales.y_scale = y
                        .iter()
                        .map(|y| y.abs())
                        .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                        .ok_or(Error::Scaling)?;
                }
                Scaling::MinMax => {
                    scales.y_min = *y
                        .iter()
                        .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                        .ok_or(Error::Scaling)?;
                    scales.y_scale = *y
                        .iter()
                        .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                        .ok_or(Error::Scaling)?
                        - scales.y_min;
                }
            },
        };
        if scales.y_scale == 0.0 {
            scales.y_scale = 1.0;
        }

        scales.start = *ds.first().ok_or(Error::NotEnoughData)?;
        scales.t_scale = (*ds.last().ok_or(Error::NotEnoughData)? - scales.start) as f64;

        for (name, regressor) in self.extra_regressors.iter_mut() {
            // Standardize if requested.
            let col = extra_regressors
                .get(name)
                .ok_or(Error::MissingRegressor(name.clone()))?;
            let mut standardize = regressor.standardize;
            // If there are 2 or fewer unique values, don't standardize.
            let mut vals = col.to_vec();
            vals.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            vals.dedup();
            if vals.len() < 2 {
                continue;
            }
            if standardize == Standardize::Auto {
                if vals.len() == 2 && vals[0] == 0.0 && vals[1] == 1.0 {
                    standardize = Standardize::No;
                } else {
                    standardize = Standardize::Yes;
                }
            }
            if standardize == Standardize::Yes {
                let mean = vals.iter().sum::<f64>() / vals.len() as f64;
                let std = vals.iter().map(|x| (x - mean).powi(2)).sum::<f64>().sqrt();
                regressor.mu = mean;
                regressor.std = std;
            }
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
        const ONE_YEAR_IN_SECONDS: f64 = 365.25 * 24.0 * 60.0 * 60.0;
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

        const ONE_WEEK_IN_SECONDS: f64 = 7.0 * 24.0 * 60.0 * 60.0;
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

        const ONE_DAY_IN_SECONDS: f64 = 24.0 * 60.0 * 60.0;
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

    fn fourier_series(
        dates: &[TimestampSeconds],
        period: PositiveFloat,
        order: NonZeroU32,
    ) -> Vec<Vec<f64>> {
        // Convert seconds to days.
        let t = dates.iter().copied().map(|ds| ds as f64 / 3600.0 / 24.0);
        // Convert to radians.
        let x_t = t.map(|x| x * std::f64::consts::PI * 2.0).collect_vec();
        // Preallocate space for the fourier components.
        let mut fourier_components = Vec::with_capacity(2 * order.get() as usize);
        // Calculate the fourier components.
        for i in 0..order.get() {
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
        let names = (0..n).map(|i| FeatureName::Seasonality(prefix.to_string(), i + 1));
        features.extend(names, new_features.into_iter());
        Ok(n)
    }

    fn construct_holidays(&self, ds: &[u64]) -> Result<HashMap<String, Holiday>, Error> {
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
            all_holidays.extend(holidays_to_add.into_iter().map(|name| {
                (
                    name.clone(),
                    Holiday {
                        ds: Vec::new(),
                        lower_window: None,
                        upper_window: None,
                        prior_scale: None,
                    },
                )
            }));
        }
        Ok(all_holidays)
    }

    fn make_holiday_features(&self, modes: &mut Modes) {
        todo!()
    }

    /// Make all features for the model.
    // This is called `make_all_seasonality_features` in the Python
    // implementation but it includes holidays and regressors too so
    // it's been renamed here for clarity.
    fn make_all_features(&self, history: &ProcessedData) -> Result<Features, Error> {
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
                name.clone(),
            )
        }

        // TODO: Add holiday features.
        // let holidays = self.construct_holidays(&history.ds)?;
        // if holidays.len() > 0 {
        //     self.make_holiday_features(
        //         &history.ds,
        //         &holidays,
        //         &mut features,
        //         &mut prior_scales,
        //         &mut modes,
        //     );
        // }

        // Add regressors.
        for (name, regressor) in &self.extra_regressors {
            let col = history
                .extra_regressors
                .get(name)
                .ok_or_else(|| Error::MissingRegressor(name.clone()))?;
            features.push(FeatureName::Regressor(name.clone()), col.clone());
            prior_scales.push(
                regressor
                    .prior_scale
                    .unwrap_or(self.opts.seasonality_prior_scale),
            );
            modes.insert(regressor.mode, name.clone());
        }

        // If there are no features, add a dummy column to prevent an empty features matrix.
        if features.is_empty() {
            features.push(FeatureName::Dummy, vec![0.0; history.ds.len()]);
            prior_scales.push(PositiveFloat::one());
        }

        let component_columns = self.regressor_column_matrix(&features.names, &mut modes);
        Ok(Features {
            features,
            prior_scales,
            component_columns,
            modes,
        })
    }

    /// Compute a matrix indicating which columns of the features matrix correspond
    /// to which seasonality/regressor components.
    fn regressor_column_matrix(
        &self,
        feature_names: &[FeatureName],
        modes: &mut Modes,
    ) -> ComponentColumns {
        // TODO: get rid of strings below, we can use a `ComponentName` enum instead.

        // Start with a vec of (col idx, component name) pairs.
        let mut components = feature_names
            .iter()
            .filter_map(|x| match x {
                FeatureName::Seasonality(name, _) => Some(name.clone()),
                FeatureName::Regressor(name) => Some(name.clone()),
                _ => None,
            })
            .enumerate()
            .collect();

        // Add total for holidays.
        if let Some(names) = &self.train_holiday_names {
            Self::add_group_component(&mut components, "holidays", names);
        }

        // Add additive and multiplicative components, and regressors.
        let (additive_regressors, multiplicative_regressors) =
            self.extra_regressors.iter().partition_map(|(name, reg)| {
                if reg.mode == FeatureMode::Additive {
                    Either::Left(name.clone())
                } else {
                    Either::Right(name.clone())
                }
            });
        Self::add_group_component(&mut components, "additive_terms", &modes.additive);
        Self::add_group_component(
            &mut components,
            "extra_regressors_additive",
            &additive_regressors,
        );
        Self::add_group_component(
            &mut components,
            "multiplicative_terms",
            &modes.multiplicative,
        );
        Self::add_group_component(
            &mut components,
            "extra_regressors_multiplicative",
            &multiplicative_regressors,
        );
        // Add the names of the group components to the modes.
        modes.additive.insert("additive_terms".to_string());
        modes
            .additive
            .insert("extra_regressors_additive".to_string());
        modes
            .multiplicative
            .insert("multiplicative_terms".to_string());
        modes
            .multiplicative
            .insert("extra_regressors_multiplicative".to_string());

        // Add holidays.
        modes.insert(self.opts.holidays_mode, "holidays".to_string());

        ComponentColumns::new(&components)
    }

    /// Add a component with the given name that contains all of the components
    /// in `group`.
    fn add_group_component(
        components: &mut Vec<(usize, String)>,
        name: &str,
        names: &HashSet<String>,
    ) {
        let group_cols = components
            .iter()
            .filter_map(|(i, n)| names.contains(n).then_some(*i))
            .dedup()
            .collect_vec();
        components.extend(group_cols.into_iter().map(|i| (i, name.to_string())));
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
            let hist_size = (ds.len() as f64 * self.opts.changepoint_range).floor() as usize;
            let mut n_changepoints = self.opts.n_changepoints as usize;
            if n_changepoints + 1 > hist_size {
                n_changepoints = hist_size - 1;
            }
            if n_changepoints > 0 {
                // Place changepoints evenly through the first `changepoint_range` percent of the history.
                let cp_indices = (0..(hist_size - 1)).step_by(n_changepoints + 1);
                cp_indices.map(|i| ds[i]).collect()
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

    /// Return `true` if the model has been fit, or `false` if not.
    pub fn is_fitted(&self) -> bool {
        self.optimized.is_some()
    }

    /// Fit the Prophet model to some training data.
    #[instrument(level = "debug", skip(self, data, opts))]
    pub fn fit(&mut self, data: TrainingData, opts: OptimizeOpts) -> Result<(), Error> {
        let preprocessed = self.preprocess(data)?;
        let init = preprocessed.calculate_initial_params(&self.opts)?;
        self.optimized = Some(
            self.optimizer
                .optimize(init.clone(), preprocessed.data.clone(), opts)
                .map_err(|e| Error::OptimizationFailed(e.to_string()))?,
        );
        self.init = Some(init);
        Ok(())
    }

    /// Predict using the Prophet model.
    ///
    /// That will fail if the model has not been fit.
    #[instrument(level = "debug", skip(self, data))]
    pub fn predict(&self, data: Option<PredictionData>) -> Result<Vec<f64>, Error> {
        // TODO!
        Err(Error::Notimplemented)
    }
}

/// Historical data after preprocessing.
struct ProcessedData {
    ds: Vec<TimestampSeconds>,
    t: Vec<f64>,
    y: Vec<f64>,
    y_scaled: Vec<f64>,
    cap: Option<Vec<f64>>,
    cap_scaled: Option<Vec<f64>>,
    floor: Vec<f64>,
    extra_regressors: HashMap<String, Vec<f64>>,
    seasonality_conditions: HashMap<String, Vec<bool>>,
}

/// Processed data used for fitting.
struct Preprocessed {
    data: Data,
    history: ProcessedData,
    history_dates: Vec<TimestampSeconds>,
}

impl Preprocessed {
    /// Calculate the initial parameters for the Stan optimization.
    fn calculate_initial_params(&self, opts: &ProphetOptions) -> Result<InitialParams, Error> {
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
            beta: vec![0.0; self.data.X.len()],
            sigma_obs: 1.0,
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
mod tests {
    use crate::{
        optimizer::dummy_optimizer::DummyOptimizer, testdata::daily_univariate_ts, Standardize,
    };

    use super::*;
    use augurs_testing::assert_approx_eq;
    use pretty_assertions::assert_eq;

    #[test]
    fn test_growth_init() {
        let mut data = daily_univariate_ts().head(468);
        let max = *data
            .y
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        data = data.with_cap(vec![max; 468]);

        let mut opts = ProphetOptions::default();
        let mut prophet = Prophet::new(opts.clone(), &DummyOptimizer);
        let preprocessed = prophet.preprocess(data.clone()).unwrap();
        let init = preprocessed.calculate_initial_params(&opts).unwrap();
        assert_approx_eq!(init.k, 0.3055671);
        assert_approx_eq!(init.m, 0.5307511);

        opts.growth = GrowthType::Logistic;
        let mut prophet = Prophet::new(opts.clone(), &DummyOptimizer);
        let preprocessed = prophet.preprocess(data).unwrap();
        let init = preprocessed.calculate_initial_params(&opts).unwrap();
        assert_approx_eq!(init.k, 1.507925);
        assert_approx_eq!(init.m, -0.08167497);

        opts.growth = GrowthType::Flat;
        let init = preprocessed.calculate_initial_params(&opts).unwrap();
        assert_approx_eq!(init.k, 0.0);
        assert_approx_eq!(init.m, 0.49335657);
    }

    #[test]
    fn test_growth_init_minmax() {
        let mut data = daily_univariate_ts().head(468);
        let max = *data
            .y
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        data = data.with_cap(vec![max; 468]);

        let mut opts = ProphetOptions {
            scaling: Scaling::MinMax,
            ..ProphetOptions::default()
        };
        let mut prophet = Prophet::new(opts.clone(), &DummyOptimizer);
        let preprocessed = prophet.preprocess(data.clone()).unwrap();
        let init = preprocessed.calculate_initial_params(&opts).unwrap();
        assert_approx_eq!(init.k, 0.4053406);
        assert_approx_eq!(init.m, 0.3775322);

        opts.growth = GrowthType::Logistic;
        let mut prophet = Prophet::new(opts.clone(), &DummyOptimizer);
        let preprocessed = prophet.preprocess(data).unwrap();
        let init = preprocessed.calculate_initial_params(&opts).unwrap();
        assert_approx_eq!(init.k, 1.782523);
        assert_approx_eq!(init.m, 0.280521);

        opts.growth = GrowthType::Flat;
        let init = preprocessed.calculate_initial_params(&opts).unwrap();
        assert_approx_eq!(init.k, 0.0);
        assert_approx_eq!(init.m, 0.32792770);
    }

    #[test]
    fn regressor_column_matrix() {
        // TODO: add holidays back in and update assertions.
        let opts = ProphetOptions::default();
        let mut prophet = Prophet::new(opts, &DummyOptimizer);
        prophet.add_regressor(
            "binary_feature".to_string(),
            Regressor::additive().with_prior_scale(0.2.try_into().unwrap()),
        );
        prophet.add_regressor(
            "numeric_feature".to_string(),
            Regressor::additive().with_prior_scale(0.5.try_into().unwrap()),
        );
        prophet.add_regressor(
            "numeric_feature2".to_string(),
            Regressor::multiplicative().with_prior_scale(0.5.try_into().unwrap()),
        );
        prophet.add_regressor(
            "binary_feature2".to_string(),
            Regressor::additive().with_standardize(Standardize::Yes),
        );
        let mut modes = Modes {
            additive: HashSet::from([
                "weekly".to_string(),
                "binary_feature".to_string(),
                "numeric_feature".to_string(),
                "binary_feature2".to_string(),
            ]),
            multiplicative: HashSet::from(["numeric_feature2".to_string()]),
        };
        let cols = prophet.regressor_column_matrix(
            &[
                FeatureName::Seasonality("weekly".to_string(), 1),
                FeatureName::Seasonality("weekly".to_string(), 2),
                FeatureName::Seasonality("weekly".to_string(), 3),
                FeatureName::Seasonality("weekly".to_string(), 4),
                FeatureName::Seasonality("weekly".to_string(), 5),
                FeatureName::Seasonality("weekly".to_string(), 6),
                FeatureName::Regressor("binary_feature".to_string()),
                FeatureName::Regressor("numeric_feature".to_string()),
                FeatureName::Regressor("numeric_feature2".to_string()),
                FeatureName::Regressor("binary_feature2".to_string()),
            ],
            &mut modes,
        );
        assert_eq!(cols.additive, vec![1, 1, 1, 1, 1, 1, 1, 1, 0, 1]);
        assert_eq!(cols.multiplicative, vec![0, 0, 0, 0, 0, 0, 0, 0, 1, 0]);
        assert_eq!(cols.holidays, vec![0; 10]);
        assert_eq!(
            cols.extra_regressors_additive,
            vec![0, 0, 0, 0, 0, 0, 1, 1, 0, 1]
        );
        assert_eq!(
            cols.extra_regressors_multiplicative,
            vec![0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
        );
        assert_eq!(cols.custom.len(), 5);
        assert_eq!(cols.custom["weekly"], &[1, 1, 1, 1, 1, 1, 0, 0, 0, 0]);
        assert_eq!(
            cols.custom["binary_feature"],
            &[0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
        );
        assert_eq!(
            cols.custom["numeric_feature"],
            &[0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
        );
        assert_eq!(
            cols.custom["numeric_feature2"],
            &[0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
        );
        assert_eq!(
            cols.custom["binary_feature2"],
            &[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        );
        assert_eq!(
            modes,
            Modes {
                additive: HashSet::from([
                    "weekly".to_string(),
                    "binary_feature".to_string(),
                    "numeric_feature".to_string(),
                    "binary_feature2".to_string(),
                    "holidays".to_string(),
                    "extra_regressors_additive".to_string(),
                    "additive_terms".to_string(),
                ]),
                multiplicative: HashSet::from([
                    "numeric_feature2".to_string(),
                    "extra_regressors_multiplicative".to_string(),
                    "multiplicative_terms".to_string(),
                ]),
            }
        );
    }

    #[test]
    fn test_add_group_component() {
        let mut components = vec![
            (0, "weekly".to_string()),
            (1, "weekly".to_string()),
            (2, "weekly".to_string()),
            (3, "weekly".to_string()),
            (4, "weekly".to_string()),
            (5, "weekly".to_string()),
            (6, "birthday".to_string()),
            (7, "birthday".to_string()),
        ];
        let names = HashSet::from(["birthday".to_string()]);
        Prophet::add_group_component(&mut components, "holidays", &names);
        assert_eq!(
            components,
            vec![
                (0, "weekly".to_string()),
                (1, "weekly".to_string()),
                (2, "weekly".to_string()),
                (3, "weekly".to_string()),
                (4, "weekly".to_string()),
                (5, "weekly".to_string()),
                (6, "birthday".to_string()),
                (7, "birthday".to_string()),
                (6, "holidays".to_string()),
                (7, "holidays".to_string()),
            ]
        );
    }

    #[test]
    fn test_component_columns() {
        let components = [
            (0, "weekly"),
            (1, "weekly"),
            (2, "weekly"),
            (3, "weekly"),
            (4, "weekly"),
            (5, "weekly"),
            (6, "birthday"),
            (7, "birthday"),
            (8, "binary_feature"),
            (9, "numeric_feature"),
            (10, "numeric_feature2"),
            (11, "binary_feature2"),
            (6, "holidays"),
            (7, "holidays"),
            (0, "additive_terms"),
            (1, "additive_terms"),
            (2, "additive_terms"),
            (3, "additive_terms"),
            (4, "additive_terms"),
            (5, "additive_terms"),
            (8, "additive_terms"),
            (9, "additive_terms"),
            (11, "additive_terms"),
            (8, "extra_regressors_additive"),
            (9, "extra_regressors_additive"),
            (11, "extra_regressors_additive"),
            (6, "multiplicative_terms"),
            (7, "multiplicative_terms"),
            (10, "multiplicative_terms"),
            (10, "extra_regressors_multiplicative"),
        ]
        .map(|(i, name)| (i, name.to_string()));
        let cols = ComponentColumns::new(&components);
        assert_eq!(cols.additive, vec![1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1]);
        assert_eq!(
            cols.multiplicative,
            vec![0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0]
        );
        assert_eq!(cols.holidays, vec![0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]);
        assert_eq!(
            cols.extra_regressors_additive,
            vec![0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1]
        );
        assert_eq!(
            cols.extra_regressors_multiplicative,
            vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
        );
        assert_eq!(cols.custom.len(), 6);
        assert_eq!(cols.custom["weekly"], &[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]);
        assert_eq!(
            cols.custom["birthday"],
            &[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
        );
        assert_eq!(
            cols.custom["binary_feature"],
            &[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
        );
        assert_eq!(
            cols.custom["numeric_feature"],
            &[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
        );
        assert_eq!(
            cols.custom["numeric_feature2"],
            &[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
        );
        assert_eq!(
            cols.custom["binary_feature2"],
            &[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        );
    }
}
