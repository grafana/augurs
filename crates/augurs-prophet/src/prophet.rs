pub(crate) mod options;

use std::{
    collections::{HashMap, HashSet},
    num::NonZeroU32,
};

use itertools::{izip, Either, Itertools, MinMaxResult};
use options::{GrowthType, ProphetOptions, Scaling, SeasonalityOption};
use rand::{distributions::Uniform, thread_rng, Rng};
use tracing::instrument;

use crate::{
    distributions::{Laplace, Normal, Poisson},
    features::RegressorScale,
    optimizer::{Data, InitialParams, OptimizeOpts, OptimizedParams, Optimizer},
    Error, FeatureMode, FloatIterExt, Holiday, PositiveFloat, PredictionData, Regressor,
    Seasonality, Standardize, TimestampSeconds, TrainingData,
};

const NO_REGRESSORS_PLACEHOLDER: &str = "__no_regressors_zeros__";

#[derive(Debug, Clone, Default)]
struct Scales {
    logistic_floor: bool,
    y_min: f64,
    y_scale: f64,
    start: TimestampSeconds,
    t_scale: f64,
    regressors: HashMap<String, RegressorScale>,
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
    regressors_additive: Vec<i32>,
    regressors_multiplicative: Vec<i32>,
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
            regressors_additive: vec![0; n_columns],
            regressors_multiplicative: vec![0; n_columns],
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
            } else if name == "regressors_additive" {
                cols.regressors_additive[i] = 1;
            } else if name == "regressors_multiplicative" {
                cols.regressors_multiplicative[i] = 1;
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

/// The prediction for a feature.
///
/// 'Feature' could refer to the forecasts themselves (`yhat`)
/// or any of the other component features which contribute to
/// the final estimate, such as trend, seasonality, seasonalities,
/// regressors or holidays.
#[derive(Debug, Default, Clone)]
pub struct FeaturePrediction {
    /// The point estimate for this feature.
    pub point: Vec<f64>,
    /// The lower estimate for this feature.
    ///
    /// Only present if `uncertainty_samples` was greater than zero
    /// when the model was created.
    pub lower: Option<Vec<f64>>,
    /// The upper estimate for this feature.
    ///
    /// Only present if `uncertainty_samples` was greater than zero
    /// when the model was created.
    pub upper: Option<Vec<f64>>,
}

#[derive(Debug, Default)]
struct FeaturePredictions {
    /// Contribution of the additive terms in the model.
    ///
    /// This includes additive seasonalities, holidays and regressors.
    additive: FeaturePrediction,
    /// Contribution of the multiplicative terms in the model.
    ///
    /// This includes multiplicative seasonalities, holidays and regressors.
    multiplicative: FeaturePrediction,
    /// Mapping from holiday name to the contribution of that holiday.
    holidays: HashMap<String, FeaturePrediction>,
    /// Mapping from regressor name to the contribution of that regressor.
    regressors: HashMap<String, FeaturePrediction>,
    /// Mapping from seasonality name to the contribution of that seasonality.
    seasonalities: HashMap<String, FeaturePrediction>,
}

/// Predictions from a Prophet model.
///
/// The `yhat` field contains the forecasts for the input time series.
/// All other fields contain individual components of the model which
/// contribute towards the final `yhat` estimate.
///
/// Certain fields (such as `cap` and `floor`) may be `None` if the
/// model did not use them (e.g. the model was not configured to use
/// logistic trend).
#[derive(Debug, Clone)]
pub struct Predictions {
    /// Forecasts of the input time series `y`.
    pub yhat: FeaturePrediction,

    /// The trend contribution at each time point.
    pub trend: FeaturePrediction,

    /// The cap for the logistic growth.
    ///
    /// Will only be `Some` if the model used [`GrowthType::Logistic`].
    pub cap: Option<Vec<f64>>,
    /// The floor for the logistic growth.
    ///
    /// Will only be `Some` if the model used [`GrowthType::Logistic`]
    /// and the floor was provided in the input data.
    pub floor: Option<Vec<f64>>,

    /// The combined combination of all _additive_ components.
    ///
    /// This includes seasonalities, holidays and regressors if their mode
    /// was configured to be [`FeatureMode::Additive`].
    pub additive: FeaturePrediction,

    /// The combined combination of all _multiplicative_ components.
    ///
    /// This includes seasonalities, holidays and regressors if their mode
    /// was configured to be [`FeatureMode::Multiplicative`].
    pub multiplicative: FeaturePrediction,

    /// Mapping from holiday name to that holiday's contribution.
    pub holidays: HashMap<String, FeaturePrediction>,

    /// Mapping from seasonality name to that seasonality's contribution.
    pub seasonalities: HashMap<String, FeaturePrediction>,

    /// Mapping from regressor name to that regressor's contribution.
    pub regressors: HashMap<String, FeaturePrediction>,
}

/// The Prophet time series forecasting model.
#[derive(Debug)]
pub struct Prophet {
    /// Options to be used for fitting.
    opts: ProphetOptions,

    /// Extra regressors.
    regressors: HashMap<String, Regressor>,

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
    optimizer: Box<dyn Optimizer>,

    /// The processed data used for fitting.
    processed: Option<Preprocessed>,

    /// The initial parameters passed to optimization.
    init: Option<InitialParams>,

    /// The optimized model, if it has been fit.
    optimized: Option<OptimizedParams>,
}

impl Prophet {
    /// Create a new Prophet model with the given options and optimizer.
    pub fn new<T: Optimizer + 'static>(opts: ProphetOptions, optimizer: T) -> Self {
        Self {
            opts,
            regressors: HashMap::new(),
            seasonalities: HashMap::new(),
            scales: None,
            changepoints: None,
            changepoints_t: None,
            component_modes: None,
            train_component_columns: None,
            train_holiday_names: None,
            optimizer: Box::new(optimizer),
            processed: None,
            init: None,
            optimized: None,
        }
    }

    /// Add a custom seasonality to the model.
    pub fn add_seasonality(&mut self, name: String, seasonality: Seasonality) -> Result<(), Error> {
        // TODO: validate name
        self.seasonalities.insert(name, seasonality);
        Ok(())
    }

    /// Add a regressor to the model.
    pub fn add_regressor(&mut self, name: String, regressor: Regressor) {
        self.regressors.insert(name, regressor);
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
            K: features.names.len() as i32,
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
                if !x.contains_key(condition_name) {
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
                            return Err(Error::Scaling);
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
            if vals.len() < 2 {
                continue;
            }

            let mut regressor_scale = RegressorScale::default();
            if regressor.standardize == Standardize::Auto {
                regressor_scale.standardize =
                    !(vals.len() == 2 && vals[0] == 0.0 && vals[1] == 1.0);
            }
            if regressor_scale.standardize {
                let mean = vals.iter().sum::<f64>() / vals.len() as f64;
                let std = vals.iter().map(|x| (x - mean).powi(2)).sum::<f64>().sqrt();
                regressor_scale.mu = mean;
                regressor_scale.std = std;
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

    /// Compute fourier series components with the specified period
    /// and order.
    ///
    /// Note: this computes the transpose of the function in the Python
    /// code for simplicity, since we need it in a columnar format anyway.
    fn fourier_series(
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

    fn construct_holidays(&self, _ds: &[u64]) -> Result<HashMap<String, Holiday>, Error> {
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
        _ds: &[TimestampSeconds],
        _holidays: HashMap<String, Holiday>,
        _features: &mut FeaturesFrame,
        _prior_scales: &mut [PositiveFloat],
        _modes: &mut Modes,
    ) {
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
        let holidays = self.construct_holidays(&history.ds)?;
        if !holidays.is_empty() {
            self.make_holiday_features(
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
                FeatureName::Seasonality { name, _id: _ } => Some(name.clone()),
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
            self.regressors.iter().partition_map(|(name, reg)| {
                if reg.mode == FeatureMode::Additive {
                    Either::Left(name.clone())
                } else {
                    Either::Right(name.clone())
                }
            });
        Self::add_group_component(&mut components, "additive_terms", &modes.additive);
        Self::add_group_component(&mut components, "regressors_additive", &additive_regressors);
        Self::add_group_component(
            &mut components,
            "multiplicative_terms",
            &modes.multiplicative,
        );
        Self::add_group_component(
            &mut components,
            "regressors_multiplicative",
            &multiplicative_regressors,
        );
        // Add the names of the group components to the modes.
        modes.additive.insert("additive_terms".to_string());
        modes.additive.insert("regressors_additive".to_string());
        modes
            .multiplicative
            .insert("multiplicative_terms".to_string());
        modes
            .multiplicative
            .insert("regressors_multiplicative".to_string());

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
        self.processed = Some(preprocessed);
        self.init = Some(init);
        Ok(())
    }

    /// Predict using the Prophet model.
    ///
    /// # Errors
    ///
    /// Returns an error if the model has not been fit.
    #[instrument(level = "debug", skip(self, data))]
    pub fn predict(&self, data: impl Into<Option<PredictionData>>) -> Result<Predictions, Error> {
        let Self {
            processed: Some(processed),
            optimized: Some(params),
            changepoints_t: Some(changepoints_t),
            scales: Some(scales),
            ..
        } = self
        else {
            return Err(Error::ModelNotFit);
        };
        let data = data.into();
        let df = data
            .map(|data| {
                let training_data = TrainingData {
                    n: data.n,
                    ds: data.ds.clone(),
                    y: vec![],
                    cap: data.cap.clone(),
                    floor: data.floor.clone(),
                    seasonality_conditions: data.seasonality_conditions.clone(),
                    x: data.x.clone(),
                };
                self.setup_dataframe(training_data, Some(scales.clone()))
                    .map(|(df, _)| df)
            })
            .transpose()?
            .unwrap_or_else(|| processed.history.clone());

        let mut trend = self.predict_trend(
            &df.t,
            &df.cap,
            &df.floor,
            changepoints_t,
            params,
            scales.y_scale,
        )?;
        let features = self.make_all_features(&df)?;
        let seasonal_components = self.predict_features(&features, params, scales.y_scale)?;

        let yhat_point = izip!(
            &trend.point,
            &seasonal_components.additive.point,
            &seasonal_components.multiplicative.point
        )
        .map(|(t, a, m)| t * (1.0 + m) + a)
        .collect();
        let mut yhat = FeaturePrediction {
            point: yhat_point,
            lower: None,
            upper: None,
        };

        if self.opts.uncertainty_samples > 0 {
            self.predict_uncertainty(
                &df,
                &features,
                params,
                changepoints_t,
                &mut yhat,
                &mut trend,
                scales.y_scale,
            )?;
        }

        Ok(Predictions {
            yhat,
            trend,
            cap: df.cap,
            floor: scales.logistic_floor.then_some(df.floor),
            additive: seasonal_components.additive,
            multiplicative: seasonal_components.multiplicative,
            holidays: seasonal_components.holidays,
            seasonalities: seasonal_components.seasonalities,
            regressors: seasonal_components.regressors,
        })
    }

    fn piecewise_linear<'a>(
        t: &'a [f64],
        deltas: &'a [f64],
        k: f64,
        m: f64,
        changepoints_t: &'a [f64],
    ) -> impl Iterator<Item = f64> + 'a {
        // `deltas_t` is a contiguous array with the changepoint delta to apply
        // delta at each time point; it has a stride of `changepoints_t.len()`,
        // since it's a 2D array in the numpy version.
        let cp_zipped = deltas.iter().zip(changepoints_t);
        let deltas_t = cp_zipped
            .cartesian_product(t)
            .map(|((delta, cp_t), t)| if cp_t <= t { *delta } else { 0.0 })
            .collect_vec();
        // `k_t` is a contiguous array with the rate to apply at each time point.
        let k_t = deltas_t
            .iter()
            .enumerate()
            .fold(vec![k; t.len()], |mut acc, (i, delta)| {
                // Add the changepoint rate to the initial rate.
                acc[i % t.len()] += *delta;
                acc
            });
        // `m_t` is a contiguous array with the offset to apply at each time point.
        let m_t = deltas_t
            .iter()
            .zip(
                // Repeat each changepoint effect `n` times so we can zip it up.
                changepoints_t
                    .iter()
                    .flat_map(|x| std::iter::repeat(*x).take(t.len())),
            )
            .enumerate()
            .fold(vec![m; t.len()], |mut acc, (i, (delta, cp_t))| {
                // Add the changepoint offset to the initial offset where applicable.
                acc[i % t.len()] += -cp_t * delta;
                acc
            });

        izip!(t, k_t, m_t).map(|(t, k, m)| t * k + m)
    }

    fn piecewise_logistic<'a>(
        t: &'a [f64],
        cap: &'a [f64],
        deltas: &'a [f64],
        k: f64,
        m: f64,
        changepoints_t: &'a [f64],
    ) -> impl Iterator<Item = f64> + 'a {
        // Compute offset changes.
        let k_cum = std::iter::once(k)
            .chain(deltas.iter().scan(k, |state, delta| {
                *state += delta;
                Some(*state)
            }))
            .collect_vec();
        let mut gammas = vec![0.0; changepoints_t.len()];
        let mut gammas_sum = 0.0;
        for (i, t_s) in changepoints_t.iter().enumerate() {
            gammas[i] = (t_s - m - gammas_sum) * (1.0 - k_cum[i] / k_cum[i + 1]);
            gammas_sum += gammas[i];
        }

        // Get cumulative rate and offset at each time point.
        let mut k_t = vec![k; t.len()];
        let mut m_t = vec![m; t.len()];
        for (s, t_s) in changepoints_t.iter().enumerate() {
            for (i, t_i) in t.iter().enumerate() {
                if t_i >= t_s {
                    k_t[i] += deltas[s];
                    m_t[i] += gammas[s];
                }
            }
        }

        izip!(cap, t, k_t, m_t).map(|(cap, t, k, m)| cap / (1.0 + (-k * (t - m)).exp()))
    }

    /// Evaluate the flat trend function.
    fn flat_trend(t: &[f64], m: f64) -> impl Iterator<Item = f64> {
        std::iter::repeat(m).take(t.len())
    }

    /// Predict trend.
    fn predict_trend(
        &self,
        t: &[f64],
        cap: &Option<Vec<f64>>,
        floor: &[f64],
        changepoints_t: &[f64],
        params: &OptimizedParams,
        y_scale: f64,
    ) -> Result<FeaturePrediction, Error> {
        let point = match (self.opts.growth, cap) {
            (GrowthType::Linear, _) => {
                Prophet::piecewise_linear(t, &params.delta, params.k, params.m, changepoints_t)
                    .zip(floor)
                    .map(|(trend, flr)| trend * y_scale + flr)
                    .collect_vec()
            }
            (GrowthType::Logistic, Some(cap)) => Prophet::piecewise_logistic(
                t,
                cap,
                &params.delta,
                params.k,
                params.m,
                changepoints_t,
            )
            .zip(floor)
            .map(|(trend, flr)| trend * y_scale + flr)
            .collect_vec(),
            (GrowthType::Logistic, None) => return Err(Error::MissingCap),
            (GrowthType::Flat, _) => Prophet::flat_trend(t, params.m)
                .zip(floor)
                .map(|(trend, flr)| trend * y_scale + flr)
                .collect_vec(),
        };
        Ok(FeaturePrediction {
            point,
            lower: None,
            upper: None,
        })
    }

    /// Predict seasonality, holidays and added regressors.
    fn predict_features(
        &self,
        features: &Features,
        params: &OptimizedParams,
        y_scale: f64,
    ) -> Result<FeaturePredictions, Error> {
        let Features {
            features,
            component_columns,
            ..
        } = features;
        // TODO: do the rest of the terms
        Ok(FeaturePredictions {
            additive: Self::predict_feature(
                &component_columns.additive,
                &features.data,
                &params.beta,
                y_scale,
                true,
            ),
            multiplicative: Self::predict_feature(
                &component_columns.multiplicative,
                &features.data,
                &params.beta,
                y_scale,
                false,
            ),
            ..Default::default()
        })
    }

    fn predict_feature(
        component_col: &[i32],
        #[allow(non_snake_case)] X: &[Vec<f64>],
        beta: &[f64],
        y_scale: f64,
        is_additive: bool,
    ) -> FeaturePrediction {
        let beta_c = component_col
            .iter()
            .copied()
            .zip(beta)
            .map(|(x, b)| x as f64 * b)
            .collect_vec();
        // Matrix multiply `beta_c` and `x`.
        let mut point = vec![0.0; X[0].len()];
        for (p, feature, b) in izip!(point.iter_mut(), X, beta_c) {
            for x in feature {
                *p += b * x;
            }
        }
        if is_additive {
            point.iter_mut().for_each(|x| *x *= y_scale);
        }
        FeaturePrediction {
            point,
            lower: None,
            upper: None,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn predict_uncertainty(
        &self,
        df: &ProcessedData,
        features: &Features,
        params: &OptimizedParams,
        changepoints_t: &[f64],
        yhat: &mut FeaturePrediction,
        trend: &mut FeaturePrediction,
        y_scale: f64,
    ) -> Result<(), Error> {
        let mut sim_values =
            self.sample_posterior_predictive(df, features, params, changepoints_t, y_scale)?;
        let lower_p = 100.0 * (1.0 - self.opts.interval_width) / 2.0;
        let upper_p = 100.0 * (1.0 + self.opts.interval_width) / 2.0;

        let mut yhat_lower = Vec::with_capacity(df.ds.len());
        let mut yhat_upper = Vec::with_capacity(df.ds.len());
        let mut trend_lower = Vec::with_capacity(df.ds.len());
        let mut trend_upper = Vec::with_capacity(df.ds.len());

        for (yhat_samples, trend_samples) in
            sim_values.yhat.iter_mut().zip(sim_values.trend.iter_mut())
        {
            // Sort, since we need to find multiple percentiles.
            yhat_samples
                .sort_unstable_by(|a, b| a.partial_cmp(b).expect("found NaN in yhat sample"));
            trend_samples
                .sort_unstable_by(|a, b| a.partial_cmp(b).expect("found NaN in yhat sample"));
            yhat_lower.push(percentile_of_sorted(yhat_samples, lower_p));
            yhat_upper.push(percentile_of_sorted(yhat_samples, upper_p));
            trend_lower.push(percentile_of_sorted(trend_samples, lower_p));
            trend_upper.push(percentile_of_sorted(trend_samples, upper_p));
        }
        yhat.lower = Some(yhat_lower);
        yhat.upper = Some(yhat_upper);
        trend.lower = Some(trend_lower);
        trend.upper = Some(trend_upper);
        Ok(())
    }

    /// Sample posterior predictive values from the model.
    fn sample_posterior_predictive(
        &self,
        df: &ProcessedData,
        features: &Features,
        params: &OptimizedParams,
        changepoints_t: &[f64],
        y_scale: f64,
    ) -> Result<PosteriorPredictiveSamples, Error> {
        // TODO: handle multiple chains.
        let n_iterations = 1;
        let samples_per_iter = usize::max(
            1,
            (self.opts.uncertainty_samples as f64 / n_iterations as f64).ceil() as usize,
        );
        let Features {
            features,
            component_columns,
            ..
        } = features;
        // We're going to generate `samples_per_iter * n_iterations` samples
        // for each of the `n` timestamps we want to predict.
        // We'll store these in a nested `Vec<Vec<f64>>`, where the outer
        // vector is indexed by the timestamps and the inner vector is
        // indexed by the samples, since we need to calculate the `p` percentile
        // of the samples for each timestamp.
        let n_timestamps = df.ds.len();
        let n_samples = samples_per_iter * n_iterations;
        let mut sim_values = PosteriorPredictiveSamples {
            yhat: std::iter::repeat_with(|| Vec::with_capacity(n_samples))
                .take(n_timestamps)
                .collect_vec(),
            trend: std::iter::repeat_with(|| Vec::with_capacity(n_samples))
                .take(n_timestamps)
                .collect_vec(),
        };
        for i in 0..n_iterations {
            for _ in 0..samples_per_iter {
                let (yhat, trend) = self.sample_model(
                    df,
                    features,
                    params,
                    changepoints_t,
                    &component_columns.additive,
                    &component_columns.multiplicative,
                    y_scale,
                    i,
                )?;
                // We have to transpose things, unfortunately.
                for ((i, yhat), trend) in yhat.into_iter().enumerate().zip(trend) {
                    sim_values.yhat[i].push(yhat);
                    sim_values.trend[i].push(trend);
                }
            }
        }
        debug_assert_eq!(sim_values.yhat.len(), n_timestamps);
        debug_assert_eq!(sim_values.trend.len(), n_timestamps);
        Ok(sim_values)
    }

    /// Simulate observations from the extrapolated model.
    #[allow(clippy::too_many_arguments)]
    fn sample_model(
        &self,
        df: &ProcessedData,
        features: &FeaturesFrame,
        params: &OptimizedParams,
        changepoints_t: &[f64],
        additive: &[i32],
        multiplicative: &[i32],
        y_scale: f64,
        iteration: usize,
    ) -> Result<(Vec<f64>, Vec<f64>), Error> {
        let n = df.ds.len();
        let trend = self.sample_predictive_trend(df, params, changepoints_t, y_scale, iteration)?;
        let beta = &params.beta;
        let mut xb_a = vec![0.0; n];
        for (feature, b, a) in izip!(&features.data, beta, additive) {
            for (p, x) in izip!(&mut xb_a, feature) {
                *p += x * b * *a as f64;
            }
        }
        xb_a.iter_mut().for_each(|x| *x *= y_scale);
        let mut xb_m = vec![0.0; n];
        for (feature, b, m) in izip!(&features.data, beta, multiplicative) {
            for (p, x) in izip!(&mut xb_m, feature) {
                *p += x * b * *m as f64;
            }
        }

        let sigma = params.sigma_obs;
        let dist = Normal::new(0.0, sigma).expect("sigma should be non-negative");
        let mut rng = thread_rng();
        let noise = (&mut rng).sample_iter(dist).take(n).map(|x| x * y_scale);

        let yhat = izip!(&trend, &xb_a, &xb_m, noise)
            .map(|(t, a, m, n)| t * (1.0 + m) + a + n)
            .collect();

        Ok((yhat, trend))
    }

    fn sample_predictive_trend(
        &self,
        df: &ProcessedData,
        params: &OptimizedParams,
        changepoints_t: &[f64],
        y_scale: f64,
        _iteration: usize, // This will be used when we implement MCMC predictions.
    ) -> Result<Vec<f64>, Error> {
        let deltas = &params.delta;

        let t_max = df.t.iter().copied().nanmax(true);

        let mut rng = thread_rng();

        let n_changes = if t_max > 1.0 {
            // Sample new changepoints from a Poisson process with rate n_cp on [1, T].
            let n_cp = changepoints_t.len() as i32;
            let lambda = n_cp as f64 * (t_max - 1.0);
            // Lambda should always be positive, so this should never fail.
            let dist = Poisson::new(lambda).expect("Valid Poisson distribution");
            rng.sample(dist).round() as usize
        } else {
            0
        };
        let changepoints_t_new = if n_changes > 0 {
            let mut cp_t_new = (&mut rng)
                .sample_iter(Uniform::new(0.0, t_max - 1.0))
                .take(n_changes)
                .map(|x| x + 1.0)
                .collect_vec();
            cp_t_new.sort_unstable_by(|a, b| {
                a.partial_cmp(b)
                    .expect("uniform distribution should not sample NaNs")
            });
            cp_t_new
        } else {
            vec![]
        };

        // Get the empirical scale of the deltas, plus epsilon to avoid NaNs.
        let mut lambda = deltas.iter().map(|x| x.abs()).nanmean(false) + 1e-8;
        if lambda.is_nan() {
            lambda = 1e-8;
        }
        // Sample deltas from a Laplace distribution with location 0 and scale lambda.
        // Lambda should always be positive and non-NaN, checked above.
        let dist = Laplace::new(0.0, lambda).expect("Valid Laplace distribution");
        let deltas_new = rng.sample_iter(dist).take(n_changes);

        // Prepend the times and deltas from the history.
        let all_changepoints_t = changepoints_t
            .iter()
            .copied()
            .chain(changepoints_t_new)
            .collect_vec();
        let all_deltas = deltas.iter().copied().chain(deltas_new).collect_vec();

        // Predict the trend.
        let new_params = OptimizedParams {
            delta: all_deltas,
            ..params.clone()
        };
        let trend = self.predict_trend(
            &df.t,
            &df.cap_scaled,
            &df.floor,
            &all_changepoints_t,
            &new_params,
            y_scale,
        )?;
        Ok(trend.point)
    }

    /// Create dates to use for predictions.
    ///
    /// # Parameters
    ///
    /// - `horizon`: The number of days to predict forward.
    /// - `include_history`: Whether to include the historical dates in the
    ///   future dataframe.
    ///
    /// # Errors
    ///
    /// Returns an error if the model has not been fit.
    pub fn make_future_dataframe(
        &self,
        horizon: NonZeroU32,
        include_history: IncludeHistory,
    ) -> Result<PredictionData, Error> {
        let Some(Preprocessed { history_dates, .. }) = &self.processed else {
            return Err(Error::ModelNotFit);
        };
        let freq = Self::infer_freq(history_dates)?;
        let last_date = *history_dates.last().ok_or(Error::NotEnoughData)?;
        let n = horizon.get() as u64 + 1;
        let dates = (last_date..last_date + n * freq)
            .step_by(freq as usize)
            .filter(|ds| *ds > last_date)
            .take(horizon.get() as usize);

        let ds = if include_history == IncludeHistory::Yes {
            history_dates.iter().copied().chain(dates).collect()
        } else {
            dates.collect()
        };
        Ok(PredictionData::new(ds))
    }

    fn infer_freq(history_dates: &[TimestampSeconds]) -> Result<TimestampSeconds, Error> {
        const INFER_N: usize = 5;
        let get_tried = || {
            history_dates
                .iter()
                .rev()
                .take(INFER_N)
                .copied()
                .collect_vec()
        };
        // Calculate diffs between the last 5 dates in the history, and
        // create a map from diffs to counts.
        let diff_counts = history_dates
            .iter()
            .rev()
            .take(INFER_N)
            .tuple_windows()
            .map(|(a, b)| a - b)
            .counts();
        // Find the max count, and return the corresponding diff, provided there
        // is exactly one diff with that count.
        let max = diff_counts
            .values()
            .copied()
            .max()
            .ok_or_else(|| Error::UnableToInferFrequency(get_tried()))?;
        diff_counts
            .into_iter()
            .filter(|(_, v)| *v == max)
            .map(|(k, _)| k)
            .exactly_one()
            .map_err(|_| Error::UnableToInferFrequency(get_tried()))
    }
}

#[derive(Debug)]
struct PosteriorPredictiveSamples {
    yhat: Vec<Vec<f64>>,
    trend: Vec<Vec<f64>>,
}

/// Whether to include the historical dates in the future dataframe for predictions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IncludeHistory {
    /// Include the historical dates in the future dataframe.
    Yes,
    /// Do not include the historical dates in the future data frame.
    No,
}

/// Historical data after preprocessing.
#[derive(Debug, Clone)]
struct ProcessedData {
    ds: Vec<TimestampSeconds>,
    t: Vec<f64>,
    y_scaled: Vec<f64>,
    cap: Option<Vec<f64>>,
    cap_scaled: Option<Vec<f64>>,
    floor: Vec<f64>,
    regressors: HashMap<String, Vec<f64>>,
    seasonality_conditions: HashMap<String, Vec<bool>>,
}

/// Processed data used for fitting.
#[derive(Debug, Clone)]
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
            beta: vec![0.0; self.data.K as usize],
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

// Taken from the Rust compiler's test suite:
// https://github.com/rust-lang/rust/blob/917b0b6c70f078cb08bbb0080c9379e4487353c3/library/test/src/stats.rs#L258-L280.
fn percentile_of_sorted(sorted_samples: &[f64], pct: f64) -> f64 {
    assert!(!sorted_samples.is_empty());
    if sorted_samples.len() == 1 {
        return sorted_samples[0];
    }
    let zero: f64 = 0.0;
    assert!(zero <= pct);
    let hundred = 100_f64;
    assert!(pct <= hundred);
    if pct == hundred {
        return sorted_samples[sorted_samples.len() - 1];
    }
    let length = (sorted_samples.len() - 1) as f64;
    let rank = (pct / hundred) * length;
    let lrank = rank.floor();
    let d = rank - lrank;
    let n = lrank as usize;
    let lo = sorted_samples[n];
    let hi = sorted_samples[n + 1];
    lo + (hi - lo) * d
}

#[cfg(test)]
mod test_trend_component {
    use std::f64::consts::PI;

    use augurs_testing::{assert_all_close, assert_approx_eq};
    use chrono::{NaiveDate, TimeDelta};

    use super::*;
    use crate::{
        optimizer::mock_optimizer::MockOptimizer,
        testdata::{daily_univariate_ts, train_test_split},
    };

    #[test]
    fn growth_init() {
        let mut data = daily_univariate_ts().head(468);
        let max = data.y.iter().copied().nanmax(true);
        data = data.with_cap(vec![max; 468]).unwrap();

        let mut opts = ProphetOptions::default();
        let mut prophet = Prophet::new(opts.clone(), MockOptimizer::new());
        let preprocessed = prophet.preprocess(data.clone()).unwrap();
        let init = preprocessed.calculate_initial_params(&opts).unwrap();
        assert_approx_eq!(init.k, 0.3055671);
        assert_approx_eq!(init.m, 0.5307511);

        opts.growth = GrowthType::Logistic;
        let mut prophet = Prophet::new(opts.clone(), MockOptimizer::new());
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
    fn growth_init_minmax() {
        let mut data = daily_univariate_ts().head(468);
        let max = data.y.iter().copied().nanmax(true);
        data = data.with_cap(vec![max; 468]).unwrap();

        let mut opts = ProphetOptions {
            scaling: Scaling::MinMax,
            ..ProphetOptions::default()
        };
        let mut prophet = Prophet::new(opts.clone(), MockOptimizer::new());
        let preprocessed = prophet.preprocess(data.clone()).unwrap();
        let init = preprocessed.calculate_initial_params(&opts).unwrap();
        assert_approx_eq!(init.k, 0.4053406);
        assert_approx_eq!(init.m, 0.3775322);

        opts.growth = GrowthType::Logistic;
        let mut prophet = Prophet::new(opts.clone(), MockOptimizer::new());
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
    fn flat_growth_absmax() {
        let opts = ProphetOptions {
            growth: GrowthType::Flat,
            scaling: Scaling::AbsMax,
            ..ProphetOptions::default()
        };
        let mut prophet = Prophet::new(opts, MockOptimizer::new());
        let x = (0..50).map(|x| x as f64 * PI * 2.0 / 50.0);
        let y = x.map(|x| 30.0 + (x * 8.0).sin()).collect_vec();
        let ds = (0..50)
            .map(|x| {
                (NaiveDate::from_ymd_opt(2020, 1, 1).unwrap() + TimeDelta::days(x))
                    .and_hms_opt(0, 0, 0)
                    .unwrap()
                    .and_utc()
                    .timestamp() as TimestampSeconds
            })
            .collect_vec();
        let data = TrainingData::new(ds, y).unwrap();
        prophet.fit(data, Default::default()).unwrap();
        let future = prophet
            .make_future_dataframe(10.try_into().unwrap(), IncludeHistory::Yes)
            .unwrap();
        let _predictions = prophet.predict(future).unwrap();
    }

    #[test]
    fn piecewise_linear() {
        let t = (0..11).map(f64::from).collect_vec();
        let m = 0.0;
        let k = 1.0;
        let deltas = vec![0.5];
        let changepoints_t = vec![5.0];
        let y = Prophet::piecewise_linear(&t, &deltas, k, m, &changepoints_t).collect_vec();
        let y_true = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.5, 8.0, 9.5, 11.0, 12.5];
        assert_eq!(y, y_true);

        let y = Prophet::piecewise_linear(&t[8..], &deltas, k, m, &changepoints_t).collect_vec();
        assert_eq!(y, y_true[8..]);

        // This test isn't in the Python version but it's worth having one with multiple
        // changepoints.
        let deltas = vec![0.4, 0.5];
        let changepoints_t = vec![4.0, 8.0];
        let y = Prophet::piecewise_linear(&t, &deltas, k, m, &changepoints_t).collect_vec();
        let y_true = &[0.0, 1.0, 2.0, 3.0, 4.0, 5.4, 6.8, 8.2, 9.6, 11.5, 13.4];
        for (a, b) in y.iter().zip(y_true) {
            assert_approx_eq!(a, b);
        }
    }

    #[test]
    fn piecewise_logistic() {
        let t = (0..11).map(f64::from).collect_vec();
        let cap = vec![10.0; 11];
        let m = 0.0;
        let k = 1.0;
        let deltas = vec![0.5];
        let changepoints_t = vec![5.0];
        let y = Prophet::piecewise_logistic(&t, &cap, &deltas, k, m, &changepoints_t).collect_vec();
        let y_true = &[
            5.000000, 7.310586, 8.807971, 9.525741, 9.820138, 9.933071, 9.984988, 9.996646,
            9.999252, 9.999833, 9.999963,
        ];
        for (a, b) in y.iter().zip(y_true) {
            assert_approx_eq!(a, b);
        }

        let y = Prophet::piecewise_logistic(&t[8..], &cap[8..], &deltas, k, m, &changepoints_t)
            .collect_vec();
        for (a, b) in y.iter().zip(&y_true[8..]) {
            assert_approx_eq!(a, b);
        }

        // This test isn't in the Python version but it's worth having one with multiple
        // changepoints.
        let deltas = vec![0.4, 0.5];
        let changepoints_t = vec![4.0, 8.0];
        let y = Prophet::piecewise_logistic(&t, &cap, &deltas, k, m, &changepoints_t).collect_vec();
        let y_true = &[
            5., 7.31058579, 8.80797078, 9.52574127, 9.8201379, 9.95503727, 9.98887464, 9.99725422,
            9.99932276, 9.9998987, 9.99998485,
        ];
        for (a, b) in y.iter().zip(y_true) {
            assert_approx_eq!(a, b);
        }
    }

    #[test]
    fn flat_trend() {
        let t = (0..11).map(f64::from).collect_vec();
        let m = 0.5;
        let y = Prophet::flat_trend(&t, m).collect_vec();
        assert_all_close(&y, &[0.5; 11]);

        let y = Prophet::flat_trend(&t[8..], m).collect_vec();
        assert_all_close(&y, &[0.5; 3]);
    }

    #[test]
    fn get_changepoints() {
        let (data, _) = train_test_split(daily_univariate_ts(), 0.5);
        let optimizer = MockOptimizer::new();
        let mut prophet = Prophet::new(ProphetOptions::default(), optimizer);
        let preprocessed = prophet.preprocess(data).unwrap();
        let history = preprocessed.history;
        let changepoints_t = prophet.changepoints_t.as_ref().unwrap();
        assert_eq!(changepoints_t.len() as u32, prophet.opts.n_changepoints,);
        // Assert that the earliest changepoint is after the first point.
        assert!(changepoints_t.iter().copied().nanmin(true) > 0.0);
        // Assert that the changepoints are less than the 80th percentile of `t`.
        let cp_idx = (history.ds.len() as f64 * 0.8).ceil() as usize;
        assert!(changepoints_t.iter().copied().nanmax(true) <= history.t[cp_idx]);
        let expected = &[
            0.03504043, 0.06738544, 0.09433962, 0.12938005, 0.16442049, 0.1967655, 0.22371968,
            0.25606469, 0.28301887, 0.3180593, 0.35040431, 0.37735849, 0.41239892, 0.45013477,
            0.48247978, 0.51752022, 0.54447439, 0.57681941, 0.61185984, 0.64150943, 0.67924528,
            0.7115903, 0.74663073, 0.77358491, 0.80592992,
        ];
        for (a, b) in changepoints_t.iter().zip(expected) {
            assert_approx_eq!(a, b);
        }
    }

    #[test]
    fn get_changepoints_range() {
        let (data, _) = train_test_split(daily_univariate_ts(), 0.5);
        let opts = ProphetOptions {
            changepoint_range: 0.4.try_into().unwrap(),
            ..ProphetOptions::default()
        };
        let mut prophet = Prophet::new(opts, MockOptimizer::new());
        let preprocessed = prophet.preprocess(data).unwrap();
        let history = preprocessed.history;
        let changepoints_t = prophet.changepoints_t.as_ref().unwrap();
        assert_eq!(changepoints_t.len() as u32, prophet.opts.n_changepoints,);
        // Assert that the earliest changepoint is after the first point.
        assert!(changepoints_t.iter().copied().nanmin(true) > 0.0);
        // Assert that the changepoints are less than the 80th percentile of `t`.
        let cp_idx = (history.ds.len() as f64 * 0.4).ceil() as usize;
        assert!(changepoints_t.iter().copied().nanmax(true) <= history.t[cp_idx]);
        let expected = &[
            0.01617251, 0.03504043, 0.05121294, 0.06738544, 0.08355795, 0.09433962, 0.11051213,
            0.12938005, 0.14555256, 0.16172507, 0.17789757, 0.18867925, 0.20754717, 0.22371968,
            0.23989218, 0.25606469, 0.2722372, 0.28301887, 0.30188679, 0.3180593, 0.33423181,
            0.35040431, 0.36657682, 0.37735849, 0.393531,
        ];
        for (a, b) in changepoints_t.iter().zip(expected) {
            assert_approx_eq!(a, b);
        }
    }

    #[test]
    fn get_zero_changepoints() {
        let (data, _) = train_test_split(daily_univariate_ts(), 0.5);
        let opts = ProphetOptions {
            n_changepoints: 0,
            ..ProphetOptions::default()
        };
        let mut prophet = Prophet::new(opts, MockOptimizer::new());
        prophet.preprocess(data).unwrap();
        let changepoints_t = prophet.changepoints_t.as_ref().unwrap();
        assert_eq!(changepoints_t.len() as u32, 1);
        assert_eq!(changepoints_t[0], 0.0);
    }

    #[test]
    fn get_n_changepoints() {
        let data = daily_univariate_ts().head(20);
        let opts = ProphetOptions {
            n_changepoints: 15,
            ..ProphetOptions::default()
        };
        let mut prophet = Prophet::new(opts, MockOptimizer::new());
        prophet.preprocess(data).unwrap();
        let changepoints_t = prophet.changepoints_t.as_ref().unwrap();
        assert_eq!(prophet.opts.n_changepoints, 15);
        assert_eq!(changepoints_t.len() as u32, 15);
    }
}

#[cfg(test)]
mod test_seasonal {
    use augurs_testing::assert_approx_eq;

    use super::*;
    use crate::testdata::daily_univariate_ts;

    #[test]
    fn fourier_series_weekly() {
        let data = daily_univariate_ts();
        let mat = Prophet::fourier_series(&data.ds, 7.0.try_into().unwrap(), 3.try_into().unwrap());
        let expected = &[
            0.7818315, 0.6234898, 0.9749279, -0.2225209, 0.4338837, -0.9009689,
        ];
        assert_eq!(mat.len(), expected.len());
        let first = mat.iter().map(|row| row[0]);
        for (a, b) in first.zip(expected) {
            assert_approx_eq!(a, b);
        }
    }

    #[test]
    fn fourier_series_yearly() {
        let data = daily_univariate_ts();
        let mat =
            Prophet::fourier_series(&data.ds, 365.25.try_into().unwrap(), 3.try_into().unwrap());
        let expected = &[
            0.7006152, -0.7135393, -0.9998330, 0.01827656, 0.7262249, 0.6874572,
        ];
        assert_eq!(mat.len(), expected.len());
        let first = mat.iter().map(|row| row[0]);
        for (a, b) in first.zip(expected) {
            assert_approx_eq!(a, b);
        }
    }
}

#[cfg(test)]
mod test_data_prep {
    use crate::{
        optimizer::mock_optimizer::MockOptimizer,
        testdata::{daily_univariate_ts, train_test_split},
        util::FloatIterExt,
        Standardize,
    };

    use super::*;
    use augurs_testing::assert_approx_eq;
    use pretty_assertions::assert_eq;

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
    fn regressor_column_matrix() {
        // TODO: add holidays back in and update assertions.
        let opts = ProphetOptions::default();
        let mut prophet = Prophet::new(opts, MockOptimizer::new());
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
        assert_eq!(cols.regressors_additive, vec![0, 0, 0, 0, 0, 0, 1, 1, 0, 1]);
        assert_eq!(
            cols.regressors_multiplicative,
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
                    "regressors_additive".to_string(),
                    "additive_terms".to_string(),
                ]),
                multiplicative: HashSet::from([
                    "numeric_feature2".to_string(),
                    "regressors_multiplicative".to_string(),
                    "multiplicative_terms".to_string(),
                ]),
            }
        );
    }

    #[test]
    fn add_group_component() {
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
            (8, "regressors_additive"),
            (9, "regressors_additive"),
            (11, "regressors_additive"),
            (6, "multiplicative_terms"),
            (7, "multiplicative_terms"),
            (10, "multiplicative_terms"),
            (10, "regressors_multiplicative"),
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
            cols.regressors_additive,
            vec![0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1]
        );
        assert_eq!(
            cols.regressors_multiplicative,
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

#[cfg(test)]
mod test_predict {
    use augurs_testing::{assert_all_close, assert_approx_eq};
    use itertools::Itertools;

    use crate::{
        optimizer::{mock_optimizer::MockOptimizer, InitialParams, OptimizedParams},
        testdata::{daily_univariate_ts, train_test_splitn},
        util::FloatIterExt,
        IncludeHistory, Prophet, ProphetOptions, TrendIndicator,
    };

    #[test]
    fn fit_predict_absmax() {
        let test_days = 30;
        let (train, test) = train_test_splitn(daily_univariate_ts(), test_days);
        let opts = ProphetOptions {
            scaling: crate::Scaling::AbsMax,
            ..Default::default()
        };
        let opt = MockOptimizer::new();
        let mut prophet = Prophet::new(opts, opt);
        prophet.fit(train.clone(), Default::default()).unwrap();
        // Make sure our optimizer was called correctly.
        let opt: &MockOptimizer = prophet.optimizer.as_any().downcast_ref().unwrap();
        let call = opt.take_call().unwrap();
        assert_eq!(
            call.init,
            InitialParams {
                beta: vec![0.0; 6],
                delta: vec![0.0; 25],
                k: 0.29834791059280863,
                m: 0.5307510759405802,
                sigma_obs: 1.0
            }
        );
        assert_eq!(call.data.T, 480);
        assert_eq!(call.data.S, 25);
        assert_eq!(call.data.K, 6);
        assert_eq!(*call.data.tau, 0.05);
        assert_eq!(call.data.trend_indicator, TrendIndicator::Linear);
        assert_eq!(call.data.y.iter().copied().nanmax(true), 1.0);
        assert_all_close(
            &call.data.y[0..5],
            &[0.530751, 0.472442, 0.430376, 0.444259, 0.458559],
        );
        assert_eq!(call.data.t.len(), train.y.len());
        assert_all_close(
            &call.data.t[0..5],
            &[0.0, 0.004298, 0.005731, 0.007163, 0.008596],
        );

        assert_eq!(call.data.cap.len(), train.y.len());
        assert_eq!(&call.data.cap, &[0.0; 480]);

        assert_eq!(
            &call.data.sigmas.iter().map(|x| **x).collect_vec(),
            &[10.0; 6]
        );
        assert_eq!(&call.data.s_a, &[1; 6]);
        assert_eq!(&call.data.s_m, &[0; 6]);
        assert_eq!(call.data.X.len(), 6);
        let first = call.data.X.iter().map(|row| row[0]).collect_vec();
        assert_all_close(
            &first,
            &[0.781831, 0.623490, 0.974928, -0.222521, 0.433884, -0.900969],
        );

        // Override optimized params since we don't have a real optimizer.
        // These were obtained from the Python version.
        prophet.optimized = Some(OptimizedParams {
            k: -1.01136,
            m: 0.460947,
            sigma_obs: 0.0451108,
            beta: vec![
                0.0205064,
                -0.0129451,
                -0.0164735,
                -0.00275837,
                0.00333371,
                0.00599414,
            ],
            delta: vec![
                3.51708e-08,
                1.17925e-09,
                -2.91421e-09,
                2.06189e-01,
                9.06870e-01,
                4.49113e-01,
                1.94664e-03,
                -1.16088e-09,
                -5.75394e-08,
                -7.90284e-06,
                -6.74530e-01,
                -5.70814e-02,
                -4.91360e-08,
                -3.53111e-09,
                1.42645e-08,
                4.50809e-05,
                8.86286e-01,
                1.14535e+00,
                4.40539e-02,
                8.17306e-09,
                -1.57715e-07,
                -5.15430e-01,
                -3.15001e-01,
                1.14429e-08,
                -2.56863e-09,
            ],
            trend: vec![
                0.460947, 0.4566, 0.455151, 0.453703, 0.452254, 0.450805, 0.445009, 0.44356,
                0.442111, 0.440662, 0.436315, 0.434866, 0.433417, 0.431968, 0.430519, 0.426173,
                0.424724, 0.423275, 0.421826, 0.420377, 0.41603, 0.414581, 0.413132, 0.411683,
                0.410234, 0.405887, 0.404438, 0.402989, 0.40154, 0.400092, 0.395745, 0.394296,
                0.391398, 0.389949, 0.385602, 0.384153, 0.382704, 0.381255, 0.379806, 0.375459,
                0.374011, 0.372562, 0.371113, 0.369664, 0.365317, 0.363868, 0.362419, 0.36097,
                0.359521, 0.355174, 0.353725, 0.352276, 0.350827, 0.349378, 0.345032, 0.343583,
                0.342134, 0.340685, 0.339236, 0.334889, 0.33344, 0.331991, 0.330838, 0.329684,
                0.326223, 0.32507, 0.323916, 0.322763, 0.321609, 0.318149, 0.316995, 0.315841,
                0.314688, 0.313534, 0.30892, 0.307767, 0.306613, 0.30546, 0.305897, 0.306042,
                0.306188, 0.306334, 0.306479, 0.306916, 0.307062, 0.307208, 0.307354, 0.307499,
                0.307936, 0.308082, 0.308228, 0.308373, 0.308519, 0.310886, 0.311676, 0.312465,
                0.313254, 0.314043, 0.31641, 0.317199, 0.317989, 0.318778, 0.319567, 0.321934,
                0.322723, 0.323512, 0.324302, 0.325091, 0.327466, 0.328258, 0.32905, 0.329842,
                0.330634, 0.334594, 0.335386, 0.336177, 0.338553, 0.339345, 0.340137, 0.340929,
                0.341721, 0.344097, 0.344888, 0.34568, 0.346472, 0.347264, 0.34964, 0.350432,
                0.351224, 0.352808, 0.355183, 0.355975, 0.356767, 0.357559, 0.358351, 0.360727,
                0.361519, 0.362311, 0.363102, 0.363894, 0.36627, 0.367062, 0.367854, 0.368646,
                0.369438, 0.371813, 0.372605, 0.373397, 0.374189, 0.374981, 0.377357, 0.378941,
                0.379733, 0.380524, 0.3829, 0.384484, 0.385276, 0.386068, 0.388443, 0.389235,
                0.390027, 0.390819, 0.391611, 0.393987, 0.394779, 0.395571, 0.396362, 0.397154,
                0.400322, 0.401114, 0.400939, 0.400765, 0.400242, 0.400067, 0.399893, 0.399718,
                0.399544, 0.39902, 0.398846, 0.398671, 0.398497, 0.398322, 0.397799, 0.397624,
                0.39745, 0.397194, 0.396937, 0.395912, 0.395656, 0.3954, 0.395144, 0.394375,
                0.394119, 0.393862, 0.393606, 0.39335, 0.392581, 0.392325, 0.392069, 0.391812,
                0.391556, 0.390787, 0.390531, 0.390275, 0.390019, 0.389762, 0.388994, 0.388737,
                0.388481, 0.388225, 0.387968, 0.3872, 0.386943, 0.386687, 0.386431, 0.385406,
                0.38515, 0.384893, 0.384637, 0.384381, 0.383612, 0.383356, 0.3831, 0.382843,
                0.382587, 0.381818, 0.381562, 0.381306, 0.38105, 0.380793, 0.380025, 0.379768,
                0.379512, 0.379256, 0.379, 0.378231, 0.377975, 0.377718, 0.377462, 0.377206,
                0.376437, 0.376181, 0.375925, 0.375668, 0.375412, 0.374643, 0.374387, 0.374131,
                0.373875, 0.373619, 0.37285, 0.372594, 0.372338, 0.372081, 0.371825, 0.3708,
                0.370544, 0.370288, 0.370032, 0.369263, 0.369007, 0.370021, 0.371034, 0.372048,
                0.375088, 0.376102, 0.377116, 0.378129, 0.379143, 0.382183, 0.383197, 0.384211,
                0.385224, 0.386238, 0.389278, 0.390292, 0.391305, 0.39396, 0.396614, 0.404578,
                0.407232, 0.409887, 0.415196, 0.423159, 0.425813, 0.428468, 0.431122, 0.433777,
                0.44174, 0.444395, 0.447049, 0.449704, 0.452421, 0.460574, 0.463291, 0.466009,
                0.468727, 0.471444, 0.479597, 0.482314, 0.485032, 0.48775, 0.490467, 0.49862,
                0.501337, 0.504055, 0.506773, 0.50949, 0.517643, 0.520361, 0.523078, 0.525796,
                0.528513, 0.536666, 0.539384, 0.542101, 0.544819, 0.547536, 0.555689, 0.558407,
                0.561124, 0.563842, 0.566559, 0.57743, 0.580147, 0.582865, 0.585582, 0.593735,
                0.596453, 0.59917, 0.601888, 0.604605, 0.612758, 0.615476, 0.618193, 0.620911,
                0.623628, 0.631781, 0.63376, 0.635739, 0.637719, 0.639698, 0.645635, 0.647614,
                0.649593, 0.651572, 0.653552, 0.659489, 0.661468, 0.663447, 0.665426, 0.667406,
                0.673343, 0.674871, 0.676399, 0.677926, 0.679454, 0.684038, 0.685566, 0.687094,
                0.688621, 0.690149, 0.694733, 0.696261, 0.697788, 0.699316, 0.700844, 0.705428,
                0.706956, 0.708483, 0.710011, 0.711539, 0.716123, 0.71765, 0.719178, 0.720706,
                0.722234, 0.726818, 0.728345, 0.729873, 0.731401, 0.732929, 0.737512, 0.73904,
                0.740568, 0.743624, 0.748207, 0.749735, 0.751263, 0.752791, 0.754319, 0.758902,
                0.76043, 0.761958, 0.763486, 0.765014, 0.769597, 0.771125, 0.772653, 0.774181,
                0.775709, 0.780292, 0.78182, 0.784876, 0.786404, 0.790987, 0.792515, 0.795571,
                0.797098, 0.801682, 0.80321, 0.804738, 0.806265, 0.807793, 0.812377, 0.813905,
                0.815433, 0.81696, 0.818488, 0.8246, 0.826127, 0.827655, 0.829183, 0.833767,
                0.835295, 0.836822, 0.83835, 0.839878, 0.844462, 0.845989, 0.847517, 0.849045,
                0.850573, 0.855157, 0.856684, 0.858212, 0.85974, 0.861268, 0.867379, 0.868907,
                0.870435, 0.871963, 0.876546, 0.878074, 0.879602, 0.88113, 0.882658, 0.887241,
                0.888769, 0.890297, 0.891825, 0.893353, 0.897936, 0.899464, 0.900992, 0.90252,
                0.904048, 0.908631, 0.910159, 0.911687, 0.913215, 0.914743, 0.919326, 0.920854,
                0.922382, 0.92391, 0.925437, 0.930021, 0.931549, 0.933077, 0.934604, 0.936132,
                0.940716, 0.942244, 0.943772, 0.945299, 0.946827, 0.951411, 0.952939, 0.954466,
            ],
        });
        let future = prophet
            .make_future_dataframe((test_days as u32).try_into().unwrap(), IncludeHistory::No)
            .unwrap();
        let predictions = prophet.predict(future).unwrap();
        assert_eq!(predictions.yhat.point.len(), test_days);
        let rmse = (predictions
            .yhat
            .point
            .iter()
            .zip(&test.y)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            / test.y.len() as f64)
            .sqrt();
        assert_approx_eq!(rmse, 10.64, 1e-1);

        let lower = predictions.yhat.lower.as_ref().unwrap();
        assert_eq!(lower.len(), predictions.yhat.point.len());
    }
}
