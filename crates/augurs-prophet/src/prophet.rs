pub(crate) mod options;
pub(crate) mod predict;
pub(crate) mod prep;

use std::{
    collections::{HashMap, HashSet},
    num::NonZeroU32,
    sync::Arc,
};

use itertools::{izip, Itertools};
use options::ProphetOptions;
use prep::{ComponentColumns, Modes, Preprocessed, Scales};

use crate::{
    forecaster::ProphetForecaster,
    optimizer::{InitialParams, OptimizeOpts, OptimizedParams, Optimizer},
    Error, EstimationMode, FeaturePrediction, IncludeHistory, IntervalWidth, PredictionData,
    Predictions, Regressor, Seasonality, TimestampSeconds, TrainingData,
};

/// The Prophet time series forecasting model.
#[derive(Debug, Clone)]
pub struct Prophet<O> {
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
    optimizer: O,

    /// The processed data used for fitting.
    processed: Option<Preprocessed>,

    /// The initial parameters passed to optimization.
    init: Option<InitialParams>,

    /// The optimized model, if it has been fit.
    optimized: Option<OptimizedParams>,
}

// All public methods should live in this `impl` even if they call
// lots of functions in private modules, so that Rustdoc shows them
// all in a single block.
impl<O> Prophet<O> {
    /// Create a new Prophet model with the given options and optimizer.
    pub fn new(opts: ProphetOptions, optimizer: O) -> Self {
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
            optimizer,
            processed: None,
            init: None,
            optimized: None,
        }
    }

    /// Add a custom seasonality to the model.
    pub fn add_seasonality(
        &mut self,
        name: String,
        seasonality: Seasonality,
    ) -> Result<&mut Self, Error> {
        if self.seasonalities.contains_key(&name) {
            return Err(Error::DuplicateSeasonality(name));
        }
        self.seasonalities.insert(name, seasonality);
        Ok(self)
    }

    /// Add a regressor to the model.
    pub fn add_regressor(&mut self, name: String, regressor: Regressor) -> &mut Self {
        self.regressors.insert(name, regressor);
        self
    }

    /// Return `true` if the model has been fit, or `false` if not.
    pub fn is_fitted(&self) -> bool {
        self.optimized.is_some()
    }

    /// Predict using the Prophet model.
    ///
    /// # Errors
    ///
    /// Returns an error if the model has not been fit.
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
            &df.cap_scaled,
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
            ds: df.ds,
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
        let n = (horizon.get() as u64 + 1) as TimestampSeconds;
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

    /// Get a reference to the Prophet options.
    pub fn opts(&self) -> &ProphetOptions {
        &self.opts
    }

    /// Get a mutable reference to the Prophet options.
    pub fn opts_mut(&mut self) -> &mut ProphetOptions {
        &mut self.opts
    }

    /// Set the width of the uncertainty intervals.
    ///
    /// The interval width does not affect training, only predictions,
    /// so this can be called after fitting the model to obtain predictions
    /// with different levels of uncertainty.
    pub fn set_interval_width(&mut self, interval_width: IntervalWidth) {
        self.opts.interval_width = interval_width;
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

impl<O: Optimizer + 'static> Prophet<O> {
    pub(crate) fn into_dyn_optimizer(self) -> Prophet<Arc<dyn Optimizer + 'static>> {
        Prophet {
            optimizer: Arc::new(self.optimizer),
            opts: self.opts,
            regressors: self.regressors,
            optimized: self.optimized,
            changepoints: self.changepoints,
            changepoints_t: self.changepoints_t,
            init: self.init,
            scales: self.scales,
            processed: self.processed,
            seasonalities: self.seasonalities,
            component_modes: self.component_modes,
            train_holiday_names: self.train_holiday_names,
            train_component_columns: self.train_component_columns,
        }
    }

    /// Create a new `ProphetForecaster` from this Prophet model.
    ///
    /// This requires the data and optimize options to be provided and sets up
    /// a `ProphetForecaster` ready to be used with the `augurs_forecaster` crate.
    pub fn into_forecaster(
        self,
        data: TrainingData,
        optimize_opts: OptimizeOpts,
    ) -> ProphetForecaster {
        ProphetForecaster::new(self, data, optimize_opts)
    }
}

impl<O: Optimizer> Prophet<O> {
    /// Fit the Prophet model to some training data.
    pub fn fit(&mut self, data: TrainingData, mut opts: OptimizeOpts) -> Result<(), Error> {
        let preprocessed = self.preprocess(data)?;
        let init = preprocessed.calculate_initial_params(&self.opts)?;
        // TODO: call `sample` if `self.opts.estimation == EstimationMode::Mcmc`.
        // We'll first need to add a `sample` method to the Optimizer trait, then accept
        // a different set of options in `opts`, and also update `OptimizedParams` to
        // include every MCMC sample.
        if opts.jacobian.is_none() {
            let use_jacobian = self.opts.estimation == EstimationMode::Map;
            opts.jacobian = Some(use_jacobian);
        }
        self.optimized = Some(
            self.optimizer
                .optimize(&init, &preprocessed.data, &opts)
                .map_err(|e| Error::OptimizationFailed(e.to_string()))?,
        );
        self.processed = Some(preprocessed);
        self.init = Some(init);
        Ok(())
    }
}

#[cfg(test)]
mod test_trend {
    use std::f64::consts::PI;

    use augurs_core::FloatIterExt;
    use augurs_testing::assert_approx_eq;
    use chrono::{NaiveDate, TimeDelta};
    use itertools::Itertools;

    use super::*;
    use crate::{
        optimizer::mock_optimizer::MockOptimizer,
        testdata::{daily_univariate_ts, train_test_split},
        GrowthType, IncludeHistory, Scaling, TrainingData,
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
        let mat =
            Prophet::<()>::fourier_series(&data.ds, 7.0.try_into().unwrap(), 3.try_into().unwrap());
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
        let mat = Prophet::<()>::fourier_series(
            &data.ds,
            365.25.try_into().unwrap(),
            3.try_into().unwrap(),
        );
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
mod test_custom_seasonal {
    use std::collections::HashMap;

    use chrono::NaiveDate;
    use itertools::Itertools;

    use crate::{
        optimizer::mock_optimizer::MockOptimizer,
        prophet::prep::{FeatureName, Features},
        testdata::daily_univariate_ts,
        FeatureMode, Holiday, HolidayOccurrence, ProphetOptions, Seasonality, SeasonalityOption,
    };

    use super::Prophet;

    #[test]
    fn custom_prior() {
        let holiday_dates = ["2017-01-02"]
            .iter()
            .map(|s| {
                HolidayOccurrence::for_day(
                    s.parse::<NaiveDate>()
                        .unwrap()
                        .and_hms_opt(0, 0, 0)
                        .unwrap()
                        .and_utc()
                        .timestamp(),
                )
            })
            .collect();

        let opts = ProphetOptions {
            holidays: [(
                "special day".to_string(),
                Holiday::new(holiday_dates).with_prior_scale(4.0.try_into().unwrap()),
            )]
            .into(),
            seasonality_mode: FeatureMode::Multiplicative,
            yearly_seasonality: SeasonalityOption::Manual(false),
            ..Default::default()
        };

        let data = daily_univariate_ts();
        let mut prophet = Prophet::new(opts, MockOptimizer::new());
        prophet
            .add_seasonality(
                "monthly".to_string(),
                Seasonality::new(30.0.try_into().unwrap(), 5.try_into().unwrap())
                    .with_prior_scale(2.0.try_into().unwrap())
                    .with_mode(FeatureMode::Additive),
            )
            .unwrap();
        prophet.fit(data, Default::default()).unwrap();
        prophet.predict(None).unwrap();

        assert_eq!(prophet.seasonalities["weekly"].mode, None);
        assert_eq!(
            prophet.seasonalities["monthly"].mode,
            Some(FeatureMode::Additive)
        );
        let Features {
            features,
            prior_scales,
            component_columns,
            ..
        } = prophet
            .make_all_features(&prophet.processed.as_ref().unwrap().history)
            .unwrap();

        assert_eq!(
            component_columns.seasonalities["monthly"]
                .iter()
                .sum::<i32>(),
            10
        );
        assert_eq!(
            component_columns.holidays["special day"]
                .iter()
                .sum::<i32>(),
            1
        );
        assert_eq!(
            component_columns.seasonalities["weekly"]
                .iter()
                .sum::<i32>(),
            6
        );
        assert_eq!(component_columns.additive.iter().sum::<i32>(), 10);
        assert_eq!(component_columns.multiplicative.iter().sum::<i32>(), 7);

        if features.names[0]
            == (FeatureName::Seasonality {
                name: "monthly".to_string(),
                _id: 1,
            })
        {
            assert_eq!(
                component_columns.seasonalities["monthly"],
                &[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            );
            assert_eq!(
                component_columns.seasonalities["weekly"],
                &[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0],
            );
            let expected_prior_scales = [
                2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 10.0, 10.0, 10.0, 10.0, 10.0,
                10.0, 4.0,
            ]
            .map(|x| x.try_into().unwrap());
            assert_eq!(&prior_scales, &expected_prior_scales);
        } else {
            assert_eq!(
                component_columns.seasonalities["monthly"],
                &[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            );
            assert_eq!(
                component_columns.seasonalities["weekly"],
                &[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            );
            let expected_prior_scales = [
                10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                2.0, 4.0,
            ]
            .map(|x| x.try_into().unwrap());
            assert_eq!(&prior_scales, &expected_prior_scales);
        }
    }

    #[test]
    fn conditional_custom_seasonality() {
        // Set up data.
        let mut data = daily_univariate_ts();
        let condition_col = [[false; 255], [true; 255]].concat();
        let conditions =
            HashMap::from([("is_conditional_week".to_string(), condition_col.clone())]);
        data = data.with_seasonality_conditions(conditions).unwrap();

        // Set up Prophet model.
        let opts = ProphetOptions {
            yearly_seasonality: SeasonalityOption::Manual(false),
            weekly_seasonality: SeasonalityOption::Manual(false),
            ..Default::default()
        };
        let mut prophet = Prophet::new(opts, MockOptimizer::new());
        prophet
            .add_seasonality(
                "conditional_weekly".to_string(),
                Seasonality::new(7.0.try_into().unwrap(), 3.try_into().unwrap())
                    .with_prior_scale(2.0.try_into().unwrap())
                    .with_condition("is_conditional_week".to_string())
                    .with_mode(FeatureMode::Additive),
            )
            .unwrap()
            .add_seasonality(
                "normal_monthly".to_string(),
                Seasonality::new(30.5.try_into().unwrap(), 5.try_into().unwrap())
                    .with_prior_scale(2.0.try_into().unwrap())
                    .with_mode(FeatureMode::Additive),
            )
            .unwrap();

        prophet.fit(data, Default::default()).unwrap();
        prophet.predict(None).unwrap();

        let Features { features, .. } = prophet
            .make_all_features(&prophet.processed.as_ref().unwrap().history)
            .unwrap();
        let condition_features = features
            .names
            .iter()
            .zip(&features.data)
            .filter(|(name, _)| {
                matches!(name, FeatureName::Seasonality { name, .. } if name == "conditional_weekly")
            })
            .collect_vec();
        // Check that each of the condition features is zero everywhere except
        // where the condition column is true.
        for (_, condition_feature) in condition_features {
            assert_eq!(condition_col.len(), condition_feature.len());
            for (cond, f) in condition_col.iter().zip(condition_feature) {
                assert_eq!(*f != 0.0, *cond);
            }
        }
    }
}

#[cfg(test)]
mod test_holidays {
    use chrono::NaiveDate;

    use crate::{
        optimizer::mock_optimizer::MockOptimizer, testdata::daily_univariate_ts, Holiday,
        HolidayOccurrence, Prophet, ProphetOptions,
    };

    #[test]
    fn fit_predict_holiday() {
        let holiday_dates = ["2012-10-09", "2013-10-09"]
            .iter()
            .map(|s| {
                HolidayOccurrence::for_day(
                    s.parse::<NaiveDate>()
                        .unwrap()
                        .and_hms_opt(0, 0, 0)
                        .unwrap()
                        .and_utc()
                        .timestamp(),
                )
            })
            .collect();
        let opts = ProphetOptions {
            holidays: [("bens-bday".to_string(), Holiday::new(holiday_dates))].into(),
            ..Default::default()
        };
        let data = daily_univariate_ts();
        let mut prophet = Prophet::new(opts, MockOptimizer::new());
        prophet.fit(data, Default::default()).unwrap();
        prophet.predict(None).unwrap();
    }
}

#[cfg(test)]
mod test_fit {
    use augurs_core::FloatIterExt;
    use augurs_testing::assert_all_close;
    use itertools::Itertools;

    use crate::{
        optimizer::{mock_optimizer::MockOptimizer, InitialParams},
        testdata::{daily_univariate_ts, train_test_splitn},
        Prophet, ProphetOptions, TrendIndicator,
    };

    /// This test is extracted from the `fit_predict` test of the Python Prophet
    /// library. Since we don't want to depend on an optimizer, this just ensures
    /// that we're correctly getting the data ready for Stan, by recording the data
    /// that's sent to the configured optimizer.
    ///
    /// There is a similar test in `predict.rs` which patches the returned
    /// optimized parameters and ensures predictions look sensible.
    #[test]
    fn fit_absmax() {
        let test_days = 30;
        let (train, _) = train_test_splitn(daily_univariate_ts(), test_days);
        let opts = ProphetOptions {
            scaling: crate::Scaling::AbsMax,
            ..Default::default()
        };
        let opt = MockOptimizer::new();
        let mut prophet = Prophet::new(opts, opt);
        prophet.fit(train.clone(), Default::default()).unwrap();
        // Make sure our optimizer was called correctly.
        let opt: &MockOptimizer = &prophet.optimizer;
        let call = opt.take_call().unwrap();
        assert_eq!(
            call.init,
            InitialParams {
                beta: vec![0.0; 6],
                delta: vec![0.0; 25],
                k: 0.29834791059280863,
                m: 0.5307510759405802,
                sigma_obs: 1.0.try_into().unwrap(),
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
        assert_eq!(call.data.X.len(), 6 * 480);
        let first = &call.data.X[..6];
        assert_all_close(
            first,
            &[0.781831, 0.623490, 0.974928, -0.222521, 0.433884, -0.900969],
        );
    }

    // Regression test for https://github.com/grafana/augurs/issues/209.
    #[test]
    fn fit_with_nans() {
        let test_days = 30;
        let (mut train, _) = train_test_splitn(daily_univariate_ts(), test_days);
        train.y[10] = f64::NAN;
        let opt = MockOptimizer::new();
        let mut prophet = Prophet::new(Default::default(), opt);
        // Should not panic.
        prophet.fit(train.clone(), Default::default()).unwrap();
    }
}
