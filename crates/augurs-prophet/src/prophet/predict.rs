use std::collections::HashMap;

use augurs_core::FloatIterExt;
use itertools::{izip, Itertools};
use rand::{distributions::Uniform, thread_rng, Rng};
use statrs::distribution::{Laplace, Normal, Poisson};

use crate::{optimizer::OptimizedParams, Error, GrowthType, Prophet, TimestampSeconds};

use super::prep::{ComponentName, Features, FeaturesFrame, Modes, ProcessedData};

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
pub(super) struct FeaturePredictions {
    /// Contribution of the additive terms in the model.
    ///
    /// This includes additive seasonalities, holidays and regressors.
    pub(super) additive: FeaturePrediction,
    /// Contribution of the multiplicative terms in the model.
    ///
    /// This includes multiplicative seasonalities, holidays and regressors.
    pub(super) multiplicative: FeaturePrediction,
    /// Mapping from holiday name to the contribution of that holiday.
    pub(super) holidays: HashMap<String, FeaturePrediction>,
    /// Mapping from regressor name to the contribution of that regressor.
    pub(super) regressors: HashMap<String, FeaturePrediction>,
    /// Mapping from seasonality name to the contribution of that seasonality.
    pub(super) seasonalities: HashMap<String, FeaturePrediction>,
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
    /// The timestamps of the forecasts.
    pub ds: Vec<TimestampSeconds>,

    /// Forecasts of the input time series `y`.
    pub yhat: FeaturePrediction,

    /// The trend contribution at each time point.
    pub trend: FeaturePrediction,

    /// The cap for the logistic growth.
    ///
    /// Will only be `Some` if the model used [`GrowthType::Logistic`](crate::GrowthType::Logistic).
    pub cap: Option<Vec<f64>>,
    /// The floor for the logistic growth.
    ///
    /// Will only be `Some` if the model used [`GrowthType::Logistic`](crate::GrowthType::Logistic)
    /// and the floor was provided in the input data.
    pub floor: Option<Vec<f64>>,

    /// The combined combination of all _additive_ components.
    ///
    /// This includes seasonalities, holidays and regressors if their mode
    /// was configured to be [`FeatureMode::Additive`](crate::FeatureMode::Additive).
    pub additive: FeaturePrediction,

    /// The combined combination of all _multiplicative_ components.
    ///
    /// This includes seasonalities, holidays and regressors if their mode
    /// was configured to be [`FeatureMode::Multiplicative`](crate::FeatureMode::Multiplicative).
    pub multiplicative: FeaturePrediction,

    /// Mapping from holiday name to that holiday's contribution.
    pub holidays: HashMap<String, FeaturePrediction>,

    /// Mapping from seasonality name to that seasonality's contribution.
    pub seasonalities: HashMap<String, FeaturePrediction>,

    /// Mapping from regressor name to that regressor's contribution.
    pub regressors: HashMap<String, FeaturePrediction>,
}

/// Whether to include the historical dates in the future dataframe for predictions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IncludeHistory {
    /// Include the historical dates in the future dataframe.
    Yes,
    /// Do not include the historical dates in the future data frame.
    No,
}

#[derive(Debug)]
pub(super) struct PosteriorPredictiveSamples {
    pub(super) yhat: Vec<Vec<f64>>,
    pub(super) trend: Vec<Vec<f64>>,
}

impl<O> Prophet<O> {
    /// Predict trend.
    pub(super) fn predict_trend(
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
                Self::piecewise_linear(t, &params.delta, params.k, params.m, changepoints_t)
                    .zip(floor)
                    .map(|(trend, flr)| trend * y_scale + flr)
                    .collect_vec()
            }
            (GrowthType::Logistic, Some(cap)) => {
                Self::piecewise_logistic(t, cap, &params.delta, params.k, params.m, changepoints_t)
                    .zip(floor)
                    .map(|(trend, flr)| trend * y_scale + flr)
                    .collect_vec()
            }
            (GrowthType::Logistic, None) => return Err(Error::MissingCap),
            (GrowthType::Flat, _) => Self::flat_trend(t, params.m)
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
            .map(|((delta, cp_t), t)| if cp_t <= t { *delta } else { 0.0 });

        // Repeat each changepoint effect `n` times so we can zip it up.
        let changepoints_repeated = changepoints_t
            .iter()
            .flat_map(|x| std::iter::repeat(*x).take(t.len()));
        let indexes = (0..t.len()).cycle();
        // `k_m_t` is a contiguous array where each element contains the rate and offset to
        // apply at each time point.
        let k_m_t = izip!(deltas_t, changepoints_repeated, indexes).fold(
            vec![(k, m); t.len()],
            |mut acc, (delta, cp_t, idx)| {
                // Add the changepoint rate to the initial rate.
                acc[idx].0 += delta;
                // Add the changepoint offset to the initial offset where applicable.
                acc[idx].1 += -cp_t * delta;
                acc
            },
        );

        izip!(t, k_m_t).map(|(t, (k, m))| t * k + m)
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

    /// Predict seasonality, holidays and added regressors.
    pub(super) fn predict_features(
        &self,
        features: &Features,
        params: &OptimizedParams,
        y_scale: f64,
    ) -> Result<FeaturePredictions, Error> {
        let Features {
            features,
            component_columns,
            modes,
            ..
        } = features;
        let predict_feature = |col, f: fn(String) -> ComponentName| {
            Self::predict_components(col, &features.data, &params.beta, y_scale, modes, f)
        };
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
            holidays: predict_feature(&component_columns.holidays, ComponentName::Holiday),
            seasonalities: predict_feature(
                &component_columns.seasonalities,
                ComponentName::Seasonality,
            ),
            regressors: predict_feature(&component_columns.regressors, ComponentName::Regressor),
        })
    }

    fn predict_components(
        component_columns: &HashMap<String, Vec<i32>>,
        #[allow(non_snake_case)] X: &[Vec<f64>],
        beta: &[f64],
        y_scale: f64,
        modes: &Modes,
        make_mode: impl Fn(String) -> ComponentName,
    ) -> HashMap<String, FeaturePrediction> {
        component_columns
            .iter()
            .map(|(name, component_col)| {
                (
                    name.clone(),
                    Self::predict_feature(
                        component_col,
                        X,
                        beta,
                        y_scale,
                        modes.additive.contains(&make_mode(name.clone())),
                    ),
                )
            })
            .collect()
    }

    pub(super) fn predict_feature(
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
        for (feature, b) in izip!(X, beta_c) {
            for (p, x) in izip!(point.iter_mut(), feature) {
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
    pub(super) fn predict_uncertainty(
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
        let lower_p = 100.0 * (1.0 - *self.opts.interval_width) / 2.0;
        let upper_p = 100.0 * (1.0 + *self.opts.interval_width) / 2.0;

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
    pub(super) fn sample_posterior_predictive(
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
        // Use temporary buffers to avoid allocating a new Vec for each
        // call to `sample_model`.
        let (mut yhat, mut trend) = (
            Vec::with_capacity(n_timestamps),
            Vec::with_capacity(n_timestamps),
        );
        for i in 0..n_iterations {
            for _ in 0..samples_per_iter {
                self.sample_model(
                    df,
                    features,
                    params,
                    changepoints_t,
                    &component_columns.additive,
                    &component_columns.multiplicative,
                    y_scale,
                    i,
                    &mut yhat,
                    &mut trend,
                )?;
                // We have to transpose things, unfortunately.
                for ((i, yhat), trend) in yhat.iter().enumerate().zip(&trend) {
                    sim_values.yhat[i].push(*yhat);
                    sim_values.trend[i].push(*trend);
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
        yhat_tmp: &mut Vec<f64>,
        trend_tmp: &mut Vec<f64>,
    ) -> Result<(), Error> {
        yhat_tmp.clear();
        trend_tmp.clear();
        let n = df.ds.len();
        *trend_tmp =
            self.sample_predictive_trend(df, params, changepoints_t, y_scale, iteration)?;
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
        let dist = Normal::new(0.0, *sigma).expect("sigma must be non-negative");
        let mut rng = thread_rng();
        let noise = (&mut rng).sample_iter(dist).take(n).map(|x| x * y_scale);

        for yhat in izip!(trend_tmp, &xb_a, &xb_m, noise).map(|(t, a, m, n)| *t * (1.0 + m) + a + n)
        {
            yhat_tmp.push(yhat);
        }

        Ok(())
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
            rng.sample::<f64, _>(dist).round() as usize
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
mod test {
    use augurs_testing::{assert_all_close, assert_approx_eq};
    use itertools::Itertools;

    use crate::{
        optimizer::{mock_optimizer::MockOptimizer, OptimizedParams},
        testdata::{daily_univariate_ts, train_test_splitn},
        IncludeHistory, Prophet, ProphetOptions,
    };

    #[test]
    fn piecewise_linear() {
        let t = (0..11).map(f64::from).collect_vec();
        let m = 0.0;
        let k = 1.0;
        let deltas = vec![0.5];
        let changepoints_t = vec![5.0];
        let y = Prophet::<()>::piecewise_linear(&t, &deltas, k, m, &changepoints_t).collect_vec();
        let y_true = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.5, 8.0, 9.5, 11.0, 12.5];
        assert_eq!(y, y_true);

        let y =
            Prophet::<()>::piecewise_linear(&t[8..], &deltas, k, m, &changepoints_t).collect_vec();
        assert_eq!(y, y_true[8..]);

        // This test isn't in the Python version but it's worth having one with multiple
        // changepoints.
        let deltas = vec![0.4, 0.5];
        let changepoints_t = vec![4.0, 8.0];
        let y = Prophet::<()>::piecewise_linear(&t, &deltas, k, m, &changepoints_t).collect_vec();
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
        let y = Prophet::<()>::piecewise_logistic(&t, &cap, &deltas, k, m, &changepoints_t)
            .collect_vec();
        let y_true = &[
            5.000000, 7.310586, 8.807971, 9.525741, 9.820138, 9.933071, 9.984988, 9.996646,
            9.999252, 9.999833, 9.999963,
        ];
        for (a, b) in y.iter().zip(y_true) {
            assert_approx_eq!(a, b);
        }

        let y =
            Prophet::<()>::piecewise_logistic(&t[8..], &cap[8..], &deltas, k, m, &changepoints_t)
                .collect_vec();
        for (a, b) in y.iter().zip(&y_true[8..]) {
            assert_approx_eq!(a, b);
        }

        // This test isn't in the Python version but it's worth having one with multiple
        // changepoints.
        let deltas = vec![0.4, 0.5];
        let changepoints_t = vec![4.0, 8.0];
        let y = Prophet::<()>::piecewise_logistic(&t, &cap, &deltas, k, m, &changepoints_t)
            .collect_vec();
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
        let y = Prophet::<()>::flat_trend(&t, m).collect_vec();
        assert_all_close(&y, &[0.5; 11]);

        let y = Prophet::<()>::flat_trend(&t[8..], m).collect_vec();
        assert_all_close(&y, &[0.5; 3]);
    }

    /// This test is extracted from the `fit_predict` test of the Python Prophet
    /// library. Since we don't want to depend on an optimizer, this test patches the
    /// optimized parameters on the Prophet object and runs `predict`, ensuring we
    /// get sensible results.
    ///
    /// There is a similar test in `fit.rs` which ensures the data being sent to
    /// Stan is correct .
    #[test]
    fn predict_absmax() {
        let test_days = 30;
        let (train, test) = train_test_splitn(daily_univariate_ts(), test_days);
        let opts = ProphetOptions {
            scaling: crate::Scaling::AbsMax,
            ..Default::default()
        };
        let opt = MockOptimizer::new();
        let mut prophet = Prophet::new(opts, opt);
        prophet.fit(train.clone(), Default::default()).unwrap();

        // Override optimized params since we don't have a real optimizer.
        // These were obtained from the Python version.
        prophet.optimized = Some(OptimizedParams {
            k: -1.01136,
            m: 0.460947,
            sigma_obs: 0.0451108.try_into().unwrap(),
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
        let upper = predictions.yhat.upper.as_ref().unwrap();
        assert_eq!(lower.len(), predictions.yhat.point.len());
        for (lower_bound, point_estimate) in lower.iter().zip(&predictions.yhat.point) {
            assert!(
                lower_bound <= point_estimate,
                "Lower bound should be less than the point estimate"
            );
        }
        for (upper_bound, point_estimate) in upper.iter().zip(&predictions.yhat.point) {
            assert!(
                upper_bound >= point_estimate,
                "Upper bound should be greater than the point estimate"
            );
        }
    }
}
