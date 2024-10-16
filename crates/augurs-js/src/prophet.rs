use std::{collections::HashMap, num::TryFromIntError};

use augurs_prophet::PositiveFloat;
use js_sys::{Float64Array, Int32Array};
use serde::{Deserialize, Serialize};
use tsify_next::Tsify;
use wasm_bindgen::prelude::*;

// We just take an untyped Function here, but we'll use the `Tsify` macro to
// customize the type in `ProphetOptions`. Here we add the JSDoc and type
// annotations for the function to make it harder to misuse.
#[wasm_bindgen(typescript_custom_section)]
const OPTIMIZER_FUNCTION: &'static str = r#"
/**
 * A function which calculates and returns the maximum likelihood estimate
 * of the Prophet Stan model parameters.
 *
 * In the R and Python libraries, and in `augurs-prophet`, the `cmdstan`
 * optimizer is used. This isn't available in a browser context, but the
 * `@bsull/augurs-prophet-wasmstan` package provides an `optimizer` function
 * which uses a version of Stan compiled to WebAssembly to run the exact
 * same procedure client-side, bypassing the filesystem.
 *
 * You will probably want to just use `@bsull/augurs-prophet-wasmstan`'s
 * `optimizer` export.
 *
 * @param init - The initial parameters for the optimization.
 * @param data - The data for the optimization.
 * @param opts - The optimization options.
 * @returns An object containing the the optimized parameters and any log
 *          messages.
 */
type OptimizerFunction = (init: InitialParams, data: Data, opts: OptimizeOptions) => OptimizeOutput;

/**
 * An optimizer for the Prophet model.
 */
interface Optimizer {
    optimize: OptimizerFunction;
}
"#;

/// An optimizer provided by Javascript.
///
/// This is used to optimize the Prophet model.
///
/// The `func` field is a Javascript function that takes the following arguments:
/// - init: the initial parameters for the optimization
/// - data: the data for the optimization
/// - opts: the optimization options
///
/// The function should return an object looking like the `OptimizedParams`
/// struct.
#[derive(Clone, Debug)]
struct JsOptimizer {
    func: js_sys::Function,
}

impl augurs_prophet::Optimizer for JsOptimizer {
    fn optimize(
        &self,
        init: &augurs_prophet::optimizer::InitialParams,
        data: &augurs_prophet::optimizer::Data,
        opts: &augurs_prophet::optimizer::OptimizeOpts,
    ) -> Result<augurs_prophet::optimizer::OptimizedParams, augurs_prophet::optimizer::Error> {
        let this = JsValue::null();
        let opts: OptimizeOptions = opts.into();
        let init: InitialParams<'_> = init.into();
        let data: Data<'_> = data.into();
        let init = serde_wasm_bindgen::to_value(&init)
            .map_err(augurs_prophet::optimizer::Error::custom)?;
        let data = serde_wasm_bindgen::to_value(&data)
            .map_err(augurs_prophet::optimizer::Error::custom)?;
        let opts = serde_wasm_bindgen::to_value(&opts)
            .map_err(augurs_prophet::optimizer::Error::custom)?;
        let result = self
            .func
            .call3(&this, &init, &data, &opts)
            .map_err(|x| augurs_prophet::optimizer::Error::string(format!("{:?}", x)))?;
        let result: OptimizeOutput = serde_wasm_bindgen::from_value(result)
            .map_err(augurs_prophet::optimizer::Error::custom)?;

        result.logs.emit();

        Ok(result.params.into())
    }
}

/// The [Prophet] time-series forecasting model.
///
/// Prophet is a forecasting procedure designed for automated forecasting
/// at scale with minimal manual input.
///
/// Create a new Prophet instance with the constructor, passing in an optimizer
/// and some other optional arguments.
///
/// # Example
///
/// ```javascript
/// import { Prophet } from '@bsull/augurs';
/// import { optimizer } from '@bsull/augurs-prophet-wasmstan';
///
/// const prophet = new Prophet({ optimizer });
/// const ds = [
///   1704067200n, 1704871384n, 1705675569n, 1706479753n, 1707283938n, 1708088123n,
///   1708892307n, 1709696492n, 1710500676n, 1711304861n, 1712109046n, 1712913230n,
/// ];
/// const y = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
/// const trainingData = { ds, y };
/// prophet.fit(trainingData);
/// const predictions = prophet.predict();
/// console.log(predictions.yhat);  // yhat is an object with 'point', 'lower' and 'upper'.
/// ```
///
/// [Prophet]: https://facebook.github.io/prophet/
#[derive(Debug)]
#[wasm_bindgen]
pub struct Prophet {
    inner: augurs_prophet::Prophet<JsOptimizer>,
}

#[wasm_bindgen]
impl Prophet {
    /// Create a new Prophet model.
    #[wasm_bindgen(constructor)]
    pub fn new(opts: ProphetOptions) -> Result<Prophet, JsError> {
        let (optimizer, opts): (JsOptimizer, augurs_prophet::OptProphetOptions) =
            opts.try_into()?;
        Ok(Self {
            inner: augurs_prophet::Prophet::new(opts.into(), optimizer),
        })
    }

    /// Fit the model to some training data.
    #[wasm_bindgen]
    pub fn fit(&mut self, data: TrainingData) -> Result<(), JsError> {
        Ok(self.inner.fit(data.try_into()?, Default::default())?)
    }

    /// Predict using the model.
    ///
    /// If `data` is omitted, predictions will be produced for the training data
    /// history.
    ///
    /// This will throw an exception if the model hasn't already been fit.
    #[wasm_bindgen]
    pub fn predict(&self, data: Option<PredictionData>) -> Result<Predictions, JsError> {
        let data: Option<augurs_prophet::PredictionData> =
            data.map(TryInto::try_into).transpose()?;
        Ok(self.inner.predict(data)?.into())
    }
}

// Intermediate structs with different serde implementations or field names, since
// the 'libprophet' / 'wasmstan' optimize function expects different field names.

/// Arguments for optimization.
#[derive(Debug, Clone, Serialize, Tsify)]
#[serde(rename_all = "camelCase")]
#[tsify(into_wasm_abi)]
struct OptimizeOptions {
    /// Algorithm to use.
    pub algorithm: Option<Algorithm>,
    /// The random seed to use for the optimization.
    pub seed: Option<u32>,
    /// The chain id to advance the PRNG.
    pub chain: Option<u32>,
    /// Line search step size for first iteration.
    pub init_alpha: Option<f64>,
    /// Convergence tolerance on changes in objective function value.
    pub tol_obj: Option<f64>,
    /// Convergence tolerance on relative changes in objective function value.
    pub tol_rel_obj: Option<f64>,
    /// Convergence tolerance on the norm of the gradient.
    pub tol_grad: Option<f64>,
    /// Convergence tolerance on the relative norm of the gradient.
    pub tol_rel_grad: Option<f64>,
    /// Convergence tolerance on changes in parameter value.
    pub tol_param: Option<f64>,
    /// Size of the history for LBFGS Hessian approximation. The value should
    /// be less than the dimensionality of the parameter space. 5-10 usually
    /// sufficient.
    pub history_size: Option<u32>,
    /// Total number of iterations.
    pub iter: Option<u32>,
    /// When `true`, use the Jacobian matrix to approximate the Hessian.
    /// Default is `false`.
    pub jacobian: Option<bool>,
    /// How frequently to emit convergence statistics, in number of iterations.
    pub refresh: Option<u32>,
}

impl From<&augurs_prophet::optimizer::OptimizeOpts> for OptimizeOptions {
    fn from(opts: &augurs_prophet::optimizer::OptimizeOpts) -> Self {
        Self {
            algorithm: opts.algorithm.map(Into::into),
            seed: opts.seed,
            chain: opts.chain,
            init_alpha: opts.init_alpha,
            tol_obj: opts.tol_obj,
            tol_rel_obj: opts.tol_rel_obj,
            tol_grad: opts.tol_grad,
            tol_rel_grad: opts.tol_rel_grad,
            tol_param: opts.tol_param,
            history_size: opts.history_size,
            iter: opts.iter,
            jacobian: opts.jacobian,
            refresh: opts.refresh,
        }
    }
}

/// The initial parameters for the optimization.
#[derive(Clone, Serialize, Tsify)]
#[serde(rename_all = "camelCase")]
#[tsify(into_wasm_abi)]
struct InitialParams<'a> {
    _phantom: std::marker::PhantomData<&'a ()>,
    /// Base trend growth rate.
    pub k: f64,
    /// Trend offset.
    pub m: f64,
    /// Trend rate adjustments, length s in data.
    #[tsify(type = "Float64Array")]
    #[serde(with = "serde_wasm_bindgen::preserve")]
    pub delta: Float64Array,
    /// Regressor coefficients, length k in data.
    #[tsify(type = "Float64Array")]
    #[serde(with = "serde_wasm_bindgen::preserve")]
    pub beta: Float64Array,
    /// Observation noise.
    pub sigma_obs: f64,
}

impl<'a> From<&'a augurs_prophet::optimizer::InitialParams> for InitialParams<'a> {
    fn from(params: &'a augurs_prophet::optimizer::InitialParams) -> Self {
        // SAFETY: We're creating a view of the `delta` field which has lifetime 'a.
        // The view is valid as long as the `InitialParams` is alive. Effectively
        // we're tying the lifetime of this struct to the lifetime of the input,
        // even though `Float64Array` doesn't have a lifetime.
        let delta = unsafe { Float64Array::view(&params.delta) };
        let beta = unsafe { Float64Array::view(&params.beta) };
        Self {
            _phantom: std::marker::PhantomData,
            k: params.k,
            m: params.m,
            delta,
            beta,
            sigma_obs: *params.sigma_obs,
        }
    }
}

/// The algorithm to use for optimization. One of: 'BFGS', 'LBFGS', 'Newton'.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Tsify)]
#[serde(rename_all = "lowercase")]
#[tsify(into_wasm_abi)]
pub enum Algorithm {
    /// Use the Newton algorithm.
    Newton,
    /// Use the Broyden-Fletcher-Goldfarb-Shanno (BFGS) algorithm.
    Bfgs,
    /// Use the Limited-memory BFGS (L-BFGS) algorithm.
    Lbfgs,
}

impl From<augurs_prophet::Algorithm> for Algorithm {
    fn from(value: augurs_prophet::Algorithm) -> Self {
        match value {
            augurs_prophet::Algorithm::Newton => Self::Newton,
            augurs_prophet::Algorithm::Bfgs => Self::Bfgs,
            augurs_prophet::Algorithm::Lbfgs => Self::Lbfgs,
        }
    }
}

/// The type of trend to use.
#[derive(Clone, Copy, Eq, Ord, PartialEq, PartialOrd, Serialize, Tsify)]
#[serde(rename_all = "camelCase")]
#[tsify(into_wasm_abi)]
enum TrendIndicator {
    /// Linear trend (default).
    Linear,
    /// Logistic trend.
    Logistic,
    /// Flat trend.
    Flat,
}

impl From<augurs_prophet::TrendIndicator> for TrendIndicator {
    fn from(value: augurs_prophet::TrendIndicator) -> Self {
        match value {
            augurs_prophet::TrendIndicator::Linear => Self::Linear,
            augurs_prophet::TrendIndicator::Logistic => Self::Logistic,
            augurs_prophet::TrendIndicator::Flat => Self::Flat,
        }
    }
}

/// Data for the Prophet model.
#[derive(Clone, Serialize, Tsify)]
#[serde(rename_all = "camelCase")]
#[tsify(into_wasm_abi)]
struct Data<'a> {
    _phantom: std::marker::PhantomData<&'a ()>,
    /// Number of time periods.
    /// This is `T` in the Prophet STAN model definition,
    /// but WIT identifiers must be lower kebab-case.
    pub n: i32,
    #[serde(with = "serde_wasm_bindgen::preserve")]
    /// Time series, length n.
    pub y: Float64Array,
    /// Time, length n.
    #[serde(with = "serde_wasm_bindgen::preserve")]
    #[tsify(type = "Float64Array")]
    pub t: Float64Array,
    /// Capacities for logistic trend, length n.
    #[serde(with = "serde_wasm_bindgen::preserve")]
    #[tsify(type = "Float64Array")]
    pub cap: Float64Array,
    /// Number of changepoints.
    /// This is 'S' in the Prophet STAN model definition,
    /// but WIT identifiers must be lower kebab-case.
    pub s: i32,
    /// Times of trend changepoints, length s.
    #[serde(with = "serde_wasm_bindgen::preserve")]
    #[tsify(type = "Float64Array")]
    pub t_change: Float64Array,
    /// The type of trend to use.
    pub trend_indicator: TrendIndicator,
    /// Number of regressors.
    /// Must be greater than or equal to 1.
    /// This is `K` in the Prophet STAN model definition,
    /// but WIT identifiers must be lower kebab-case.
    pub k: i32,
    /// Indicator of additive features, length k.
    /// This is `s_a` in the Prophet STAN model definition,
    /// but WIT identifiers must be lower kebab-case.
    #[serde(with = "serde_wasm_bindgen::preserve")]
    #[tsify(type = "Int32Array")]
    pub s_a: Int32Array,
    /// Indicator of multiplicative features, length k.
    /// This is `s_m` in the Prophet STAN model definition,
    /// but WIT identifiers must be lower kebab-case.
    #[serde(with = "serde_wasm_bindgen::preserve")]
    #[tsify(type = "Int32Array")]
    pub s_m: Int32Array,
    /// Regressors.
    /// This is `X` in the Prophet STAN model definition,
    /// but WIT identifiers must be lower kebab-case.
    /// This is passed as a flat array but should be treated as
    /// a matrix with shape (n, k) (i.e. strides of length n).
    #[serde(with = "serde_wasm_bindgen::preserve")]
    #[tsify(type = "Float64Array")]
    pub x: Float64Array,
    /// Scale on seasonality prior.
    #[serde(with = "serde_wasm_bindgen::preserve")]
    #[tsify(type = "Float64Array")]
    pub sigmas: Float64Array,
    /// Scale on changepoints prior.
    /// Must be greater than 0.
    pub tau: f64,
}

impl<'a> From<&'a augurs_prophet::optimizer::Data> for Data<'a> {
    fn from(data: &'a augurs_prophet::optimizer::Data) -> Self {
        let sigmas: Vec<_> = data.sigmas.iter().map(|x| **x).collect();
        // Be sure to use `from` instead of `view` here, as we've allocated
        // and the vec will be dropped at the end of this function.
        let sigmas = Float64Array::from(sigmas.as_slice());
        // SAFETY: We're creating a view of these fields which have lifetime 'a.
        // The view is valid as long as the `Data` is alive. Effectively
        // we're tying the lifetime of this struct to the lifetime of the input,
        // even though `Float64Array` doesn't have a lifetime.
        // We also have to be careful not to allocate after this!
        let y = unsafe { Float64Array::view(&data.y) };
        let t = unsafe { Float64Array::view(&data.t) };
        let cap = unsafe { Float64Array::view(&data.cap) };
        let t_change = unsafe { Float64Array::view(&data.t_change) };
        let s_a = unsafe { Int32Array::view(&data.s_a) };
        let s_m = unsafe { Int32Array::view(&data.s_m) };
        let x = unsafe { Float64Array::view(&data.X) };
        Self {
            _phantom: std::marker::PhantomData,
            n: data.T,
            y,
            t,
            cap,
            s: data.S,
            t_change,
            trend_indicator: data.trend_indicator.into(),
            k: data.K,
            s_a,
            s_m,
            x,
            sigmas,
            tau: *data.tau,
        }
    }
}

/// Log messages from the optimizer.
#[derive(Debug, Clone, Deserialize, Tsify)]
#[serde(rename_all = "camelCase")]
#[tsify(from_wasm_abi)]
struct Logs {
    /// Debug logs.
    pub debug: String,
    /// Info logs.
    pub info: String,
    /// Warning logs.
    pub warn: String,
    /// Error logs.
    pub error: String,
    /// Fatal logs.
    pub fatal: String,
}

impl Logs {
    fn emit(self) {
        for line in self.debug.lines() {
            tracing::trace!(target: "augurs::prophet::stan::optimize", "{}", line);
        }
        for line in self.info.lines().filter(|line| !line.contains("Iter")) {
            match ConvergenceLog::new(line) {
                Some(log) => {
                    tracing::debug!(
                        target: "augurs::prophet::stan::optimize::progress",
                        iter = log.iter,
                        log_prob = log.log_prob,
                        dx = log.dx,
                        grad = log.grad,
                        alpha = log.alpha,
                        alpha0 = log.alpha0,
                        evals = log.evals,
                        notes = log.notes,
                    );
                }
                None => {
                    tracing::debug!(target: "augurs::prophet::stan::optimize", "{}", line);
                }
            }
        }
        for line in self.warn.lines() {
            tracing::warn!(target: "augurs::prophet::stan::optimize", "{}", line);
        }
        for line in self.error.lines() {
            tracing::error!(target: "augurs::prophet::stan::optimize", "{}", line);
        }
        for line in self.fatal.lines() {
            tracing::error!(target: "augurs::prophet::stan::optimize", "{}", line);
        }
    }
}

/// The output of the optimizer.
#[derive(Debug, Clone, Deserialize, Tsify)]
#[serde(rename_all = "camelCase")]
#[tsify(from_wasm_abi)]
struct OptimizeOutput {
    /// Logs emitted by the optimizer, split by log level.
    pub logs: Logs,
    /// The optimized parameters.
    pub params: OptimizedParams,
}

/// The optimal parameters found by the optimizer.
#[derive(Debug, Clone, Deserialize, Tsify)]
#[serde(rename_all = "camelCase")]
#[tsify(from_wasm_abi)]
struct OptimizedParams {
    /// Base trend growth rate.
    pub k: f64,
    /// Trend offset.
    pub m: f64,
    /// Observation noise.
    #[tsify(type = "number")]
    pub sigma_obs: PositiveFloat,
    /// Trend rate adjustments.
    #[tsify(type = "Float64Array")]
    pub delta: Vec<f64>,
    /// Regressor coefficients.
    #[tsify(type = "Float64Array")]
    pub beta: Vec<f64>,
    /// Transformed trend.
    #[tsify(type = "Float64Array")]
    pub trend: Vec<f64>,
}

impl From<OptimizedParams> for augurs_prophet::optimizer::OptimizedParams {
    fn from(x: OptimizedParams) -> Self {
        augurs_prophet::optimizer::OptimizedParams {
            k: x.k,
            m: x.m,
            sigma_obs: x.sigma_obs,
            delta: x.delta,
            beta: x.beta,
            trend: x.trend,
        }
    }
}

#[tsify_next::declare]
type TimestampSeconds = i64;

/// The data needed to train a Prophet model.
///
/// Seasonality conditions, regressors,
/// floor and cap columns.
#[derive(Clone, Debug, Default, Deserialize, Tsify)]
#[serde(rename_all = "camelCase")]
#[tsify(from_wasm_abi)]
pub struct TrainingData {
    pub ds: Vec<TimestampSeconds>,
    pub y: Vec<f64>,
    #[tsify(optional)]
    pub cap: Option<Vec<f64>>,
    #[tsify(optional)]
    pub floor: Option<Vec<f64>>,
    #[tsify(optional)]
    pub seasonality_conditions: Option<HashMap<String, Vec<bool>>>,
    #[tsify(optional)]
    pub x: Option<HashMap<String, Vec<f64>>>,
}

impl TryFrom<TrainingData> for augurs_prophet::TrainingData {
    type Error = augurs_prophet::Error;

    fn try_from(value: TrainingData) -> Result<Self, Self::Error> {
        let mut td = Self::new(value.ds, value.y)?;
        if let Some(cap) = value.cap {
            td = td.with_cap(cap)?;
        }
        if let Some(floor) = value.floor {
            td = td.with_floor(floor)?;
        }
        if let Some(seasonality_conditions) = value.seasonality_conditions {
            td = td.with_seasonality_conditions(seasonality_conditions)?;
        }
        if let Some(x) = value.x {
            td = td.with_regressors(x)?;
        }
        Ok(td)
    }
}

/// The data needed to predict with a Prophet model.
///
/// The structure of the prediction data must be the same as the
/// training data used to train the model, with the exception of
/// `y` (which is being predicted).
///
/// That is, if your model used certain seasonality conditions or
/// regressors, you must include them in the prediction data.
#[derive(Clone, Debug, Default, Deserialize, Tsify)]
#[serde(rename_all = "camelCase")]
#[tsify(from_wasm_abi)]
pub struct PredictionData {
    pub ds: Vec<TimestampSeconds>,
    #[tsify(optional)]
    pub cap: Option<Vec<f64>>,
    #[tsify(optional)]
    pub floor: Option<Vec<f64>>,
    #[tsify(optional)]
    pub seasonality_conditions: Option<HashMap<String, Vec<bool>>>,
    #[tsify(optional)]
    pub x: Option<HashMap<String, Vec<f64>>>,
}

impl TryFrom<PredictionData> for augurs_prophet::PredictionData {
    type Error = augurs_prophet::Error;

    fn try_from(value: PredictionData) -> Result<Self, Self::Error> {
        let mut pd = Self::new(value.ds);
        if let Some(cap) = value.cap {
            pd = pd.with_cap(cap)?;
        }
        if let Some(floor) = value.floor {
            pd = pd.with_floor(floor)?;
        }
        if let Some(seasonality_conditions) = value.seasonality_conditions {
            pd = pd.with_seasonality_conditions(seasonality_conditions)?;
        }
        if let Some(x) = value.x {
            pd = pd.with_regressors(x)?;
        }
        Ok(pd)
    }
}

/// The prediction for a feature.
///
/// 'Feature' could refer to the forecasts themselves (`yhat`)
/// or any of the other component features which contribute to
/// the final estimate, such as trend, seasonality, seasonalities,
/// regressors or holidays.
#[derive(Clone, Debug, Default, Serialize, Tsify)]
#[serde(rename_all = "camelCase")]
#[tsify(into_wasm_abi)]
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

impl From<augurs_prophet::FeaturePrediction> for FeaturePrediction {
    fn from(value: augurs_prophet::FeaturePrediction) -> Self {
        Self {
            point: value.point,
            lower: value.lower,
            upper: value.upper,
        }
    }
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
#[derive(Clone, Debug, Default, Serialize, Tsify)]
#[serde(rename_all = "camelCase")]
#[tsify(into_wasm_abi)]
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

impl From<augurs_prophet::Predictions> for Predictions {
    fn from(value: augurs_prophet::Predictions) -> Self {
        Self {
            ds: value.ds,
            yhat: value.yhat.into(),
            trend: value.trend.into(),
            cap: value.cap,
            floor: value.floor,
            additive: value.additive.into(),
            multiplicative: value.multiplicative.into(),
            holidays: value
                .holidays
                .into_iter()
                .map(|(k, v)| (k, v.into()))
                .collect(),
            seasonalities: value
                .seasonalities
                .into_iter()
                .map(|(k, v)| (k, v.into()))
                .collect(),
            regressors: value
                .regressors
                .into_iter()
                .map(|(k, v)| (k, v.into()))
                .collect(),
        }
    }
}

/// Options for Prophet, after applying defaults.
///
/// The only required field is `optimizer`. See the documentation for
/// `Optimizer` for more details.
///
/// All other options are treated exactly the same as the original
/// Prophet library; see its [documentation] for more detail.
///
/// [documentation]: https://facebook.github.io/prophet/docs/quick_start.html
#[derive(Clone, Debug, Default, Deserialize, Tsify)]
#[serde(rename_all = "camelCase")]
#[tsify(from_wasm_abi)]
pub struct ProphetOptions {
    /// Optimizer, used to find the maximum likelihood estimate of the
    /// Prophet Stan model parameters.
    ///
    /// See the documentation for `ProphetOptions` for more details.
    #[serde(with = "serde_wasm_bindgen::preserve")]
    #[tsify(type = "Optimizer")]
    pub optimizer: js_sys::Object,

    /// The type of growth (trend) to use.
    ///
    /// Defaults to [`GrowthType::Linear`].
    #[tsify(optional)]
    pub growth: Option<GrowthType>,

    /// An optional list of changepoints.
    ///
    /// If not provided, changepoints will be automatically selected.
    #[tsify(optional)]
    pub changepoints: Option<Vec<TimestampSeconds>>,

    /// The number of potential changepoints to include.
    ///
    /// Not used if `changepoints` is provided.
    ///
    /// If provided and `changepoints` is not provided, then
    /// `n_changepoints` potential changepoints will be selected
    /// uniformly from the first `changepoint_range` proportion of
    /// the history.
    ///
    /// Defaults to 25.
    #[tsify(optional)]
    pub n_changepoints: Option<u32>,

    /// The proportion of the history to consider for potential changepoints.
    ///
    /// Not used if `changepoints` is provided.
    ///
    /// Defaults to `0.8` for the first 80% of the data.
    #[tsify(optional)]
    pub changepoint_range: Option<f64>,

    /// How to fit yearly seasonality.
    ///
    /// Defaults to [`SeasonalityOption::Auto`].
    #[tsify(optional)]
    pub yearly_seasonality: Option<SeasonalityOption>,
    /// How to fit weekly seasonality.
    ///
    /// Defaults to [`SeasonalityOption::Auto`].
    #[tsify(optional)]
    pub weekly_seasonality: Option<SeasonalityOption>,
    /// How to fit daily seasonality.
    ///
    /// Defaults to [`SeasonalityOption::Auto`].
    #[tsify(optional)]
    pub daily_seasonality: Option<SeasonalityOption>,

    /// How to model seasonality.
    ///
    /// Defaults to [`FeatureMode::Additive`].
    #[tsify(optional)]
    pub seasonality_mode: Option<FeatureMode>,

    /// The prior scale for seasonality.
    ///
    /// This modulates the strength of seasonality,
    /// with larger values allowing the model to fit
    /// larger seasonal fluctuations and smaller values
    /// dampening the seasonality.
    ///
    /// Can be specified for individual seasonalities
    /// using [`Prophet::add_seasonality`](crate::Prophet::add_seasonality).
    ///
    /// Defaults to `10.0`.
    #[tsify(optional)]
    pub seasonality_prior_scale: Option<f64>,

    /// The prior scale for changepoints.
    ///
    /// This modulates the flexibility of the automatic
    /// changepoint selection. Large values will allow many
    /// changepoints, while small values will allow few
    /// changepoints.
    ///
    /// Defaults to `0.05`.
    #[tsify(optional)]
    pub changepoint_prior_scale: Option<f64>,

    /// How to perform parameter estimation.
    ///
    /// When [`EstimationMode::Mle`] or [`EstimationMode::Map`]
    /// are used then no MCMC samples are taken.
    ///
    /// Defaults to [`EstimationMode::Mle`].
    #[tsify(optional)]
    pub estimation: Option<EstimationMode>,

    /// The width of the uncertainty intervals.
    ///
    /// Must be between `0.0` and `1.0`. Common values are
    /// `0.8` (80%), `0.9` (90%) and `0.95` (95%).
    ///
    /// Defaults to `0.8` for 80% intervals.
    #[tsify(optional)]
    pub interval_width: Option<f64>,

    /// The number of simulated draws used to estimate uncertainty intervals.
    ///
    /// Setting this value to `0` will disable uncertainty
    /// estimation and speed up the calculation.
    ///
    /// Defaults to `1000`.
    #[tsify(optional)]
    pub uncertainty_samples: Option<u32>,

    /// How to scale the data prior to fitting the model.
    ///
    /// Defaults to [`Scaling::AbsMax`].
    #[tsify(optional)]
    pub scaling: Option<Scaling>,

    /// Holidays to include in the model.
    #[tsify(optional)]
    pub holidays: Option<HashMap<String, Holiday>>,
    /// Prior scale for holidays.
    ///
    /// This parameter modulates the strength of the holiday
    /// components model, unless overridden in each individual
    /// holiday's input.
    ///
    /// Defaults to `100.0`.
    #[tsify(optional)]
    pub holidays_prior_scale: Option<f64>,

    /// How to model holidays.
    ///
    /// Defaults to the same value as [`ProphetOptions::seasonality_mode`].
    #[tsify(optional)]
    pub holidays_mode: Option<FeatureMode>,
}

impl TryFrom<ProphetOptions> for (JsOptimizer, augurs_prophet::OptProphetOptions) {
    type Error = JsError;

    fn try_from(value: ProphetOptions) -> Result<Self, Self::Error> {
        let Ok(val) = js_sys::Reflect::get(&value.optimizer, &js_sys::JsString::from("optimize"))
        else {
            return Err(JsError::new("optimizer does not have `optimize` property"));
        };
        if !val.is_function() {
            return Err(JsError::new("optimizer.optimize is not a function"));
        };
        let optimizer = JsOptimizer { func: val.into() };
        let opts = augurs_prophet::OptProphetOptions {
            growth: value.growth.map(Into::into),
            changepoints: value.changepoints,
            n_changepoints: value.n_changepoints,
            changepoint_range: value.changepoint_range.map(TryInto::try_into).transpose()?,
            yearly_seasonality: value
                .yearly_seasonality
                .map(TryInto::try_into)
                .transpose()?,
            weekly_seasonality: value
                .weekly_seasonality
                .map(TryInto::try_into)
                .transpose()?,
            daily_seasonality: value.daily_seasonality.map(TryInto::try_into).transpose()?,
            seasonality_mode: value.seasonality_mode.map(Into::into),
            seasonality_prior_scale: value
                .seasonality_prior_scale
                .map(TryInto::try_into)
                .transpose()?,
            changepoint_prior_scale: value
                .changepoint_prior_scale
                .map(TryInto::try_into)
                .transpose()?,
            estimation: value.estimation.map(Into::into),
            interval_width: value.interval_width.map(TryInto::try_into).transpose()?,
            uncertainty_samples: value.uncertainty_samples,
            scaling: value.scaling.map(Into::into),
            holidays: value
                .holidays
                .map(|x| {
                    x.into_iter()
                        .map(|(k, v)| Ok((k, v.try_into()?)))
                        .collect::<Result<_, JsError>>()
                })
                .transpose()?,
            holidays_prior_scale: value
                .holidays_prior_scale
                .map(TryInto::try_into)
                .transpose()?,
            holidays_mode: value.holidays_mode.map(TryInto::try_into).transpose()?,
        };
        Ok((optimizer, opts))
    }
}

/// The type of growth to use.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Default, Deserialize, Tsify)]
#[serde(rename_all = "camelCase")]
#[tsify(from_wasm_abi)]
pub enum GrowthType {
    /// Linear growth (default).
    #[default]
    Linear,
    /// Logistic growth.
    Logistic,
    /// Flat growth.
    Flat,
}

impl From<GrowthType> for augurs_prophet::GrowthType {
    fn from(value: GrowthType) -> Self {
        match value {
            GrowthType::Linear => Self::Linear,
            GrowthType::Logistic => Self::Logistic,
            GrowthType::Flat => Self::Flat,
        }
    }
}

/// Define whether to include a specific seasonality, and how it should be specified.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Default, Deserialize, Tsify)]
#[serde(rename_all = "camelCase")]
#[tsify(from_wasm_abi)]
pub enum SeasonalityOption {
    /// Automatically determine whether to include this seasonality.
    ///
    /// Yearly seasonality is automatically included if there is >=2
    /// years of history.
    ///
    /// Weekly seasonality is automatically included if there is >=2
    /// weeks of history, and the spacing between the dates in the
    /// data is <7 days.
    ///
    /// Daily seasonality is automatically included if there is >=2
    /// days of history, and the spacing between the dates in the
    /// data is <1 day.
    #[default]
    Auto,
    /// Manually specify whether to include this seasonality.
    Manual(bool),
    /// Enable this seasonality and use the provided number of Fourier terms.
    Fourier(u32),
}

impl TryFrom<SeasonalityOption> for augurs_prophet::SeasonalityOption {
    type Error = TryFromIntError;

    fn try_from(value: SeasonalityOption) -> Result<Self, Self::Error> {
        match value {
            SeasonalityOption::Auto => Ok(Self::Auto),
            SeasonalityOption::Manual(b) => Ok(Self::Manual(b)),
            SeasonalityOption::Fourier(n) => Ok(Self::Fourier(n.try_into()?)),
        }
    }
}

/// How to scale the data prior to fitting the model.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Default, Deserialize, Tsify)]
#[serde(rename_all = "camelCase")]
#[tsify(from_wasm_abi)]
pub enum Scaling {
    /// Use abs-max scaling (the default).
    #[default]
    AbsMax,
    /// Use min-max scaling.
    MinMax,
}

impl From<Scaling> for augurs_prophet::Scaling {
    fn from(value: Scaling) -> Self {
        match value {
            Scaling::AbsMax => Self::AbsMax,
            Scaling::MinMax => Self::MinMax,
        }
    }
}

/// How to do parameter estimation.
///
/// Note: for now, only MLE/MAP estimation is supported, i.e. there
/// is no support for MCMC sampling. This will be added in the future!
/// The enum will be marked as `non_exhaustive` until that point.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Default, Deserialize, Tsify)]
#[serde(rename_all = "camelCase")]
#[tsify(from_wasm_abi)]
pub enum EstimationMode {
    /// Use MLE estimation.
    #[default]
    Mle,
    /// Use MAP estimation.
    Map,
    // This is not yet implemented. We need to add a new `Sampler` trait and
    // implement it, then handle the different number outputs when predicting,
    // before this can be enabled.
    // /// Do full Bayesian inference with the specified number of MCMC samples.
    //
    // Mcmc(u32),
}

impl From<EstimationMode> for augurs_prophet::EstimationMode {
    fn from(value: EstimationMode) -> Self {
        match value {
            EstimationMode::Mle => Self::Mle,
            EstimationMode::Map => Self::Map,
        }
    }
}

/// The mode of a seasonality, regressor, or holiday.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Default, Deserialize, Tsify)]
#[serde(rename_all = "camelCase")]
#[tsify(from_wasm_abi)]
pub enum FeatureMode {
    /// Additive mode.
    #[default]
    Additive,
    /// Multiplicative mode.
    Multiplicative,
}

impl From<FeatureMode> for augurs_prophet::FeatureMode {
    fn from(value: FeatureMode) -> Self {
        match value {
            FeatureMode::Additive => Self::Additive,
            FeatureMode::Multiplicative => Self::Multiplicative,
        }
    }
}

/// A holiday to be considered by the Prophet model.
#[derive(Clone, Debug, Deserialize, Tsify)]
#[serde(rename_all = "camelCase")]
#[tsify(from_wasm_abi)]
pub struct Holiday {
    /// The dates of the holiday.
    pub ds: Vec<TimestampSeconds>,

    /// The lower window for the holiday.
    ///
    /// The lower window is the number of days before the holiday
    /// that it is observed. For example, if the holiday is on
    /// 2023-01-01 and the lower window is -1, then the holiday will
    /// _also_ be observed on 2022-12-31.
    pub lower_window: Option<Vec<i32>>,

    /// The upper window for the holiday.
    ///
    /// The upper window is the number of days after the holiday
    /// that it is observed. For example, if the holiday is on
    /// 2023-01-01 and the upper window is 1, then the holiday will
    /// _also_ be observed on 2023-01-02.
    pub upper_window: Option<Vec<i32>>,

    /// The prior scale for the holiday.
    pub prior_scale: Option<f64>,
}

impl TryFrom<Holiday> for augurs_prophet::Holiday {
    type Error = JsError;

    fn try_from(value: Holiday) -> Result<Self, Self::Error> {
        let mut holiday = Self::new(value.ds);
        if let Some(lower_window) = value.lower_window {
            holiday = holiday.with_lower_window(lower_window)?;
        }
        if let Some(upper_window) = value.upper_window {
            holiday = holiday.with_upper_window(upper_window)?;
        }
        if let Some(prior_scale) = value.prior_scale {
            holiday = holiday.with_prior_scale(prior_scale.try_into()?);
        }
        Ok(holiday)
    }
}

/// Struct representing a convergence log line, such as
/// `      718       140.457   7.62967e-09       191.549      0.8667      0.8667      809`
struct ConvergenceLog<'a> {
    iter: usize,
    log_prob: f64,
    dx: f64,
    grad: f64,
    alpha: f64,
    alpha0: f64,
    evals: usize,
    notes: &'a str,
}

impl<'a> ConvergenceLog<'a> {
    fn new(s: &'a str) -> Option<Self> {
        let mut split = s.split_whitespace();
        let iter = split.next()?.parse::<usize>().ok()?;
        let log_prob = split.next()?.parse::<f64>().ok()?;
        let dx = split.next()?.parse::<f64>().ok()?;
        let grad = split.next()?.parse::<f64>().ok()?;
        let alpha = split.next()?.parse::<f64>().ok()?;
        let alpha0 = split.next()?.parse::<f64>().ok()?;
        let evals = split.next()?.parse::<usize>().ok()?;
        let notes = split.next().unwrap_or_default();
        Some(Self {
            iter,
            log_prob,
            dx,
            grad,
            alpha,
            alpha0,
            evals,
            notes,
        })
    }
}
