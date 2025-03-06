//! JS bindings for the Prophet model.
use std::{
    collections::HashMap,
    num::{NonZeroU32, TryFromIntError},
};

use js_sys::Float64Array;
use serde::{Deserialize, Serialize};
use tsify_next::Tsify;
use wasm_bindgen::prelude::*;

use augurs_core_js::{Forecast, ForecastIntervals};
use augurs_prophet::PositiveFloat;

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
 * @param data - JSON representation of the data for the optimization.
 * @param opts - The optimization options.
 * @returns An object containing the the optimized parameters and any log
 *          messages.
 */
type ProphetOptimizerFunction = (init: ProphetInitialParams, data: ProphetStanDataJSON, opts: ProphetOptimizeOptions) => ProphetOptimizeOutput;

/**
 * An optimizer for the Prophet model.
 */
interface ProphetOptimizer {
    optimize: ProphetOptimizerFunction;
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
        let init: InitialParams = init.into();
        let init = serde_wasm_bindgen::to_value(&init)
            .map_err(augurs_prophet::optimizer::Error::custom)?;
        let data =
            serde_json::to_string(&data).map_err(augurs_prophet::optimizer::Error::custom)?;
        let data_s = JsValue::from_str(&data);
        let opts = serde_wasm_bindgen::to_value(&opts)
            .map_err(augurs_prophet::optimizer::Error::custom)?;
        let result = self
            .func
            .call3(&this, &init, &data_s, &opts)
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
    level: Option<f64>,
}

#[wasm_bindgen]
impl Prophet {
    /// Create a new Prophet model.
    #[wasm_bindgen(constructor)]
    pub fn new(opts: Options) -> Result<Prophet, JsError> {
        let (optimizer, opts): (JsOptimizer, augurs_prophet::OptProphetOptions) =
            opts.try_into()?;
        let level = opts.interval_width.map(Into::into);
        Ok(Self {
            inner: augurs_prophet::Prophet::new(opts.into(), optimizer),
            level,
        })
    }

    /// Fit the model to some training data.
    #[wasm_bindgen]
    pub fn fit(
        &mut self,
        data: TrainingData,
        opts: Option<OptimizeOptions>,
    ) -> Result<(), JsError> {
        Ok(self
            .inner
            .fit(data.try_into()?, opts.map(Into::into).unwrap_or_default())?)
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
        let predictions = self.inner.predict(data)?;
        Ok(Predictions::from((self.level, predictions)))
    }

    /// Add a custom seasonality to the model
    #[wasm_bindgen(js_name = "addSeasonality")]
    pub fn add_seasonality(
        &mut self,
        name: String,
        seasonality: Seasonality,
    ) -> Result<(), JsError> {
        self.inner.add_seasonality(name, seasonality.into())?;
        Ok(())
    }

    /// Add a regressor to the model. Name should be one of the column names.
    /// The extra regressor must be known for both the history and for future dates.
    #[wasm_bindgen(js_name = "addRegressor")]
    pub fn add_regressor(
        &mut self,
        name: String,
        regressor: Option<Regressor>,
    ) -> Result<(), JsError> {
        self.inner
            .add_regressor(name, regressor.unwrap_or_default().into());
        Ok(())
    }
}

/// Can be used to specify custom seasonality
#[derive(Debug, Clone, Deserialize, Serialize, Tsify)]
#[serde(rename_all = "camelCase")]
#[tsify(from_wasm_abi, into_wasm_abi, type_prefix = "Prophet")]
pub struct Seasonality {
    /// The period of the seasonality we expect the time series to have.
    /// For e.g. 365.25 for yearly seasonality and 7 for weekly seasonality,
    /// when the time variable is scaled to days
    #[tsify(type = "number")]
    period: PositiveFloat,

    /// Fourier order for the seasonality.
    /// Increasing this allows for fitting seasonal patterns that change more quickly,
    /// albeit with increased risk of overfitting.
    #[tsify(type = "number")]
    fourier_order: NonZeroU32,

    /// The prior scale for the seasonality.
    /// A large value allows the seasonality to fit large fluctuations,
    /// a small value shrinks the magnitude of the seasonality.
    #[tsify(optional, type = "number")]
    prior_scale: Option<PositiveFloat>,

    /// The mode of the seasonality.
    #[tsify(optional)]
    mode: Option<FeatureMode>,

    /// The condition column name for the seasonality.
    /// The seasonality will only be applied to dates where the condition name column is True
    #[tsify(optional)]
    condition_name: Option<String>,
}

impl From<Seasonality> for augurs_prophet::Seasonality {
    fn from(seasonality: Seasonality) -> Self {
        let mut s = Self::new(seasonality.period, seasonality.fourier_order);
        if let Some(ps) = seasonality.prior_scale {
            s = s.with_prior_scale(ps);
        }
        if let Some(mode) = seasonality.mode {
            s = s.with_mode(mode.into());
        }
        if let Some(condition_name) = seasonality.condition_name {
            s = s.with_condition(condition_name);
        }
        s
    }
}

/// Can be used to specify custom regressors
#[derive(Clone, Debug, Default, Deserialize, Serialize, Tsify)]
#[serde(rename_all = "camelCase")]
#[tsify(from_wasm_abi, into_wasm_abi, type_prefix = "Prophet")]
pub struct Regressor {
    /// The mode of this regressor.
    ///
    /// Defaults to "additive".
    #[serde(default)]
    mode: FeatureMode,

    /// The prior scale of this regressor.
    ///
    /// Defaults to the `seasonality_prior_scale` of the Prophet model
    /// the regressor is added to.
    #[tsify(optional, type = "number")]
    prior_scale: Option<PositiveFloat>,

    /// Whether to standardize this regressor.
    ///
    /// The default is to use automatic standardization, which will standardize
    /// numeric regressors and leave binary regressors alone.
    #[serde(default)]
    standardize: Standardize,
}

impl From<Regressor> for augurs_prophet::Regressor {
    fn from(regressor: Regressor) -> Self {
        let mut r;
        match regressor.mode {
            FeatureMode::Additive => r = Self::additive(),
            FeatureMode::Multiplicative => r = Self::multiplicative(),
        }
        if let Some(ps) = regressor.prior_scale {
            r = r.with_prior_scale(ps);
        }
        r = r.with_standardize(regressor.standardize.into());
        r
    }
}

/// Whether to standardize a regressor.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Default, Deserialize, Serialize, Tsify)]
#[serde(rename_all = "camelCase")]
#[tsify(from_wasm_abi, type_prefix = "Prophet")]
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

impl From<Standardize> for augurs_prophet::Standardize {
    fn from(standardize: Standardize) -> Self {
        match standardize {
            Standardize::Auto => Self::Auto,
            Standardize::Yes => Self::Yes,
            Standardize::No => Self::No,
        }
    }
}

// Intermediate structs with different serde implementations or field names, since
// the 'libprophet' / 'wasmstan' optimize function expects different field names.

/// Arguments for optimization.
#[derive(Debug, Clone, Deserialize, Serialize, Tsify)]
#[serde(rename_all = "camelCase")]
#[tsify(from_wasm_abi, into_wasm_abi, type_prefix = "Prophet")]
pub struct OptimizeOptions {
    /// Algorithm to use.
    #[tsify(optional)]
    pub algorithm: Option<Algorithm>,
    /// The random seed to use for the optimization.
    #[tsify(optional)]
    pub seed: Option<u32>,
    /// The chain id to advance the PRNG.
    #[tsify(optional)]
    pub chain: Option<u32>,
    /// Line search step size for first iteration.
    #[tsify(optional)]
    pub init_alpha: Option<f64>,
    /// Convergence tolerance on changes in objective function value.
    #[tsify(optional)]
    pub tol_obj: Option<f64>,
    /// Convergence tolerance on relative changes in objective function value.
    #[tsify(optional)]
    pub tol_rel_obj: Option<f64>,
    /// Convergence tolerance on the norm of the gradient.
    #[tsify(optional)]
    pub tol_grad: Option<f64>,
    /// Convergence tolerance on the relative norm of the gradient.
    #[tsify(optional)]
    pub tol_rel_grad: Option<f64>,
    /// Convergence tolerance on changes in parameter value.
    #[tsify(optional)]
    pub tol_param: Option<f64>,
    /// Size of the history for LBFGS Hessian approximation. The value should
    /// be less than the dimensionality of the parameter space. 5-10 usually
    /// sufficient.
    #[tsify(optional)]
    pub history_size: Option<u32>,
    /// Total number of iterations.
    #[tsify(optional)]
    pub iter: Option<u32>,
    /// When `true`, use the Jacobian matrix to approximate the Hessian.
    /// Default is `false`.
    #[tsify(optional)]
    pub jacobian: Option<bool>,
    /// How frequently to emit convergence statistics, in number of iterations.
    #[tsify(optional)]
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

impl From<OptimizeOptions> for augurs_prophet::optimizer::OptimizeOpts {
    fn from(opts: OptimizeOptions) -> Self {
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
#[tsify(into_wasm_abi, type_prefix = "Prophet")]
struct InitialParams {
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

impl From<&augurs_prophet::optimizer::InitialParams> for InitialParams {
    fn from(params: &augurs_prophet::optimizer::InitialParams) -> Self {
        let delta = Float64Array::new_with_length(params.delta.len() as u32);
        delta.copy_from(&params.delta);
        let beta = Float64Array::new_with_length(params.beta.len() as u32);
        beta.copy_from(&params.beta);
        Self {
            k: params.k,
            m: params.m,
            delta,
            beta,
            sigma_obs: *params.sigma_obs,
        }
    }
}

/// The algorithm to use for optimization. One of: 'BFGS', 'LBFGS', 'Newton'.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize, Tsify)]
#[serde(rename_all = "lowercase")]
#[tsify(into_wasm_abi, type_prefix = "Prophet")]
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

impl From<Algorithm> for augurs_prophet::Algorithm {
    fn from(value: Algorithm) -> Self {
        match value {
            Algorithm::Newton => Self::Newton,
            Algorithm::Bfgs => Self::Bfgs,
            Algorithm::Lbfgs => Self::Lbfgs,
        }
    }
}

/// The type of trend to use.
#[derive(Debug, Clone, Copy, Eq, Ord, PartialEq, PartialOrd, Serialize, Tsify)]
#[serde(rename_all = "camelCase")]
#[tsify(into_wasm_abi, type_prefix = "Prophet")]
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
// Copy/pasted from augurs-prophet/src/optimizer.rs, only used
// here for the generated TS definitions, and hopefully temporarily.
#[derive(Clone, Debug, PartialEq, Serialize, Tsify)]
#[allow(non_snake_case)]
#[tsify(into_wasm_abi, type_prefix = "ProphetStan")]
pub struct Data {
    /// Number of time periods.
    pub T: i32,
    /// Time series, length n.
    pub y: Vec<f64>,
    /// Time, length n.
    pub t: Vec<f64>,
    /// Capacities for logistic trend, length n.
    pub cap: Vec<f64>,
    /// Number of changepoints.
    pub S: i32,
    /// Times of trend changepoints, length s.
    pub t_change: Vec<f64>,
    /// The type of trend to use.
    ///
    /// Possible values are:
    /// - 0 for linear trend
    /// - 1 for logistic trend
    /// - 2 for flat trend.
    pub trend_indicator: u8,
    /// Number of regressors.
    ///
    /// Must be greater than or equal to 1.
    pub K: i32,
    /// Indicator of additive features, length k.
    pub s_a: Vec<i32>,
    /// Indicator of multiplicative features, length k.
    pub s_m: Vec<i32>,
    /// Regressors, shape (n, k).
    pub X: Vec<f64>,
    /// Scale on seasonality prior.
    ///
    /// Must all be greater than zero.
    pub sigmas: Vec<f64>,
    /// Scale on changepoints prior.
    /// Must be greater than 0.
    pub tau: f64,
}

/// Data for the Prophet Stan model, in JSON format.
///
/// The JSON should represent an object of type `ProphetStanData`.
#[tsify_next::declare]
#[allow(dead_code)]
type ProphetStanDataJSON = String;

// This is unused as of #145 because it's difficult to
// use this struct correctly from C++ in the WASM component.
// We're just passing JSON instead, which is not great at all
// but at least works correctly. In future I'd like to reuse
// it, hence commenting rather than deleting.

// /// Data for the Prophet model.
// #[derive(Clone, Serialize, Tsify)]
// #[serde(rename_all = "camelCase")]
// #[tsify(into_wasm_abi, type_prefix = "ProphetStan")]
// struct Data<'a> {
//     _phantom: std::marker::PhantomData<&'a ()>,
//     /// Number of time periods.
//     /// This is `T` in the Prophet STAN model definition,
//     /// but WIT identifiers must be lower kebab-case.
//     pub n: i32,
//     #[serde(with = "serde_wasm_bindgen::preserve")]
//     #[tsify(type = "Float64Array")]
//     /// Time series, length n.
//     pub y: Float64Array,
//     /// Time, length n.
//     #[serde(with = "serde_wasm_bindgen::preserve")]
//     #[tsify(type = "Float64Array")]
//     pub t: Float64Array,
//     /// Capacities for logistic trend, length n.
//     #[serde(with = "serde_wasm_bindgen::preserve")]
//     #[tsify(type = "Float64Array")]
//     pub cap: Float64Array,
//     /// Number of changepoints.
//     /// This is 'S' in the Prophet STAN model definition,
//     /// but WIT identifiers must be lower kebab-case.
//     pub s: i32,
//     /// Times of trend changepoints, length s.
//     #[serde(with = "serde_wasm_bindgen::preserve")]
//     #[tsify(type = "Float64Array")]
//     pub t_change: Float64Array,
//     /// The type of trend to use.
//     #[tsify(type = "ProphetTrendIndicator")]
//     pub trend_indicator: TrendIndicator,
//     /// Number of regressors.
//     /// Must be greater than or equal to 1.
//     /// This is `K` in the Prophet STAN model definition,
//     /// but WIT identifiers must be lower kebab-case.
//     pub k: i32,
//     /// Indicator of additive features, length k.
//     /// This is `s_a` in the Prophet STAN model definition,
//     /// but WIT identifiers must be lower kebab-case.
//     #[serde(with = "serde_wasm_bindgen::preserve")]
//     #[tsify(type = "Int32Array")]
//     pub s_a: Int32Array,
//     /// Indicator of multiplicative features, length k.
//     /// This is `s_m` in the Prophet STAN model definition,
//     /// but WIT identifiers must be lower kebab-case.
//     #[serde(with = "serde_wasm_bindgen::preserve")]
//     #[tsify(type = "Int32Array")]
//     pub s_m: Int32Array,
//     /// Regressors.
//     /// This is `X` in the Prophet STAN model definition,
//     /// but WIT identifiers must be lower kebab-case.
//     /// This is passed as a flat array but should be treated as
//     /// a matrix with shape (n, k) (i.e. strides of length n).
//     #[serde(with = "serde_wasm_bindgen::preserve")]
//     #[tsify(type = "Float64Array")]
//     pub x: Float64Array,
//     /// Scale on seasonality prior.
//     #[serde(with = "serde_wasm_bindgen::preserve")]
//     #[tsify(type = "Float64Array")]
//     pub sigmas: Float64Array,
//     /// Scale on changepoints prior.
//     /// Must be greater than 0.
//     pub tau: f64,
// }

// impl<'a> From<&'a augurs_prophet::optimizer::Data> for Data<'a> {
//     fn from(data: &'a augurs_prophet::optimizer::Data) -> Self {
//         let sigmas: Vec<_> = data.sigmas.iter().map(|x| **x).collect();
//         // Be sure to use `from` instead of `view` here, as we've allocated
//         // and the vec will be dropped at the end of this function.
//         let sigmas = Float64Array::from(sigmas.as_slice());
//         // SAFETY: We're creating a view of these fields which have lifetime 'a.
//         // The view is valid as long as the `Data` is alive. Effectively
//         // we're tying the lifetime of this struct to the lifetime of the input,
//         // even though `Float64Array` doesn't have a lifetime.
//         // We also have to be careful not to allocate after this!
//         let y = unsafe { Float64Array::view(&data.y) };
//         let t = unsafe { Float64Array::view(&data.t) };
//         let cap = unsafe { Float64Array::view(&data.cap) };
//         let t_change = unsafe { Float64Array::view(&data.t_change) };
//         let s_a = unsafe { Int32Array::view(&data.s_a) };
//         let s_m = unsafe { Int32Array::view(&data.s_m) };
//         let x = unsafe { Float64Array::view(&data.X) };
//         Self {
//             _phantom: std::marker::PhantomData,
//             n: data.T,
//             y,
//             t,
//             cap,
//             s: data.S,
//             t_change,
//             trend_indicator: data.trend_indicator.into(),
//             k: data.K,
//             s_a,
//             s_m,
//             x,
//             sigmas,
//             tau: *data.tau,
//         }
//     }
// }

/// Log messages from the optimizer.
#[derive(Debug, Clone, Deserialize, Tsify)]
#[serde(rename_all = "camelCase")]
#[tsify(from_wasm_abi, type_prefix = "Prophet")]
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
#[tsify(from_wasm_abi, type_prefix = "Prophet")]
struct OptimizeOutput {
    /// Logs emitted by the optimizer, split by log level.
    pub logs: Logs,
    /// The optimized parameters.
    pub params: OptimizedParams,
}

/// The optimal parameters found by the optimizer.
#[derive(Debug, Clone, Deserialize, Serialize, Tsify)]
#[serde(rename_all = "camelCase")]
#[tsify(into_wasm_abi, from_wasm_abi, type_prefix = "Prophet")]
pub struct OptimizedParams {
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

impl From<augurs_prophet::optimizer::OptimizedParams> for OptimizedParams {
    fn from(x: augurs_prophet::optimizer::OptimizedParams) -> Self {
        OptimizedParams {
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
#[tsify(from_wasm_abi, type_prefix = "Prophet")]
pub struct TrainingData {
    /// The timestamps of the time series.
    ///
    /// These should be in seconds since the epoch.
    #[tsify(type = "TimestampSeconds[] | BigInt64Array")]
    pub ds: Vec<TimestampSeconds>,

    /// The time series values to fit the model to.
    #[tsify(type = "number[] | Float64Array")]
    pub y: Vec<f64>,

    /// Optionally, an upper bound (cap) on the values of the time series.
    ///
    /// Only used if the model's growth type is `logistic`.
    #[tsify(optional, type = "number[] | Float64Array")]
    pub cap: Option<Vec<f64>>,

    /// Optionally, a lower bound (floor) on the values of the time series.
    ///
    /// Only used if the model's growth type is `logistic`.
    #[tsify(optional, type = "number[] | Float64Array")]
    pub floor: Option<Vec<f64>>,

    /// Optional indicator variables for conditional seasonalities.
    ///
    /// The keys of the map are the names of the seasonality components,
    /// and the values are boolean arrays of length `T` where `true` indicates
    /// that the component is active for the corresponding time point.
    ///
    /// There must be a key in this map for each seasonality component
    /// that is marked as conditional in the model.
    #[tsify(optional)]
    pub seasonality_conditions: Option<HashMap<String, Vec<bool>>>,

    /// Optional exogynous regressors.
    #[tsify(optional, type = "Map<string, number[] | Float64Array>")]
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
#[tsify(from_wasm_abi, type_prefix = "Prophet")]
pub struct PredictionData {
    /// The timestamps of the time series.
    ///
    /// These should be in seconds since the epoch.
    #[tsify(type = "TimestampSeconds[]")]
    pub ds: Vec<TimestampSeconds>,

    /// Optionally, an upper bound (cap) on the values of the time series.
    ///
    /// Only used if the model's growth type is `logistic`.
    #[tsify(optional)]
    pub cap: Option<Vec<f64>>,

    /// Optionally, a lower bound (floor) on the values of the time series.
    ///
    /// Only used if the model's growth type is `logistic`.
    #[tsify(optional)]
    pub floor: Option<Vec<f64>>,

    /// Optional indicator variables for conditional seasonalities.
    ///
    /// The keys of the map are the names of the seasonality components,
    /// and the values are boolean arrays of length `T` where `true` indicates
    /// that the component is active for the corresponding time point.
    ///
    /// There must be a key in this map for each seasonality component
    /// that is marked as conditional in the model.
    #[tsify(optional)]
    pub seasonality_conditions: Option<HashMap<String, Vec<bool>>>,

    /// Optional exogynous regressors.
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

fn make_forecast(level: Option<f64>, predictions: augurs_prophet::FeaturePrediction) -> Forecast {
    Forecast {
        point: predictions.point,
        intervals: level.zip(predictions.lower).zip(predictions.upper).map(
            |((level, lower), upper)| ForecastIntervals {
                level,
                lower,
                upper,
            },
        ),
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
#[derive(Clone, Debug, Serialize, Tsify)]
#[serde(rename_all = "camelCase")]
#[tsify(into_wasm_abi, type_prefix = "Prophet")]
pub struct Predictions {
    #[tsify(type = "TimestampSeconds[]")]
    /// The timestamps of the forecasts.
    pub ds: Vec<TimestampSeconds>,

    /// Forecasts of the input time series `y`.
    #[tsify(type = "Forecast")]
    pub yhat: Forecast,

    /// The trend contribution at each time point.
    #[tsify(type = "Forecast")]
    pub trend: Forecast,

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
    #[tsify(type = "Forecast")]
    pub additive: Forecast,

    /// The combined combination of all _multiplicative_ components.
    ///
    /// This includes seasonalities, holidays and regressors if their mode
    /// was configured to be [`FeatureMode::Multiplicative`](crate::FeatureMode::Multiplicative).
    #[tsify(type = "Forecast")]
    pub multiplicative: Forecast,

    /// Mapping from holiday name to that holiday's contribution.
    #[tsify(type = "Map<string, Forecast>")]
    pub holidays: HashMap<String, Forecast>,

    /// Mapping from seasonality name to that seasonality's contribution.
    #[tsify(type = "Map<string, Forecast>")]
    pub seasonalities: HashMap<String, Forecast>,

    /// Mapping from regressor name to that regressor's contribution.
    #[tsify(type = "Map<string, Forecast>")]
    pub regressors: HashMap<String, Forecast>,
}

impl From<(Option<f64>, augurs_prophet::Predictions)> for Predictions {
    fn from((level, value): (Option<f64>, augurs_prophet::Predictions)) -> Self {
        Self {
            ds: value.ds,
            yhat: make_forecast(level, value.yhat),
            trend: make_forecast(level, value.trend),
            cap: value.cap,
            floor: value.floor,
            additive: make_forecast(level, value.additive),
            multiplicative: make_forecast(level, value.multiplicative),
            holidays: value
                .holidays
                .into_iter()
                .map(|(k, v)| (k, make_forecast(level, v)))
                .collect(),
            seasonalities: value
                .seasonalities
                .into_iter()
                .map(|(k, v)| (k, make_forecast(level, v)))
                .collect(),
            regressors: value
                .regressors
                .into_iter()
                .map(|(k, v)| (k, make_forecast(level, v)))
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
#[tsify(from_wasm_abi, type_prefix = "Prophet")]
pub struct Options {
    /// Optimizer, used to find the maximum likelihood estimate of the
    /// Prophet Stan model parameters.
    ///
    /// See the documentation for `ProphetOptions` for more details.
    #[serde(with = "serde_wasm_bindgen::preserve")]
    #[tsify(type = "ProphetOptimizer")]
    pub optimizer: js_sys::Object,

    /// The type of growth (trend) to use.
    ///
    /// Defaults to [`GrowthType::Linear`].
    #[tsify(optional)]
    pub growth: Option<GrowthType>,

    /// An optional list of changepoints.
    ///
    /// If not provided, changepoints will be automatically selected.
    #[tsify(optional, type = "TimestampSeconds[]")]
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

impl TryFrom<Options> for (JsOptimizer, augurs_prophet::OptProphetOptions) {
    type Error = JsError;

    fn try_from(value: Options) -> Result<Self, Self::Error> {
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
#[tsify(from_wasm_abi, type_prefix = "Prophet")]
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
#[serde(rename_all = "camelCase", tag = "type")]
#[tsify(from_wasm_abi, type_prefix = "Prophet")]
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
    Manual {
        /// Whether to include this seasonality.
        enabled: bool,
    },
    /// Enable this seasonality and use the provided number of Fourier terms.
    Fourier {
        /// The order of the Fourier terms to use.
        order: u32,
    },
}

impl TryFrom<SeasonalityOption> for augurs_prophet::SeasonalityOption {
    type Error = TryFromIntError;

    fn try_from(value: SeasonalityOption) -> Result<Self, Self::Error> {
        match value {
            SeasonalityOption::Auto => Ok(Self::Auto),
            SeasonalityOption::Manual { enabled } => Ok(Self::Manual(enabled)),
            SeasonalityOption::Fourier { order } => Ok(Self::Fourier(order.try_into()?)),
        }
    }
}

/// How to scale the data prior to fitting the model.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Default, Deserialize, Tsify)]
#[serde(rename_all = "camelCase")]
#[tsify(from_wasm_abi, type_prefix = "Prophet")]
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
#[tsify(from_wasm_abi, type_prefix = "Prophet")]
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
#[derive(Clone, Copy, Debug, Eq, PartialEq, Default, Deserialize, Serialize, Tsify)]
#[serde(rename_all = "camelCase")]
#[tsify(from_wasm_abi, type_prefix = "Prophet")]
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

/// An occurrence of a holiday.
///
/// Each occurrence has a start and end time represented as
/// a Unix timestamp. Holiday occurrences are therefore
/// timestamp-unaware and can therefore span multiple days
/// or even sub-daily periods.
///
/// This differs from the Python and R Prophet implementations,
/// which require all holidays to be day-long events.
///
/// The caller is responsible for ensuring that the start
/// and end time provided are in the correct timezone.
#[derive(Clone, Debug, Deserialize, Tsify)]
#[serde(rename_all = "camelCase")]
#[tsify(from_wasm_abi, type_prefix = "Prophet")]
pub struct HolidayOccurrence {
    /// The start of the holiday, as a Unix timestamp in seconds.
    #[tsify(type = "TimestampSeconds")]
    pub start: TimestampSeconds,
    /// The end of the holiday, as a Unix timestamp in seconds.
    #[tsify(type = "TimestampSeconds")]
    pub end: TimestampSeconds,
}

impl From<HolidayOccurrence> for augurs_prophet::HolidayOccurrence {
    fn from(value: HolidayOccurrence) -> Self {
        Self::new(value.start, value.end)
    }
}

/// A holiday to be considered by the Prophet model.
#[derive(Clone, Debug, Deserialize, Tsify)]
#[serde(rename_all = "camelCase")]
#[tsify(from_wasm_abi, type_prefix = "Prophet")]
pub struct Holiday {
    /// The occurrences of the holiday.
    pub occurrences: Vec<HolidayOccurrence>,

    /// The prior scale for the holiday.
    #[tsify(optional)]
    pub prior_scale: Option<f64>,
}

impl TryFrom<Holiday> for augurs_prophet::Holiday {
    type Error = JsError;

    fn try_from(value: Holiday) -> Result<Self, Self::Error> {
        let mut holiday = Self::new(value.occurrences.into_iter().map(|x| x.into()).collect());
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
