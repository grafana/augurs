/* tslint:disable */
/* eslint-disable */
/**
* Initialize logging.
*
* You can use this to emit logs from augurs to the browser console.
* The default is to log everything to the console, but you can
* change the log level and whether logs are emitted to the console
* or to the browser's performance timeline.
*
* IMPORTANT: this function should only be called once. It will throw
* an exception if called more than once.
* @param {LogConfig | undefined} [config]
*/
export function initLogging(config?: LogConfig): void;
/**
* Initialize the logger and panic hook.
*
* This will be called automatically when the module is imported.
* It sets the default tracing subscriber to `tracing-wasm`, and
* sets WASM panics to print to the console with a helpful error
* message.
*/
export function custom_init(): void;

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


/**
 * Arguments for optimization.
 */
export interface ProphetOptimizeOptions {
    /**
     * Algorithm to use.
     */
    algorithm?: ProphetAlgorithm;
    /**
     * The random seed to use for the optimization.
     */
    seed?: number;
    /**
     * The chain id to advance the PRNG.
     */
    chain?: number;
    /**
     * Line search step size for first iteration.
     */
    initAlpha?: number;
    /**
     * Convergence tolerance on changes in objective function value.
     */
    tolObj?: number;
    /**
     * Convergence tolerance on relative changes in objective function value.
     */
    tolRelObj?: number;
    /**
     * Convergence tolerance on the norm of the gradient.
     */
    tolGrad?: number;
    /**
     * Convergence tolerance on the relative norm of the gradient.
     */
    tolRelGrad?: number;
    /**
     * Convergence tolerance on changes in parameter value.
     */
    tolParam?: number;
    /**
     * Size of the history for LBFGS Hessian approximation. The value should
     * be less than the dimensionality of the parameter space. 5-10 usually
     * sufficient.
     */
    historySize?: number;
    /**
     * Total number of iterations.
     */
    iter?: number;
    /**
     * When `true`, use the Jacobian matrix to approximate the Hessian.
     * Default is `false`.
     */
    jacobian?: boolean;
    /**
     * How frequently to emit convergence statistics, in number of iterations.
     */
    refresh?: number;
}

/**
 * The initial parameters for the optimization.
 */
export interface ProphetInitialParams {
    /**
     * Base trend growth rate.
     */
    k: number;
    /**
     * Trend offset.
     */
    m: number;
    /**
     * Trend rate adjustments, length s in data.
     */
    delta: Float64Array;
    /**
     * Regressor coefficients, length k in data.
     */
    beta: Float64Array;
    /**
     * Observation noise.
     */
    sigmaObs: number;
}

/**
 * The algorithm to use for optimization. One of: \'BFGS\', \'LBFGS\', \'Newton\'.
 */
export type ProphetAlgorithm = "newton" | "bfgs" | "lbfgs";

/**
 * The type of trend to use.
 */
export type ProphetTrendIndicator = "linear" | "logistic" | "flat";

/**
 * Data for the Prophet model.
 */
export interface ProphetStanData {
    /**
     * Number of time periods.
     */
    T: number;
    /**
     * Time series, length n.
     */
    y: number[];
    /**
     * Time, length n.
     */
    t: number[];
    /**
     * Capacities for logistic trend, length n.
     */
    cap: number[];
    /**
     * Number of changepoints.
     */
    S: number;
    /**
     * Times of trend changepoints, length s.
     */
    t_change: number[];
    /**
     * The type of trend to use.
     *
     * Possible values are:
     * - 0 for linear trend
     * - 1 for logistic trend
     * - 2 for flat trend.
     */
    trend_indicator: number;
    /**
     * Number of regressors.
     *
     * Must be greater than or equal to 1.
     */
    K: number;
    /**
     * Indicator of additive features, length k.
     */
    s_a: number[];
    /**
     * Indicator of multiplicative features, length k.
     */
    s_m: number[];
    /**
     * Regressors, shape (n, k).
     */
    X: number[];
    /**
     * Scale on seasonality prior.
     *
     * Must all be greater than zero.
     */
    sigmas: number[];
    /**
     * Scale on changepoints prior.
     * Must be greater than 0.
     */
    tau: number;
}

/**
 * Data for the Prophet Stan model, in JSON format.
 *
 * The JSON should represent an object of type `ProphetStanData`.
 */
export type ProphetStanDataJSON = string;

/**
 * Log messages from the optimizer.
 */
export interface ProphetLogs {
    /**
     * Debug logs.
     */
    debug: string;
    /**
     * Info logs.
     */
    info: string;
    /**
     * Warning logs.
     */
    warn: string;
    /**
     * Error logs.
     */
    error: string;
    /**
     * Fatal logs.
     */
    fatal: string;
}

/**
 * The output of the optimizer.
 */
export interface ProphetOptimizeOutput {
    /**
     * Logs emitted by the optimizer, split by log level.
     */
    logs: ProphetLogs;
    /**
     * The optimized parameters.
     */
    params: ProphetOptimizedParams;
}

/**
 * The optimal parameters found by the optimizer.
 */
export interface ProphetOptimizedParams {
    /**
     * Base trend growth rate.
     */
    k: number;
    /**
     * Trend offset.
     */
    m: number;
    /**
     * Observation noise.
     */
    sigmaObs: number;
    /**
     * Trend rate adjustments.
     */
    delta: Float64Array;
    /**
     * Regressor coefficients.
     */
    beta: Float64Array;
    /**
     * Transformed trend.
     */
    trend: Float64Array;
}

export type TimestampSeconds = number;

/**
 * The data needed to train a Prophet model.
 *
 * Seasonality conditions, regressors,
 * floor and cap columns.
 */
export interface ProphetTrainingData {
    /**
     * The timestamps of the time series.
     *
     * These should be in seconds since the epoch.
     */
    ds: TimestampSeconds[] | BigInt64Array;
    /**
     * The time series values to fit the model to.
     */
    y: number[] | Float64Array;
    /**
     * Optionally, an upper bound (cap) on the values of the time series.
     *
     * Only used if the model\'s growth type is `logistic`.
     */
    cap?: number[] | Float64Array;
    /**
     * Optionally, a lower bound (floor) on the values of the time series.
     *
     * Only used if the model\'s growth type is `logistic`.
     */
    floor?: number[] | Float64Array;
    /**
     * Optional indicator variables for conditional seasonalities.
     *
     * The keys of the map are the names of the seasonality components,
     * and the values are boolean arrays of length `T` where `true` indicates
     * that the component is active for the corresponding time point.
     *
     * There must be a key in this map for each seasonality component
     * that is marked as conditional in the model.
     */
    seasonalityConditions?: Map<string, boolean[]>;
    /**
     * Optional exogynous regressors.
     */
    x?: Map<string, number[] | Float64Array>;
}

/**
 * The data needed to predict with a Prophet model.
 *
 * The structure of the prediction data must be the same as the
 * training data used to train the model, with the exception of
 * `y` (which is being predicted).
 *
 * That is, if your model used certain seasonality conditions or
 * regressors, you must include them in the prediction data.
 */
export interface ProphetPredictionData {
    /**
     * The timestamps of the time series.
     *
     * These should be in seconds since the epoch.
     */
    ds: TimestampSeconds[];
    /**
     * Optionally, an upper bound (cap) on the values of the time series.
     *
     * Only used if the model\'s growth type is `logistic`.
     */
    cap?: number[];
    /**
     * Optionally, a lower bound (floor) on the values of the time series.
     *
     * Only used if the model\'s growth type is `logistic`.
     */
    floor?: number[];
    /**
     * Optional indicator variables for conditional seasonalities.
     *
     * The keys of the map are the names of the seasonality components,
     * and the values are boolean arrays of length `T` where `true` indicates
     * that the component is active for the corresponding time point.
     *
     * There must be a key in this map for each seasonality component
     * that is marked as conditional in the model.
     */
    seasonalityConditions?: Map<string, boolean[]>;
    /**
     * Optional exogynous regressors.
     */
    x?: Map<string, number[]>;
}

/**
 * Predictions from a Prophet model.
 *
 * The `yhat` field contains the forecasts for the input time series.
 * All other fields contain individual components of the model which
 * contribute towards the final `yhat` estimate.
 *
 * Certain fields (such as `cap` and `floor`) may be `None` if the
 * model did not use them (e.g. the model was not configured to use
 * logistic trend).
 */
export interface ProphetPredictions {
    /**
     * The timestamps of the forecasts.
     */
    ds: TimestampSeconds[];
    /**
     * Forecasts of the input time series `y`.
     */
    yhat: Forecast;
    /**
     * The trend contribution at each time point.
     */
    trend: Forecast;
    /**
     * The cap for the logistic growth.
     *
     * Will only be `Some` if the model used [`GrowthType::Logistic`](crate::GrowthType::Logistic).
     */
    cap: number[] | undefined;
    /**
     * The floor for the logistic growth.
     *
     * Will only be `Some` if the model used [`GrowthType::Logistic`](crate::GrowthType::Logistic)
     * and the floor was provided in the input data.
     */
    floor: number[] | undefined;
    /**
     * The combined combination of all _additive_ components.
     *
     * This includes seasonalities, holidays and regressors if their mode
     * was configured to be [`FeatureMode::Additive`](crate::FeatureMode::Additive).
     */
    additive: Forecast;
    /**
     * The combined combination of all _multiplicative_ components.
     *
     * This includes seasonalities, holidays and regressors if their mode
     * was configured to be [`FeatureMode::Multiplicative`](crate::FeatureMode::Multiplicative).
     */
    multiplicative: Forecast;
    /**
     * Mapping from holiday name to that holiday\'s contribution.
     */
    holidays: Map<string, Forecast>;
    /**
     * Mapping from seasonality name to that seasonality\'s contribution.
     */
    seasonalities: Map<string, Forecast>;
    /**
     * Mapping from regressor name to that regressor\'s contribution.
     */
    regressors: Map<string, Forecast>;
}

/**
 * Options for Prophet, after applying defaults.
 *
 * The only required field is `optimizer`. See the documentation for
 * `Optimizer` for more details.
 *
 * All other options are treated exactly the same as the original
 * Prophet library; see its [documentation] for more detail.
 *
 * [documentation]: https://facebook.github.io/prophet/docs/quick_start.html
 */
export interface ProphetOptions {
    /**
     * Optimizer, used to find the maximum likelihood estimate of the
     * Prophet Stan model parameters.
     *
     * See the documentation for `ProphetOptions` for more details.
     */
    optimizer: ProphetOptimizer;
    /**
     * The type of growth (trend) to use.
     *
     * Defaults to [`GrowthType::Linear`].
     */
    growth?: ProphetGrowthType;
    /**
     * An optional list of changepoints.
     *
     * If not provided, changepoints will be automatically selected.
     */
    changepoints?: TimestampSeconds[];
    /**
     * The number of potential changepoints to include.
     *
     * Not used if `changepoints` is provided.
     *
     * If provided and `changepoints` is not provided, then
     * `n_changepoints` potential changepoints will be selected
     * uniformly from the first `changepoint_range` proportion of
     * the history.
     *
     * Defaults to 25.
     */
    nChangepoints?: number;
    /**
     * The proportion of the history to consider for potential changepoints.
     *
     * Not used if `changepoints` is provided.
     *
     * Defaults to `0.8` for the first 80% of the data.
     */
    changepointRange?: number;
    /**
     * How to fit yearly seasonality.
     *
     * Defaults to [`SeasonalityOption::Auto`].
     */
    yearlySeasonality?: ProphetSeasonalityOption;
    /**
     * How to fit weekly seasonality.
     *
     * Defaults to [`SeasonalityOption::Auto`].
     */
    weeklySeasonality?: ProphetSeasonalityOption;
    /**
     * How to fit daily seasonality.
     *
     * Defaults to [`SeasonalityOption::Auto`].
     */
    dailySeasonality?: ProphetSeasonalityOption;
    /**
     * How to model seasonality.
     *
     * Defaults to [`FeatureMode::Additive`].
     */
    seasonalityMode?: ProphetFeatureMode;
    /**
     * The prior scale for seasonality.
     *
     * This modulates the strength of seasonality,
     * with larger values allowing the model to fit
     * larger seasonal fluctuations and smaller values
     * dampening the seasonality.
     *
     * Can be specified for individual seasonalities
     * using [`Prophet::add_seasonality`](crate::Prophet::add_seasonality).
     *
     * Defaults to `10.0`.
     */
    seasonalityPriorScale?: number;
    /**
     * The prior scale for changepoints.
     *
     * This modulates the flexibility of the automatic
     * changepoint selection. Large values will allow many
     * changepoints, while small values will allow few
     * changepoints.
     *
     * Defaults to `0.05`.
     */
    changepointPriorScale?: number;
    /**
     * How to perform parameter estimation.
     *
     * When [`EstimationMode::Mle`] or [`EstimationMode::Map`]
     * are used then no MCMC samples are taken.
     *
     * Defaults to [`EstimationMode::Mle`].
     */
    estimation?: ProphetEstimationMode;
    /**
     * The width of the uncertainty intervals.
     *
     * Must be between `0.0` and `1.0`. Common values are
     * `0.8` (80%), `0.9` (90%) and `0.95` (95%).
     *
     * Defaults to `0.8` for 80% intervals.
     */
    intervalWidth?: number;
    /**
     * The number of simulated draws used to estimate uncertainty intervals.
     *
     * Setting this value to `0` will disable uncertainty
     * estimation and speed up the calculation.
     *
     * Defaults to `1000`.
     */
    uncertaintySamples?: number;
    /**
     * How to scale the data prior to fitting the model.
     *
     * Defaults to [`Scaling::AbsMax`].
     */
    scaling?: ProphetScaling;
    /**
     * Holidays to include in the model.
     */
    holidays?: Map<string, ProphetHoliday>;
    /**
     * Prior scale for holidays.
     *
     * This parameter modulates the strength of the holiday
     * components model, unless overridden in each individual
     * holiday\'s input.
     *
     * Defaults to `100.0`.
     */
    holidaysPriorScale?: number;
    /**
     * How to model holidays.
     *
     * Defaults to the same value as [`ProphetOptions::seasonality_mode`].
     */
    holidaysMode?: ProphetFeatureMode;
}

/**
 * The type of growth to use.
 */
export type ProphetGrowthType = "linear" | "logistic" | "flat";

/**
 * Define whether to include a specific seasonality, and how it should be specified.
 */
export type ProphetSeasonalityOption = { type: "auto" } | { type: "manual"; enabled: boolean } | { type: "fourier"; order: number };

/**
 * How to scale the data prior to fitting the model.
 */
export type ProphetScaling = "absMax" | "minMax";

/**
 * How to do parameter estimation.
 *
 * Note: for now, only MLE/MAP estimation is supported, i.e. there
 * is no support for MCMC sampling. This will be added in the future!
 * The enum will be marked as `non_exhaustive` until that point.
 */
export type ProphetEstimationMode = "mle" | "map";

/**
 * The mode of a seasonality, regressor, or holiday.
 */
export type ProphetFeatureMode = "additive" | "multiplicative";

/**
 * A holiday to be considered by the Prophet model.
 */
export interface ProphetHoliday {
    /**
     * The dates of the holiday.
     */
    ds: TimestampSeconds[];
    /**
     * The lower window for the holiday.
     *
     * The lower window is the number of days before the holiday
     * that it is observed. For example, if the holiday is on
     * 2023-01-01 and the lower window is -1, then the holiday will
     * _also_ be observed on 2022-12-31.
     */
    lowerWindow?: number[];
    /**
     * The upper window for the holiday.
     *
     * The upper window is the number of days after the holiday
     * that it is observed. For example, if the holiday is on
     * 2023-01-01 and the upper window is 1, then the holiday will
     * _also_ be observed on 2023-01-02.
     */
    upperWindow?: number[];
    /**
     * The prior scale for the holiday.
     */
    priorScale?: number;
}

/**
 * The maximum log level to emit.
 *
 * The default is `Level::Info`.
 */
export type Level = "trace" | "debug" | "info" | "warn" | "error";

/**
 * The target for augurs log events.
 */
export type LogTarget = "console" | "performance";

/**
 * Log configuration.
 */
export interface LogConfig {
    /**
     * The maximum log level to emit.
     *
     * Defaults to `INFO`.
     */
    maxLevel?: Level;
    /**
     * The target for augurs log events.
     *
     * Defaults to logging to the browser console.
     */
    target?: LogTarget;
    /**
     * Whether to emit coloured logs.
     *
     * Defaults to `true`.
     */
    color?: boolean;
    /**
     * Whether to show detailed fields such as augurs\' file names and line numbers
     * in the logs.
     *
     * Probably not wise in production.
     *
     * Defaults to `false`.
     */
    showDetailedFields?: boolean;
}

/**
 * Forecast intervals.
 */
export interface ForecastIntervals {
    /**
     * The confidence level for the intervals.
     */
    level: number;
    /**
     * The lower prediction intervals.
     */
    lower: number[];
    /**
     * The upper prediction intervals.
     */
    upper: number[];
}

/**
 * A forecast containing point forecasts and, optionally, prediction intervals.
 */
export interface Forecast {
    /**
     * The point forecasts.
     */
    point: number[];
    /**
     * The forecast intervals, if requested and supported
     * by the trend model.
     */
    intervals: ForecastIntervals | undefined;
}

/**
* The [Prophet] time-series forecasting model.
*
* Prophet is a forecasting procedure designed for automated forecasting
* at scale with minimal manual input.
*
* Create a new Prophet instance with the constructor, passing in an optimizer
* and some other optional arguments.
*
* # Example
*
* ```javascript
* import { Prophet } from '@bsull/augurs';
* import { optimizer } from '@bsull/augurs-prophet-wasmstan';
*
* const prophet = new Prophet({ optimizer });
* const ds = [
*   1704067200n, 1704871384n, 1705675569n, 1706479753n, 1707283938n, 1708088123n,
*   1708892307n, 1709696492n, 1710500676n, 1711304861n, 1712109046n, 1712913230n,
* ];
* const y = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
* const trainingData = { ds, y };
* prophet.fit(trainingData);
* const predictions = prophet.predict();
* console.log(predictions.yhat);  // yhat is an object with 'point', 'lower' and 'upper'.
* ```
*
* [Prophet]: https://facebook.github.io/prophet/
*/
export class Prophet {
  free(): void;
/**
* Create a new Prophet model.
* @param {ProphetOptions} opts
*/
  constructor(opts: ProphetOptions);
/**
* Fit the model to some training data.
* @param {ProphetTrainingData} data
* @param {ProphetOptimizeOptions | undefined} [opts]
*/
  fit(data: ProphetTrainingData, opts?: ProphetOptimizeOptions): void;
/**
* Predict using the model.
*
* If `data` is omitted, predictions will be produced for the training data
* history.
*
* This will throw an exception if the model hasn't already been fit.
* @param {ProphetPredictionData | undefined} [data]
* @returns {ProphetPredictions}
*/
  predict(data?: ProphetPredictionData): ProphetPredictions;
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
  readonly memory: WebAssembly.Memory;
  readonly __wbg_prophet_free: (a: number, b: number) => void;
  readonly prophet_new: (a: number, b: number) => void;
  readonly prophet_fit: (a: number, b: number, c: number, d: number) => void;
  readonly prophet_predict: (a: number, b: number, c: number) => void;
  readonly initLogging: (a: number, b: number) => void;
  readonly custom_init: () => void;
  readonly __wbindgen_malloc: (a: number, b: number) => number;
  readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
  readonly __wbindgen_add_to_stack_pointer: (a: number) => number;
  readonly __wbindgen_exn_store: (a: number) => void;
  readonly __wbindgen_free: (a: number, b: number, c: number) => void;
  readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;
/**
* Instantiates the given `module`, which can either be bytes or
* a precompiled `WebAssembly.Module`.
*
* @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
*
* @returns {InitOutput}
*/
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
* If `module_or_path` is {RequestInfo} or {URL}, makes a request and
* for everything else, calls `WebAssembly.instantiate` directly.
*
* @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
*
* @returns {Promise<InitOutput>}
*/
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
