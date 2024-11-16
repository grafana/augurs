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
 * The type of changepoint detector to use.
 */
export type ChangepointDetectorType = "normal-gamma" | "default-argpcp";

/**
 * Parameters for the Normal Gamma prior.
 * Options for the ETS MSTL model.
 */
export interface NormalGammaParameters {
    /**
     * The prior mean.
     *
     * Defaults to 0.0.
     */
    mu?: number;
    /**
     * The relative precision of μ versus data.
     *
     * Defaults to 1.0.
     */
    rho?: number;
    /**
     * The mean of rho (the precision) is v/s.
     *
     * Defaults to 1.0.
     */
    s?: number;
    /**
     * The degrees of freedom of precision of rho.
     *
     * Defaults to 1.0.
     */
    v?: number;
}

/**
 * Options for the Normal Gamma changepoint detector.
 */
export interface NormalGammaDetectorOptions {
    /**
     * The hazard lambda.
     *
     * `1/hazard` is the probability of the next step being a changepoint.
     * Therefore, the larger the value, the lower the prior probability
     * is for the any point to be a change-point.
     * Mean run-length is lambda - 1.
     *
     * Defaults to 250.0.
     */
    hazardLambda?: number;
    /**
     * The prior for the Normal distribution.
     */
    prior?: NormalGammaParameters;
}

/**
 * Options for the default Autoregressive Gaussian Process detector.
 */
export interface DefaultArgpcpDetectorOptions {
    /**
     * The value of the constant kernel.
     */
    constantValue?: number;
    /**
     * The length scale of the RBF kernel.
     */
    lengthScale?: number;
    /**
     * The noise level of the white kernel.
     */
    noiseLevel?: number;
    /**
     * The maximum autoregressive lag.
     */
    maxLag?: number;
    /**
     * Scale Gamma distribution alpha parameter.
     */
    alpha0?: number;
    /**
     * Scale Gamma distribution beta parameter.
     */
    beta0?: number;
    logisticHazardH?: number;
    logisticHazardA?: number;
    logisticHazardB?: number;
}

/**
 * Changepoints detected in a time series.
 */
export interface Changepoints {
    /**
     * The indices of the most likely changepoints.
     */
    indices: number[];
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
* A changepoint detector.
*/
export class ChangepointDetector {
  free(): void;
/**
* @param {ChangepointDetectorType} detectorType
*/
  constructor(detectorType: ChangepointDetectorType);
/**
* Create a new Bayesian Online changepoint detector with a Normal Gamma prior.
* @param {NormalGammaDetectorOptions | undefined} [opts]
* @returns {ChangepointDetector}
*/
  static normalGamma(opts?: NormalGammaDetectorOptions): ChangepointDetector;
/**
* Create a new Autoregressive Gaussian Process changepoint detector
* with the default kernel and parameters.
* @param {DefaultArgpcpDetectorOptions | undefined} [opts]
* @returns {ChangepointDetector}
*/
  static defaultArgpcp(opts?: DefaultArgpcpDetectorOptions): ChangepointDetector;
/**
* Detect changepoints in the given time series.
* @param {number[] | Float64Array} y
* @returns {Changepoints}
*/
  detectChangepoints(y: number[] | Float64Array): Changepoints;
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
  readonly memory: WebAssembly.Memory;
  readonly __wbg_changepointdetector_free: (a: number, b: number) => void;
  readonly changepointdetector_new: (a: number, b: number) => void;
  readonly changepointdetector_normalGamma: (a: number, b: number) => void;
  readonly changepointdetector_defaultArgpcp: (a: number, b: number) => void;
  readonly changepointdetector_detectChangepoints: (a: number, b: number, c: number) => void;
  readonly initLogging: (a: number, b: number) => void;
  readonly custom_init: () => void;
  readonly __wbindgen_malloc: (a: number, b: number) => number;
  readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
  readonly __wbindgen_add_to_stack_pointer: (a: number) => number;
  readonly __wbindgen_free: (a: number, b: number, c: number) => void;
  readonly __wbindgen_exn_store: (a: number) => void;
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
