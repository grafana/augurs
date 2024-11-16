/* tslint:disable */
/* eslint-disable */
/**
* Create a new MSTL model with the given periods using the `AutoETS` trend model.
*
* @deprecated use `MSTL.ets` instead
* @param {number[] | Uint32Array} periods
* @param {ETSOptions | undefined} [options]
* @returns {MSTL}
*/
export function ets(periods: number[] | Uint32Array, options?: ETSOptions): MSTL;
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
 * The type of trend forecaster to use.
 */
export type MSTLTrendModel = "ets";

/**
 * Options for the ETS MSTL model.
 */
export interface ETSOptions {
    /**
     * Whether to impute missing values.
     */
    impute?: boolean;
    /**
     * Whether to logit-transform the data before forecasting.
     *
     * If `true`, the training data will be transformed using the logit function.
     * Forecasts will be back-transformed using the logistic function.
     */
    logitTransform?: boolean;
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
* A MSTL model.
*/
export class MSTL {
  free(): void;
/**
* Create a new MSTL model with the given periods using the given trend model.
* @param {MSTLTrendModel} trend_forecaster
* @param {number[] | Uint32Array} periods
* @param {ETSOptions | undefined} [options]
*/
  constructor(trend_forecaster: MSTLTrendModel, periods: number[] | Uint32Array, options?: ETSOptions);
/**
* Create a new MSTL model with the given periods using the `AutoETS` trend model.
* @param {number[] | Uint32Array} periods
* @param {ETSOptions | undefined} [options]
* @returns {MSTL}
*/
  static ets(periods: number[] | Uint32Array, options?: ETSOptions): MSTL;
/**
* Fit the model to the given time series.
* @param {number[] | Float64Array} y
*/
  fit(y: number[] | Float64Array): void;
/**
* Predict the next `horizon` values, optionally including prediction
* intervals at the given level.
*
* If provided, `level` must be a float between 0 and 1.
* @param {number} horizon
* @param {number | undefined} [level]
* @returns {Forecast}
*/
  predict(horizon: number, level?: number): Forecast;
/**
* Produce in-sample forecasts, optionally including prediction
* intervals at the given level.
*
* If provided, `level` must be a float between 0 and 1.
* @param {number | undefined} [level]
* @returns {Forecast}
*/
  predictInSample(level?: number): Forecast;
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
  readonly memory: WebAssembly.Memory;
  readonly __wbg_mstl_free: (a: number, b: number) => void;
  readonly mstl_new: (a: number, b: number, c: number, d: number) => void;
  readonly mstl_fit: (a: number, b: number, c: number) => void;
  readonly mstl_predict: (a: number, b: number, c: number, d: number, e: number) => void;
  readonly mstl_predictInSample: (a: number, b: number, c: number, d: number) => void;
  readonly ets: (a: number, b: number, c: number) => void;
  readonly mstl_ets: (a: number, b: number, c: number) => void;
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
