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
 * Options for the dynamic time warping calculation.
 */
export interface DtwOptions {
    /**
     * The size of the Sakoe-Chiba band.
     */
    window?: number;
    /**
     * The maximum distance permitted between two points.
     *
     * If the distance between two points exceeds this value, the algorithm will
     * early abandon and use `maxDistance`.
     *
     * Only used when calculating distance matrices using [`Dtw::distanceMatrix`],
     * not when calculating the distance between two series.
     */
    maxDistance?: number;
    /**
     * The lower bound, used for early abandoning.
     * If specified, before calculating the DTW (which can be expensive), check if the
     * lower bound of the DTW is greater than this distance; if so, skip the DTW
     * calculation and return this bound instead.
     */
    lowerBound?: number;
    /**
     * The upper bound, used for early abandoning.
     * If specified, before calculating the DTW (which can be expensive), check if the
     * upper bound of the DTW is less than this distance; if so, skip the DTW
     * calculation and return this bound instead.
     */
    upperBound?: number;
}

/**
 * The distance function to use for Dynamic Time Warping.
 */
export type DistanceFunction = "euclidean" | "manhattan";

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
* Dynamic Time Warping.
*
* The `window` parameter can be used to specify the Sakoe-Chiba band size.
* The distance function depends on the constructor used; `euclidean` and
* `manhattan` are available, `euclidean` being the default.
*/
export class Dtw {
  free(): void;
/**
* Create a new `Dtw` instance.
* @param {DistanceFunction} distanceFunction
* @param {DtwOptions | undefined} [opts]
*/
  constructor(distanceFunction: DistanceFunction, opts?: DtwOptions);
/**
* Create a new `Dtw` instance using the Euclidean distance.
* @param {DtwOptions | undefined} [opts]
* @returns {Dtw}
*/
  static euclidean(opts?: DtwOptions): Dtw;
/**
* Create a new `Dtw` instance using the Manhattan distance.
* @param {DtwOptions | undefined} [opts]
* @returns {Dtw}
*/
  static manhattan(opts?: DtwOptions): Dtw;
/**
* Calculate the distance between two arrays under Dynamic Time Warping.
* @param {number[] | Float64Array} a
* @param {number[] | Float64Array} b
* @returns {number}
*/
  distance(a: number[] | Float64Array, b: number[] | Float64Array): number;
/**
* Compute the distance matrix between all pairs of series.
*
* The series do not all have to be the same length.
* @param {number[][] | Float64Array[]} series
* @returns {(Float64Array)[]}
*/
  distanceMatrix(series: number[][] | Float64Array[]): (Float64Array)[];
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
  readonly memory: WebAssembly.Memory;
  readonly __wbg_dtw_free: (a: number, b: number) => void;
  readonly dtw_new: (a: number, b: number) => number;
  readonly dtw_euclidean: (a: number) => number;
  readonly dtw_manhattan: (a: number) => number;
  readonly dtw_distance: (a: number, b: number, c: number, d: number) => void;
  readonly dtw_distanceMatrix: (a: number, b: number, c: number) => void;
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
