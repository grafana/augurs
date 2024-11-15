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
 * Options for the DBSCAN outlier detector.
 */
export interface OutlierDetectorOptions {
    /**
     * A scale-invariant sensitivity parameter.
     *
     * This must be in (0, 1) and will be used to estimate a sensible
     * value of epsilon based on the data.
     */
    sensitivity: number;
}

/**
 * The type of outlier detector to use.
 */
export type OutlierDetectorType = "dbscan" | "mad";

/**
 * A band indicating the min and max value considered outlying
 * at each timestamp.
 */
export interface ClusterBand {
    /**
     * The minimum value considered outlying at each timestamp.
     */
    min: number[];
    /**
     * The maximum value considered outlying at each timestamp.
     */
    max: number[];
}

/**
 * A potentially outlying series.
 */
export interface OutlierSeries {
    /**
     * Whether the series is an outlier for at least one of the samples.
     */
    isOutlier: boolean;
    /**
     * The intervals of the series that are considered outliers.
     */
    outlierIntervals: OutlierInterval[];
    /**
     * The outlier scores of the series for each sample.
     */
    scores: number[];
}

/**
 * An interval for which a series is outlying.
 */
export interface OutlierInterval {
    /**
     * The start index of the interval.
     */
    start: number;
    /**
     * The end index of the interval, if any.
     */
    end: number | undefined;
}

/**
 * The result of applying an outlier detection algorithm to a group of time series.
 */
export interface OutlierOutput {
    /**
     * The indexes of the series considered outliers.
     */
    outlyingSeries: number[];
    /**
     * The results of the detection for each series.
     */
    seriesResults: OutlierSeries[];
    /**
     * The band indicating the min and max value considered outlying
     * at each timestamp.
     *
     * This may be undefined if no cluster was found (for example if
     * there were fewer than 3 series in the input data in the case of
     * DBSCAN).
     */
    clusterBand: ClusterBand | undefined;
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
* A 'loaded' outlier detector, ready to detect outliers.
*
* This is returned by the `preprocess` method of `OutlierDetector`,
* and holds the preprocessed data for the detector.
*/
export class LoadedOutlierDetector {
  free(): void;
/**
* Detect outliers in the given time series.
* @returns {OutlierOutput}
*/
  detect(): OutlierOutput;
/**
* Update the detector with new options.
*
* # Errors
*
* This method will return an error if the detector and options types
* are incompatible.
* @param {OutlierDetectorOptions} options
*/
  updateDetector(options: OutlierDetectorOptions): void;
}
/**
* A detector for detecting outlying time series in a group of series.
*/
export class OutlierDetector {
  free(): void;
/**
* Create a new outlier detector.
* @param {OutlierDetectorType} detectorType
* @param {OutlierDetectorOptions} options
*/
  constructor(detectorType: OutlierDetectorType, options: OutlierDetectorOptions);
/**
* Create a new outlier detector using the DBSCAN algorithm.
* @param {OutlierDetectorOptions} options
* @returns {OutlierDetector}
*/
  static dbscan(options: OutlierDetectorOptions): OutlierDetector;
/**
* Create a new outlier detector using the MAD algorithm.
* @param {OutlierDetectorOptions} options
* @returns {OutlierDetector}
*/
  static mad(options: OutlierDetectorOptions): OutlierDetector;
/**
* Detect outlying time series in a group of series.
*
* Note: if you plan to run the detector multiple times on the same data,
* you should use the `preprocess` method to cache the preprocessed data,
* then call `detect` on the `LoadedOutlierDetector` returned by `preprocess`.
* @param {number[][] | Float64Array[]} y
* @returns {OutlierOutput}
*/
  detect(y: number[][] | Float64Array[]): OutlierOutput;
/**
* Preprocess the data for the detector.
*
* The returned value is a 'loaded' outlier detector, which can be used
* to detect outliers without needing to preprocess the data again.
*
* This is useful if you plan to run the detector multiple times on the same data.
* @param {number[][] | Float64Array[]} y
* @returns {LoadedOutlierDetector}
*/
  preprocess(y: number[][] | Float64Array[]): LoadedOutlierDetector;
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
  readonly memory: WebAssembly.Memory;
  readonly __wbg_outlierdetector_free: (a: number, b: number) => void;
  readonly outlierdetector_new: (a: number, b: number, c: number) => void;
  readonly outlierdetector_dbscan: (a: number, b: number) => void;
  readonly outlierdetector_mad: (a: number, b: number) => void;
  readonly outlierdetector_detect: (a: number, b: number, c: number) => void;
  readonly outlierdetector_preprocess: (a: number, b: number, c: number) => void;
  readonly __wbg_loadedoutlierdetector_free: (a: number, b: number) => void;
  readonly loadedoutlierdetector_detect: (a: number, b: number) => void;
  readonly loadedoutlierdetector_updateDetector: (a: number, b: number, c: number) => void;
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
