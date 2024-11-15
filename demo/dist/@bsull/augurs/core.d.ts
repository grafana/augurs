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


export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
  readonly memory: WebAssembly.Memory;
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
