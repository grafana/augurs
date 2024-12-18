//! Logging utilities.
//!
//! Currently uses `tracing-wasm` to emit logs to the browser console
//! or the browser's performance timeline.

use serde::Deserialize;
use tracing_subscriber::{layer::SubscriberExt, Registry};
use tsify_next::Tsify;
use wasm_bindgen::prelude::*;
use wasm_tracing::{ConsoleConfig, WasmLayer, WasmLayerConfig};

/// The maximum log level to emit.
///
/// The default is `Level::Info`.
#[derive(Debug, Default, Clone, Copy, Deserialize, Tsify)]
#[serde(rename_all = "camelCase")]
#[tsify(from_wasm_abi)]
pub enum Level {
    /// Emit logs at or above the `TRACE` level.
    Trace,
    /// Emit logs at or above the `DEBUG` level.
    Debug,
    /// Emit logs at or above the `INFO` level.
    #[default]
    Info,
    /// Emit logs at or above the `WARN` level.
    Warn,
    /// Emit logs at or above the `ERROR` level.
    Error,
}

impl From<Level> for tracing::Level {
    fn from(value: Level) -> Self {
        match value {
            Level::Trace => tracing::Level::TRACE,
            Level::Debug => tracing::Level::DEBUG,
            Level::Info => tracing::Level::INFO,
            Level::Warn => tracing::Level::WARN,
            Level::Error => tracing::Level::ERROR,
        }
    }
}

/// The target for augurs log events.
#[derive(Debug, Default, Clone, Copy, Deserialize, Tsify)]
#[serde(rename_all = "camelCase")]
#[tsify(from_wasm_abi)]
pub enum LogTarget {
    /// Emit logs to the browser console.
    #[default]
    Console,
    /// Emit logs to the browser's performance timeline.
    Performance,
}

fn default_coloured_logs() -> bool {
    true
}

/// Log configuration.
#[derive(Debug, Default, Clone, Deserialize, Tsify)]
#[serde(rename_all = "camelCase")]
#[tsify(from_wasm_abi)]
pub struct LogConfig {
    /// The maximum log level to emit.
    ///
    /// Defaults to `INFO`.
    #[serde(default)]
    pub max_level: Level,

    /// The target for augurs log events.
    ///
    /// Defaults to logging to the browser console.
    #[serde(default)]
    pub target: LogTarget,

    /// Whether to emit coloured logs.
    ///
    /// Defaults to `true`.
    #[serde(alias = "colour", default = "default_coloured_logs")]
    pub color: bool,

    /// Whether to show detailed fields such as augurs' file names and line numbers
    /// in the logs.
    ///
    /// Probably not wise in production.
    ///
    /// Defaults to `false`.
    #[serde(default)]
    pub show_detailed_fields: bool,
}

/// Initialize logging.
///
/// You can use this to emit logs from augurs to the browser console.
/// The default is to log everything to the console, but you can
/// change the log level and whether logs are emitted to the console
/// or to the browser's performance timeline.
///
/// IMPORTANT: this function should only be called once. It will throw
/// an exception if called more than once.
#[wasm_bindgen(js_name = "initLogging")]
pub fn init_logging(config: Option<LogConfig>) -> Result<(), JsError> {
    let config = config.unwrap_or_default();
    let config = WasmLayerConfig {
        report_logs_in_timings: matches!(config.target, LogTarget::Performance),
        show_fields: config.show_detailed_fields,
        console: if config.color {
            ConsoleConfig::ReportWithConsoleColor
        } else {
            ConsoleConfig::ReportWithoutConsoleColor
        },
        max_level: config.max_level.into(),
        show_origin: false,
    };
    tracing::subscriber::set_global_default(Registry::default().with(WasmLayer::new(config)))
        .map_err(|_| JsError::new("logging already initialized"))
}
