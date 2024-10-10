//! Use `cmdstan` to optimize or sample from the Prophet model.
//!
//! This module provides an [`Optimizer`] implementation that uses `cmdstan` to
//! optimize or sample from the Prophet model. `cmdstan` is a C++ program provided
//! by Stan which is used to compile the Stan model into a binary executable.
//! That executable can then be used to optimize or sample from the model,
//! passing data in and out over the filesystem.
//!
//! # Usage
//!
//! There are two ways to use this optimizer:
//!
//! 1. Use the `compile-cmdstan` feature to build the Prophet Stan model at build time.
//!    This requires a working installation of `stan` and setting the `STAN_PATH`
//!    environment variable to the path to the `stan` installation.
//!    This will embed the Prophet model binary and the libtbb dynamic library into
//!    the final executable, which will increase the size of the final executable by about
//!    2MB, but the final binary won't require any additional dependencies.
//! 2. Use [`CmdstanOptimizer::with_prophet_path`] to create a new `CmdstanOptimizer` using
//!    a precompiled Prophet Stan model. This model can be obtained by either manually building
//!    the Prophet model (which still requires a working Stan installation) or extracting it from
//!    the Prophet Python package for your target platform.
//!    The `download-stan-model` binary of this crate can be used to do the latter easily.
//!    It will download the precompiled model for the current architecture and OS, and
//!    extract it to the `prophet_stan_model` directory. This won't work if there is no
//!    wheel for the current architecture and OS, though.
//!
//!    For example:
//!
//!    ```sh
//!    $ cargo install --bin download-stan-model augurs-prophet
//!    $ download-stan-model
//!    $ ls prophet_stan_model
//!    ```
//!
//! # Gotchas
//!
//! - the `compile-cmdstan` feature may not work on all platforms. If you encounter
//!   problems, try using the second method instead.
//! - the Prophet model binary is not statically linked to `libtbb`, since doing so is
//!   not recommended by the Stan developers or TBB developers. This means that the
//!   final binary will require the `libtbb` dynamic library to be present at runtime.
//!   The first method takes care of this for you, but if manually providing the path
//!   using the second you'll need to make sure the `libtbb` dynamic library is
//!   available at runtime. You can do this by copying the `libtbb` dynamic library
//!   to a known directory and setting the `LD_LIBRARY_PATH` environment variable
//!   to that directory. The `download-stan-model` tool adds this library to the
//!   `lib` subdirectory next to the Prophet binary, and the `LD_LIBRARY_PATH`
//!   environment variable is set to that directory by default.
use std::{
    io,
    num::{ParseFloatError, ParseIntError},
    path::{Path, PathBuf},
    process::Command,
};

use rand::{thread_rng, Rng};

use crate::{optimizer, Optimizer, PositiveFloat, TryFromFloatError};

/// Errors that can occur when trying to run optimization using cmdstan.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The provided path to the Prophet binary has no parent directory.
    #[error("Prophet path {0} has no parent")]
    ProphetPathHasNoParent(PathBuf),
    /// An I/O error occurred when trying to run the Prophet model executable.
    #[error("Error running Prophet at {command}: {source}")]
    ProphetIOError {
        /// The stringified command that was attempted.
        command: String,
        /// The source error.
        #[source]
        source: io::Error,
    },
    /// An error occurred when running the Prophet model executable.
    #[error("Error {}running Prophet command ({command}): {stdout}\n{stderr}", code.map(|c| format!("code {}", c)).unwrap_or_default())]
    ProphetExeError {
        /// The stringified command that was attempted.
        command: String,
        /// The status code, if provided by the OS.
        code: Option<i32>,
        /// Stdout from the command.
        stdout: String,
        /// Stderr from the command.
        stderr: String,
    },
    /// An error occurred when creating a temporary directory.
    #[error("Error creating temporary directory: {0}")]
    CreateTempDir(
        /// The source error.
        #[from]
        io::Error,
    ),
    /// An error occurred when serializing the initial parameters to JSON.
    #[error("Error serializing initial parameters to JSON: {0}")]
    SerializeInits(
        /// The source error.
        #[source]
        serde_json::Error,
    ),
    /// An error occurred when writing the initial parameters to a temporary file.
    #[error("Error writing initial parameters to temporary file: {0}")]
    WriteInits(
        /// The source error.
        #[source]
        io::Error,
    ),
    /// An error occurred when serializing the data to JSON.
    #[error("Error serializing data to JSON: {0}")]
    SerializeData(
        /// The source error.
        #[source]
        serde_json::Error,
    ),
    /// An error occurred when writing the data to a temporary file.
    #[error("Error writing data to temporary file: {0}")]
    WriteData(
        /// The source error.
        #[source]
        io::Error,
    ),
    /// An error occurred when reading the output from Stan.
    #[error("Error reading output from Prophet at {path}: {source}")]
    ReadOutput {
        /// The path to the output file.
        path: PathBuf,
        /// The source error.
        #[source]
        source: io::Error,
    },
    /// No header row was found in the output file.
    #[error("No header row was found in the output file")]
    NoHeader,
    /// No data row was found in the output file.
    #[error("No data row was found in the output file")]
    NoData,
    /// An invalid integer was found in the output header row.
    ///
    /// Vector parameters should appear in the form `<name>.<index>`
    /// where `<index>` is a one-indexed integer.
    #[error("Invalid int in output: {source}")]
    InvalidInt {
        /// The source error.
        #[from]
        source: ParseIntError,
    },
    /// An invalid float was found in the output data row.
    #[error("Invalid float in output: {source}")]
    InvalidFloat {
        /// The source error.
        #[from]
        source: ParseFloatError,
    },
    /// An invalid positive float was found in the output data row.
    #[error("Invalid positive float in output: {source}")]
    InvalidPositiveFloat {
        /// The source error.
        #[from]
        source: TryFromFloatError,
    },
    /// Some parameters were missing from the output.
    #[error("Some parameters were not found in the output: {0:?}")]
    MissingOptimizedParameters(OptimizedParamsFound),
}

/// Helper struct to track which parameters were found in the output.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct OptimizedParamsFound {
    /// The `k` parameter was found in the output.
    pub k: bool,
    /// The `m` parameter was found in the output.
    pub m: bool,
    /// The `sigma_obs` parameter was found in the output.
    pub sigma_obs: bool,
    /// The `delta` parameter was found in the output.
    pub delta: bool,
    /// The `beta` parameter was found in the output.
    pub beta: bool,
    /// The `trend` parameter was found in the output.
    pub trend: bool,
}

#[derive(Debug)]
struct ProphetInstallation {
    #[cfg(feature = "compile-cmdstan")]
    _dir: Option<tempfile::TempDir>,
    lib_dir: PathBuf,
    prophet_binary_path: PathBuf,
}

impl Clone for ProphetInstallation {
    fn clone(&self) -> Self {
        Self {
            // We can't clone the temporary file but we also don't really need it.
            // We just need the path to the Prophet binary and library directory.
            #[cfg(feature = "compile-cmdstan")]
            _dir: None,
            lib_dir: self.lib_dir.clone(),
            prophet_binary_path: self.prophet_binary_path.clone(),
        }
    }
}

impl ProphetInstallation {
    fn command(&self) -> Command {
        let mut cmd = Command::new(&self.prophet_binary_path);
        // Use the provided LD_LIBRARY_PATH if it's set, adding on the configured lib
        // dir for good measure.
        if let Ok(ld_library_path) = std::env::var("LD_LIBRARY_PATH") {
            let path = format!("{}:{}", ld_library_path, self.lib_dir.display());
            cmd.env("LD_LIBRARY_PATH", path);
        }
        // Otherwise, just use our lib_dir.
        else {
            cmd.env("LD_LIBRARY_PATH", &self.lib_dir);
        }
        cmd
    }
}

struct OptimizeCommand<'a> {
    installation: &'a ProphetInstallation,
    init: &'a optimizer::InitialParams,
    data: &'a optimizer::Data,
    opts: &'a optimizer::OptimizeOpts,
}

impl<'a> OptimizeCommand<'a> {
    fn run(&self) -> Result<optimizer::OptimizedParams, Error> {
        // Set up temp dir and files.
        let tempdir = tempfile::tempdir()?;
        let init_path = tempdir.path().join("init.json");
        std::fs::write(
            &init_path,
            serde_json::to_vec(&self.init).map_err(Error::SerializeInits)?,
        )
        .map_err(Error::WriteInits)?;
        let data_path = tempdir.path().join("data.json");
        std::fs::write(
            &data_path,
            serde_json::to_vec(&self.data).map_err(Error::SerializeData)?,
        )
        .map_err(Error::WriteData)?;
        let output_path = tempdir.path().join("prophet_output.csv");

        // Run the command.
        let mut command = self.command(&data_path, &init_path, &output_path);
        let output = command.output().map_err(|source| Error::ProphetIOError {
            command: format!("{:?}", command),
            source,
        })?;
        if !output.status.success() {
            return Err(Error::ProphetExeError {
                command: format!("{:?}", command),
                code: output.status.code(),
                stdout: String::from_utf8_lossy(&output.stdout).to_string(),
                stderr: String::from_utf8_lossy(&output.stderr).to_string(),
            });
        }

        self.parse_output(output_path)
    }

    fn parse_output(&self, output_path: PathBuf) -> Result<optimizer::OptimizedParams, Error> {
        let output = std::fs::read_to_string(&output_path).map_err(|source| Error::ReadOutput {
            path: output_path,
            source,
        })?;
        let mut lines = output.lines().skip_while(|line| line.starts_with('#'));
        let header = lines.next().ok_or(Error::NoHeader)?.split(',');
        let data = lines
            .next()
            .ok_or(Error::NoData)?
            .split(',')
            .map(|val| val.parse());

        let mut delta_indices: Vec<usize> = Vec::new();
        let mut beta_indices: Vec<usize> = Vec::new();
        let mut trend_indices: Vec<usize> = Vec::new();
        let mut out = optimizer::OptimizedParams {
            k: 0.0,
            m: 0.0,
            sigma_obs: PositiveFloat::one(),
            delta: Vec::new(),
            beta: Vec::new(),
            trend: Vec::new(),
        };
        let mut found = OptimizedParamsFound::default();
        for (name, val) in header.zip(data) {
            match name.split_once('.') {
                Some(("delta", i)) => {
                    found.delta = true;
                    delta_indices.push(i.parse()?);
                    out.delta.push(val?);
                }
                Some(("beta", i)) => {
                    found.beta = true;
                    beta_indices.push(i.parse()?);
                    out.beta.push(val?);
                }
                Some(("trend", i)) => {
                    found.trend = true;
                    trend_indices.push(i.parse()?);
                    out.trend.push(val?);
                }
                None | Some((_, _)) => match name {
                    "k" => {
                        found.k = true;
                        out.k = val?;
                    }
                    "m" => {
                        found.m = true;
                        out.m = val?;
                    }
                    "sigma_obs" => {
                        found.sigma_obs = true;
                        out.sigma_obs = val?.try_into()?;
                    }
                    _ => {}
                },
            }
        }

        if !(found.k && found.m && found.sigma_obs && found.delta && found.beta && found.trend) {
            return Err(Error::MissingOptimizedParameters(found));
        }

        // Sort the vector params by their indices.
        // We need to subtract 1 from the indices because the params
        // returned by Stan are 1-indexed.
        out.delta = delta_indices
            .into_iter()
            .map(|i| out.delta[i - 1])
            .collect();
        out.beta = beta_indices.into_iter().map(|i| out.beta[i - 1]).collect();
        out.trend = trend_indices
            .into_iter()
            .map(|i| out.trend[i - 1])
            .collect();
        Ok(out)
    }

    fn command(&self, data_path: &Path, init_path: &Path, output_path: &Path) -> Command {
        let mut command = self.installation.command();
        command.arg("random");
        command.arg(format!(
            "seed={}",
            self.opts.seed.unwrap_or_else(|| {
                let mut rng = thread_rng();
                rng.gen_range(1..99999)
            })
        ));
        command.arg("data");
        command.args([
            format!("file={}", data_path.display()),
            format!("init={}", init_path.display()),
        ]);
        command.arg("output");
        command.arg(format!("file={}", output_path.display()));
        command.arg("method=optimize");
        command.arg(format!(
            "algorithm={}",
            self.opts.algorithm.unwrap_or(crate::Algorithm::Lbfgs)
        ));
        command
    }
}

/// Optimizer that calls out to a compiled Stan model using `cmdstan`.
///
/// See the module level documentation for more information on how to use this.
#[derive(Debug, Clone)]
pub struct CmdstanOptimizer {
    prophet_installation: ProphetInstallation,
}

impl CmdstanOptimizer {
    /// Create a new [`CmdstanOptimizer`] using the Prophet model compiled at build-time.
    ///
    /// This works by embedding the built Prophet model binary into the executable and
    /// writing it to a temporary file at runtime when this method is first called.
    ///
    /// This is only available if the `compile-cmdstan` feature is enabled.
    ///
    /// It will fail at compile-time if the Prophet model wasn't built by the build
    /// script. Generally this shouldn't ever happen (since the build script will fail),
    /// but there is always a chance that the built file is deleted in between
    /// the build script running and compilation!
    ///
    /// # Panics
    ///
    /// This function will panic if the temporary file could not be created, or if the
    /// Prophet model binary could not be written to the temporary file.
    #[cfg(feature = "compile-cmdstan")]
    pub fn new_embedded() -> Self {
        static PROPHET_INSTALLATION: std::sync::LazyLock<ProphetInstallation> =
            std::sync::LazyLock::new(|| {
                static PROPHET_BINARY: &[u8] = include_bytes!("../build/prophet");
                static LIBTBB_SO: &[u8] = include_bytes!("../build/libtbb.so.12");

                let dir = tempfile::tempdir().expect("could not create temporary directory");
                let lib_dir = dir.path().join("lib");
                std::fs::create_dir_all(&lib_dir).expect("could not create lib directory");
                let prophet_binary_path = dir.path().join("prophet");
                let libtbb_path = lib_dir.join("libtbb.so.12");

                // Write the Prophet model binary to the temporary directory.
                std::fs::write(&prophet_binary_path, PROPHET_BINARY)
                    .expect("could not write prophet model to temporary file");
                // Write the libtbb.so file to the temporary directory.
                std::fs::write(&libtbb_path, LIBTBB_SO)
                    .expect("could not write libtbb to temporary file");

                #[cfg(unix)]
                {
                    use std::os::unix::fs::PermissionsExt;
                    std::fs::set_permissions(
                        &prophet_binary_path,
                        std::fs::Permissions::from_mode(0o755),
                    )
                    .expect("could not set permissions on Prophet binary");
                }

                ProphetInstallation {
                    _dir: Some(dir),
                    prophet_binary_path,
                    lib_dir,
                }
            });

        Self {
            prophet_installation: PROPHET_INSTALLATION.clone(),
        }
    }

    /// Create a new [`CmdstanOptimizer`] using the Prophet model found the provided path.
    pub fn with_prophet_path(prophet_path: impl Into<PathBuf>) -> Result<Self, Error> {
        let prophet_binary_path = prophet_path.into();
        // Assume that the libtbb library is at the `lib` subdirectory of
        // the Prophet binary, as arranged by the `download-stan-model`
        // convenience script.
        let lib_dir = prophet_binary_path
            .parent()
            .ok_or_else(|| Error::ProphetPathHasNoParent(prophet_binary_path.to_path_buf()))?
            .join("lib");
        let prophet_installation = ProphetInstallation {
            #[cfg(feature = "compile-cmdstan")]
            _dir: None,
            lib_dir,
            prophet_binary_path,
        };
        // Test that the command can be executed.
        let mut command = prophet_installation.command();
        command.arg("help");
        let output = command.output().map_err(|source| Error::ProphetIOError {
            command: format!("{:?}", command),
            source,
        })?;
        if !output.status.success() {
            return Err(Error::ProphetExeError {
                command: format!("{:?}", command),
                code: output.status.code(),
                stdout: String::from_utf8_lossy(&output.stdout).to_string(),
                stderr: String::from_utf8_lossy(&output.stderr).to_string(),
            });
        }
        Ok(Self {
            prophet_installation,
        })
    }
}

impl Optimizer for CmdstanOptimizer {
    fn optimize(
        &self,
        init: &optimizer::InitialParams,
        data: &optimizer::Data,
        opts: &optimizer::OptimizeOpts,
    ) -> Result<optimizer::OptimizedParams, optimizer::Error> {
        OptimizeCommand {
            installation: &self.prophet_installation,
            init,
            data,
            opts,
        }
        .run()
        .map_err(optimizer::Error::custom)
    }
}
