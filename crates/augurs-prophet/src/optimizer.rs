//! Methods for optimizing the Prophet model.
//!
//! This module contains the `Optimize` trait, which represents
//! a way of finding the optimal parameters for the Prophet model
//! given the data.
//!
//! The original Prophet library uses Stan for this; specifically,
//! it uses the `optimize` command of Stan to find the maximum
//! likelihood estimate (or maximum a-priori estimates) of the
//! parameters.
//!
//! The `cmdstan` feature of this crate provides an implementation
//! of the `Optimize` trait that uses `cmdstan` to do the same.
//! This requires a working installation of `cmdstan`.
//!
//! The `libstan` feature uses FFI calls to call out to the Stan
//! C++ library to do the same. This requires a C++ compiler.
//!
// TODO: actually add these features.
// TODO: come up with a way of doing something in WASM. Maybe
//       WASM Components?
// TODO: write a pure Rust optimizer for the default case.

use crate::positive_float::PositiveFloat;

/// The initial parameters for the optimization.
#[derive(Clone, Debug, PartialEq)]
pub struct InitialParams {
    /// Base trend growth rate.
    pub k: f64,
    /// Trend offset.
    pub m: f64,
    /// Trend rate adjustments, length s in data.
    pub delta: Vec<f64>,
    /// Regressor coefficients, length k in data.
    pub beta: Vec<f64>,
    /// Observation noise.
    pub sigma_obs: f64,
}

/// The type of trend to use.
#[derive(Clone, Debug, Copy, Eq, PartialEq)]
pub enum TrendIndicator {
    /// Linear trend (default).
    Linear,
    /// Logistic trend.
    Logistic,
    /// Flat trend.
    Flat,
}

/// Data for the Prophet model.
#[derive(Clone, Debug, PartialEq)]
#[allow(non_snake_case)]
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
    pub trend_indicator: TrendIndicator,
    /// Number of regressors.
    /// Must be greater than or equal to 1.
    pub K: i32,
    /// Indicator of additive features, length k.
    pub s_a: Vec<i32>,
    /// Indicator of multiplicative features, length k.
    pub s_m: Vec<i32>,
    /// Regressors, shape (n, k).
    pub X: Vec<Vec<f64>>,
    /// Scale on seasonality prior.
    pub sigmas: Vec<PositiveFloat>,
    /// Scale on changepoints prior.
    /// Must be greater than 0.
    pub tau: PositiveFloat,
}

/// The algorithm to use for optimization. One of: 'BFGS', 'LBFGS', 'Newton'.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum Algorithm {
    /// Use the Newton algorithm.
    Newton,
    /// Use the Broyden-Fletcher-Goldfarb-Shanno (BFGS) algorithm.
    Bfgs,
    /// Use the Limited-memory BFGS (L-BFGS) algorithm.
    Lbfgs,
}

/// Arguments for optimization.
#[derive(Default, Debug, Clone, Copy)]
pub struct OptimizeOpts {
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
}

/// The optimized parameters.
#[derive(Debug, Clone)]
pub struct OptimizedParams {
    /// Base trend growth rate.
    pub k: f64,
    /// Trend offset.
    pub m: f64,
    /// Observation noise.
    pub sigma_obs: f64,
    /// Trend rate adjustments.
    pub delta: Vec<f64>,
    /// Regressor coefficients.
    pub beta: Vec<f64>,
    /// Transformed trend.
    pub trend: Vec<f64>,
}

/// An error that occurred during the optimization procedure.
#[derive(Debug, thiserror::Error)]
#[error(transparent)]
pub struct Error(
    /// The kind of error that occurred.
    ///
    /// This is a private field so that we can evolve
    /// the `ErrorKind` enum without breaking changes.
    #[from]
    ErrorKind,
);

impl Error {
    /// A static string error.
    pub fn static_str(s: &'static str) -> Self {
        Self(ErrorKind::StaticStr(s))
    }

    /// A string error.
    pub fn string(s: String) -> Self {
        Self(ErrorKind::String(s))
    }

    /// A custom error, which is any type that implements `std::error::Error`.
    pub fn custom<E: std::error::Error + 'static>(e: E) -> Self {
        Self(ErrorKind::Custom(Box::new(e)))
    }
}

#[derive(Debug, thiserror::Error)]
enum ErrorKind {
    #[error("Error in optimization: {0}")]
    StaticStr(&'static str),
    #[error("Error in optimization: {0}")]
    String(String),
    #[error("Error in optimization: {0}")]
    Custom(#[from] Box<dyn std::error::Error>),
}

/// A type that can run maximum likelihood estimation optimization
/// for the Prophet model.
pub trait Optimizer: std::fmt::Debug {
    /// Find the maximum likelihood estimate of the parameters given the
    /// data, initial parameters and optimization options.
    fn optimize(
        &self,
        init: InitialParams,
        data: Data,
        opts: OptimizeOpts,
    ) -> Result<OptimizedParams, Error>;
}

#[cfg(test)]
pub mod mock_optimizer {
    use std::cell::RefCell;

    use super::*;

    #[derive(Debug, Clone)]
    pub(crate) struct OptimizeCall {
        pub init: InitialParams,
        pub data: Data,
        pub _opts: OptimizeOpts,
    }

    /// A mock optimizer that records the optimization call.
    #[derive(Clone, Debug)]
    pub(crate) struct MockOptimizer {
        /// The optimization call.
        ///
        /// This will be updated by the mock optimizer when
        /// [`Optimizer::optimize`] is called.
        // [`Optimizer::optimize`] takes self by reference,
        // so we need to store the call in a RefCell.
        pub call: RefCell<Option<OptimizeCall>>,
    }

    impl MockOptimizer {
        /// Create a new mock optimizer.
        pub(crate) fn new() -> Self {
            Self {
                call: RefCell::new(None),
            }
        }

        /// Take the optimization call out of the mock.
        pub(crate) fn take_call(&self) -> Option<OptimizeCall> {
            self.call.borrow_mut().take()
        }
    }

    impl Optimizer for MockOptimizer {
        fn optimize(
            &self,
            init: InitialParams,
            data: Data,
            opts: OptimizeOpts,
        ) -> Result<OptimizedParams, Error> {
            *self.call.borrow_mut() = Some(OptimizeCall {
                init: init.clone(),
                data,
                _opts: opts,
            });
            Ok(OptimizedParams {
                k: init.k,
                m: init.m,
                sigma_obs: init.sigma_obs,
                delta: init.delta,
                beta: init.beta,
                trend: Vec::new(),
            })
        }
    }
}
