use augurs_prophet::optimizer::{
    Algorithm, Data, Error, InitialParams, OptimizeOpts, OptimizedParams, Optimizer, TrendIndicator,
};
use bytemuck::cast_slice;

use crate::bindings::augurs::prophet_wasmstan::{self, optimizer::optimize};

#[derive(Debug)]
pub struct StanOptimizer;

impl Optimizer for StanOptimizer {
    fn optimize(
        &self,
        init: &InitialParams,
        data: &Data,
        opts: &OptimizeOpts,
    ) -> Result<OptimizedParams, Error> {
        optimize(init.into(), data.into(), opts.into())
            .map_err(Error::string)?
            .params
            .try_into()
    }
}

// Add a bunch of type conversions from `prophet_core` to `prophet_wasmstan` or vice versa.

impl<'a> From<&'a InitialParams> for prophet_wasmstan::types::InitsParam<'a> {
    fn from(params: &'a InitialParams) -> Self {
        Self {
            k: params.k,
            m: params.m,
            delta: &params.delta,
            beta: &params.beta,
            sigma_obs: *params.sigma_obs,
        }
    }
}

impl<'a> From<&'a Data> for prophet_wasmstan::types::DataParam<'a> {
    fn from(data: &'a Data) -> Self {
        // Sigmas from augurs are `PositiveFloat`s, but we need to pass
        // the f64 into `prophet_wasmstan`. Since `PositiveFloat` is just a
        // newtype wrapper with `#[repr(transparent)]` we can use
        // bytemuck to cast the slice and reinterpret it as a slice of
        // f64s.
        let sigmas = cast_slice(data.sigmas.as_slice());
        Self {
            n: data.T,
            y: &data.y,
            t: &data.t,
            cap: &data.cap,
            s: data.S,
            t_change: &data.t_change,
            trend_indicator: data.trend_indicator.into(),
            k: data.K,
            s_a: &data.s_a,
            s_m: &data.s_m,
            x: &data.X,
            sigmas,
            tau: data.tau.into(),
        }
    }
}

impl From<&OptimizeOpts> for prophet_wasmstan::types::OptimizeOpts {
    fn from(opts: &OptimizeOpts) -> Self {
        Self {
            algorithm: opts.algorithm.map(|x| x.into()),
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

impl From<prophet_wasmstan::types::OptimizeOpts> for OptimizeOpts {
    fn from(opts: prophet_wasmstan::types::OptimizeOpts) -> Self {
        Self {
            algorithm: opts.algorithm.map(|x| x.into()),
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

impl From<Algorithm> for prophet_wasmstan::types::Algorithm {
    fn from(algorithm: Algorithm) -> Self {
        match algorithm {
            Algorithm::Newton => Self::Newton,
            Algorithm::Bfgs => Self::Bfgs,
            Algorithm::Lbfgs => Self::Lbfgs,
        }
    }
}

impl From<prophet_wasmstan::types::Algorithm> for Algorithm {
    fn from(algorithm: prophet_wasmstan::types::Algorithm) -> Self {
        match algorithm {
            prophet_wasmstan::types::Algorithm::Newton => Self::Newton,
            prophet_wasmstan::types::Algorithm::Bfgs => Self::Bfgs,
            prophet_wasmstan::types::Algorithm::Lbfgs => Self::Lbfgs,
        }
    }
}

impl From<TrendIndicator> for prophet_wasmstan::types::TrendIndicator {
    fn from(trend: TrendIndicator) -> Self {
        match trend {
            TrendIndicator::Linear => Self::Linear,
            TrendIndicator::Logistic => Self::Logistic,
            TrendIndicator::Flat => Self::Flat,
        }
    }
}

impl TryFrom<prophet_wasmstan::types::OptimizedParams>
    for augurs_prophet::optimizer::OptimizedParams
{
    type Error = Error;
    fn try_from(x: prophet_wasmstan::types::OptimizedParams) -> Result<Self, Self::Error> {
        Ok(augurs_prophet::optimizer::OptimizedParams {
            k: x.k,
            m: x.m,
            sigma_obs: x.sigma_obs.try_into().map_err(Error::custom)?,
            delta: x.delta,
            beta: x.beta,
            trend: x.trend,
        })
    }
}
