#[allow(dead_code)]
pub mod augurs {
    #[allow(dead_code)]
    pub mod prophet_wasmstan {
        #[allow(dead_code, clippy::all)]
        pub mod types {
            #[used]
            #[doc(hidden)]
            static __FORCE_SECTION_REF: fn() = super::super::super::__link_custom_section_describing_imports;
            use super::super::super::_rt;
            /// The initial parameters for the optimization.
            #[derive(Clone)]
            pub struct InitsResult {
                /// Base trend growth rate.
                pub k: f64,
                /// Trend offset.
                pub m: f64,
                /// Trend rate adjustments, length s in data.
                pub delta: _rt::Vec<f64>,
                /// Regressor coefficients, length k in data.
                pub beta: _rt::Vec<f64>,
                /// Observation noise.
                pub sigma_obs: f64,
            }
            impl ::core::fmt::Debug for InitsResult {
                fn fmt(
                    &self,
                    f: &mut ::core::fmt::Formatter<'_>,
                ) -> ::core::fmt::Result {
                    f.debug_struct("InitsResult")
                        .field("k", &self.k)
                        .field("m", &self.m)
                        .field("delta", &self.delta)
                        .field("beta", &self.beta)
                        .field("sigma-obs", &self.sigma_obs)
                        .finish()
                }
            }
            /// The initial parameters for the optimization.
            #[derive(Clone)]
            pub struct InitsParam<'a> {
                /// Base trend growth rate.
                pub k: f64,
                /// Trend offset.
                pub m: f64,
                /// Trend rate adjustments, length s in data.
                pub delta: &'a [f64],
                /// Regressor coefficients, length k in data.
                pub beta: &'a [f64],
                /// Observation noise.
                pub sigma_obs: f64,
            }
            impl<'a> ::core::fmt::Debug for InitsParam<'a> {
                fn fmt(
                    &self,
                    f: &mut ::core::fmt::Formatter<'_>,
                ) -> ::core::fmt::Result {
                    f.debug_struct("InitsParam")
                        .field("k", &self.k)
                        .field("m", &self.m)
                        .field("delta", &self.delta)
                        .field("beta", &self.beta)
                        .field("sigma-obs", &self.sigma_obs)
                        .finish()
                }
            }
            /// The type of trend to use.
            #[repr(u8)]
            #[derive(Clone, Copy, Eq, Ord, PartialEq, PartialOrd)]
            pub enum TrendIndicator {
                /// Linear trend (default).
                Linear,
                /// 0
                /// Logistic trend.
                Logistic,
                /// 1
                /// Flat trend.
                Flat,
            }
            impl ::core::fmt::Debug for TrendIndicator {
                fn fmt(
                    &self,
                    f: &mut ::core::fmt::Formatter<'_>,
                ) -> ::core::fmt::Result {
                    match self {
                        TrendIndicator::Linear => {
                            f.debug_tuple("TrendIndicator::Linear").finish()
                        }
                        TrendIndicator::Logistic => {
                            f.debug_tuple("TrendIndicator::Logistic").finish()
                        }
                        TrendIndicator::Flat => {
                            f.debug_tuple("TrendIndicator::Flat").finish()
                        }
                    }
                }
            }
            impl TrendIndicator {
                #[doc(hidden)]
                pub unsafe fn _lift(val: u8) -> TrendIndicator {
                    if !cfg!(debug_assertions) {
                        return ::core::mem::transmute(val);
                    }
                    match val {
                        0 => TrendIndicator::Linear,
                        1 => TrendIndicator::Logistic,
                        2 => TrendIndicator::Flat,
                        _ => panic!("invalid enum discriminant"),
                    }
                }
            }
            /// Data for the Prophet model.
            #[derive(Clone)]
            pub struct DataResult {
                /// Number of time periods.
                /// This is `T` in the Prophet STAN model definition,
                /// but WIT identifiers must be lower kebab-case.
                pub n: i32,
                /// Time series, length n.
                pub y: _rt::Vec<f64>,
                /// Time, length n.
                pub t: _rt::Vec<f64>,
                /// Capacities for logistic trend, length n.
                pub cap: _rt::Vec<f64>,
                /// Number of changepoints.
                /// This is 'S' in the Prophet STAN model definition,
                /// but WIT identifiers must be lower kebab-case.
                pub s: i32,
                /// Times of trend changepoints, length s.
                pub t_change: _rt::Vec<f64>,
                /// The type of trend to use.
                pub trend_indicator: TrendIndicator,
                /// Number of regressors.
                /// Must be greater than or equal to 1.
                /// This is `K` in the Prophet STAN model definition,
                /// but WIT identifiers must be lower kebab-case.
                pub k: i32,
                /// Indicator of additive features, length k.
                /// This is `s_a` in the Prophet STAN model definition,
                /// but WIT identifiers must be lower kebab-case.
                pub s_a: _rt::Vec<i32>,
                /// Indicator of multiplicative features, length k.
                /// This is `s_m` in the Prophet STAN model definition,
                /// but WIT identifiers must be lower kebab-case.
                pub s_m: _rt::Vec<i32>,
                /// Regressors.
                /// This is `X` in the Prophet STAN model definition,
                /// but WIT identifiers must be lower kebab-case.
                /// This is passed as a flat array but should be treated as
                /// a matrix with shape (n, k) (i.e. strides of length n).
                pub x: _rt::Vec<f64>,
                /// Scale on seasonality prior.
                pub sigmas: _rt::Vec<f64>,
                /// Scale on changepoints prior.
                /// Must be greater than 0.
                pub tau: f64,
            }
            impl ::core::fmt::Debug for DataResult {
                fn fmt(
                    &self,
                    f: &mut ::core::fmt::Formatter<'_>,
                ) -> ::core::fmt::Result {
                    f.debug_struct("DataResult")
                        .field("n", &self.n)
                        .field("y", &self.y)
                        .field("t", &self.t)
                        .field("cap", &self.cap)
                        .field("s", &self.s)
                        .field("t-change", &self.t_change)
                        .field("trend-indicator", &self.trend_indicator)
                        .field("k", &self.k)
                        .field("s-a", &self.s_a)
                        .field("s-m", &self.s_m)
                        .field("x", &self.x)
                        .field("sigmas", &self.sigmas)
                        .field("tau", &self.tau)
                        .finish()
                }
            }
            /// Data for the Prophet model.
            #[derive(Clone)]
            pub struct DataParam<'a> {
                /// Number of time periods.
                /// This is `T` in the Prophet STAN model definition,
                /// but WIT identifiers must be lower kebab-case.
                pub n: i32,
                /// Time series, length n.
                pub y: &'a [f64],
                /// Time, length n.
                pub t: &'a [f64],
                /// Capacities for logistic trend, length n.
                pub cap: &'a [f64],
                /// Number of changepoints.
                /// This is 'S' in the Prophet STAN model definition,
                /// but WIT identifiers must be lower kebab-case.
                pub s: i32,
                /// Times of trend changepoints, length s.
                pub t_change: &'a [f64],
                /// The type of trend to use.
                pub trend_indicator: TrendIndicator,
                /// Number of regressors.
                /// Must be greater than or equal to 1.
                /// This is `K` in the Prophet STAN model definition,
                /// but WIT identifiers must be lower kebab-case.
                pub k: i32,
                /// Indicator of additive features, length k.
                /// This is `s_a` in the Prophet STAN model definition,
                /// but WIT identifiers must be lower kebab-case.
                pub s_a: &'a [i32],
                /// Indicator of multiplicative features, length k.
                /// This is `s_m` in the Prophet STAN model definition,
                /// but WIT identifiers must be lower kebab-case.
                pub s_m: &'a [i32],
                /// Regressors.
                /// This is `X` in the Prophet STAN model definition,
                /// but WIT identifiers must be lower kebab-case.
                /// This is passed as a flat array but should be treated as
                /// a matrix with shape (n, k) (i.e. strides of length n).
                pub x: &'a [f64],
                /// Scale on seasonality prior.
                pub sigmas: &'a [f64],
                /// Scale on changepoints prior.
                /// Must be greater than 0.
                pub tau: f64,
            }
            impl<'a> ::core::fmt::Debug for DataParam<'a> {
                fn fmt(
                    &self,
                    f: &mut ::core::fmt::Formatter<'_>,
                ) -> ::core::fmt::Result {
                    f.debug_struct("DataParam")
                        .field("n", &self.n)
                        .field("y", &self.y)
                        .field("t", &self.t)
                        .field("cap", &self.cap)
                        .field("s", &self.s)
                        .field("t-change", &self.t_change)
                        .field("trend-indicator", &self.trend_indicator)
                        .field("k", &self.k)
                        .field("s-a", &self.s_a)
                        .field("s-m", &self.s_m)
                        .field("x", &self.x)
                        .field("sigmas", &self.sigmas)
                        .field("tau", &self.tau)
                        .finish()
                }
            }
            /// The algorithm to use for optimization. One of: 'BFGS', 'LBFGS', 'Newton'.
            #[repr(u8)]
            #[derive(Clone, Copy, Eq, Ord, PartialEq, PartialOrd)]
            pub enum Algorithm {
                /// Use the Newton algorithm.
                Newton,
                /// Use the Broyden-Fletcher-Goldfarb-Shanno (BFGS) algorithm.
                Bfgs,
                /// Use the Limited-memory BFGS (L-BFGS) algorithm.
                Lbfgs,
            }
            impl ::core::fmt::Debug for Algorithm {
                fn fmt(
                    &self,
                    f: &mut ::core::fmt::Formatter<'_>,
                ) -> ::core::fmt::Result {
                    match self {
                        Algorithm::Newton => f.debug_tuple("Algorithm::Newton").finish(),
                        Algorithm::Bfgs => f.debug_tuple("Algorithm::Bfgs").finish(),
                        Algorithm::Lbfgs => f.debug_tuple("Algorithm::Lbfgs").finish(),
                    }
                }
            }
            impl Algorithm {
                #[doc(hidden)]
                pub unsafe fn _lift(val: u8) -> Algorithm {
                    if !cfg!(debug_assertions) {
                        return ::core::mem::transmute(val);
                    }
                    match val {
                        0 => Algorithm::Newton,
                        1 => Algorithm::Bfgs,
                        2 => Algorithm::Lbfgs,
                        _ => panic!("invalid enum discriminant"),
                    }
                }
            }
            /// Arguments for optimization.
            #[repr(C)]
            #[derive(Clone, Copy)]
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
                /// How frequently to update the log message, in number of iterations.
                pub refresh: Option<u32>,
            }
            impl ::core::fmt::Debug for OptimizeOpts {
                fn fmt(
                    &self,
                    f: &mut ::core::fmt::Formatter<'_>,
                ) -> ::core::fmt::Result {
                    f.debug_struct("OptimizeOpts")
                        .field("algorithm", &self.algorithm)
                        .field("seed", &self.seed)
                        .field("chain", &self.chain)
                        .field("init-alpha", &self.init_alpha)
                        .field("tol-obj", &self.tol_obj)
                        .field("tol-rel-obj", &self.tol_rel_obj)
                        .field("tol-grad", &self.tol_grad)
                        .field("tol-rel-grad", &self.tol_rel_grad)
                        .field("tol-param", &self.tol_param)
                        .field("history-size", &self.history_size)
                        .field("iter", &self.iter)
                        .field("jacobian", &self.jacobian)
                        .field("refresh", &self.refresh)
                        .finish()
                }
            }
            /// Log lines produced during optimization.
            #[derive(Clone)]
            pub struct Logs {
                /// Debug log lines.
                pub debug: _rt::String,
                /// Info log lines.
                pub info: _rt::String,
                /// Warning log lines.
                pub warn: _rt::String,
                /// Error log lines.
                pub error: _rt::String,
                /// Fatal log lines.
                pub fatal: _rt::String,
            }
            impl ::core::fmt::Debug for Logs {
                fn fmt(
                    &self,
                    f: &mut ::core::fmt::Formatter<'_>,
                ) -> ::core::fmt::Result {
                    f.debug_struct("Logs")
                        .field("debug", &self.debug)
                        .field("info", &self.info)
                        .field("warn", &self.warn)
                        .field("error", &self.error)
                        .field("fatal", &self.fatal)
                        .finish()
                }
            }
            /// The optimal parameter values found by optimization.
            #[derive(Clone)]
            pub struct OptimizedParams {
                /// Base trend growth rate.
                pub k: f64,
                /// Trend offset.
                pub m: f64,
                /// Trend rate adjustments, length s in data.
                pub delta: _rt::Vec<f64>,
                /// Regressor coefficients, length k in data.
                pub beta: _rt::Vec<f64>,
                /// Observation noise.
                pub sigma_obs: f64,
                /// Transformed trend.
                pub trend: _rt::Vec<f64>,
            }
            impl ::core::fmt::Debug for OptimizedParams {
                fn fmt(
                    &self,
                    f: &mut ::core::fmt::Formatter<'_>,
                ) -> ::core::fmt::Result {
                    f.debug_struct("OptimizedParams")
                        .field("k", &self.k)
                        .field("m", &self.m)
                        .field("delta", &self.delta)
                        .field("beta", &self.beta)
                        .field("sigma-obs", &self.sigma_obs)
                        .field("trend", &self.trend)
                        .finish()
                }
            }
            /// The result of optimization.
            ///
            /// This includes both the parameters and any logs produced by the
            /// process.
            #[derive(Clone)]
            pub struct OptimizeOutput {
                /// Logs produced by the optimization process.
                pub logs: Logs,
                /// The optimized parameters.
                pub params: OptimizedParams,
            }
            impl ::core::fmt::Debug for OptimizeOutput {
                fn fmt(
                    &self,
                    f: &mut ::core::fmt::Formatter<'_>,
                ) -> ::core::fmt::Result {
                    f.debug_struct("OptimizeOutput")
                        .field("logs", &self.logs)
                        .field("params", &self.params)
                        .finish()
                }
            }
        }
        #[allow(dead_code, clippy::all)]
        pub mod optimizer {
            #[used]
            #[doc(hidden)]
            static __FORCE_SECTION_REF: fn() = super::super::super::__link_custom_section_describing_imports;
            use super::super::super::_rt;
            pub type InitsResult = super::super::super::augurs::prophet_wasmstan::types::InitsResult;
            pub type InitsParam<'a> = super::super::super::augurs::prophet_wasmstan::types::InitsParam<
                'a,
            >;
            pub type DataResult = super::super::super::augurs::prophet_wasmstan::types::DataResult;
            pub type DataParam<'a> = super::super::super::augurs::prophet_wasmstan::types::DataParam<
                'a,
            >;
            pub type OptimizeOpts = super::super::super::augurs::prophet_wasmstan::types::OptimizeOpts;
            pub type OptimizeOutput = super::super::super::augurs::prophet_wasmstan::types::OptimizeOutput;
            #[allow(unused_unsafe, clippy::all)]
            pub fn optimize(
                init: InitsParam<'_>,
                data: DataParam<'_>,
                opts: OptimizeOpts,
            ) -> Result<OptimizeOutput, _rt::String> {
                unsafe {
                    #[repr(align(8))]
                    struct RetArea([::core::mem::MaybeUninit<u8>; 280]);
                    let mut ret_area = RetArea(
                        [::core::mem::MaybeUninit::uninit(); 280],
                    );
                    let ptr0 = ret_area.0.as_mut_ptr().cast::<u8>();
                    let super::super::super::augurs::prophet_wasmstan::types::InitsParam {
                        k: k1,
                        m: m1,
                        delta: delta1,
                        beta: beta1,
                        sigma_obs: sigma_obs1,
                    } = init;
                    *ptr0.add(0).cast::<f64>() = _rt::as_f64(k1);
                    *ptr0.add(8).cast::<f64>() = _rt::as_f64(m1);
                    let vec2 = delta1;
                    let ptr2 = vec2.as_ptr().cast::<u8>();
                    let len2 = vec2.len();
                    *ptr0.add(20).cast::<usize>() = len2;
                    *ptr0.add(16).cast::<*mut u8>() = ptr2.cast_mut();
                    let vec3 = beta1;
                    let ptr3 = vec3.as_ptr().cast::<u8>();
                    let len3 = vec3.len();
                    *ptr0.add(28).cast::<usize>() = len3;
                    *ptr0.add(24).cast::<*mut u8>() = ptr3.cast_mut();
                    *ptr0.add(32).cast::<f64>() = _rt::as_f64(sigma_obs1);
                    let super::super::super::augurs::prophet_wasmstan::types::DataParam {
                        n: n4,
                        y: y4,
                        t: t4,
                        cap: cap4,
                        s: s4,
                        t_change: t_change4,
                        trend_indicator: trend_indicator4,
                        k: k4,
                        s_a: s_a4,
                        s_m: s_m4,
                        x: x4,
                        sigmas: sigmas4,
                        tau: tau4,
                    } = data;
                    *ptr0.add(40).cast::<i32>() = _rt::as_i32(n4);
                    let vec5 = y4;
                    let ptr5 = vec5.as_ptr().cast::<u8>();
                    let len5 = vec5.len();
                    *ptr0.add(48).cast::<usize>() = len5;
                    *ptr0.add(44).cast::<*mut u8>() = ptr5.cast_mut();
                    let vec6 = t4;
                    let ptr6 = vec6.as_ptr().cast::<u8>();
                    let len6 = vec6.len();
                    *ptr0.add(56).cast::<usize>() = len6;
                    *ptr0.add(52).cast::<*mut u8>() = ptr6.cast_mut();
                    let vec7 = cap4;
                    let ptr7 = vec7.as_ptr().cast::<u8>();
                    let len7 = vec7.len();
                    *ptr0.add(64).cast::<usize>() = len7;
                    *ptr0.add(60).cast::<*mut u8>() = ptr7.cast_mut();
                    *ptr0.add(68).cast::<i32>() = _rt::as_i32(s4);
                    let vec8 = t_change4;
                    let ptr8 = vec8.as_ptr().cast::<u8>();
                    let len8 = vec8.len();
                    *ptr0.add(76).cast::<usize>() = len8;
                    *ptr0.add(72).cast::<*mut u8>() = ptr8.cast_mut();
                    *ptr0.add(80).cast::<u8>() = (trend_indicator4.clone() as i32) as u8;
                    *ptr0.add(84).cast::<i32>() = _rt::as_i32(k4);
                    let vec9 = s_a4;
                    let ptr9 = vec9.as_ptr().cast::<u8>();
                    let len9 = vec9.len();
                    *ptr0.add(92).cast::<usize>() = len9;
                    *ptr0.add(88).cast::<*mut u8>() = ptr9.cast_mut();
                    let vec10 = s_m4;
                    let ptr10 = vec10.as_ptr().cast::<u8>();
                    let len10 = vec10.len();
                    *ptr0.add(100).cast::<usize>() = len10;
                    *ptr0.add(96).cast::<*mut u8>() = ptr10.cast_mut();
                    let vec11 = x4;
                    let ptr11 = vec11.as_ptr().cast::<u8>();
                    let len11 = vec11.len();
                    *ptr0.add(108).cast::<usize>() = len11;
                    *ptr0.add(104).cast::<*mut u8>() = ptr11.cast_mut();
                    let vec12 = sigmas4;
                    let ptr12 = vec12.as_ptr().cast::<u8>();
                    let len12 = vec12.len();
                    *ptr0.add(116).cast::<usize>() = len12;
                    *ptr0.add(112).cast::<*mut u8>() = ptr12.cast_mut();
                    *ptr0.add(120).cast::<f64>() = _rt::as_f64(tau4);
                    let super::super::super::augurs::prophet_wasmstan::types::OptimizeOpts {
                        algorithm: algorithm13,
                        seed: seed13,
                        chain: chain13,
                        init_alpha: init_alpha13,
                        tol_obj: tol_obj13,
                        tol_rel_obj: tol_rel_obj13,
                        tol_grad: tol_grad13,
                        tol_rel_grad: tol_rel_grad13,
                        tol_param: tol_param13,
                        history_size: history_size13,
                        iter: iter13,
                        jacobian: jacobian13,
                        refresh: refresh13,
                    } = opts;
                    match algorithm13 {
                        Some(e) => {
                            *ptr0.add(128).cast::<u8>() = (1i32) as u8;
                            *ptr0.add(129).cast::<u8>() = (e.clone() as i32) as u8;
                        }
                        None => {
                            *ptr0.add(128).cast::<u8>() = (0i32) as u8;
                        }
                    };
                    match seed13 {
                        Some(e) => {
                            *ptr0.add(132).cast::<u8>() = (1i32) as u8;
                            *ptr0.add(136).cast::<i32>() = _rt::as_i32(e);
                        }
                        None => {
                            *ptr0.add(132).cast::<u8>() = (0i32) as u8;
                        }
                    };
                    match chain13 {
                        Some(e) => {
                            *ptr0.add(140).cast::<u8>() = (1i32) as u8;
                            *ptr0.add(144).cast::<i32>() = _rt::as_i32(e);
                        }
                        None => {
                            *ptr0.add(140).cast::<u8>() = (0i32) as u8;
                        }
                    };
                    match init_alpha13 {
                        Some(e) => {
                            *ptr0.add(152).cast::<u8>() = (1i32) as u8;
                            *ptr0.add(160).cast::<f64>() = _rt::as_f64(e);
                        }
                        None => {
                            *ptr0.add(152).cast::<u8>() = (0i32) as u8;
                        }
                    };
                    match tol_obj13 {
                        Some(e) => {
                            *ptr0.add(168).cast::<u8>() = (1i32) as u8;
                            *ptr0.add(176).cast::<f64>() = _rt::as_f64(e);
                        }
                        None => {
                            *ptr0.add(168).cast::<u8>() = (0i32) as u8;
                        }
                    };
                    match tol_rel_obj13 {
                        Some(e) => {
                            *ptr0.add(184).cast::<u8>() = (1i32) as u8;
                            *ptr0.add(192).cast::<f64>() = _rt::as_f64(e);
                        }
                        None => {
                            *ptr0.add(184).cast::<u8>() = (0i32) as u8;
                        }
                    };
                    match tol_grad13 {
                        Some(e) => {
                            *ptr0.add(200).cast::<u8>() = (1i32) as u8;
                            *ptr0.add(208).cast::<f64>() = _rt::as_f64(e);
                        }
                        None => {
                            *ptr0.add(200).cast::<u8>() = (0i32) as u8;
                        }
                    };
                    match tol_rel_grad13 {
                        Some(e) => {
                            *ptr0.add(216).cast::<u8>() = (1i32) as u8;
                            *ptr0.add(224).cast::<f64>() = _rt::as_f64(e);
                        }
                        None => {
                            *ptr0.add(216).cast::<u8>() = (0i32) as u8;
                        }
                    };
                    match tol_param13 {
                        Some(e) => {
                            *ptr0.add(232).cast::<u8>() = (1i32) as u8;
                            *ptr0.add(240).cast::<f64>() = _rt::as_f64(e);
                        }
                        None => {
                            *ptr0.add(232).cast::<u8>() = (0i32) as u8;
                        }
                    };
                    match history_size13 {
                        Some(e) => {
                            *ptr0.add(248).cast::<u8>() = (1i32) as u8;
                            *ptr0.add(252).cast::<i32>() = _rt::as_i32(e);
                        }
                        None => {
                            *ptr0.add(248).cast::<u8>() = (0i32) as u8;
                        }
                    };
                    match iter13 {
                        Some(e) => {
                            *ptr0.add(256).cast::<u8>() = (1i32) as u8;
                            *ptr0.add(260).cast::<i32>() = _rt::as_i32(e);
                        }
                        None => {
                            *ptr0.add(256).cast::<u8>() = (0i32) as u8;
                        }
                    };
                    match jacobian13 {
                        Some(e) => {
                            *ptr0.add(264).cast::<u8>() = (1i32) as u8;
                            *ptr0.add(265).cast::<u8>() = (match e {
                                true => 1,
                                false => 0,
                            }) as u8;
                        }
                        None => {
                            *ptr0.add(264).cast::<u8>() = (0i32) as u8;
                        }
                    };
                    match refresh13 {
                        Some(e) => {
                            *ptr0.add(268).cast::<u8>() = (1i32) as u8;
                            *ptr0.add(272).cast::<i32>() = _rt::as_i32(e);
                        }
                        None => {
                            *ptr0.add(268).cast::<u8>() = (0i32) as u8;
                        }
                    };
                    let ptr14 = ret_area.0.as_mut_ptr().cast::<u8>();
                    #[cfg(target_arch = "wasm32")]
                    #[link(wasm_import_module = "augurs:prophet-wasmstan/optimizer")]
                    extern "C" {
                        #[link_name = "optimize"]
                        fn wit_import(_: *mut u8, _: *mut u8);
                    }
                    #[cfg(not(target_arch = "wasm32"))]
                    fn wit_import(_: *mut u8, _: *mut u8) {
                        unreachable!()
                    }
                    wit_import(ptr0, ptr14);
                    let l15 = i32::from(*ptr14.add(0).cast::<u8>());
                    match l15 {
                        0 => {
                            let e = {
                                let l16 = *ptr14.add(8).cast::<*mut u8>();
                                let l17 = *ptr14.add(12).cast::<usize>();
                                let len18 = l17;
                                let bytes18 = _rt::Vec::from_raw_parts(
                                    l16.cast(),
                                    len18,
                                    len18,
                                );
                                let l19 = *ptr14.add(16).cast::<*mut u8>();
                                let l20 = *ptr14.add(20).cast::<usize>();
                                let len21 = l20;
                                let bytes21 = _rt::Vec::from_raw_parts(
                                    l19.cast(),
                                    len21,
                                    len21,
                                );
                                let l22 = *ptr14.add(24).cast::<*mut u8>();
                                let l23 = *ptr14.add(28).cast::<usize>();
                                let len24 = l23;
                                let bytes24 = _rt::Vec::from_raw_parts(
                                    l22.cast(),
                                    len24,
                                    len24,
                                );
                                let l25 = *ptr14.add(32).cast::<*mut u8>();
                                let l26 = *ptr14.add(36).cast::<usize>();
                                let len27 = l26;
                                let bytes27 = _rt::Vec::from_raw_parts(
                                    l25.cast(),
                                    len27,
                                    len27,
                                );
                                let l28 = *ptr14.add(40).cast::<*mut u8>();
                                let l29 = *ptr14.add(44).cast::<usize>();
                                let len30 = l29;
                                let bytes30 = _rt::Vec::from_raw_parts(
                                    l28.cast(),
                                    len30,
                                    len30,
                                );
                                let l31 = *ptr14.add(48).cast::<f64>();
                                let l32 = *ptr14.add(56).cast::<f64>();
                                let l33 = *ptr14.add(64).cast::<*mut u8>();
                                let l34 = *ptr14.add(68).cast::<usize>();
                                let len35 = l34;
                                let l36 = *ptr14.add(72).cast::<*mut u8>();
                                let l37 = *ptr14.add(76).cast::<usize>();
                                let len38 = l37;
                                let l39 = *ptr14.add(80).cast::<f64>();
                                let l40 = *ptr14.add(88).cast::<*mut u8>();
                                let l41 = *ptr14.add(92).cast::<usize>();
                                let len42 = l41;
                                super::super::super::augurs::prophet_wasmstan::types::OptimizeOutput {
                                    logs: super::super::super::augurs::prophet_wasmstan::types::Logs {
                                        debug: _rt::string_lift(bytes18),
                                        info: _rt::string_lift(bytes21),
                                        warn: _rt::string_lift(bytes24),
                                        error: _rt::string_lift(bytes27),
                                        fatal: _rt::string_lift(bytes30),
                                    },
                                    params: super::super::super::augurs::prophet_wasmstan::types::OptimizedParams {
                                        k: l31,
                                        m: l32,
                                        delta: _rt::Vec::from_raw_parts(l33.cast(), len35, len35),
                                        beta: _rt::Vec::from_raw_parts(l36.cast(), len38, len38),
                                        sigma_obs: l39,
                                        trend: _rt::Vec::from_raw_parts(l40.cast(), len42, len42),
                                    },
                                }
                            };
                            Ok(e)
                        }
                        1 => {
                            let e = {
                                let l43 = *ptr14.add(8).cast::<*mut u8>();
                                let l44 = *ptr14.add(12).cast::<usize>();
                                let len45 = l44;
                                let bytes45 = _rt::Vec::from_raw_parts(
                                    l43.cast(),
                                    len45,
                                    len45,
                                );
                                _rt::string_lift(bytes45)
                            };
                            Err(e)
                        }
                        _ => _rt::invalid_enum_discriminant(),
                    }
                }
            }
        }
    }
}
#[allow(dead_code)]
pub mod exports {
    #[allow(dead_code)]
    pub mod augurs {
        #[allow(dead_code)]
        pub mod prophet {
            #[allow(dead_code, clippy::all)]
            pub mod prophet {
                #[used]
                #[doc(hidden)]
                static __FORCE_SECTION_REF: fn() = super::super::super::super::__link_custom_section_describing_imports;
                use super::super::super::super::_rt;
                pub type OptimizeOpts = super::super::super::super::augurs::prophet_wasmstan::types::OptimizeOpts;
                /// A timestamp in seconds since the epoch.
                pub type TimestampSeconds = i64;
                /// The type of growth to use.
                #[repr(u8)]
                #[derive(Clone, Copy, Eq, Ord, PartialEq, PartialOrd)]
                pub enum GrowthType {
                    /// Linear growth (default).
                    Linear,
                    /// 0
                    /// Logistic growth.
                    Logistic,
                    /// 1
                    /// Flat growth.
                    Flat,
                }
                impl ::core::fmt::Debug for GrowthType {
                    fn fmt(
                        &self,
                        f: &mut ::core::fmt::Formatter<'_>,
                    ) -> ::core::fmt::Result {
                        match self {
                            GrowthType::Linear => {
                                f.debug_tuple("GrowthType::Linear").finish()
                            }
                            GrowthType::Logistic => {
                                f.debug_tuple("GrowthType::Logistic").finish()
                            }
                            GrowthType::Flat => {
                                f.debug_tuple("GrowthType::Flat").finish()
                            }
                        }
                    }
                }
                impl GrowthType {
                    #[doc(hidden)]
                    pub unsafe fn _lift(val: u8) -> GrowthType {
                        if !cfg!(debug_assertions) {
                            return ::core::mem::transmute(val);
                        }
                        match val {
                            0 => GrowthType::Linear,
                            1 => GrowthType::Logistic,
                            2 => GrowthType::Flat,
                            _ => panic!("invalid enum discriminant"),
                        }
                    }
                }
                /// Whether to include a specific seasonality, and if so, how
                /// many Fourier terms to use.
                #[derive(Clone, Copy)]
                pub enum SeasonalityOption {
                    /// Automatically determine whether to include this seasonality.
                    Auto,
                    /// Manually specify whether to include this seasonality.
                    Manual(bool),
                    /// Enable this seasonality and use the provided number of Fourier terms.
                    Fourier(u32),
                }
                impl ::core::fmt::Debug for SeasonalityOption {
                    fn fmt(
                        &self,
                        f: &mut ::core::fmt::Formatter<'_>,
                    ) -> ::core::fmt::Result {
                        match self {
                            SeasonalityOption::Auto => {
                                f.debug_tuple("SeasonalityOption::Auto").finish()
                            }
                            SeasonalityOption::Manual(e) => {
                                f.debug_tuple("SeasonalityOption::Manual").field(e).finish()
                            }
                            SeasonalityOption::Fourier(e) => {
                                f.debug_tuple("SeasonalityOption::Fourier")
                                    .field(e)
                                    .finish()
                            }
                        }
                    }
                }
                /// How to model seasonality.
                #[repr(u8)]
                #[derive(Clone, Copy, Eq, Ord, PartialEq, PartialOrd)]
                pub enum SeasonalityMode {
                    /// Additive seasonality (the default).
                    Additive,
                    /// Multiplicative seasonality.
                    Multiplicative,
                }
                impl ::core::fmt::Debug for SeasonalityMode {
                    fn fmt(
                        &self,
                        f: &mut ::core::fmt::Formatter<'_>,
                    ) -> ::core::fmt::Result {
                        match self {
                            SeasonalityMode::Additive => {
                                f.debug_tuple("SeasonalityMode::Additive").finish()
                            }
                            SeasonalityMode::Multiplicative => {
                                f.debug_tuple("SeasonalityMode::Multiplicative").finish()
                            }
                        }
                    }
                }
                impl SeasonalityMode {
                    #[doc(hidden)]
                    pub unsafe fn _lift(val: u8) -> SeasonalityMode {
                        if !cfg!(debug_assertions) {
                            return ::core::mem::transmute(val);
                        }
                        match val {
                            0 => SeasonalityMode::Additive,
                            1 => SeasonalityMode::Multiplicative,
                            _ => panic!("invalid enum discriminant"),
                        }
                    }
                }
                /// How to do parameter estimation.
                #[derive(Clone, Copy)]
                pub enum EstimationMode {
                    /// Use MAP estimation.
                    Map,
                    /// Use maximum likelihood estimation.
                    Mle,
                }
                impl ::core::fmt::Debug for EstimationMode {
                    fn fmt(
                        &self,
                        f: &mut ::core::fmt::Formatter<'_>,
                    ) -> ::core::fmt::Result {
                        match self {
                            EstimationMode::Map => {
                                f.debug_tuple("EstimationMode::Map").finish()
                            }
                            EstimationMode::Mle => {
                                f.debug_tuple("EstimationMode::Mle").finish()
                            }
                        }
                    }
                }
                /// How to scale the data prior to fitting the model.
                #[repr(u8)]
                #[derive(Clone, Copy, Eq, Ord, PartialEq, PartialOrd)]
                pub enum Scaling {
                    /// Use abs-max scaling (the default).
                    AbsMax,
                    /// Use min-max scaling.
                    MinMax,
                }
                impl ::core::fmt::Debug for Scaling {
                    fn fmt(
                        &self,
                        f: &mut ::core::fmt::Formatter<'_>,
                    ) -> ::core::fmt::Result {
                        match self {
                            Scaling::AbsMax => f.debug_tuple("Scaling::AbsMax").finish(),
                            Scaling::MinMax => f.debug_tuple("Scaling::MinMax").finish(),
                        }
                    }
                }
                impl Scaling {
                    #[doc(hidden)]
                    pub unsafe fn _lift(val: u8) -> Scaling {
                        if !cfg!(debug_assertions) {
                            return ::core::mem::transmute(val);
                        }
                        match val {
                            0 => Scaling::AbsMax,
                            1 => Scaling::MinMax,
                            _ => panic!("invalid enum discriminant"),
                        }
                    }
                }
                /// A holiday to include in the model.
                #[derive(Clone)]
                pub struct Holiday {
                    /// The name of the holiday.
                    pub name: _rt::String,
                    /// The date of the holiday.
                    pub ds: _rt::Vec<TimestampSeconds>,
                    /// The lower window of the holiday.
                    pub lower_window: Option<_rt::Vec<i32>>,
                    /// The upper window of the holiday.
                    pub upper_window: Option<_rt::Vec<i32>>,
                    /// The prior scale of the holiday.
                    pub prior_scale: Option<f64>,
                }
                impl ::core::fmt::Debug for Holiday {
                    fn fmt(
                        &self,
                        f: &mut ::core::fmt::Formatter<'_>,
                    ) -> ::core::fmt::Result {
                        f.debug_struct("Holiday")
                            .field("name", &self.name)
                            .field("ds", &self.ds)
                            .field("lower-window", &self.lower_window)
                            .field("upper-window", &self.upper_window)
                            .field("prior-scale", &self.prior_scale)
                            .finish()
                    }
                }
                /// Options for the Prophet model.
                #[derive(Clone)]
                pub struct ProphetOpts {
                    /// The type of growth to use.
                    pub growth: Option<GrowthType>,
                    /// List of dates at which to include potential changepoints. If
                    /// not specified, potential changepoints are selected automatically.
                    pub changepoints: Option<_rt::Vec<TimestampSeconds>>,
                    /// Number of potential changepoints to include. Not used
                    /// if input `changepoints` is supplied. If `changepoints` is not supplied,
                    /// then `n-changepoints` potential changepoints are selected uniformly from
                    /// the first `changepoint_range` proportion of the history.
                    pub n_changepoints: Option<u32>,
                    /// Proportion of history in which trend changepoints will
                    /// be estimated. Defaults to 0.8 for the first 80%. Not used if
                    /// `changepoints` is specified.
                    pub changepoint_range: Option<f64>,
                    /// Fit yearly seasonality.
                    pub yearly_seasonality: Option<SeasonalityOption>,
                    /// Fit weekly seasonality.
                    pub weekly_seasonality: Option<SeasonalityOption>,
                    /// Fit daily seasonality.
                    pub daily_seasonality: Option<SeasonalityOption>,
                    /// Whether to treat seasonality as additive (default) or multiplicative.
                    pub seasonality_mode: Option<SeasonalityMode>,
                    /// Parameter modulating the strength of the
                    /// seasonality model. Larger values allow the model to fit larger seasonal
                    /// fluctuations, smaller values dampen the seasonality. Can be specified
                    /// for individual seasonalities using add_seasonality.
                    pub seasonality_prior_scale: Option<f64>,
                    /// Parameter modulating the flexibility of the
                    /// automatic changepoint selection. Large values will allow many
                    /// changepoints, small values will allow few changepoints.
                    pub changepoint_prior_scale: Option<f64>,
                    /// If supplied and greater than 0, will do full Bayesian inference
                    /// with the specified number of MCMC samples. If 0, will do MAP
                    /// estimation.
                    /// Corresponds to the `mcmc_samples` argument in Prophet.
                    pub estimation: Option<EstimationMode>,
                    /// Width of the uncertainty intervals provided for the forecast.
                    /// If estimation=map, this will be only the uncertainty in the trend
                    /// using the MAP estimate of the extrapolated generative model.
                    /// If estimation=mcmc this will be integrated over all model
                    /// parameters, which will include uncertainty in seasonality.
                    pub interval_width: Option<f64>,
                    /// Number of simulated draws used to estimate uncertainty intervals.
                    /// Settings this value to 0 or False will disable uncertainty estimation
                    /// and speed up the calculation.
                    pub uncertainty_samples: Option<u32>,
                    /// How to scale the data prior to fitting.
                    pub scaling: Option<Scaling>,
                    /// Holidays to include in the model.
                    pub holidays: Option<_rt::Vec<Holiday>>,
                    /// How to treat holidays.
                    pub holidays_mode: Option<SeasonalityMode>,
                    /// Prior scale for holidays.
                    ///
                    /// This parameter modulates the strength of the holiday
                    /// components model, unless overridden in each individual
                    /// holiday's input.
                    pub holidays_prior_scale: Option<f64>,
                }
                impl ::core::fmt::Debug for ProphetOpts {
                    fn fmt(
                        &self,
                        f: &mut ::core::fmt::Formatter<'_>,
                    ) -> ::core::fmt::Result {
                        f.debug_struct("ProphetOpts")
                            .field("growth", &self.growth)
                            .field("changepoints", &self.changepoints)
                            .field("n-changepoints", &self.n_changepoints)
                            .field("changepoint-range", &self.changepoint_range)
                            .field("yearly-seasonality", &self.yearly_seasonality)
                            .field("weekly-seasonality", &self.weekly_seasonality)
                            .field("daily-seasonality", &self.daily_seasonality)
                            .field("seasonality-mode", &self.seasonality_mode)
                            .field(
                                "seasonality-prior-scale",
                                &self.seasonality_prior_scale,
                            )
                            .field(
                                "changepoint-prior-scale",
                                &self.changepoint_prior_scale,
                            )
                            .field("estimation", &self.estimation)
                            .field("interval-width", &self.interval_width)
                            .field("uncertainty-samples", &self.uncertainty_samples)
                            .field("scaling", &self.scaling)
                            .field("holidays", &self.holidays)
                            .field("holidays-mode", &self.holidays_mode)
                            .field("holidays-prior-scale", &self.holidays_prior_scale)
                            .finish()
                    }
                }
                /// A seasonality condition.
                #[derive(Clone)]
                pub struct SeasonalityCondition {
                    /// The name of the seasonality condition.
                    /// This must match the `condition_name` provided when
                    /// the seasonality was added to the Prophet model.
                    pub name: _rt::String,
                    /// A list of booleans indicating whether the seasonality is active
                    /// on each date.
                    pub is_active: _rt::Vec<bool>,
                }
                impl ::core::fmt::Debug for SeasonalityCondition {
                    fn fmt(
                        &self,
                        f: &mut ::core::fmt::Formatter<'_>,
                    ) -> ::core::fmt::Result {
                        f.debug_struct("SeasonalityCondition")
                            .field("name", &self.name)
                            .field("is-active", &self.is_active)
                            .finish()
                    }
                }
                /// An external regressor.
                #[derive(Clone)]
                pub struct Regressor {
                    /// The name of the regressor.
                    pub name: _rt::String,
                    /// The regressor values.
                    pub values: _rt::Vec<f64>,
                }
                impl ::core::fmt::Debug for Regressor {
                    fn fmt(
                        &self,
                        f: &mut ::core::fmt::Formatter<'_>,
                    ) -> ::core::fmt::Result {
                        f.debug_struct("Regressor")
                            .field("name", &self.name)
                            .field("values", &self.values)
                            .finish()
                    }
                }
                /// Data to fit the Prophet model.
                #[derive(Clone)]
                pub struct TrainingData {
                    /// The date-time column.
                    pub ds: _rt::Vec<TimestampSeconds>,
                    /// The time series.
                    pub y: _rt::Vec<f64>,
                    /// The capacity of the logistic growth component.
                    /// Required if the Prophet model's `growth-type` is `logistic`.
                    pub cap: Option<_rt::Vec<f64>>,
                    /// The floor of the logistic growth component.
                    /// Optional, only used if the Prophet model's `growth-type` is `logistic`.
                    pub floor: Option<_rt::Vec<f64>>,
                    /// Seasonality conditions.
                    /// Optional, only used if a seasonality added to the Prophet model was
                    /// marked as conditional.
                    pub seasonality_conditions: Option<_rt::Vec<SeasonalityCondition>>,
                    /// External regressors.
                    pub regressors: Option<_rt::Vec<Regressor>>,
                }
                impl ::core::fmt::Debug for TrainingData {
                    fn fmt(
                        &self,
                        f: &mut ::core::fmt::Formatter<'_>,
                    ) -> ::core::fmt::Result {
                        f.debug_struct("TrainingData")
                            .field("ds", &self.ds)
                            .field("y", &self.y)
                            .field("cap", &self.cap)
                            .field("floor", &self.floor)
                            .field(
                                "seasonality-conditions",
                                &self.seasonality_conditions,
                            )
                            .field("regressors", &self.regressors)
                            .finish()
                    }
                }
                /// Data to predict using the Prophet model.
                ///
                /// The prediction data must have the same fields as the training data used
                /// to fit the model, except that the `y` column is not required.
                #[derive(Clone)]
                pub struct PredictionData {
                    /// The date-time column.
                    pub ds: _rt::Vec<TimestampSeconds>,
                    /// The capacity of the logistic growth component.
                    pub cap: Option<_rt::Vec<f64>>,
                    /// The floor of the logistic growth component.
                    pub floor: Option<_rt::Vec<f64>>,
                    /// Seasonality conditions.
                    pub seasonality_conditions: Option<_rt::Vec<SeasonalityCondition>>,
                    /// External regressors.
                    pub regressors: Option<_rt::Vec<Regressor>>,
                }
                impl ::core::fmt::Debug for PredictionData {
                    fn fmt(
                        &self,
                        f: &mut ::core::fmt::Formatter<'_>,
                    ) -> ::core::fmt::Result {
                        f.debug_struct("PredictionData")
                            .field("ds", &self.ds)
                            .field("cap", &self.cap)
                            .field("floor", &self.floor)
                            .field(
                                "seasonality-conditions",
                                &self.seasonality_conditions,
                            )
                            .field("regressors", &self.regressors)
                            .finish()
                    }
                }
                /// A prediction for a component.
                #[derive(Clone)]
                pub struct Prediction {
                    pub point: _rt::Vec<f64>,
                    /// The lower bound of the prediction interval.
                    ///
                    /// Will be `none` if `uncertainty-samples` was
                    /// set to `none` or zero.
                    pub lower: Option<_rt::Vec<f64>>,
                    /// The upper bound of the prediction interval.
                    ///
                    /// Will be `none` if `uncertainty-samples` was
                    /// set to `none` or zero.
                    pub upper: Option<_rt::Vec<f64>>,
                }
                impl ::core::fmt::Debug for Prediction {
                    fn fmt(
                        &self,
                        f: &mut ::core::fmt::Formatter<'_>,
                    ) -> ::core::fmt::Result {
                        f.debug_struct("Prediction")
                            .field("point", &self.point)
                            .field("lower", &self.lower)
                            .field("upper", &self.upper)
                            .finish()
                    }
                }
                /// Predictions made by Prophet.
                #[derive(Clone)]
                pub struct Predictions {
                    /// The date-time column.
                    pub ds: _rt::Vec<TimestampSeconds>,
                    /// The predicted values.
                    pub yhat: Prediction,
                    /// The trend component.
                    pub trend: Prediction,
                    /// The logistic cap, if applicable.
                    pub cap: Option<_rt::Vec<f64>>,
                    /// The logistic floor, if applicable.
                    pub floor: Option<_rt::Vec<f64>>,
                    /// The contributions from additive terms.
                    pub additive_terms: Prediction,
                    /// The contributions from multiplicative terms.
                    pub multiplicative_terms: Prediction,
                }
                impl ::core::fmt::Debug for Predictions {
                    fn fmt(
                        &self,
                        f: &mut ::core::fmt::Formatter<'_>,
                    ) -> ::core::fmt::Result {
                        f.debug_struct("Predictions")
                            .field("ds", &self.ds)
                            .field("yhat", &self.yhat)
                            .field("trend", &self.trend)
                            .field("cap", &self.cap)
                            .field("floor", &self.floor)
                            .field("additive-terms", &self.additive_terms)
                            .field("multiplicative-terms", &self.multiplicative_terms)
                            .finish()
                    }
                }
                /// A Prophet model.
                #[derive(Debug)]
                #[repr(transparent)]
                pub struct Prophet {
                    handle: _rt::Resource<Prophet>,
                }
                type _ProphetRep<T> = Option<T>;
                impl Prophet {
                    /// Creates a new resource from the specified representation.
                    ///
                    /// This function will create a new resource handle by moving `val` onto
                    /// the heap and then passing that heap pointer to the component model to
                    /// create a handle. The owned handle is then returned as `Prophet`.
                    pub fn new<T: GuestProphet>(val: T) -> Self {
                        Self::type_guard::<T>();
                        let val: _ProphetRep<T> = Some(val);
                        let ptr: *mut _ProphetRep<T> = _rt::Box::into_raw(
                            _rt::Box::new(val),
                        );
                        unsafe { Self::from_handle(T::_resource_new(ptr.cast())) }
                    }
                    /// Gets access to the underlying `T` which represents this resource.
                    pub fn get<T: GuestProphet>(&self) -> &T {
                        let ptr = unsafe { &*self.as_ptr::<T>() };
                        ptr.as_ref().unwrap()
                    }
                    /// Gets mutable access to the underlying `T` which represents this
                    /// resource.
                    pub fn get_mut<T: GuestProphet>(&mut self) -> &mut T {
                        let ptr = unsafe { &mut *self.as_ptr::<T>() };
                        ptr.as_mut().unwrap()
                    }
                    /// Consumes this resource and returns the underlying `T`.
                    pub fn into_inner<T: GuestProphet>(self) -> T {
                        let ptr = unsafe { &mut *self.as_ptr::<T>() };
                        ptr.take().unwrap()
                    }
                    #[doc(hidden)]
                    pub unsafe fn from_handle(handle: u32) -> Self {
                        Self {
                            handle: _rt::Resource::from_handle(handle),
                        }
                    }
                    #[doc(hidden)]
                    pub fn take_handle(&self) -> u32 {
                        _rt::Resource::take_handle(&self.handle)
                    }
                    #[doc(hidden)]
                    pub fn handle(&self) -> u32 {
                        _rt::Resource::handle(&self.handle)
                    }
                    #[doc(hidden)]
                    fn type_guard<T: 'static>() {
                        use core::any::TypeId;
                        static mut LAST_TYPE: Option<TypeId> = None;
                        unsafe {
                            assert!(! cfg!(target_feature = "atomics"));
                            let id = TypeId::of::<T>();
                            match LAST_TYPE {
                                Some(ty) => {
                                    assert!(
                                        ty == id, "cannot use two types with this resource type"
                                    )
                                }
                                None => LAST_TYPE = Some(id),
                            }
                        }
                    }
                    #[doc(hidden)]
                    pub unsafe fn dtor<T: 'static>(handle: *mut u8) {
                        Self::type_guard::<T>();
                        let _ = _rt::Box::from_raw(handle as *mut _ProphetRep<T>);
                    }
                    fn as_ptr<T: GuestProphet>(&self) -> *mut _ProphetRep<T> {
                        Prophet::type_guard::<T>();
                        T::_resource_rep(self.handle()).cast()
                    }
                }
                /// A borrowed version of [`Prophet`] which represents a borrowed value
                /// with the lifetime `'a`.
                #[derive(Debug)]
                #[repr(transparent)]
                pub struct ProphetBorrow<'a> {
                    rep: *mut u8,
                    _marker: core::marker::PhantomData<&'a Prophet>,
                }
                impl<'a> ProphetBorrow<'a> {
                    #[doc(hidden)]
                    pub unsafe fn lift(rep: usize) -> Self {
                        Self {
                            rep: rep as *mut u8,
                            _marker: core::marker::PhantomData,
                        }
                    }
                    /// Gets access to the underlying `T` in this resource.
                    pub fn get<T: GuestProphet>(&self) -> &T {
                        let ptr = unsafe { &mut *self.as_ptr::<T>() };
                        ptr.as_ref().unwrap()
                    }
                    fn as_ptr<T: 'static>(&self) -> *mut _ProphetRep<T> {
                        Prophet::type_guard::<T>();
                        self.rep.cast()
                    }
                }
                unsafe impl _rt::WasmResource for Prophet {
                    #[inline]
                    unsafe fn drop(_handle: u32) {
                        #[cfg(not(target_arch = "wasm32"))]
                        unreachable!();
                        #[cfg(target_arch = "wasm32")]
                        {
                            #[link(
                                wasm_import_module = "[export]augurs:prophet/prophet"
                            )]
                            extern "C" {
                                #[link_name = "[resource-drop]prophet"]
                                fn drop(_: u32);
                            }
                            drop(_handle);
                        }
                    }
                }
                #[doc(hidden)]
                #[allow(non_snake_case)]
                pub unsafe fn _export_static_prophet_new_cabi<T: GuestProphet>(
                    arg0: *mut u8,
                ) -> *mut u8 {
                    #[cfg(target_arch = "wasm32")] _rt::run_ctors_once();
                    let l0 = i32::from(*arg0.add(0).cast::<u8>());
                    let result65 = T::new(
                        match l0 {
                            0 => None,
                            1 => {
                                let e = {
                                    let l1 = i32::from(*arg0.add(8).cast::<u8>());
                                    let l3 = i32::from(*arg0.add(12).cast::<u8>());
                                    let l7 = i32::from(*arg0.add(24).cast::<u8>());
                                    let l9 = i32::from(*arg0.add(32).cast::<u8>());
                                    let l11 = i32::from(*arg0.add(48).cast::<u8>());
                                    let l16 = i32::from(*arg0.add(60).cast::<u8>());
                                    let l21 = i32::from(*arg0.add(72).cast::<u8>());
                                    let l26 = i32::from(*arg0.add(84).cast::<u8>());
                                    let l28 = i32::from(*arg0.add(88).cast::<u8>());
                                    let l30 = i32::from(*arg0.add(104).cast::<u8>());
                                    let l32 = i32::from(*arg0.add(120).cast::<u8>());
                                    let l35 = i32::from(*arg0.add(128).cast::<u8>());
                                    let l37 = i32::from(*arg0.add(144).cast::<u8>());
                                    let l39 = i32::from(*arg0.add(152).cast::<u8>());
                                    let l41 = i32::from(*arg0.add(156).cast::<u8>());
                                    let l61 = i32::from(*arg0.add(168).cast::<u8>());
                                    let l63 = i32::from(*arg0.add(176).cast::<u8>());
                                    ProphetOpts {
                                        growth: match l1 {
                                            0 => None,
                                            1 => {
                                                let e = {
                                                    let l2 = i32::from(*arg0.add(9).cast::<u8>());
                                                    GrowthType::_lift(l2 as u8)
                                                };
                                                Some(e)
                                            }
                                            _ => _rt::invalid_enum_discriminant(),
                                        },
                                        changepoints: match l3 {
                                            0 => None,
                                            1 => {
                                                let e = {
                                                    let l4 = *arg0.add(16).cast::<*mut u8>();
                                                    let l5 = *arg0.add(20).cast::<usize>();
                                                    let len6 = l5;
                                                    _rt::Vec::from_raw_parts(l4.cast(), len6, len6)
                                                };
                                                Some(e)
                                            }
                                            _ => _rt::invalid_enum_discriminant(),
                                        },
                                        n_changepoints: match l7 {
                                            0 => None,
                                            1 => {
                                                let e = {
                                                    let l8 = *arg0.add(28).cast::<i32>();
                                                    l8 as u32
                                                };
                                                Some(e)
                                            }
                                            _ => _rt::invalid_enum_discriminant(),
                                        },
                                        changepoint_range: match l9 {
                                            0 => None,
                                            1 => {
                                                let e = {
                                                    let l10 = *arg0.add(40).cast::<f64>();
                                                    l10
                                                };
                                                Some(e)
                                            }
                                            _ => _rt::invalid_enum_discriminant(),
                                        },
                                        yearly_seasonality: match l11 {
                                            0 => None,
                                            1 => {
                                                let e = {
                                                    let l12 = i32::from(*arg0.add(52).cast::<u8>());
                                                    let v15 = match l12 {
                                                        0 => SeasonalityOption::Auto,
                                                        1 => {
                                                            let e15 = {
                                                                let l13 = i32::from(*arg0.add(56).cast::<u8>());
                                                                _rt::bool_lift(l13 as u8)
                                                            };
                                                            SeasonalityOption::Manual(e15)
                                                        }
                                                        n => {
                                                            debug_assert_eq!(n, 2, "invalid enum discriminant");
                                                            let e15 = {
                                                                let l14 = *arg0.add(56).cast::<i32>();
                                                                l14 as u32
                                                            };
                                                            SeasonalityOption::Fourier(e15)
                                                        }
                                                    };
                                                    v15
                                                };
                                                Some(e)
                                            }
                                            _ => _rt::invalid_enum_discriminant(),
                                        },
                                        weekly_seasonality: match l16 {
                                            0 => None,
                                            1 => {
                                                let e = {
                                                    let l17 = i32::from(*arg0.add(64).cast::<u8>());
                                                    let v20 = match l17 {
                                                        0 => SeasonalityOption::Auto,
                                                        1 => {
                                                            let e20 = {
                                                                let l18 = i32::from(*arg0.add(68).cast::<u8>());
                                                                _rt::bool_lift(l18 as u8)
                                                            };
                                                            SeasonalityOption::Manual(e20)
                                                        }
                                                        n => {
                                                            debug_assert_eq!(n, 2, "invalid enum discriminant");
                                                            let e20 = {
                                                                let l19 = *arg0.add(68).cast::<i32>();
                                                                l19 as u32
                                                            };
                                                            SeasonalityOption::Fourier(e20)
                                                        }
                                                    };
                                                    v20
                                                };
                                                Some(e)
                                            }
                                            _ => _rt::invalid_enum_discriminant(),
                                        },
                                        daily_seasonality: match l21 {
                                            0 => None,
                                            1 => {
                                                let e = {
                                                    let l22 = i32::from(*arg0.add(76).cast::<u8>());
                                                    let v25 = match l22 {
                                                        0 => SeasonalityOption::Auto,
                                                        1 => {
                                                            let e25 = {
                                                                let l23 = i32::from(*arg0.add(80).cast::<u8>());
                                                                _rt::bool_lift(l23 as u8)
                                                            };
                                                            SeasonalityOption::Manual(e25)
                                                        }
                                                        n => {
                                                            debug_assert_eq!(n, 2, "invalid enum discriminant");
                                                            let e25 = {
                                                                let l24 = *arg0.add(80).cast::<i32>();
                                                                l24 as u32
                                                            };
                                                            SeasonalityOption::Fourier(e25)
                                                        }
                                                    };
                                                    v25
                                                };
                                                Some(e)
                                            }
                                            _ => _rt::invalid_enum_discriminant(),
                                        },
                                        seasonality_mode: match l26 {
                                            0 => None,
                                            1 => {
                                                let e = {
                                                    let l27 = i32::from(*arg0.add(85).cast::<u8>());
                                                    SeasonalityMode::_lift(l27 as u8)
                                                };
                                                Some(e)
                                            }
                                            _ => _rt::invalid_enum_discriminant(),
                                        },
                                        seasonality_prior_scale: match l28 {
                                            0 => None,
                                            1 => {
                                                let e = {
                                                    let l29 = *arg0.add(96).cast::<f64>();
                                                    l29
                                                };
                                                Some(e)
                                            }
                                            _ => _rt::invalid_enum_discriminant(),
                                        },
                                        changepoint_prior_scale: match l30 {
                                            0 => None,
                                            1 => {
                                                let e = {
                                                    let l31 = *arg0.add(112).cast::<f64>();
                                                    l31
                                                };
                                                Some(e)
                                            }
                                            _ => _rt::invalid_enum_discriminant(),
                                        },
                                        estimation: match l32 {
                                            0 => None,
                                            1 => {
                                                let e = {
                                                    let l33 = i32::from(*arg0.add(121).cast::<u8>());
                                                    let v34 = match l33 {
                                                        0 => EstimationMode::Map,
                                                        n => {
                                                            debug_assert_eq!(n, 1, "invalid enum discriminant");
                                                            EstimationMode::Mle
                                                        }
                                                    };
                                                    v34
                                                };
                                                Some(e)
                                            }
                                            _ => _rt::invalid_enum_discriminant(),
                                        },
                                        interval_width: match l35 {
                                            0 => None,
                                            1 => {
                                                let e = {
                                                    let l36 = *arg0.add(136).cast::<f64>();
                                                    l36
                                                };
                                                Some(e)
                                            }
                                            _ => _rt::invalid_enum_discriminant(),
                                        },
                                        uncertainty_samples: match l37 {
                                            0 => None,
                                            1 => {
                                                let e = {
                                                    let l38 = *arg0.add(148).cast::<i32>();
                                                    l38 as u32
                                                };
                                                Some(e)
                                            }
                                            _ => _rt::invalid_enum_discriminant(),
                                        },
                                        scaling: match l39 {
                                            0 => None,
                                            1 => {
                                                let e = {
                                                    let l40 = i32::from(*arg0.add(153).cast::<u8>());
                                                    Scaling::_lift(l40 as u8)
                                                };
                                                Some(e)
                                            }
                                            _ => _rt::invalid_enum_discriminant(),
                                        },
                                        holidays: match l41 {
                                            0 => None,
                                            1 => {
                                                let e = {
                                                    let l42 = *arg0.add(160).cast::<*mut u8>();
                                                    let l43 = *arg0.add(164).cast::<usize>();
                                                    let base60 = l42;
                                                    let len60 = l43;
                                                    let mut result60 = _rt::Vec::with_capacity(len60);
                                                    for i in 0..len60 {
                                                        let base = base60.add(i * 56);
                                                        let e60 = {
                                                            let l44 = *base.add(0).cast::<*mut u8>();
                                                            let l45 = *base.add(4).cast::<usize>();
                                                            let len46 = l45;
                                                            let bytes46 = _rt::Vec::from_raw_parts(
                                                                l44.cast(),
                                                                len46,
                                                                len46,
                                                            );
                                                            let l47 = *base.add(8).cast::<*mut u8>();
                                                            let l48 = *base.add(12).cast::<usize>();
                                                            let len49 = l48;
                                                            let l50 = i32::from(*base.add(16).cast::<u8>());
                                                            let l54 = i32::from(*base.add(28).cast::<u8>());
                                                            let l58 = i32::from(*base.add(40).cast::<u8>());
                                                            Holiday {
                                                                name: _rt::string_lift(bytes46),
                                                                ds: _rt::Vec::from_raw_parts(l47.cast(), len49, len49),
                                                                lower_window: match l50 {
                                                                    0 => None,
                                                                    1 => {
                                                                        let e = {
                                                                            let l51 = *base.add(20).cast::<*mut u8>();
                                                                            let l52 = *base.add(24).cast::<usize>();
                                                                            let len53 = l52;
                                                                            _rt::Vec::from_raw_parts(l51.cast(), len53, len53)
                                                                        };
                                                                        Some(e)
                                                                    }
                                                                    _ => _rt::invalid_enum_discriminant(),
                                                                },
                                                                upper_window: match l54 {
                                                                    0 => None,
                                                                    1 => {
                                                                        let e = {
                                                                            let l55 = *base.add(32).cast::<*mut u8>();
                                                                            let l56 = *base.add(36).cast::<usize>();
                                                                            let len57 = l56;
                                                                            _rt::Vec::from_raw_parts(l55.cast(), len57, len57)
                                                                        };
                                                                        Some(e)
                                                                    }
                                                                    _ => _rt::invalid_enum_discriminant(),
                                                                },
                                                                prior_scale: match l58 {
                                                                    0 => None,
                                                                    1 => {
                                                                        let e = {
                                                                            let l59 = *base.add(48).cast::<f64>();
                                                                            l59
                                                                        };
                                                                        Some(e)
                                                                    }
                                                                    _ => _rt::invalid_enum_discriminant(),
                                                                },
                                                            }
                                                        };
                                                        result60.push(e60);
                                                    }
                                                    _rt::cabi_dealloc(base60, len60 * 56, 8);
                                                    result60
                                                };
                                                Some(e)
                                            }
                                            _ => _rt::invalid_enum_discriminant(),
                                        },
                                        holidays_mode: match l61 {
                                            0 => None,
                                            1 => {
                                                let e = {
                                                    let l62 = i32::from(*arg0.add(169).cast::<u8>());
                                                    SeasonalityMode::_lift(l62 as u8)
                                                };
                                                Some(e)
                                            }
                                            _ => _rt::invalid_enum_discriminant(),
                                        },
                                        holidays_prior_scale: match l63 {
                                            0 => None,
                                            1 => {
                                                let e = {
                                                    let l64 = *arg0.add(184).cast::<f64>();
                                                    l64
                                                };
                                                Some(e)
                                            }
                                            _ => _rt::invalid_enum_discriminant(),
                                        },
                                    }
                                };
                                Some(e)
                            }
                            _ => _rt::invalid_enum_discriminant(),
                        },
                    );
                    _rt::cabi_dealloc(arg0, 192, 8);
                    let ptr66 = _RET_AREA.0.as_mut_ptr().cast::<u8>();
                    match result65 {
                        Ok(e) => {
                            *ptr66.add(0).cast::<u8>() = (0i32) as u8;
                            *ptr66.add(4).cast::<i32>() = (e).take_handle() as i32;
                        }
                        Err(e) => {
                            *ptr66.add(0).cast::<u8>() = (1i32) as u8;
                            let vec67 = (e.into_bytes()).into_boxed_slice();
                            let ptr67 = vec67.as_ptr().cast::<u8>();
                            let len67 = vec67.len();
                            ::core::mem::forget(vec67);
                            *ptr66.add(8).cast::<usize>() = len67;
                            *ptr66.add(4).cast::<*mut u8>() = ptr67.cast_mut();
                        }
                    };
                    ptr66
                }
                #[doc(hidden)]
                #[allow(non_snake_case)]
                pub unsafe fn __post_return_static_prophet_new<T: GuestProphet>(
                    arg0: *mut u8,
                ) {
                    let l0 = i32::from(*arg0.add(0).cast::<u8>());
                    match l0 {
                        0 => {}
                        _ => {
                            let l1 = *arg0.add(4).cast::<*mut u8>();
                            let l2 = *arg0.add(8).cast::<usize>();
                            _rt::cabi_dealloc(l1, l2, 1);
                        }
                    }
                }
                #[doc(hidden)]
                #[allow(non_snake_case)]
                pub unsafe fn _export_method_prophet_fit_cabi<T: GuestProphet>(
                    arg0: *mut u8,
                ) -> *mut u8 {
                    #[cfg(target_arch = "wasm32")] _rt::run_ctors_once();
                    let l0 = *arg0.add(0).cast::<i32>();
                    let l1 = *arg0.add(4).cast::<*mut u8>();
                    let l2 = *arg0.add(8).cast::<usize>();
                    let len3 = l2;
                    let l4 = *arg0.add(12).cast::<*mut u8>();
                    let l5 = *arg0.add(16).cast::<usize>();
                    let len6 = l5;
                    let l7 = i32::from(*arg0.add(20).cast::<u8>());
                    let l11 = i32::from(*arg0.add(32).cast::<u8>());
                    let l15 = i32::from(*arg0.add(44).cast::<u8>());
                    let l26 = i32::from(*arg0.add(56).cast::<u8>());
                    let l36 = i32::from(*arg0.add(72).cast::<u8>());
                    let result63 = T::fit(
                        ProphetBorrow::lift(l0 as u32 as usize).get(),
                        TrainingData {
                            ds: _rt::Vec::from_raw_parts(l1.cast(), len3, len3),
                            y: _rt::Vec::from_raw_parts(l4.cast(), len6, len6),
                            cap: match l7 {
                                0 => None,
                                1 => {
                                    let e = {
                                        let l8 = *arg0.add(24).cast::<*mut u8>();
                                        let l9 = *arg0.add(28).cast::<usize>();
                                        let len10 = l9;
                                        _rt::Vec::from_raw_parts(l8.cast(), len10, len10)
                                    };
                                    Some(e)
                                }
                                _ => _rt::invalid_enum_discriminant(),
                            },
                            floor: match l11 {
                                0 => None,
                                1 => {
                                    let e = {
                                        let l12 = *arg0.add(36).cast::<*mut u8>();
                                        let l13 = *arg0.add(40).cast::<usize>();
                                        let len14 = l13;
                                        _rt::Vec::from_raw_parts(l12.cast(), len14, len14)
                                    };
                                    Some(e)
                                }
                                _ => _rt::invalid_enum_discriminant(),
                            },
                            seasonality_conditions: match l15 {
                                0 => None,
                                1 => {
                                    let e = {
                                        let l16 = *arg0.add(48).cast::<*mut u8>();
                                        let l17 = *arg0.add(52).cast::<usize>();
                                        let base25 = l16;
                                        let len25 = l17;
                                        let mut result25 = _rt::Vec::with_capacity(len25);
                                        for i in 0..len25 {
                                            let base = base25.add(i * 16);
                                            let e25 = {
                                                let l18 = *base.add(0).cast::<*mut u8>();
                                                let l19 = *base.add(4).cast::<usize>();
                                                let len20 = l19;
                                                let bytes20 = _rt::Vec::from_raw_parts(
                                                    l18.cast(),
                                                    len20,
                                                    len20,
                                                );
                                                let l21 = *base.add(8).cast::<*mut u8>();
                                                let l22 = *base.add(12).cast::<usize>();
                                                let base24 = l21;
                                                let len24 = l22;
                                                let mut result24 = _rt::Vec::with_capacity(len24);
                                                for i in 0..len24 {
                                                    let base = base24.add(i * 1);
                                                    let e24 = {
                                                        let l23 = i32::from(*base.add(0).cast::<u8>());
                                                        _rt::bool_lift(l23 as u8)
                                                    };
                                                    result24.push(e24);
                                                }
                                                _rt::cabi_dealloc(base24, len24 * 1, 1);
                                                SeasonalityCondition {
                                                    name: _rt::string_lift(bytes20),
                                                    is_active: result24,
                                                }
                                            };
                                            result25.push(e25);
                                        }
                                        _rt::cabi_dealloc(base25, len25 * 16, 4);
                                        result25
                                    };
                                    Some(e)
                                }
                                _ => _rt::invalid_enum_discriminant(),
                            },
                            regressors: match l26 {
                                0 => None,
                                1 => {
                                    let e = {
                                        let l27 = *arg0.add(60).cast::<*mut u8>();
                                        let l28 = *arg0.add(64).cast::<usize>();
                                        let base35 = l27;
                                        let len35 = l28;
                                        let mut result35 = _rt::Vec::with_capacity(len35);
                                        for i in 0..len35 {
                                            let base = base35.add(i * 16);
                                            let e35 = {
                                                let l29 = *base.add(0).cast::<*mut u8>();
                                                let l30 = *base.add(4).cast::<usize>();
                                                let len31 = l30;
                                                let bytes31 = _rt::Vec::from_raw_parts(
                                                    l29.cast(),
                                                    len31,
                                                    len31,
                                                );
                                                let l32 = *base.add(8).cast::<*mut u8>();
                                                let l33 = *base.add(12).cast::<usize>();
                                                let len34 = l33;
                                                Regressor {
                                                    name: _rt::string_lift(bytes31),
                                                    values: _rt::Vec::from_raw_parts(l32.cast(), len34, len34),
                                                }
                                            };
                                            result35.push(e35);
                                        }
                                        _rt::cabi_dealloc(base35, len35 * 16, 4);
                                        result35
                                    };
                                    Some(e)
                                }
                                _ => _rt::invalid_enum_discriminant(),
                            },
                        },
                        match l36 {
                            0 => None,
                            1 => {
                                let e = {
                                    let l37 = i32::from(*arg0.add(80).cast::<u8>());
                                    let l39 = i32::from(*arg0.add(84).cast::<u8>());
                                    let l41 = i32::from(*arg0.add(92).cast::<u8>());
                                    let l43 = i32::from(*arg0.add(104).cast::<u8>());
                                    let l45 = i32::from(*arg0.add(120).cast::<u8>());
                                    let l47 = i32::from(*arg0.add(136).cast::<u8>());
                                    let l49 = i32::from(*arg0.add(152).cast::<u8>());
                                    let l51 = i32::from(*arg0.add(168).cast::<u8>());
                                    let l53 = i32::from(*arg0.add(184).cast::<u8>());
                                    let l55 = i32::from(*arg0.add(200).cast::<u8>());
                                    let l57 = i32::from(*arg0.add(208).cast::<u8>());
                                    let l59 = i32::from(*arg0.add(216).cast::<u8>());
                                    let l61 = i32::from(*arg0.add(220).cast::<u8>());
                                    super::super::super::super::augurs::prophet_wasmstan::types::OptimizeOpts {
                                        algorithm: match l37 {
                                            0 => None,
                                            1 => {
                                                let e = {
                                                    let l38 = i32::from(*arg0.add(81).cast::<u8>());
                                                    super::super::super::super::augurs::prophet_wasmstan::types::Algorithm::_lift(
                                                        l38 as u8,
                                                    )
                                                };
                                                Some(e)
                                            }
                                            _ => _rt::invalid_enum_discriminant(),
                                        },
                                        seed: match l39 {
                                            0 => None,
                                            1 => {
                                                let e = {
                                                    let l40 = *arg0.add(88).cast::<i32>();
                                                    l40 as u32
                                                };
                                                Some(e)
                                            }
                                            _ => _rt::invalid_enum_discriminant(),
                                        },
                                        chain: match l41 {
                                            0 => None,
                                            1 => {
                                                let e = {
                                                    let l42 = *arg0.add(96).cast::<i32>();
                                                    l42 as u32
                                                };
                                                Some(e)
                                            }
                                            _ => _rt::invalid_enum_discriminant(),
                                        },
                                        init_alpha: match l43 {
                                            0 => None,
                                            1 => {
                                                let e = {
                                                    let l44 = *arg0.add(112).cast::<f64>();
                                                    l44
                                                };
                                                Some(e)
                                            }
                                            _ => _rt::invalid_enum_discriminant(),
                                        },
                                        tol_obj: match l45 {
                                            0 => None,
                                            1 => {
                                                let e = {
                                                    let l46 = *arg0.add(128).cast::<f64>();
                                                    l46
                                                };
                                                Some(e)
                                            }
                                            _ => _rt::invalid_enum_discriminant(),
                                        },
                                        tol_rel_obj: match l47 {
                                            0 => None,
                                            1 => {
                                                let e = {
                                                    let l48 = *arg0.add(144).cast::<f64>();
                                                    l48
                                                };
                                                Some(e)
                                            }
                                            _ => _rt::invalid_enum_discriminant(),
                                        },
                                        tol_grad: match l49 {
                                            0 => None,
                                            1 => {
                                                let e = {
                                                    let l50 = *arg0.add(160).cast::<f64>();
                                                    l50
                                                };
                                                Some(e)
                                            }
                                            _ => _rt::invalid_enum_discriminant(),
                                        },
                                        tol_rel_grad: match l51 {
                                            0 => None,
                                            1 => {
                                                let e = {
                                                    let l52 = *arg0.add(176).cast::<f64>();
                                                    l52
                                                };
                                                Some(e)
                                            }
                                            _ => _rt::invalid_enum_discriminant(),
                                        },
                                        tol_param: match l53 {
                                            0 => None,
                                            1 => {
                                                let e = {
                                                    let l54 = *arg0.add(192).cast::<f64>();
                                                    l54
                                                };
                                                Some(e)
                                            }
                                            _ => _rt::invalid_enum_discriminant(),
                                        },
                                        history_size: match l55 {
                                            0 => None,
                                            1 => {
                                                let e = {
                                                    let l56 = *arg0.add(204).cast::<i32>();
                                                    l56 as u32
                                                };
                                                Some(e)
                                            }
                                            _ => _rt::invalid_enum_discriminant(),
                                        },
                                        iter: match l57 {
                                            0 => None,
                                            1 => {
                                                let e = {
                                                    let l58 = *arg0.add(212).cast::<i32>();
                                                    l58 as u32
                                                };
                                                Some(e)
                                            }
                                            _ => _rt::invalid_enum_discriminant(),
                                        },
                                        jacobian: match l59 {
                                            0 => None,
                                            1 => {
                                                let e = {
                                                    let l60 = i32::from(*arg0.add(217).cast::<u8>());
                                                    _rt::bool_lift(l60 as u8)
                                                };
                                                Some(e)
                                            }
                                            _ => _rt::invalid_enum_discriminant(),
                                        },
                                        refresh: match l61 {
                                            0 => None,
                                            1 => {
                                                let e = {
                                                    let l62 = *arg0.add(224).cast::<i32>();
                                                    l62 as u32
                                                };
                                                Some(e)
                                            }
                                            _ => _rt::invalid_enum_discriminant(),
                                        },
                                    }
                                };
                                Some(e)
                            }
                            _ => _rt::invalid_enum_discriminant(),
                        },
                    );
                    _rt::cabi_dealloc(arg0, 232, 8);
                    let ptr64 = _RET_AREA.0.as_mut_ptr().cast::<u8>();
                    match result63 {
                        Ok(_) => {
                            *ptr64.add(0).cast::<u8>() = (0i32) as u8;
                        }
                        Err(e) => {
                            *ptr64.add(0).cast::<u8>() = (1i32) as u8;
                            let vec65 = (e.into_bytes()).into_boxed_slice();
                            let ptr65 = vec65.as_ptr().cast::<u8>();
                            let len65 = vec65.len();
                            ::core::mem::forget(vec65);
                            *ptr64.add(8).cast::<usize>() = len65;
                            *ptr64.add(4).cast::<*mut u8>() = ptr65.cast_mut();
                        }
                    };
                    ptr64
                }
                #[doc(hidden)]
                #[allow(non_snake_case)]
                pub unsafe fn __post_return_method_prophet_fit<T: GuestProphet>(
                    arg0: *mut u8,
                ) {
                    let l0 = i32::from(*arg0.add(0).cast::<u8>());
                    match l0 {
                        0 => {}
                        _ => {
                            let l1 = *arg0.add(4).cast::<*mut u8>();
                            let l2 = *arg0.add(8).cast::<usize>();
                            _rt::cabi_dealloc(l1, l2, 1);
                        }
                    }
                }
                #[doc(hidden)]
                #[allow(non_snake_case)]
                pub unsafe fn _export_method_prophet_predict_cabi<T: GuestProphet>(
                    arg0: *mut u8,
                    arg1: i32,
                    arg2: *mut u8,
                    arg3: usize,
                    arg4: i32,
                    arg5: *mut u8,
                    arg6: usize,
                    arg7: i32,
                    arg8: *mut u8,
                    arg9: usize,
                    arg10: i32,
                    arg11: *mut u8,
                    arg12: usize,
                    arg13: i32,
                    arg14: *mut u8,
                    arg15: usize,
                ) -> *mut u8 {
                    #[cfg(target_arch = "wasm32")] _rt::run_ctors_once();
                    let result18 = T::predict(
                        ProphetBorrow::lift(arg0 as u32 as usize).get(),
                        match arg1 {
                            0 => None,
                            1 => {
                                let e = {
                                    let len0 = arg3;
                                    PredictionData {
                                        ds: _rt::Vec::from_raw_parts(arg2.cast(), len0, len0),
                                        cap: match arg4 {
                                            0 => None,
                                            1 => {
                                                let e = {
                                                    let len1 = arg6;
                                                    _rt::Vec::from_raw_parts(arg5.cast(), len1, len1)
                                                };
                                                Some(e)
                                            }
                                            _ => _rt::invalid_enum_discriminant(),
                                        },
                                        floor: match arg7 {
                                            0 => None,
                                            1 => {
                                                let e = {
                                                    let len2 = arg9;
                                                    _rt::Vec::from_raw_parts(arg8.cast(), len2, len2)
                                                };
                                                Some(e)
                                            }
                                            _ => _rt::invalid_enum_discriminant(),
                                        },
                                        seasonality_conditions: match arg10 {
                                            0 => None,
                                            1 => {
                                                let e = {
                                                    let base10 = arg11;
                                                    let len10 = arg12;
                                                    let mut result10 = _rt::Vec::with_capacity(len10);
                                                    for i in 0..len10 {
                                                        let base = base10.add(i * 16);
                                                        let e10 = {
                                                            let l3 = *base.add(0).cast::<*mut u8>();
                                                            let l4 = *base.add(4).cast::<usize>();
                                                            let len5 = l4;
                                                            let bytes5 = _rt::Vec::from_raw_parts(
                                                                l3.cast(),
                                                                len5,
                                                                len5,
                                                            );
                                                            let l6 = *base.add(8).cast::<*mut u8>();
                                                            let l7 = *base.add(12).cast::<usize>();
                                                            let base9 = l6;
                                                            let len9 = l7;
                                                            let mut result9 = _rt::Vec::with_capacity(len9);
                                                            for i in 0..len9 {
                                                                let base = base9.add(i * 1);
                                                                let e9 = {
                                                                    let l8 = i32::from(*base.add(0).cast::<u8>());
                                                                    _rt::bool_lift(l8 as u8)
                                                                };
                                                                result9.push(e9);
                                                            }
                                                            _rt::cabi_dealloc(base9, len9 * 1, 1);
                                                            SeasonalityCondition {
                                                                name: _rt::string_lift(bytes5),
                                                                is_active: result9,
                                                            }
                                                        };
                                                        result10.push(e10);
                                                    }
                                                    _rt::cabi_dealloc(base10, len10 * 16, 4);
                                                    result10
                                                };
                                                Some(e)
                                            }
                                            _ => _rt::invalid_enum_discriminant(),
                                        },
                                        regressors: match arg13 {
                                            0 => None,
                                            1 => {
                                                let e = {
                                                    let base17 = arg14;
                                                    let len17 = arg15;
                                                    let mut result17 = _rt::Vec::with_capacity(len17);
                                                    for i in 0..len17 {
                                                        let base = base17.add(i * 16);
                                                        let e17 = {
                                                            let l11 = *base.add(0).cast::<*mut u8>();
                                                            let l12 = *base.add(4).cast::<usize>();
                                                            let len13 = l12;
                                                            let bytes13 = _rt::Vec::from_raw_parts(
                                                                l11.cast(),
                                                                len13,
                                                                len13,
                                                            );
                                                            let l14 = *base.add(8).cast::<*mut u8>();
                                                            let l15 = *base.add(12).cast::<usize>();
                                                            let len16 = l15;
                                                            Regressor {
                                                                name: _rt::string_lift(bytes13),
                                                                values: _rt::Vec::from_raw_parts(l14.cast(), len16, len16),
                                                            }
                                                        };
                                                        result17.push(e17);
                                                    }
                                                    _rt::cabi_dealloc(base17, len17 * 16, 4);
                                                    result17
                                                };
                                                Some(e)
                                            }
                                            _ => _rt::invalid_enum_discriminant(),
                                        },
                                    }
                                };
                                Some(e)
                            }
                            _ => _rt::invalid_enum_discriminant(),
                        },
                    );
                    let ptr19 = _RET_AREA.0.as_mut_ptr().cast::<u8>();
                    match result18 {
                        Ok(e) => {
                            *ptr19.add(0).cast::<u8>() = (0i32) as u8;
                            let Predictions {
                                ds: ds20,
                                yhat: yhat20,
                                trend: trend20,
                                cap: cap20,
                                floor: floor20,
                                additive_terms: additive_terms20,
                                multiplicative_terms: multiplicative_terms20,
                            } = e;
                            let vec21 = (ds20).into_boxed_slice();
                            let ptr21 = vec21.as_ptr().cast::<u8>();
                            let len21 = vec21.len();
                            ::core::mem::forget(vec21);
                            *ptr19.add(8).cast::<usize>() = len21;
                            *ptr19.add(4).cast::<*mut u8>() = ptr21.cast_mut();
                            let Prediction {
                                point: point22,
                                lower: lower22,
                                upper: upper22,
                            } = yhat20;
                            let vec23 = (point22).into_boxed_slice();
                            let ptr23 = vec23.as_ptr().cast::<u8>();
                            let len23 = vec23.len();
                            ::core::mem::forget(vec23);
                            *ptr19.add(16).cast::<usize>() = len23;
                            *ptr19.add(12).cast::<*mut u8>() = ptr23.cast_mut();
                            match lower22 {
                                Some(e) => {
                                    *ptr19.add(20).cast::<u8>() = (1i32) as u8;
                                    let vec24 = (e).into_boxed_slice();
                                    let ptr24 = vec24.as_ptr().cast::<u8>();
                                    let len24 = vec24.len();
                                    ::core::mem::forget(vec24);
                                    *ptr19.add(28).cast::<usize>() = len24;
                                    *ptr19.add(24).cast::<*mut u8>() = ptr24.cast_mut();
                                }
                                None => {
                                    *ptr19.add(20).cast::<u8>() = (0i32) as u8;
                                }
                            };
                            match upper22 {
                                Some(e) => {
                                    *ptr19.add(32).cast::<u8>() = (1i32) as u8;
                                    let vec25 = (e).into_boxed_slice();
                                    let ptr25 = vec25.as_ptr().cast::<u8>();
                                    let len25 = vec25.len();
                                    ::core::mem::forget(vec25);
                                    *ptr19.add(40).cast::<usize>() = len25;
                                    *ptr19.add(36).cast::<*mut u8>() = ptr25.cast_mut();
                                }
                                None => {
                                    *ptr19.add(32).cast::<u8>() = (0i32) as u8;
                                }
                            };
                            let Prediction {
                                point: point26,
                                lower: lower26,
                                upper: upper26,
                            } = trend20;
                            let vec27 = (point26).into_boxed_slice();
                            let ptr27 = vec27.as_ptr().cast::<u8>();
                            let len27 = vec27.len();
                            ::core::mem::forget(vec27);
                            *ptr19.add(48).cast::<usize>() = len27;
                            *ptr19.add(44).cast::<*mut u8>() = ptr27.cast_mut();
                            match lower26 {
                                Some(e) => {
                                    *ptr19.add(52).cast::<u8>() = (1i32) as u8;
                                    let vec28 = (e).into_boxed_slice();
                                    let ptr28 = vec28.as_ptr().cast::<u8>();
                                    let len28 = vec28.len();
                                    ::core::mem::forget(vec28);
                                    *ptr19.add(60).cast::<usize>() = len28;
                                    *ptr19.add(56).cast::<*mut u8>() = ptr28.cast_mut();
                                }
                                None => {
                                    *ptr19.add(52).cast::<u8>() = (0i32) as u8;
                                }
                            };
                            match upper26 {
                                Some(e) => {
                                    *ptr19.add(64).cast::<u8>() = (1i32) as u8;
                                    let vec29 = (e).into_boxed_slice();
                                    let ptr29 = vec29.as_ptr().cast::<u8>();
                                    let len29 = vec29.len();
                                    ::core::mem::forget(vec29);
                                    *ptr19.add(72).cast::<usize>() = len29;
                                    *ptr19.add(68).cast::<*mut u8>() = ptr29.cast_mut();
                                }
                                None => {
                                    *ptr19.add(64).cast::<u8>() = (0i32) as u8;
                                }
                            };
                            match cap20 {
                                Some(e) => {
                                    *ptr19.add(76).cast::<u8>() = (1i32) as u8;
                                    let vec30 = (e).into_boxed_slice();
                                    let ptr30 = vec30.as_ptr().cast::<u8>();
                                    let len30 = vec30.len();
                                    ::core::mem::forget(vec30);
                                    *ptr19.add(84).cast::<usize>() = len30;
                                    *ptr19.add(80).cast::<*mut u8>() = ptr30.cast_mut();
                                }
                                None => {
                                    *ptr19.add(76).cast::<u8>() = (0i32) as u8;
                                }
                            };
                            match floor20 {
                                Some(e) => {
                                    *ptr19.add(88).cast::<u8>() = (1i32) as u8;
                                    let vec31 = (e).into_boxed_slice();
                                    let ptr31 = vec31.as_ptr().cast::<u8>();
                                    let len31 = vec31.len();
                                    ::core::mem::forget(vec31);
                                    *ptr19.add(96).cast::<usize>() = len31;
                                    *ptr19.add(92).cast::<*mut u8>() = ptr31.cast_mut();
                                }
                                None => {
                                    *ptr19.add(88).cast::<u8>() = (0i32) as u8;
                                }
                            };
                            let Prediction {
                                point: point32,
                                lower: lower32,
                                upper: upper32,
                            } = additive_terms20;
                            let vec33 = (point32).into_boxed_slice();
                            let ptr33 = vec33.as_ptr().cast::<u8>();
                            let len33 = vec33.len();
                            ::core::mem::forget(vec33);
                            *ptr19.add(104).cast::<usize>() = len33;
                            *ptr19.add(100).cast::<*mut u8>() = ptr33.cast_mut();
                            match lower32 {
                                Some(e) => {
                                    *ptr19.add(108).cast::<u8>() = (1i32) as u8;
                                    let vec34 = (e).into_boxed_slice();
                                    let ptr34 = vec34.as_ptr().cast::<u8>();
                                    let len34 = vec34.len();
                                    ::core::mem::forget(vec34);
                                    *ptr19.add(116).cast::<usize>() = len34;
                                    *ptr19.add(112).cast::<*mut u8>() = ptr34.cast_mut();
                                }
                                None => {
                                    *ptr19.add(108).cast::<u8>() = (0i32) as u8;
                                }
                            };
                            match upper32 {
                                Some(e) => {
                                    *ptr19.add(120).cast::<u8>() = (1i32) as u8;
                                    let vec35 = (e).into_boxed_slice();
                                    let ptr35 = vec35.as_ptr().cast::<u8>();
                                    let len35 = vec35.len();
                                    ::core::mem::forget(vec35);
                                    *ptr19.add(128).cast::<usize>() = len35;
                                    *ptr19.add(124).cast::<*mut u8>() = ptr35.cast_mut();
                                }
                                None => {
                                    *ptr19.add(120).cast::<u8>() = (0i32) as u8;
                                }
                            };
                            let Prediction {
                                point: point36,
                                lower: lower36,
                                upper: upper36,
                            } = multiplicative_terms20;
                            let vec37 = (point36).into_boxed_slice();
                            let ptr37 = vec37.as_ptr().cast::<u8>();
                            let len37 = vec37.len();
                            ::core::mem::forget(vec37);
                            *ptr19.add(136).cast::<usize>() = len37;
                            *ptr19.add(132).cast::<*mut u8>() = ptr37.cast_mut();
                            match lower36 {
                                Some(e) => {
                                    *ptr19.add(140).cast::<u8>() = (1i32) as u8;
                                    let vec38 = (e).into_boxed_slice();
                                    let ptr38 = vec38.as_ptr().cast::<u8>();
                                    let len38 = vec38.len();
                                    ::core::mem::forget(vec38);
                                    *ptr19.add(148).cast::<usize>() = len38;
                                    *ptr19.add(144).cast::<*mut u8>() = ptr38.cast_mut();
                                }
                                None => {
                                    *ptr19.add(140).cast::<u8>() = (0i32) as u8;
                                }
                            };
                            match upper36 {
                                Some(e) => {
                                    *ptr19.add(152).cast::<u8>() = (1i32) as u8;
                                    let vec39 = (e).into_boxed_slice();
                                    let ptr39 = vec39.as_ptr().cast::<u8>();
                                    let len39 = vec39.len();
                                    ::core::mem::forget(vec39);
                                    *ptr19.add(160).cast::<usize>() = len39;
                                    *ptr19.add(156).cast::<*mut u8>() = ptr39.cast_mut();
                                }
                                None => {
                                    *ptr19.add(152).cast::<u8>() = (0i32) as u8;
                                }
                            };
                        }
                        Err(e) => {
                            *ptr19.add(0).cast::<u8>() = (1i32) as u8;
                            let vec40 = (e.into_bytes()).into_boxed_slice();
                            let ptr40 = vec40.as_ptr().cast::<u8>();
                            let len40 = vec40.len();
                            ::core::mem::forget(vec40);
                            *ptr19.add(8).cast::<usize>() = len40;
                            *ptr19.add(4).cast::<*mut u8>() = ptr40.cast_mut();
                        }
                    };
                    ptr19
                }
                #[doc(hidden)]
                #[allow(non_snake_case)]
                pub unsafe fn __post_return_method_prophet_predict<T: GuestProphet>(
                    arg0: *mut u8,
                ) {
                    let l0 = i32::from(*arg0.add(0).cast::<u8>());
                    match l0 {
                        0 => {
                            let l1 = *arg0.add(4).cast::<*mut u8>();
                            let l2 = *arg0.add(8).cast::<usize>();
                            let base3 = l1;
                            let len3 = l2;
                            _rt::cabi_dealloc(base3, len3 * 8, 8);
                            let l4 = *arg0.add(12).cast::<*mut u8>();
                            let l5 = *arg0.add(16).cast::<usize>();
                            let base6 = l4;
                            let len6 = l5;
                            _rt::cabi_dealloc(base6, len6 * 8, 8);
                            let l7 = i32::from(*arg0.add(20).cast::<u8>());
                            match l7 {
                                0 => {}
                                _ => {
                                    let l8 = *arg0.add(24).cast::<*mut u8>();
                                    let l9 = *arg0.add(28).cast::<usize>();
                                    let base10 = l8;
                                    let len10 = l9;
                                    _rt::cabi_dealloc(base10, len10 * 8, 8);
                                }
                            }
                            let l11 = i32::from(*arg0.add(32).cast::<u8>());
                            match l11 {
                                0 => {}
                                _ => {
                                    let l12 = *arg0.add(36).cast::<*mut u8>();
                                    let l13 = *arg0.add(40).cast::<usize>();
                                    let base14 = l12;
                                    let len14 = l13;
                                    _rt::cabi_dealloc(base14, len14 * 8, 8);
                                }
                            }
                            let l15 = *arg0.add(44).cast::<*mut u8>();
                            let l16 = *arg0.add(48).cast::<usize>();
                            let base17 = l15;
                            let len17 = l16;
                            _rt::cabi_dealloc(base17, len17 * 8, 8);
                            let l18 = i32::from(*arg0.add(52).cast::<u8>());
                            match l18 {
                                0 => {}
                                _ => {
                                    let l19 = *arg0.add(56).cast::<*mut u8>();
                                    let l20 = *arg0.add(60).cast::<usize>();
                                    let base21 = l19;
                                    let len21 = l20;
                                    _rt::cabi_dealloc(base21, len21 * 8, 8);
                                }
                            }
                            let l22 = i32::from(*arg0.add(64).cast::<u8>());
                            match l22 {
                                0 => {}
                                _ => {
                                    let l23 = *arg0.add(68).cast::<*mut u8>();
                                    let l24 = *arg0.add(72).cast::<usize>();
                                    let base25 = l23;
                                    let len25 = l24;
                                    _rt::cabi_dealloc(base25, len25 * 8, 8);
                                }
                            }
                            let l26 = i32::from(*arg0.add(76).cast::<u8>());
                            match l26 {
                                0 => {}
                                _ => {
                                    let l27 = *arg0.add(80).cast::<*mut u8>();
                                    let l28 = *arg0.add(84).cast::<usize>();
                                    let base29 = l27;
                                    let len29 = l28;
                                    _rt::cabi_dealloc(base29, len29 * 8, 8);
                                }
                            }
                            let l30 = i32::from(*arg0.add(88).cast::<u8>());
                            match l30 {
                                0 => {}
                                _ => {
                                    let l31 = *arg0.add(92).cast::<*mut u8>();
                                    let l32 = *arg0.add(96).cast::<usize>();
                                    let base33 = l31;
                                    let len33 = l32;
                                    _rt::cabi_dealloc(base33, len33 * 8, 8);
                                }
                            }
                            let l34 = *arg0.add(100).cast::<*mut u8>();
                            let l35 = *arg0.add(104).cast::<usize>();
                            let base36 = l34;
                            let len36 = l35;
                            _rt::cabi_dealloc(base36, len36 * 8, 8);
                            let l37 = i32::from(*arg0.add(108).cast::<u8>());
                            match l37 {
                                0 => {}
                                _ => {
                                    let l38 = *arg0.add(112).cast::<*mut u8>();
                                    let l39 = *arg0.add(116).cast::<usize>();
                                    let base40 = l38;
                                    let len40 = l39;
                                    _rt::cabi_dealloc(base40, len40 * 8, 8);
                                }
                            }
                            let l41 = i32::from(*arg0.add(120).cast::<u8>());
                            match l41 {
                                0 => {}
                                _ => {
                                    let l42 = *arg0.add(124).cast::<*mut u8>();
                                    let l43 = *arg0.add(128).cast::<usize>();
                                    let base44 = l42;
                                    let len44 = l43;
                                    _rt::cabi_dealloc(base44, len44 * 8, 8);
                                }
                            }
                            let l45 = *arg0.add(132).cast::<*mut u8>();
                            let l46 = *arg0.add(136).cast::<usize>();
                            let base47 = l45;
                            let len47 = l46;
                            _rt::cabi_dealloc(base47, len47 * 8, 8);
                            let l48 = i32::from(*arg0.add(140).cast::<u8>());
                            match l48 {
                                0 => {}
                                _ => {
                                    let l49 = *arg0.add(144).cast::<*mut u8>();
                                    let l50 = *arg0.add(148).cast::<usize>();
                                    let base51 = l49;
                                    let len51 = l50;
                                    _rt::cabi_dealloc(base51, len51 * 8, 8);
                                }
                            }
                            let l52 = i32::from(*arg0.add(152).cast::<u8>());
                            match l52 {
                                0 => {}
                                _ => {
                                    let l53 = *arg0.add(156).cast::<*mut u8>();
                                    let l54 = *arg0.add(160).cast::<usize>();
                                    let base55 = l53;
                                    let len55 = l54;
                                    _rt::cabi_dealloc(base55, len55 * 8, 8);
                                }
                            }
                        }
                        _ => {
                            let l56 = *arg0.add(4).cast::<*mut u8>();
                            let l57 = *arg0.add(8).cast::<usize>();
                            _rt::cabi_dealloc(l56, l57, 1);
                        }
                    }
                }
                pub trait Guest {
                    type Prophet: GuestProphet;
                }
                pub trait GuestProphet: 'static {
                    #[doc(hidden)]
                    unsafe fn _resource_new(val: *mut u8) -> u32
                    where
                        Self: Sized,
                    {
                        #[cfg(not(target_arch = "wasm32"))]
                        {
                            let _ = val;
                            unreachable!();
                        }
                        #[cfg(target_arch = "wasm32")]
                        {
                            #[link(
                                wasm_import_module = "[export]augurs:prophet/prophet"
                            )]
                            extern "C" {
                                #[link_name = "[resource-new]prophet"]
                                fn new(_: *mut u8) -> u32;
                            }
                            new(val)
                        }
                    }
                    #[doc(hidden)]
                    fn _resource_rep(handle: u32) -> *mut u8
                    where
                        Self: Sized,
                    {
                        #[cfg(not(target_arch = "wasm32"))]
                        {
                            let _ = handle;
                            unreachable!();
                        }
                        #[cfg(target_arch = "wasm32")]
                        {
                            #[link(
                                wasm_import_module = "[export]augurs:prophet/prophet"
                            )]
                            extern "C" {
                                #[link_name = "[resource-rep]prophet"]
                                fn rep(_: u32) -> *mut u8;
                            }
                            unsafe { rep(handle) }
                        }
                    }
                    /// Construct a Prophet model.
                    ///
                    /// The `opts` argument is optional, and if not supplied, will use the
                    /// default options.
                    fn new(opts: Option<ProphetOpts>) -> Result<Prophet, _rt::String>;
                    /// Fit the Prophet model to the given data.
                    ///
                    /// The `opts` argument is optional, and if not supplied, will use the
                    /// default options.
                    fn fit(
                        &self,
                        data: TrainingData,
                        opts: Option<OptimizeOpts>,
                    ) -> Result<(), _rt::String>;
                    /// Predict future values.
                    ///
                    /// If `data` is not supplied, the model will predict values for
                    /// the training data.
                    fn predict(
                        &self,
                        data: Option<PredictionData>,
                    ) -> Result<Predictions, _rt::String>;
                }
                #[doc(hidden)]
                macro_rules! __export_augurs_prophet_prophet_cabi {
                    ($ty:ident with_types_in $($path_to_types:tt)*) => {
                        const _ : () = { #[export_name =
                        "augurs:prophet/prophet#[static]prophet.new"] unsafe extern "C"
                        fn export_static_prophet_new(arg0 : * mut u8,) -> * mut u8 {
                        $($path_to_types)*:: _export_static_prophet_new_cabi::<<$ty as
                        $($path_to_types)*:: Guest >::Prophet > (arg0) } #[export_name =
                        "cabi_post_augurs:prophet/prophet#[static]prophet.new"] unsafe
                        extern "C" fn _post_return_static_prophet_new(arg0 : * mut u8,) {
                        $($path_to_types)*:: __post_return_static_prophet_new::<<$ty as
                        $($path_to_types)*:: Guest >::Prophet > (arg0) } #[export_name =
                        "augurs:prophet/prophet#[method]prophet.fit"] unsafe extern "C"
                        fn export_method_prophet_fit(arg0 : * mut u8,) -> * mut u8 {
                        $($path_to_types)*:: _export_method_prophet_fit_cabi::<<$ty as
                        $($path_to_types)*:: Guest >::Prophet > (arg0) } #[export_name =
                        "cabi_post_augurs:prophet/prophet#[method]prophet.fit"] unsafe
                        extern "C" fn _post_return_method_prophet_fit(arg0 : * mut u8,) {
                        $($path_to_types)*:: __post_return_method_prophet_fit::<<$ty as
                        $($path_to_types)*:: Guest >::Prophet > (arg0) } #[export_name =
                        "augurs:prophet/prophet#[method]prophet.predict"] unsafe extern
                        "C" fn export_method_prophet_predict(arg0 : * mut u8, arg1 : i32,
                        arg2 : * mut u8, arg3 : usize, arg4 : i32, arg5 : * mut u8, arg6
                        : usize, arg7 : i32, arg8 : * mut u8, arg9 : usize, arg10 : i32,
                        arg11 : * mut u8, arg12 : usize, arg13 : i32, arg14 : * mut u8,
                        arg15 : usize,) -> * mut u8 { $($path_to_types)*::
                        _export_method_prophet_predict_cabi::<<$ty as
                        $($path_to_types)*:: Guest >::Prophet > (arg0, arg1, arg2, arg3,
                        arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, arg12, arg13,
                        arg14, arg15) } #[export_name =
                        "cabi_post_augurs:prophet/prophet#[method]prophet.predict"]
                        unsafe extern "C" fn _post_return_method_prophet_predict(arg0 : *
                        mut u8,) { $($path_to_types)*::
                        __post_return_method_prophet_predict::<<$ty as
                        $($path_to_types)*:: Guest >::Prophet > (arg0) } const _ : () = {
                        #[doc(hidden)] #[export_name =
                        "augurs:prophet/prophet#[dtor]prophet"] #[allow(non_snake_case)]
                        unsafe extern "C" fn dtor(rep : * mut u8) { $($path_to_types)*::
                        Prophet::dtor::< <$ty as $($path_to_types)*:: Guest >::Prophet >
                        (rep) } }; };
                    };
                }
                #[doc(hidden)]
                pub(crate) use __export_augurs_prophet_prophet_cabi;
                #[repr(align(4))]
                struct _RetArea([::core::mem::MaybeUninit<u8>; 164]);
                static mut _RET_AREA: _RetArea = _RetArea(
                    [::core::mem::MaybeUninit::uninit(); 164],
                );
            }
        }
    }
}
mod _rt {
    pub use alloc_crate::vec::Vec;
    pub use alloc_crate::string::String;
    pub fn as_f64<T: AsF64>(t: T) -> f64 {
        t.as_f64()
    }
    pub trait AsF64 {
        fn as_f64(self) -> f64;
    }
    impl<'a, T: Copy + AsF64> AsF64 for &'a T {
        fn as_f64(self) -> f64 {
            (*self).as_f64()
        }
    }
    impl AsF64 for f64 {
        #[inline]
        fn as_f64(self) -> f64 {
            self as f64
        }
    }
    pub fn as_i32<T: AsI32>(t: T) -> i32 {
        t.as_i32()
    }
    pub trait AsI32 {
        fn as_i32(self) -> i32;
    }
    impl<'a, T: Copy + AsI32> AsI32 for &'a T {
        fn as_i32(self) -> i32 {
            (*self).as_i32()
        }
    }
    impl AsI32 for i32 {
        #[inline]
        fn as_i32(self) -> i32 {
            self as i32
        }
    }
    impl AsI32 for u32 {
        #[inline]
        fn as_i32(self) -> i32 {
            self as i32
        }
    }
    impl AsI32 for i16 {
        #[inline]
        fn as_i32(self) -> i32 {
            self as i32
        }
    }
    impl AsI32 for u16 {
        #[inline]
        fn as_i32(self) -> i32 {
            self as i32
        }
    }
    impl AsI32 for i8 {
        #[inline]
        fn as_i32(self) -> i32 {
            self as i32
        }
    }
    impl AsI32 for u8 {
        #[inline]
        fn as_i32(self) -> i32 {
            self as i32
        }
    }
    impl AsI32 for char {
        #[inline]
        fn as_i32(self) -> i32 {
            self as i32
        }
    }
    impl AsI32 for usize {
        #[inline]
        fn as_i32(self) -> i32 {
            self as i32
        }
    }
    pub unsafe fn string_lift(bytes: Vec<u8>) -> String {
        if cfg!(debug_assertions) {
            String::from_utf8(bytes).unwrap()
        } else {
            String::from_utf8_unchecked(bytes)
        }
    }
    pub unsafe fn invalid_enum_discriminant<T>() -> T {
        if cfg!(debug_assertions) {
            panic!("invalid enum discriminant")
        } else {
            core::hint::unreachable_unchecked()
        }
    }
    use core::fmt;
    use core::marker;
    use core::sync::atomic::{AtomicU32, Ordering::Relaxed};
    /// A type which represents a component model resource, either imported or
    /// exported into this component.
    ///
    /// This is a low-level wrapper which handles the lifetime of the resource
    /// (namely this has a destructor). The `T` provided defines the component model
    /// intrinsics that this wrapper uses.
    ///
    /// One of the chief purposes of this type is to provide `Deref` implementations
    /// to access the underlying data when it is owned.
    ///
    /// This type is primarily used in generated code for exported and imported
    /// resources.
    #[repr(transparent)]
    pub struct Resource<T: WasmResource> {
        handle: AtomicU32,
        _marker: marker::PhantomData<T>,
    }
    /// A trait which all wasm resources implement, namely providing the ability to
    /// drop a resource.
    ///
    /// This generally is implemented by generated code, not user-facing code.
    #[allow(clippy::missing_safety_doc)]
    pub unsafe trait WasmResource {
        /// Invokes the `[resource-drop]...` intrinsic.
        unsafe fn drop(handle: u32);
    }
    impl<T: WasmResource> Resource<T> {
        #[doc(hidden)]
        pub unsafe fn from_handle(handle: u32) -> Self {
            debug_assert!(handle != u32::MAX);
            Self {
                handle: AtomicU32::new(handle),
                _marker: marker::PhantomData,
            }
        }
        /// Takes ownership of the handle owned by `resource`.
        ///
        /// Note that this ideally would be `into_handle` taking `Resource<T>` by
        /// ownership. The code generator does not enable that in all situations,
        /// unfortunately, so this is provided instead.
        ///
        /// Also note that `take_handle` is in theory only ever called on values
        /// owned by a generated function. For example a generated function might
        /// take `Resource<T>` as an argument but then call `take_handle` on a
        /// reference to that argument. In that sense the dynamic nature of
        /// `take_handle` should only be exposed internally to generated code, not
        /// to user code.
        #[doc(hidden)]
        pub fn take_handle(resource: &Resource<T>) -> u32 {
            resource.handle.swap(u32::MAX, Relaxed)
        }
        #[doc(hidden)]
        pub fn handle(resource: &Resource<T>) -> u32 {
            resource.handle.load(Relaxed)
        }
    }
    impl<T: WasmResource> fmt::Debug for Resource<T> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_struct("Resource").field("handle", &self.handle).finish()
        }
    }
    impl<T: WasmResource> Drop for Resource<T> {
        fn drop(&mut self) {
            unsafe {
                match self.handle.load(Relaxed) {
                    u32::MAX => {}
                    other => T::drop(other),
                }
            }
        }
    }
    pub use alloc_crate::boxed::Box;
    #[cfg(target_arch = "wasm32")]
    pub fn run_ctors_once() {
        wit_bindgen_rt::run_ctors_once();
    }
    pub unsafe fn bool_lift(val: u8) -> bool {
        if cfg!(debug_assertions) {
            match val {
                0 => false,
                1 => true,
                _ => panic!("invalid bool discriminant"),
            }
        } else {
            val != 0
        }
    }
    pub unsafe fn cabi_dealloc(ptr: *mut u8, size: usize, align: usize) {
        if size == 0 {
            return;
        }
        let layout = alloc::Layout::from_size_align_unchecked(size, align);
        alloc::dealloc(ptr, layout);
    }
    extern crate alloc as alloc_crate;
    pub use alloc_crate::alloc;
}
/// Generates `#[no_mangle]` functions to export the specified type as the
/// root implementation of all generated traits.
///
/// For more information see the documentation of `wit_bindgen::generate!`.
///
/// ```rust
/// # macro_rules! export{ ($($t:tt)*) => (); }
/// # trait Guest {}
/// struct MyType;
///
/// impl Guest for MyType {
///     // ...
/// }
///
/// export!(MyType);
/// ```
#[allow(unused_macros)]
#[doc(hidden)]
macro_rules! __export_component_impl {
    ($ty:ident) => {
        self::export!($ty with_types_in self);
    };
    ($ty:ident with_types_in $($path_to_types_root:tt)*) => {
        $($path_to_types_root)*::
        exports::augurs::prophet::prophet::__export_augurs_prophet_prophet_cabi!($ty
        with_types_in $($path_to_types_root)*:: exports::augurs::prophet::prophet);
    };
}
#[doc(inline)]
pub(crate) use __export_component_impl as export;
#[cfg(target_arch = "wasm32")]
#[link_section = "component-type:wit-bindgen:0.31.0:augurs:prophet:component:encoded world"]
#[doc(hidden)]
pub static __WIT_BINDGEN_COMPONENT_TYPE: [u8; 2209] = *b"\
\0asm\x0d\0\x01\0\0\x19\x16wit-component-encoding\x04\0\x07\xa1\x10\x01A\x02\x01\
A\x0a\x01B\x16\x01pu\x01r\x05\x01ku\x01mu\x05delta\0\x04beta\0\x09sigma-obsu\x04\
\0\x05inits\x03\0\x01\x01m\x03\x06linear\x08logistic\x04flat\x04\0\x0ftrend-indi\
cator\x03\0\x03\x01pz\x01r\x0d\x01nz\x01y\0\x01t\0\x03cap\0\x01sz\x08t-change\0\x0f\
trend-indicator\x04\x01kz\x03s-a\x05\x03s-m\x05\x01x\0\x06sigmas\0\x03tauu\x04\0\
\x04data\x03\0\x06\x01m\x03\x06newton\x04bfgs\x05lbfgs\x04\0\x09algorithm\x03\0\x08\
\x01k\x09\x01ky\x01ku\x01k\x7f\x01r\x0d\x09algorithm\x0a\x04seed\x0b\x05chain\x0b\
\x0ainit-alpha\x0c\x07tol-obj\x0c\x0btol-rel-obj\x0c\x08tol-grad\x0c\x0ctol-rel-\
grad\x0c\x09tol-param\x0c\x0chistory-size\x0b\x04iter\x0b\x08jacobian\x0d\x07ref\
resh\x0b\x04\0\x0doptimize-opts\x03\0\x0e\x01r\x05\x05debugs\x04infos\x04warns\x05\
errors\x05fatals\x04\0\x04logs\x03\0\x10\x01r\x06\x01ku\x01mu\x05delta\0\x04beta\
\0\x09sigma-obsu\x05trend\0\x04\0\x10optimized-params\x03\0\x12\x01r\x02\x04logs\
\x11\x06params\x13\x04\0\x0foptimize-output\x03\0\x14\x03\x01\x1daugurs:prophet-\
wasmstan/types\x05\0\x02\x03\0\0\x05inits\x02\x03\0\0\x04data\x02\x03\0\0\x0dopt\
imize-opts\x02\x03\0\0\x0foptimize-output\x01B\x0b\x02\x03\x02\x01\x01\x04\0\x05\
inits\x03\0\0\x02\x03\x02\x01\x02\x04\0\x04data\x03\0\x02\x02\x03\x02\x01\x03\x04\
\0\x0doptimize-opts\x03\0\x04\x02\x03\x02\x01\x04\x04\0\x0foptimize-output\x03\0\
\x06\x01j\x01\x07\x01s\x01@\x03\x04init\x01\x04data\x03\x04opts\x05\0\x08\x04\0\x08\
optimize\x01\x09\x03\x01!augurs:prophet-wasmstan/optimizer\x05\x05\x01BA\x02\x03\
\x02\x01\x03\x04\0\x0doptimize-opts\x03\0\0\x01x\x04\0\x11timestamp-seconds\x03\0\
\x02\x01m\x03\x06linear\x08logistic\x04flat\x04\0\x0bgrowth-type\x03\0\x04\x01q\x03\
\x04auto\0\0\x06manual\x01\x7f\0\x07fourier\x01y\0\x04\0\x12seasonality-option\x03\
\0\x06\x01m\x02\x08additive\x0emultiplicative\x04\0\x10seasonality-mode\x03\0\x08\
\x01q\x02\x03map\0\0\x03mle\0\0\x04\0\x0festimation-mode\x03\0\x0a\x01m\x02\x07a\
bs-max\x07min-max\x04\0\x07scaling\x03\0\x0c\x01p\x03\x01pz\x01k\x0f\x01ku\x01r\x05\
\x04names\x02ds\x0e\x0clower-window\x10\x0cupper-window\x10\x0bprior-scale\x11\x04\
\0\x07holiday\x03\0\x12\x01k\x05\x01k\x0e\x01ky\x01k\x07\x01k\x09\x01k\x0b\x01k\x0d\
\x01p\x13\x01k\x1b\x01r\x11\x06growth\x14\x0cchangepoints\x15\x0en-changepoints\x16\
\x11changepoint-range\x11\x12yearly-seasonality\x17\x12weekly-seasonality\x17\x11\
daily-seasonality\x17\x10seasonality-mode\x18\x17seasonality-prior-scale\x11\x17\
changepoint-prior-scale\x11\x0aestimation\x19\x0einterval-width\x11\x13uncertain\
ty-samples\x16\x07scaling\x1a\x08holidays\x1c\x0dholidays-mode\x18\x14holidays-p\
rior-scale\x11\x04\0\x0cprophet-opts\x03\0\x1d\x01p\x7f\x01r\x02\x04names\x09is-\
active\x1f\x04\0\x15seasonality-condition\x03\0\x20\x01pu\x01r\x02\x04names\x06v\
alues\"\x04\0\x09regressor\x03\0#\x01k\"\x01p!\x01k&\x01p$\x01k(\x01r\x06\x02ds\x0e\
\x01y\"\x03cap%\x05floor%\x16seasonality-conditions'\x0aregressors)\x04\0\x0dtra\
ining-data\x03\0*\x01r\x05\x02ds\x0e\x03cap%\x05floor%\x16seasonality-conditions\
'\x0aregressors)\x04\0\x0fprediction-data\x03\0,\x01r\x03\x05point\"\x05lower%\x05\
upper%\x04\0\x0aprediction\x03\0.\x01r\x07\x02ds\x0e\x04yhat/\x05trend/\x03cap%\x05\
floor%\x0eadditive-terms/\x14multiplicative-terms/\x04\0\x0bpredictions\x03\00\x04\
\0\x07prophet\x03\x01\x01k\x1e\x01i2\x01j\x014\x01s\x01@\x01\x04opts3\05\x04\0\x13\
[static]prophet.new\x016\x01h2\x01k\x01\x01j\0\x01s\x01@\x03\x04self7\x04data+\x04\
opts8\09\x04\0\x13[method]prophet.fit\x01:\x01k-\x01j\x011\x01s\x01@\x02\x04self\
7\x04data;\0<\x04\0\x17[method]prophet.predict\x01=\x04\x01\x16augurs:prophet/pr\
ophet\x05\x06\x04\x01\x18augurs:prophet/component\x04\0\x0b\x0f\x01\0\x09compone\
nt\x03\0\0\0G\x09producers\x01\x0cprocessed-by\x02\x0dwit-component\x070.216.0\x10\
wit-bindgen-rust\x060.31.0";
#[inline(never)]
#[doc(hidden)]
pub fn __link_custom_section_describing_imports() {
    wit_bindgen_rt::maybe_link_cabi_realloc();
}
