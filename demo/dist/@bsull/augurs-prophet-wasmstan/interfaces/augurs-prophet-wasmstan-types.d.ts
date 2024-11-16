export namespace AugursProphetWasmstanTypes {
}
/**
 * The initial parameters for the optimization.
 */
export interface Inits {
  /**
   * Base trend growth rate.
   */
  k: number,
  /**
   * Trend offset.
   */
  m: number,
  /**
   * Trend rate adjustments, length s in data.
   */
  delta: Float64Array,
  /**
   * Regressor coefficients, length k in data.
   */
  beta: Float64Array,
  /**
   * Observation noise.
   */
  sigmaObs: number,
}
/**
 * The type of trend to use.
 * # Variants
 * 
 * ## `"linear"`
 * 
 * Linear trend (default).
 * ## `"logistic"`
 * 
 * 0
 * Logistic trend.
 * ## `"flat"`
 * 
 * 1
 * Flat trend.
 */
export type TrendIndicator = 'linear' | 'logistic' | 'flat';
/**
 * Data for the Prophet model.
 */
export interface Data {
  /**
   * Number of time periods.
   * This is `T` in the Prophet STAN model definition,
   * but WIT identifiers must be lower kebab-case.
   */
  n: number,
  /**
   * Time series, length n.
   */
  y: Float64Array,
  /**
   * Time, length n.
   */
  t: Float64Array,
  /**
   * Capacities for logistic trend, length n.
   */
  cap: Float64Array,
  /**
   * Number of changepoints.
   * This is 'S' in the Prophet STAN model definition,
   * but WIT identifiers must be lower kebab-case.
   */
  s: number,
  /**
   * Times of trend changepoints, length s.
   */
  tChange: Float64Array,
  /**
   * The type of trend to use.
   */
  trendIndicator: TrendIndicator,
  /**
   * Number of regressors.
   * Must be greater than or equal to 1.
   * This is `K` in the Prophet STAN model definition,
   * but WIT identifiers must be lower kebab-case.
   */
  k: number,
  /**
   * Indicator of additive features, length k.
   * This is `s_a` in the Prophet STAN model definition,
   * but WIT identifiers must be lower kebab-case.
   */
  sA: Int32Array,
  /**
   * Indicator of multiplicative features, length k.
   * This is `s_m` in the Prophet STAN model definition,
   * but WIT identifiers must be lower kebab-case.
   */
  sM: Int32Array,
  /**
   * Regressors.
   * This is `X` in the Prophet STAN model definition,
   * but WIT identifiers must be lower kebab-case.
   * This is passed as a flat array but should be treated as
   * a matrix with shape (n, k) (i.e. strides of length n).
   */
  x: Float64Array,
  /**
   * Scale on seasonality prior.
   */
  sigmas: Float64Array,
  /**
   * Scale on changepoints prior.
   * Must be greater than 0.
   */
  tau: number,
}
/**
 * JSON representation of the Prophet data to pass to Stan.
 * 
 * This should be a string containing a JSONified `Data`.
 */
export type DataJson = string;
/**
 * The algorithm to use for optimization. One of: 'BFGS', 'LBFGS', 'Newton'.
 * # Variants
 * 
 * ## `"newton"`
 * 
 * Use the Newton algorithm.
 * ## `"bfgs"`
 * 
 * Use the Broyden-Fletcher-Goldfarb-Shanno (BFGS) algorithm.
 * ## `"lbfgs"`
 * 
 * Use the Limited-memory BFGS (L-BFGS) algorithm.
 */
export type Algorithm = 'newton' | 'bfgs' | 'lbfgs';
/**
 * Arguments for optimization.
 */
export interface OptimizeOpts {
  /**
   * Algorithm to use.
   */
  algorithm?: Algorithm,
  /**
   * The random seed to use for the optimization.
   */
  seed?: number,
  /**
   * The chain id to advance the PRNG.
   */
  chain?: number,
  /**
   * Line search step size for first iteration.
   */
  initAlpha?: number,
  /**
   * Convergence tolerance on changes in objective function value.
   */
  tolObj?: number,
  /**
   * Convergence tolerance on relative changes in objective function value.
   */
  tolRelObj?: number,
  /**
   * Convergence tolerance on the norm of the gradient.
   */
  tolGrad?: number,
  /**
   * Convergence tolerance on the relative norm of the gradient.
   */
  tolRelGrad?: number,
  /**
   * Convergence tolerance on changes in parameter value.
   */
  tolParam?: number,
  /**
   * Size of the history for LBFGS Hessian approximation. The value should
   * be less than the dimensionality of the parameter space. 5-10 usually
   * sufficient.
   */
  historySize?: number,
  /**
   * Total number of iterations.
   */
  iter?: number,
  /**
   * When `true`, use the Jacobian matrix to approximate the Hessian.
   * Default is `false`.
   */
  jacobian?: boolean,
  /**
   * How frequently to update the log message, in number of iterations.
   */
  refresh?: number,
}
/**
 * Log lines produced during optimization.
 */
export interface Logs {
  /**
   * Debug log lines.
   */
  debug: string,
  /**
   * Info log lines.
   */
  info: string,
  /**
   * Warning log lines.
   */
  warn: string,
  /**
   * Error log lines.
   */
  error: string,
  /**
   * Fatal log lines.
   */
  fatal: string,
}
/**
 * The optimal parameter values found by optimization.
 */
export interface OptimizedParams {
  /**
   * Base trend growth rate.
   */
  k: number,
  /**
   * Trend offset.
   */
  m: number,
  /**
   * Trend rate adjustments, length s in data.
   */
  delta: Float64Array,
  /**
   * Regressor coefficients, length k in data.
   */
  beta: Float64Array,
  /**
   * Observation noise.
   */
  sigmaObs: number,
  /**
   * Transformed trend.
   */
  trend: Float64Array,
}
/**
 * The result of optimization.
 * 
 * This includes both the parameters and any logs produced by the
 * process.
 */
export interface OptimizeOutput {
  /**
   * Logs produced by the optimization process.
   */
  logs: Logs,
  /**
   * The optimized parameters.
   */
  params: OptimizedParams,
}
