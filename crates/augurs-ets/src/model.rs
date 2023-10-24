//! A single model of the ETS family.
//!
//! This module contains the `ETSModel` struct, which represents a single model of the ETS family.

use std::fmt::{self, Write};

use augurs_core::ForecastIntervals;
use itertools::Itertools;
use nalgebra::{DMatrix, DVector};
use rand_distr::{Distribution, Normal};
use tracing::instrument;

use crate::{
    ets::{Ets, FitState},
    stat::StatExt,
    Error,
};

/// The type of error component used by the model.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum ErrorComponent {
    /// Additive error component.
    Additive,
    /// Multiplicative error component.
    Multiplicative,
}

impl fmt::Display for ErrorComponent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Additive => f.write_char('A'),
            Self::Multiplicative => f.write_char('M'),
        }
    }
}

/// The type of trend component included in the model.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum TrendComponent {
    /// No trend component.
    None,
    /// Additive trend component.
    Additive,
    /// Multiplicative trend component.
    Multiplicative,
}

impl TrendComponent {
    /// Whether this component will be included in a model.
    pub fn included(&self) -> bool {
        *self != TrendComponent::None
    }
}

impl fmt::Display for TrendComponent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::None => f.write_char('N'),
            Self::Additive => f.write_char('A'),
            Self::Multiplicative => f.write_char('M'),
        }
    }
}

/// The type of trend component included in the model.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum SeasonalComponent {
    /// No seasonal component.
    None,
    /// Additive seasonal component.
    Additive {
        /// The number of observations in a seasonal cycle.
        ///
        /// This was called `m` in the original `ets` R code.
        season_length: usize,
    },
    /// Multiplicative seasonal component.
    Multiplicative {
        /// The number of observations in a seasonal cycle.
        ///
        /// This was called `m` in the original `ets` R code.
        season_length: usize,
    },
}

impl SeasonalComponent {
    /// Whether this component will be included in a model.
    pub fn included(&self) -> bool {
        *self != SeasonalComponent::None
    }

    /// The number of observations in a seasonal cycle.
    ///
    /// This will be `1` if the component is `None`, otherwise it will be the
    /// `season_length` of the variant.
    pub fn season_length(&self) -> usize {
        match self {
            SeasonalComponent::None => 1,
            SeasonalComponent::Additive { season_length } => *season_length,
            SeasonalComponent::Multiplicative { season_length } => *season_length,
        }
    }
}

impl fmt::Display for SeasonalComponent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::None => f.write_char('N'),
            Self::Additive { .. } => f.write_char('A'),
            Self::Multiplicative { .. } => f.write_char('M'),
        }
    }
}

/// The upper and lower bounds to use with [`Bounds::Usual`] and [`Bounds::Both`].
#[derive(Clone, PartialEq, Debug)]
pub struct UpperLowerBounds {
    lower: [f64; 4],
    upper: [f64; 4],
}

impl UpperLowerBounds {
    /// Create a new set of bounds.
    ///
    /// # Errors
    ///
    /// Returns an error if any of the lower bounds are greater than the
    /// corresponding upper bounds.
    pub fn new(lower: [f64; 4], upper: [f64; 4]) -> Result<Self, Error> {
        if lower.iter().zip(&upper).any(|(l, u)| l > u) {
            Err(Error::InconsistentBounds)
        } else {
            Ok(Self { lower, upper })
        }
    }
}

impl Default for UpperLowerBounds {
    fn default() -> Self {
        Self {
            lower: [0.0001, 0.0001, 0.0001, 0.8],
            upper: [0.9999, 0.9999, 0.9999, 0.98],
        }
    }
}

/// The type of parameter space to impose.
#[derive(Clone, Debug)]
pub enum Bounds {
    /// All parameters must lie in the admissible space.
    Admissible,
    /// All parameters must lie between specified lower and upper bounds.
    Usual(UpperLowerBounds),
    /// The intersection of `Admissible` and `Usual`. This is the default.
    Both(UpperLowerBounds),
}

impl Bounds {
    fn for_optimizer(
        &self,
        opt_params: &OptimizeParams,
        n_states: usize,
    ) -> Option<(Vec<f64>, Vec<f64>)> {
        match self {
            Self::Admissible => None,
            Self::Usual(bounds) | Self::Both(bounds) => {
                let n_params = opt_params.n_included();
                let mut lower = Vec::with_capacity(n_params + n_states);
                let mut upper = Vec::with_capacity(n_params + n_states);
                if opt_params.alpha {
                    lower.push(bounds.lower[0]);
                    upper.push(bounds.upper[0]);
                }
                if opt_params.beta {
                    lower.push(bounds.lower[1]);
                    upper.push(bounds.upper[1]);
                }
                if opt_params.gamma {
                    lower.push(bounds.lower[2]);
                    upper.push(bounds.upper[2]);
                }
                if opt_params.phi {
                    lower.push(bounds.lower[3]);
                    upper.push(bounds.upper[3]);
                }
                for _ in 0..n_states {
                    lower.push(f64::NEG_INFINITY);
                    upper.push(f64::INFINITY);
                }
                Some((lower, upper))
            }
        }
    }
}

impl Default for Bounds {
    fn default() -> Self {
        Self::Both(UpperLowerBounds::default())
    }
}

/// The optimization criterion to use when fitting the model.
///
/// Defaults to [`OptimizationCriteria::Likelihood`].
#[derive(Debug, Copy, Clone, Default)]
pub enum OptimizationCriteria {
    /// Log-likelihood.
    #[default]
    Likelihood,
    /// Mean squared error.
    MSE,
    /// Average mean squared error over the first `nmse` forecast horizons.
    AMSE,
    /// Standard deviation of the residuals.
    Sigma,
    /// Mean absolute error.
    MAE,
}

/// The type of ETS model.
///
/// ETS models are defined by the type of error, trend, and seasonal components
/// included in the model. These components can be excluded, included additively,
/// or included multiplicatively. Some combinations of components are not
/// allowed due to identifiability issues; these will be excluded
/// from the search space of [`crate::AutoETS`].
#[derive(Debug, Clone, Copy)]
pub struct ModelType {
    /// The type of error component.
    pub error: ErrorComponent,
    /// The type of trend component.
    pub trend: TrendComponent,
    /// The type of seasonal component.
    pub season: SeasonalComponent,
}

impl fmt::Display for ModelType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.error.fmt(f)?;
        self.trend.fmt(f)?;
        self.season.fmt(f)?;
        Ok(())
    }
}

/// The parameters of an ETS model.
#[derive(Debug, Clone)]
pub struct Params {
    /// The value of the smoothing parameter for the level.
    ///
    /// If `alpha = 0`, the level will not change over time.
    /// Conversely, if `alpha = 1` the level will update similarly to a random walk process.
    pub alpha: f64,
    /// The value of the smoothing parameter for the slope.
    ///
    /// If `beta = 0`, the slope will not change over time.
    /// Conversely, if `beta = 1` the slope will have no memory of past slopes.
    pub beta: f64,
    /// The value of the smoothing parameter for the seasonal pattern.
    /// If `gamma = 0`, the seasonal pattern will not change over time.
    /// Conversely, if `gamma = 1` the seasonality will have no memory of past seasonal periods.
    pub gamma: f64,
    /// The value of the dampening parameter for the slope.
    /// If `phi = 0`, the slope will be dampened immediately (no slope).
    /// Conversely, if `phi = 1` the slope will not be dampened.
    pub phi: f64,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            alpha: f64::NAN,
            beta: f64::NAN,
            gamma: f64::NAN,
            phi: f64::NAN,
        }
    }
}

/// Parameters to be optimized by the optimizer.
///
/// If parameters are explicitly specified, they won't be included
/// in the Nelder-Mead optimization, and the specified values will be used.
/// Otherwise the parameters will be optimized.
///
/// By default, all parameters relevant to the model are optimized
/// (i.e. `gamma` is only included for seasonal models; `phi` is
/// only included for damped trend models; etc).
#[derive(Debug, Default, Clone)]
pub(crate) struct OptimizeParams {
    /// Optimize `alpha`.
    pub alpha: bool,
    /// Optimize `beta`.
    pub beta: bool,
    /// Optimize `gamma`.
    pub gamma: bool,
    /// Optimize `phi`.
    pub phi: bool,
}

impl OptimizeParams {
    pub fn n_included(&self) -> usize {
        self.alpha as usize + self.beta as usize + self.gamma as usize + self.phi as usize
    }
}

/// Returns `x` if `x` is not NaN, otherwise returns `default`.
fn not_nan_or(x: f64, default: f64) -> f64 {
    if x.is_nan() {
        default
    } else {
        x
    }
}

/// An ETS model that has not been fit.
#[derive(Debug, Clone)]
pub struct Unfit {
    /// The type of model to be used.
    model_type: ModelType,

    /// Whether or not the model uses a damped trend.
    ///
    /// Defaults to `false`.
    damped: bool,

    /// Number of steps over which to calculate the average MSE.
    ///
    /// Will be constrained to the range [1, 30].
    ///
    /// Defaults to 3.
    nmse: usize,

    /// The bounds on parameters.
    ///
    /// Defaults to [`Bounds::Both`] with lower limits of
    /// `[0.0001, 0.0001, 0.0001, 0.8]` and upper limits of
    /// `[0.9999, 0.9999, 0.9999, 0.98]`.
    bounds: Bounds,

    /// The parameters of the model.
    ///
    /// Defaults to [`Params::default()`], meaning the parameters will be
    /// determined and optimized by the optimizer.
    params: Params,

    /// Optimization criteria to use.
    ///
    /// Defaults to [`OptimizationCriteria::Likelihood`].
    opt_crit: OptimizationCriteria,

    /// Maximum number of iterations to use in the optimizer.
    ///
    /// Defaults to 2,000.
    max_iter: usize,
}

impl Unfit {
    /// Creates a new ETS model with the given type.
    pub fn new(model_type: ModelType) -> Self {
        Self {
            model_type,
            damped: false,
            bounds: Bounds::default(),
            nmse: 3,
            params: Params::default(),
            opt_crit: OptimizationCriteria::default(),
            max_iter: 2_000,
        }
    }

    /// Set the parameters of the model.
    ///
    /// To leave parameters unspecified, leave them set to `f64::NAN`.
    pub fn params(self, params: Params) -> Self {
        Self { params, ..self }
    }

    /// Set the number of steps over which to calculate the average MSE.
    pub fn nmse(self, nmse: usize) -> Self {
        Self { nmse, ..self }
    }

    /// Set the optimization criteria to use.
    pub fn opt_crit(self, opt_crit: OptimizationCriteria) -> Self {
        Self { opt_crit, ..self }
    }

    /// Set the maximum number of iterations to use in the optimizer.
    pub fn max_iterations(self, max_iterations: usize) -> Self {
        Self {
            max_iter: max_iterations,
            ..self
        }
    }

    /// Set the model to use a damped trend or not.
    pub fn damped(self, damped: bool) -> Self {
        Self { damped, ..self }
    }

    /// Select a sensible initial value for the `alpha` parameter.
    fn select_alpha(lower: &[f64; 4], upper: &[f64; 4], alpha: f64, m: usize) -> f64 {
        if alpha.is_nan() {
            let mut alpha = lower[0] + 0.2 * (upper[0] - lower[0]) / m as f64;
            if !(0.0..=1.0).contains(&alpha) {
                alpha = lower[0] + 2e-3;
            }
            alpha
        } else {
            alpha
        }
    }

    /// Select a sensible initial value for the `beta` parameter.
    fn select_beta(
        lower: &[f64; 4],
        upper: &mut [f64; 4],
        trend: TrendComponent,
        alpha: f64,
        beta: f64,
    ) -> f64 {
        if trend != TrendComponent::None && beta.is_nan() {
            // Ensure beta < alpha.
            upper[1] = upper[1].min(alpha);
            let mut beta = lower[1] + 0.1 * (upper[1] - lower[1]);
            if beta < 0.0 || beta > alpha {
                beta = alpha - 1e-3;
            }
            beta
        } else {
            beta
        }
    }

    /// Select a sensible initial value for the `gamma` parameter.
    fn select_gamma(
        lower: &[f64; 4],
        upper: &mut [f64; 4],
        season: SeasonalComponent,
        alpha: f64,
        gamma: f64,
    ) -> f64 {
        if season != SeasonalComponent::None && gamma.is_nan() {
            upper[2] = upper[2].min(1.0 - alpha);
            let mut gamma = lower[2] + 0.05 * (upper[2] - lower[2]);
            if gamma < 0.0 || gamma > 1.0 - alpha {
                gamma = 1.0 - alpha - 1e-3;
            }
            gamma
        } else {
            gamma
        }
    }

    /// Select a sensible initial value for the `phi` parameter.
    fn select_phi(lower: &[f64; 4], upper: &[f64; 4], damped: bool, phi: f64) -> f64 {
        if damped && phi.is_nan() {
            let mut phi = lower[3] + 0.99 * (upper[3] - lower[3]);
            if !(0.0..=1.0).contains(&phi) {
                phi = upper[3] - 1e-3;
            }
            phi
        } else {
            phi
        }
    }

    /// Initialize the parameters for the model.
    fn initial_params(&mut self) -> Params {
        // These dummy parameters aren't used, they're just here to placate the borrow checker.
        let (mut dummy_lower, mut dummy_upper) = ([0.0; 4], [1e-3; 4]);
        let (lower, upper) = match &mut self.bounds {
            Bounds::Admissible => (&mut dummy_lower, &mut dummy_upper),
            Bounds::Usual(UpperLowerBounds { lower, upper }) => (lower, upper),
            Bounds::Both(UpperLowerBounds { lower, upper }) => (lower, upper),
        };
        let alpha = Self::select_alpha(
            lower,
            upper,
            self.params.alpha,
            self.model_type.season.season_length(),
        );
        let beta = Self::select_beta(lower, upper, self.model_type.trend, alpha, self.params.beta);
        let gamma = Self::select_gamma(
            lower,
            upper,
            self.model_type.season,
            alpha,
            self.params.gamma,
        );
        let phi = Self::select_phi(lower, upper, self.damped, self.params.phi);
        Params {
            alpha,
            beta,
            gamma,
            phi,
        }
    }

    /// Initialize the state for the model.
    fn initial_state(&self, y: &[f64]) -> Result<Vec<f64>, Error> {
        let n = y.len();
        let (m, y_sa) = if self.model_type.season == SeasonalComponent::None {
            (1, y.to_vec())
        } else {
            unimplemented!("seasonal component not implemented yet")
            // if n < 4 {
            //     return Err(Error::NotEnoughData);
            // }
            // let y_d = if n < 3 * self.m {
            //     let fourier_y = fourier(self.y, &[self.m], &[1]);
            //     // TODO: remove these copies.
            //     let mut fourier_X = DMatrix::from_element(n, 4, f64::NAN);
            //     fourier_X.set_column(0, &DVector::from_element(n, 1.0));
            //     fourier_X.set_column(1, &DVector::from_iterator(n, (0..n).map(|x| x as f64)));
            //     fourier_X.set_column(2, &fourier_y.column(0));
            //     fourier_X.set_column(3, &fourier_y.column(1));
            //     let coefs = lstsq(&fourier_X, &self.y, 1e-6)?;
            //     if self.season == ComponentSpec::Additive {
            //         let mut y_d = self.y.clone();
            //         for (i, &x) in fourier_X.column(2).iter().enumerate() {
            //             y_d[i] -= coefs[2] * x;
            //         }
            //         for (i, &x) in fourier_X.column(3).iter().enumerate() {
            //             y_d[i] -= coefs[3] * x;
            //         }
            //         y_d
            //     } else {
            //         let mut y_d = self.y.clone();
            //         for (i, &x) in fourier_X.column(2).iter().enumerate() {
            //             y_d[i] /= coefs[2] * x;
            //         }
            //         for (i, &x) in fourier_X.column(3).iter().enumerate() {
            //             y_d[i] /= coefs[3] * x;
            //         }
            //         y_d
            //     }
            // } else {
            //     seasonal_decompose(
            //         self.y,
            //         self.m,
            //         if self.season == ComponentSpec::Additive {
            //             ModelType::Additive
            //         } else {
            //             ModelType::Multiplicative
            //         },
            //     )
            // };
        };
        let max_n = 10.clamp(m, n);
        match self.model_type.trend {
            TrendComponent::None => {
                let l0 = y_sa.iter().take(max_n).sum::<f64>() / max_n as f64;
                Ok(vec![l0])
            }
            _ => {
                #[allow(non_snake_case)]
                let X = DMatrix::from_iterator(
                    max_n,
                    2,
                    std::iter::repeat(1.0)
                        .take(max_n)
                        .chain((1..(max_n + 1)).map(|x| x as f64)),
                );
                let y = DVector::from_row_slice(&y_sa[..max_n]);
                let lstsq = lstsq::lstsq(&X, &y, f64::EPSILON).map_err(Error::LeastSquares)?;
                let (l, b) = (lstsq.solution[0], lstsq.solution[1]);
                if self.model_type.trend == TrendComponent::Additive {
                    let (mut l0, mut b0) = (l, b);
                    if (l0 + b0).abs() < 1e-8 {
                        l0 *= 1.0 + 1e-3;
                        b0 *= 1.0 + 1e-3;
                    }
                    Ok(vec![l0, b0])
                } else {
                    let mut l0 = l + b;
                    if l0.abs() < 1e-8 {
                        l0 *= 1.0 + 1e-3;
                    }
                    let mut b0: f64 = (l + 2.0 * b) / l0;
                    let div = if b0.abs() < 1e-8 { 1e-8 } else { b0 };
                    l0 /= div;
                    if b0.abs() > 1e10 {
                        b0 = b0.signum() * 1e10;
                    }
                    if l0 < 1e-8 || b0 < 1e-8 {
                        // simple linear approximation didn't work
                        l0 = y_sa[0].max(1e-3);
                        let div = if y_sa[0].abs() < 1e-8 { 1e-8 } else { y_sa[0] };
                        b0 = (y_sa[1] / div).max(1e-3);
                    }
                    Ok(vec![l0, b0])
                }
            }
        }
    }

    /// Fit the ETS model to the data, returning a fitted [`Model`].
    #[instrument(skip_all)]
    pub fn fit(mut self, y: &[f64]) -> Result<Model, Error> {
        self.nmse = self.nmse.min(30);
        let season_length = self.model_type.season.season_length();

        let n_states = season_length * self.model_type.season.included() as usize
            + 1
            + self.model_type.trend.included() as usize;

        // Store the original parameters.
        let par_noopt = self.params.clone();
        let par_ = self.initial_params();
        let alpha = not_nan_or(par_.alpha, par_noopt.alpha);
        let beta = not_nan_or(par_.beta, par_noopt.beta);
        let gamma = not_nan_or(par_.gamma, par_noopt.gamma);
        let phi = not_nan_or(par_.phi, par_noopt.phi);
        if !check_params(
            &self.bounds,
            season_length,
            Params {
                alpha,
                beta,
                gamma,
                phi,
            },
        ) {
            return Err(Error::ParamsOutOfRange);
        }

        let initial_state = self.initial_state(y)?;
        let param_arr = [alpha, beta, gamma, phi];

        let x0: Vec<_> = param_arr
            .iter()
            .copied()
            .filter(|&x| !x.is_nan())
            .chain(initial_state.iter().copied())
            .collect();
        let np_ = x0.len();
        if np_ >= y.len() - 1 {
            return Err(Error::NotEnoughData);
        }
        let opt_params = OptimizeParams {
            alpha: !alpha.is_nan(),
            beta: !beta.is_nan(),
            gamma: !gamma.is_nan(),
            phi: !phi.is_nan(),
        };

        let params = Params {
            alpha,
            beta: if self.model_type.trend.included() {
                beta
            } else {
                0.0
            },
            phi: if self.damped { phi } else { 1.0 },
            gamma: if self.model_type.season.included() {
                gamma
            } else {
                0.0
            },
        };

        let opt_bounds = self.bounds.for_optimizer(&opt_params, n_states);
        // Construct the problem.
        let ets = Ets::new(
            self.model_type,
            self.damped,
            self.nmse,
            n_states,
            params,
            opt_params,
            self.opt_crit,
        );
        let mut problem = ETSProblem::new(y, ets);
        // Set up the input simplex for Nelder-Mead.
        let simplex = self.param_vecs(x0, opt_bounds.as_ref());
        // Run Nelder-Mead.
        let best_params = self.nelder_mead(&mut problem, simplex, opt_bounds.as_ref());

        // Rerun the model with the best parameters.
        problem.amse.fill(0.0);
        problem.denom.fill(0.0);
        let fit = problem.ets.pegels_resid_in(
            y,
            &best_params,
            problem.x,
            problem.ets.params.clone(),
            problem.residuals,
            problem.forecasts,
            problem.amse,
            problem.denom,
        );
        let sigma_squared = y
            .iter()
            .zip(fit.fitted())
            .map(|(y, f)| (y - f).powi(2))
            .sum::<f64>()
            / (y.len() - fit.n_params() - 1) as f64;
        Ok(Model::new(problem.ets, fit, sigma_squared.sqrt()))
    }

    /// Generate the initial simplex.
    ///
    /// The original article suggested a simplex where an initial point is given
    /// as x0 with the others generated a fixed step along each dimension in turn.
    #[instrument(skip_all)]
    fn param_vecs(&self, mut x0: Vec<f64>, bounds: Option<&(Vec<f64>, Vec<f64>)>) -> Vec<Vec<f64>> {
        if let Some((lower, upper)) = bounds {
            Self::restrict_to_bounds(&mut x0, lower, upper);
        }
        let n = x0.len();

        let mut simplex = vec![x0; n + 1];
        let diag = simplex
            .iter_mut()
            .take(n)
            .enumerate()
            .map(|(i, row)| &mut row[i]);
        for el in diag {
            if el.abs() < 1e-8 {
                *el = 1e-4;
            } else {
                *el *= 1.05;
            }
        }
        if let Some((lower, upper)) = bounds {
            for row in simplex.iter_mut() {
                Self::restrict_to_bounds(row, lower, upper)
            }
        }
        simplex
    }

    const TOL_STD: f64 = 1e-4;

    /// Run the Nelder-Mead algorithm.
    ///
    /// This is a custom implementation of the Nelder-Mead algorithm, which is
    /// based on the implementation in the `statsforecast` Python package.
    /// It implements bounds checks and a custom stopping criterion.
    ///
    /// It could be generalised by making `problem` a generic type but I can't
    /// see that being needed.
    #[instrument(skip_all)]
    fn nelder_mead(
        &self,
        problem: &mut ETSProblem<'_>,
        mut simplex: Vec<Vec<f64>>,
        bounds: Option<&(Vec<f64>, Vec<f64>)>,
    ) -> Vec<f64> {
        let n_u = simplex[0].len();
        let n = simplex[0].len() as f64;

        let alpha = 1.0;
        let gamma = 1.0 + 2.0 / n;
        let rho = 0.75 - 1.0 / (2.0 * n);
        let sigma = 1.0 - 1.0 / n;

        let mut f_simplex: Vec<_> = simplex.iter().map(|x| problem.cost(x)).collect();
        let mut costs_sorted: Vec<_> = f_simplex.iter().copied().enumerate().collect();
        let mut order_f: Vec<_> = costs_sorted.iter().map(|(i, _)| *i).collect();
        let mut best_idx = order_f[0];
        let mut x_o: Vec<_>;
        let mut x_r: Vec<_>;
        let mut x_e: Vec<_>;
        let mut x_oc: Vec<_>;
        let mut x_ic: Vec<_>;
        for _ in 0..self.max_iter {
            costs_sorted.clear();
            costs_sorted.extend(f_simplex.iter().copied().enumerate());
            costs_sorted.sort_unstable_by(|(_, a), (_, b)| {
                a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
            });
            order_f.clear();
            order_f.extend(costs_sorted.iter().map(|(i, _)| *i));

            best_idx = order_f[0];
            let worst_idx = order_f[order_f.len() - 1];
            let second_worst_idx = order_f[order_f.len() - 2];

            // Check stopping criteria.
            if f_simplex.std(0) < Self::TOL_STD {
                break;
            }

            // Calculate centroid except argmax f_simplex.
            x_o = vec![0.0; n_u];
            for x in simplex
                .iter()
                .enumerate()
                .filter_map(|(i, x)| (i != worst_idx).then_some(x))
            {
                for (i, el) in x.iter().enumerate() {
                    x_o[i] += el;
                }
            }
            for x in x_o.iter_mut() {
                *x /= n;
            }

            // Step 2: Reflection, Compute reflected point
            x_r = x_o
                .iter()
                .zip(&simplex[worst_idx])
                .map(|(x_0, x)| x_0 + alpha * (x_0 - x))
                .collect();
            if let Some((lower, upper)) = &bounds {
                Self::restrict_to_bounds(&mut x_r, lower, upper);
            }
            let f_r = problem.cost(&x_r);
            if f_simplex[best_idx] <= f_r && f_r < f_simplex[second_worst_idx] {
                simplex[worst_idx] = x_r;
                f_simplex[worst_idx] = f_r;
                continue;
            }

            // Step 3: Expansion, reflected point is the best point so far
            if f_r < f_simplex[best_idx] {
                x_e = x_o
                    .iter()
                    .zip(&x_r)
                    .map(|(x_o, x_r)| x_o + gamma * (x_r - x_o))
                    .collect();
                if let Some((lower, upper)) = &bounds {
                    Self::restrict_to_bounds(&mut x_e, lower, upper);
                }
                let f_e = problem.cost(&x_e);
                if f_e < f_r {
                    simplex[worst_idx] = x_e;
                    f_simplex[worst_idx] = f_e;
                } else {
                    simplex[worst_idx] = x_r;
                    f_simplex[worst_idx] = f_r;
                }
                continue;
            }

            // Step 4: outside Contraction
            if f_simplex[second_worst_idx] <= f_r && f_r < f_simplex[worst_idx] {
                x_oc = x_o
                    .iter()
                    .zip(&x_r)
                    .map(|(x_o, x_r)| x_o + rho * (x_r - x_o))
                    .collect();
                if let Some((lower, upper)) = &bounds {
                    Self::restrict_to_bounds(&mut x_oc, lower, upper);
                }
                let f_oc = problem.cost(&x_oc);
                if f_oc <= f_r {
                    simplex[worst_idx] = x_oc;
                    f_simplex[worst_idx] = f_oc;
                    continue;
                }
            } else {
                // Step 5: inside contraction
                x_ic = x_o
                    .iter()
                    .zip(&x_r)
                    .map(|(x_o, x_r)| x_o - rho * (x_r - x_o))
                    .collect();
                if let Some((lower, upper)) = &bounds {
                    Self::restrict_to_bounds(&mut x_ic, lower, upper);
                }
                let f_ic = problem.cost(&x_ic);
                if f_ic < f_simplex[worst_idx] {
                    simplex[worst_idx] = x_ic;
                    f_simplex[worst_idx] = f_ic;
                    continue;
                }
            }

            // Step 6: shrink
            let best = simplex[best_idx].clone();
            simplex.iter_mut().enumerate().for_each(|(i, x)| {
                if i != best_idx {
                    x.iter_mut()
                        .zip(&best)
                        .for_each(|(x, x_best)| *x = x_best + sigma * (*x - x_best));
                    if let Some((lower, upper)) = &bounds {
                        Self::restrict_to_bounds(&mut x_r, lower, upper);
                    }
                    f_simplex[i] = problem.cost(x);
                }
            });
        }
        simplex[best_idx].clone()
    }

    /// Restrict `x0` to the bounds given by `lower` and `upper`.
    fn restrict_to_bounds(x0: &mut [f64], lower: &[f64], upper: &[f64]) {
        x0.iter_mut()
            .zip(lower)
            .zip(upper)
            .for_each(|((x, &l), &u)| {
                *x = x.clamp(l, u);
            });
    }
}

// This was generated by ChatGPT, we should probably check it...
// In particular the `roots` part is unclear since the `roots` crate only returns real roots,
// but the R/Python implementations reference complex roots too.
fn admissible(alpha: f64, mut beta: f64, gamma: f64, mut phi: f64, m: usize) -> bool {
    const EPSILON: f64 = 1e-8;
    if phi.is_nan() {
        phi = 1.0;
    }
    if !(0.0..=1.0 + EPSILON).contains(&phi) {
        return false;
    }
    if gamma.is_nan() {
        if alpha < 1.0 - 1.0 / phi || alpha > 1.0 + 1.0 / phi {
            return false;
        }
        if !beta.is_nan() && (beta < alpha * (phi - 1.0) || beta > (1.0 + phi) * (2.0 - alpha)) {
            return false;
        }
    } else if m > 1 {
        if beta.is_nan() {
            beta = 0.0;
        }
        if gamma < f64::max(1.0 - 1.0 / phi - alpha, 0.0) || gamma > 1.0 + 1.0 / phi - alpha {
            return false;
        }
        if alpha
            < 1.0
                - 1.0 / phi
                - gamma * (1.0 - m as f64 + phi + phi * m as f64) / (2.0 * phi * m as f64)
        {
            return false;
        }
        if beta < -(1.0 - phi) * (gamma / m as f64 + alpha) {
            return false;
        }
        let mut p: Vec<f64> = vec![f64::NAN; 2 + m];
        p[0] = phi * (1.0 - alpha - gamma);
        p[1] = alpha + beta - alpha * phi + gamma - 1.0;
        p[2..m].fill(alpha + beta - alpha * phi);
        p[m..].fill(alpha + beta - phi);
        p[m + 1] = 1.0;
        let roots = roots::find_roots_eigen(p);
        let max_ = roots
            .into_iter()
            .fold(f64::NEG_INFINITY, |max_, r| r.abs().max(max_));
        if max_ > 1.0 + 1e-10 {
            return false;
        }
    }
    true
}

/// A 'problem' for the Nelder-Mead algorithm.
///
/// This just groups together and holds several pieces of data that are used in the
/// cost function called by the Nelder-Mead algorithm. It saves us from having to
/// pass around a bunch of arguments to the Nelder-Mead function.
pub(crate) struct ETSProblem<'a> {
    y: &'a [f64],
    ets: Ets,
    x: Vec<f64>,
    residuals: Vec<f64>,
    forecasts: Vec<f64>,
    amse: Vec<f64>,
    denom: Vec<f64>,
}

impl<'a> ETSProblem<'a> {
    /// Create a new problem.
    ///
    /// The `y` argument is the time series to fit.
    /// The `ets` argument is the ETS model to fit.
    ///
    /// The returned problem is ready to be passed to the Nelder-Mead algorithm.
    /// Each of the vectors in the problem is pre-allocated to the correct size.
    pub(crate) fn new(y: &'a [f64], ets: Ets) -> Self {
        let nmse = ets.nmse;
        let x_len = ets.n_states * (y.len() + 1);
        Self {
            y,
            ets,
            x: vec![0.0; x_len],
            residuals: vec![0.0; y.len()],
            forecasts: vec![0.0; nmse],
            amse: vec![0.0; nmse],
            denom: vec![0.0; nmse],
        }
    }

    /// Calculate the cost function.
    ///
    /// The first `self.n_states` elements of `param` are the initial values of the parameters.
    /// The remaining elements are the initial state.
    fn cost(&mut self, inputs: &[f64]) -> f64 {
        let Ets {
            params,
            opt_params,
            opt_crit,
            n_states,
            ..
        } = &self.ets;
        let mut params = params.clone();

        // If we're optimizing params, they'll be included the inputs to the
        // optimizer, so use them to override the defaults.
        let mut i = 0;
        if opt_params.alpha {
            params.alpha = inputs[i];
            i += 1;
        }
        if opt_params.beta {
            params.beta = inputs[i];
            i += 1;
        }
        if opt_params.gamma {
            params.gamma = inputs[i];
            i += 1;
        }
        if opt_params.phi {
            params.phi = inputs[i];
            i += 1;
        }

        // The remaining parameters are the initial state.
        let state_inputs = &inputs[i..];
        self.x.truncate(state_inputs.len());
        self.x.copy_from_slice(state_inputs);
        self.x.resize(n_states * (self.y.len() + 1), 0.0);
        // TODO: add extra state for seasonality?

        // Calculate the cost.
        let fit = self.ets.etscalc_in(
            self.y,
            &mut self.x,
            params,
            &mut self.residuals,
            &mut self.forecasts,
            &mut self.amse,
            &mut self.denom,
            // We only need to update the AMSE if we're optimizing using
            // AMSE-based criteria.
            matches!(
                opt_crit,
                OptimizationCriteria::MSE | OptimizationCriteria::AMSE
            ),
        );
        match opt_crit {
            OptimizationCriteria::Likelihood => fit.likelihood(),
            OptimizationCriteria::MSE => fit.mse(),
            OptimizationCriteria::AMSE => fit.amse(),
            OptimizationCriteria::Sigma => fit.sigma_squared(),
            OptimizationCriteria::MAE => fit.mae(),
        }
    }
}

/// Check that the parameters are within the bounds.
fn check_params(bounds: &Bounds, season_length: usize, params: Params) -> bool {
    let Params {
        alpha,
        beta,
        gamma,
        phi,
    } = params;
    if let Bounds::Usual(UpperLowerBounds {
        lower: [lower_a, lower_b, lower_g, lower_p],
        upper: [upper_a, upper_b, upper_g, upper_p],
    })
    | Bounds::Both(UpperLowerBounds {
        lower: [lower_a, lower_b, lower_g, lower_p],
        upper: [upper_a, upper_b, upper_g, upper_p],
    }) = bounds
    {
        if !(alpha.is_nan() || alpha >= *lower_a && alpha <= *upper_a) {
            return false;
        }
        if !(beta.is_nan() || beta >= *lower_b && beta <= alpha && beta <= *upper_b) {
            return false;
        }
        if !(gamma.is_nan() || gamma >= *lower_g && gamma <= 1.0 - alpha && gamma <= *upper_g) {
            return false;
        }
        if !(phi.is_nan() || phi >= *lower_p && phi <= *upper_p) {
            return false;
        }
    }
    if !matches!(bounds, Bounds::Usual(_)) {
        return admissible(alpha, beta, gamma, phi, season_length);
    }
    true
}

/// A fitted ETS model.
#[derive(Debug, Clone)]
pub struct Model {
    /// The original model.
    ets: Ets,

    /// The fitted model state, parameters and likelihood.
    model_fit: FitState,

    /// The standard error of the residuals.
    ///
    /// This is used when calculating prediction intervals for in-sample
    /// predictions.
    sigma: f64,
}

impl Model {
    fn new(ets: Ets, fit: FitState, sigma: f64) -> Model {
        Self {
            ets,
            model_fit: fit,
            sigma,
        }
    }

    fn pegels_forecast(&self, horizon: usize) -> Vec<f64> {
        let mut forecasts = vec![0.0; horizon];
        let states = self.model_fit.states().last().unwrap();
        let phi = if self.ets.damped {
            self.model_fit.params().phi
        } else {
            1.0
        };
        let b = if self.ets.model_type.trend.included() {
            Some(states[1])
        } else {
            None
        };
        self.ets
            .forecast(phi, states[0], b, &mut forecasts, horizon);
        forecasts
    }

    /// The log-likelihood of the model.
    pub fn log_likelihood(&self) -> f64 {
        -0.5 * self.model_fit.likelihood()
    }

    /// The Akaike Information Criterion (AIC) of the model.
    pub fn aic(&self) -> f64 {
        self.model_fit.likelihood() + 2.0 * self.model_fit.n_params() as f64
    }

    /// The corrected Akaike Information Criterion (AICC) of the model.
    pub fn aicc(&self) -> f64 {
        let n_y = self.model_fit.residuals().len();
        let n_params = self.model_fit.n_params() + 1;
        let aic = self.aic();
        let denom = n_y - n_params - 1;
        if denom != 0 {
            aic + 2.0 * n_params as f64 * (n_params as f64 + 1.0) / denom as f64
        } else {
            f64::INFINITY
        }
    }

    /// The Bayesian Information Criterion (BIC) of the model.
    pub fn bic(&self) -> f64 {
        self.model_fit.likelihood()
            + (self.model_fit.n_params() as f64 + 1.0)
                * ((self.model_fit.residuals().len() as f64).ln())
    }

    /// The mean squared error (MSE) of the model.
    pub fn mse(&self) -> f64 {
        self.model_fit.mse()
    }

    /// The average mean squared error (AMSE) of the model.
    ///
    /// This is the average of the MSE over the number of forecasting horizons (`nmse`).
    pub fn amse(&self) -> f64 {
        self.model_fit.amse()
    }

    /// Predict the next `horizon` values using the model.
    pub fn predict(&self, horizon: usize, level: impl Into<Option<f64>>) -> augurs_core::Forecast {
        self.predict_impl(horizon, level.into()).0
    }

    fn predict_impl(&self, horizon: usize, level: Option<f64>) -> Forecast {
        // Short-circuit if horizon is zero.
        if horizon == 0 {
            return Forecast(augurs_core::Forecast {
                point: vec![],
                intervals: level.map(ForecastIntervals::empty),
            });
        }

        let mut f = Forecast(augurs_core::Forecast {
            point: self.pegels_forecast(horizon),
            intervals: None,
        });
        if let Some(level) = level {
            f.calculate_intervals(&self.ets, &self.model_fit, horizon, level);
        }
        f
    }

    /// Return the model's predictions for the in-sample data.
    pub fn predict_in_sample(&self, level: impl Into<Option<f64>>) -> augurs_core::Forecast {
        self.predict_in_sample_impl(level.into()).0
    }

    fn predict_in_sample_impl(&self, level: Option<f64>) -> Forecast {
        let mut f = Forecast(augurs_core::Forecast {
            point: self.model_fit.fitted().to_vec(),
            intervals: None,
        });
        if let Some(level) = level {
            f.calculate_in_sample_intervals(self.sigma, level);
        }
        f
    }

    /// The model type.
    pub fn model_type(&self) -> ModelType {
        self.ets.model_type
    }

    /// Whether the model uses damped trend.
    pub fn damped(&self) -> bool {
        self.ets.damped
    }
}

struct Forecast(augurs_core::Forecast);

impl Forecast {
    /// Calculate the prediction intervals for the forecast.
    fn calculate_intervals(&mut self, ets: &Ets, fit: &FitState, horizon: usize, level: f64) {
        let sigma = fit.sigma_squared();
        let season_length = ets.model_type.season.season_length();
        let season_length_f = season_length as f64;

        let ModelType {
            error,
            trend,
            season,
        } = ets.model_type;
        let steps: Vec<_> = (1..(horizon + 1)).map(|x| x as f64).collect();
        let hm = ((horizon - 1) as f64 / season_length_f).floor();

        let Params {
            alpha,
            beta,
            gamma,
            phi,
        } = fit.params();

        let alpha_2 = alpha.powi(2);
        let phi_2 = phi.powi(2);

        let exp3 = 2.0 * alpha * (1.0 - phi) + beta * phi;
        let (exp1, exp2, exp4, exp5): (Vec<_>, Vec<_>, Vec<_>, Vec<_>) = steps
            .iter()
            .copied()
            .map(|s| {
                let phi_s = phi.powi(s as i32);
                (
                    alpha_2 + alpha * beta * s + (1.0 / 6.0) * beta.powi(2) * s * (2.0 * s - 1.0),
                    (beta * phi * s) / (1.0 - phi).powi(2),
                    (beta * phi * (1.0 - phi_s)) / ((1.0 - phi).powi(2) * (1.0 - phi_2)),
                    2.0 * alpha * (1.0 - phi_2) + beta * phi * (1.0 + 2.0 * phi - phi_s),
                )
            })
            .multiunzip();

        use {ErrorComponent as EC, SeasonalComponent as SC, TrendComponent as TC};
        let (lower, upper) =
            match (error, trend, season, ets.damped) {
                // Class 1 models.
                // ANN
                (EC::Additive, TC::None, SC::None, false) => {
                    let sigma_h = steps
                        .iter()
                        .map(|s| (((s - 1.0) * alpha.powi(2) + 1.0) * sigma).sqrt());
                    self.compute_intervals(level, sigma_h)
                }
                // AAN
                (EC::Additive, TC::Additive, SC::None, false) => {
                    let sigma_h = steps
                        .iter()
                        .zip(&exp1)
                        .map(|(s, e)| ((1.0 + (s - 1.0) * e) * sigma).sqrt());
                    self.compute_intervals(level, sigma_h)
                }
                // AAdN
                (EC::Additive, TC::Additive, SC::None, true) => {
                    let sigma_h =
                        steps
                            .iter()
                            .zip(&exp2)
                            .zip(&exp4)
                            .zip(&exp5)
                            .map(|(((s, e2), e4), e5)| {
                                ((1.0 + alpha_2 * (s - 1.0) + e2 * exp3 - e4 * e5) * sigma).sqrt()
                            });
                    self.compute_intervals(level, sigma_h)
                }
                // ANA
                (EC::Additive, TC::None, SC::Additive { .. }, false) => {
                    let sigma_h = steps.iter().map(|s| {
                        ((1.0 + alpha_2 * (s - 1.0) + gamma * hm * (2.0 * alpha * gamma)) * sigma)
                            .sqrt()
                    });
                    self.compute_intervals(level, sigma_h)
                }
                // AAA
                (EC::Additive, TC::Additive, SC::Additive { .. }, false) => {
                    let sigma_h = steps.iter().zip(&exp1).map(|(s, e1)| {
                        let e6 = 2.0 * alpha + gamma + beta * season_length_f * (hm + 1.0);
                        ((1.0 + (s - 1.0) * e1 * gamma * hm * e6) * sigma).sqrt()
                    });
                    self.compute_intervals(level, sigma_h)
                }
                // AAdA
                (EC::Additive, TC::Additive, SC::Additive { season_length }, true) => {
                    let sigma_h = steps.iter().zip(&exp2).zip(&exp4).zip(&exp5).map(
                        |(((&s, e2), e4), e5)| {
                            let phi_s = phi.powi(s as i32);
                            let e7 = (2.0 * beta * gamma * phi) / ((1.0 - phi) * (1.0 - phi_s));
                            let e8 = hm * (1.0 - phi_s)
                                - phi_s * (1.0 - phi.powi(season_length as i32 * hm as i32));
                            ((1.0 + alpha_2 * (s - 1.0) + e2 * exp3 - e4 * e5
                                + gamma * hm * (2.0 * alpha + gamma)
                                + e7 * e8)
                                * sigma)
                                .sqrt()
                        },
                    );
                    self.compute_intervals(level, sigma_h)
                }
                // Class 2 models.
                // MNN
                (EC::Multiplicative, TC::None, SC::None, false) => {
                    let cvals = std::iter::repeat(*alpha).take(horizon);
                    let sigma_h = self.compute_sigma_h(sigma, cvals, horizon);
                    self.compute_intervals(level, sigma_h.into_iter())
                }
                // MAN
                (EC::Multiplicative, TC::Additive, SC::None, false) => {
                    let cvals = steps.iter().map(|s| alpha + beta * s);
                    let sigma_h = self.compute_sigma_h(sigma, cvals, horizon);
                    self.compute_intervals(level, sigma_h.into_iter())
                }
                // MAdN
                (EC::Multiplicative, TC::Additive, SC::None, true) => {
                    let mut cvals: Vec<_> = vec![f64::NAN; horizon];
                    for k in 1..(horizon + 1) {
                        let sum_phi = (1..(k + 1)).map(|j| phi.powi(j as i32)).sum::<f64>();
                        cvals[k - 1] = alpha + beta * sum_phi;
                    }
                    let sigma_h = self.compute_sigma_h(sigma, cvals.into_iter(), horizon);
                    self.compute_intervals(level, sigma_h.into_iter())
                }
                // TODO: all below models, once we do seasonality.
                // MNA
                (EC::Multiplicative, TC::None, SC::Additive { .. }, false) => todo!(),
                // MAA
                (EC::Multiplicative, TC::Additive, SC::Additive { .. }, false) => todo!(),
                // MAdA
                (EC::Multiplicative, TC::Additive, SC::Additive { .. }, true) => todo!(),
                // Class 3 models.
                // Anything with multiplicative error and seasonality?
                (EC::Multiplicative, _, SC::Multiplicative { .. }, _) => {
                    unimplemented!(
                        "Prediction intervals for class 3 models are not implemented yet"
                    )
                }
                // Class 4 or 5 models without seasonality.
                // In future we should also handle those with seasonality.
                (_, _, SC::None, _) => {
                    // Simulate.
                    self.simulate(ets, fit, horizon, level)
                }
                // Any other models aren't yet implemented.
                _ => unimplemented!("Prediction intervals for this model are not implemented yet"),
            };
        self.0.intervals = Some(ForecastIntervals {
            level,
            lower,
            upper,
        });
    }

    /// Compute the prediction intervals for a given level.
    ///
    /// `level` should be a number between 0 and 1.
    /// `sigma_h` is the standard deviation of the residuals.
    fn compute_intervals(
        &self,
        level: f64,
        sigma_h: impl Iterator<Item = f64>,
    ) -> (Vec<f64>, Vec<f64>) {
        let z = distrs::Normal::ppf(0.5 + level / 2.0, 0.0, 1.0);
        self.0
            .point
            .iter()
            .zip(sigma_h)
            .map(|(p, s)| (p - z * s, p + z * s))
            .unzip()
    }

    /// Compute the standard deviations of the residuals given the model's
    /// overall standard deviation and some critical values.
    fn compute_sigma_h(
        &self,
        sigma: f64,
        cvals: impl Iterator<Item = f64>,
        horizon: usize,
    ) -> Vec<f64> {
        let cvals_squared: Vec<_> = cvals.map(|c| c.powi(2)).collect();
        let theta =
            // Iterate over each point estimate, up to `horizon`.
            &self
                .0
                .point
                .iter()
                // `point` should always have length == horizon, but `take` just in case
                .take(horizon)
                .fold(Vec::with_capacity(horizon), |mut acc, p| {
                    // For each point estimate, accumulate a vec of
                    // errors so far, by iterating the current accumulator,
                    // zipping with the reversed critical values, and multiplying.
                    // Sum the totals up until this point, then multiply with sigma
                    // and add that onto the accumulator.
                    let t = p.powi(2)
                        + acc
                            .iter()
                            .rev()
                            .zip(&cvals_squared)
                            .map(|(t, c)| t * c)
                            .sum::<f64>()
                            * sigma;
                    acc.push(t);
                    acc
                });
        theta
            .iter()
            .zip(&self.0.point)
            .map(|(t, p)| ((1.0 + sigma) * t - p.powi(2)).sqrt())
            .collect()
    }

    fn simulate(
        &self,
        ets: &Ets,
        fit: &FitState,
        horizon: usize,
        level: f64,
    ) -> (Vec<f64>, Vec<f64>) {
        let n_sim = 5000;
        let last_state = fit.last_state();
        let mut y_path = vec![vec![0.0; horizon]; n_sim];
        let params = fit.params();
        let beta = if params.beta.is_nan() {
            0.0
        } else {
            params.beta
        };
        let gamma = if params.gamma.is_nan() {
            0.0
        } else {
            params.gamma
        };
        let phi = if params.phi.is_nan() { 0.0 } else { params.phi };
        let rng = &mut rand::thread_rng();
        let normal = Normal::new(0.0, fit.sigma_squared().sqrt()).unwrap();
        // Use the same `f` vector for each simulation to avoid re-allocating.
        // For some reason statsforecast uses a length of 10 for `f`?
        let mut f = vec![0.0; 10];
        for y_path_k in &mut y_path {
            let e: Vec<_> = (0..horizon).map(|_| normal.sample(rng)).collect();
            ets.etssimulate(
                last_state,
                Params {
                    alpha: params.alpha,
                    beta,
                    gamma,
                    phi,
                },
                &e,
                &mut f,
                y_path_k,
            );
            f.iter_mut().for_each(|f| *f = 0.0);
        }
        y_path
            .into_iter()
            .map(|mut yhat| {
                yhat.sort_by(|a, b| a.partial_cmp(b).unwrap());
                (
                    percentile_of_sorted(&yhat, 0.5 - level / 2.0),
                    percentile_of_sorted(&yhat, 0.5 + level / 2.0),
                )
            })
            .unzip()
    }

    fn calculate_in_sample_intervals(&mut self, sigma: f64, level: f64) {
        let (lower, upper) = self.compute_intervals(level, std::iter::repeat(sigma));
        self.0.intervals = Some(ForecastIntervals {
            level,
            lower,
            upper,
        });
    }
}

// Taken from the Rust compiler's test suite:
// https://github.com/rust-lang/rust/blob/917b0b6c70f078cb08bbb0080c9379e4487353c3/library/test/src/stats.rs#L258-L280.
fn percentile_of_sorted(sorted_samples: &[f64], pct: f64) -> f64 {
    assert!(!sorted_samples.is_empty());
    if sorted_samples.len() == 1 {
        return sorted_samples[0];
    }
    let zero: f64 = 0.0;
    assert!(zero <= pct);
    let hundred = 100_f64;
    assert!(pct <= hundred);
    if pct == hundred {
        return sorted_samples[sorted_samples.len() - 1];
    }
    let length = (sorted_samples.len() - 1) as f64;
    let rank = (pct / hundred) * length;
    let lrank = rank.floor();
    let d = rank - lrank;
    let n = lrank as usize;
    let lo = sorted_samples[n];
    let hi = sorted_samples[n + 1];
    lo + (hi - lo) * d
}

#[cfg(test)]
mod test {
    use assert_approx_eq::assert_approx_eq;

    use crate::{
        assert_closeish,
        data::AIR_PASSENGERS as AP,
        model::{
            ErrorComponent, ForecastIntervals, ModelType, SeasonalComponent, TrendComponent, Unfit,
        },
    };

    #[test]
    fn initial_params() {
        let mut unfit = Unfit::new(ModelType {
            error: ErrorComponent::Additive,
            trend: TrendComponent::None,
            season: SeasonalComponent::None,
        });
        let initial_params = unfit.initial_params();
        assert_approx_eq!(initial_params.alpha, 0.20006);
        assert!(initial_params.beta.is_nan());
        assert!(initial_params.gamma.is_nan());
        assert!(initial_params.phi.is_nan());
    }

    #[test]
    fn air_passengers_fit_aan() {
        let unfit = Unfit::new(ModelType {
            error: ErrorComponent::Additive,
            trend: TrendComponent::Additive,
            season: SeasonalComponent::None,
        })
        .damped(true);
        let model = unfit.fit(&AP[AP.len() - 20..]).unwrap();
        assert_closeish!(model.log_likelihood(), -109.6248525790271, 0.01);
        assert_closeish!(model.aic(), 231.2497051580542, 0.01);
        assert_closeish!(model.bic(), 237.22409879937817, 0.01);
        assert_closeish!(model.aicc(), 237.71124361959266, 0.01);
        assert_closeish!(model.mse(), 2883.47944444736, 0.01);
        assert_closeish!(model.amse(), 8292.71075580747, 0.01);
    }

    #[test]
    fn air_passengers_fit_man() {
        let unfit = Unfit::new(ModelType {
            error: ErrorComponent::Multiplicative,
            trend: TrendComponent::Additive,
            season: SeasonalComponent::None,
        });
        let model = unfit.fit(&AP).unwrap();
        assert_closeish!(model.log_likelihood(), -831.4883541595792, 0.01);
        assert_closeish!(model.aic(), 1672.9767083191584, 0.01);
        assert_closeish!(model.bic(), 1687.8257748170383, 0.01);
        assert_closeish!(model.aicc(), 1673.4114909278542, 0.01);
        assert_closeish!(model.mse(), 1127.443938773091, 0.01);
        assert_closeish!(model.amse(), 2888.3802507845635, 0.01);
    }

    #[test]
    fn air_passengers_forecast_aan() {
        let unfit = Unfit::new(ModelType {
            error: ErrorComponent::Additive,
            trend: TrendComponent::Additive,
            season: SeasonalComponent::None,
        })
        .damped(true);
        let model = unfit.fit(&AP[AP.len() - 20..]).unwrap();
        let forecasts = model.predict(10, 0.95);
        let expected_p = [
            432.26645246,
            432.53827337,
            432.75575609,
            432.92976307,
            433.0689853,
            433.18037639,
            433.26949992,
            433.34080727,
            433.39785997,
            433.44350758,
        ];
        assert_eq!(forecasts.point.len(), 10);
        for (actual, expected) in forecasts.point.iter().zip(expected_p.iter()) {
            assert_approx_eq!(actual, expected);
        }

        let expected_l = [
            301.72457857,
            247.92511851,
            206.64496117,
            171.83062947,
            141.14177344,
            113.38060224,
            87.83698619,
            64.04903959,
            41.69638225,
            20.54598327,
        ];
        let ForecastIntervals { lower, upper, .. } = forecasts.intervals.unwrap();
        assert_eq!(lower.len(), 10);
        for (actual, expected) in lower.iter().zip(expected_l.iter()) {
            assert_approx_eq!(actual, expected);
        }
        let expected_u = [
            562.80832636,
            617.15142823,
            658.86655102,
            694.02889667,
            724.99619716,
            752.98015054,
            778.70201365,
            802.63257495,
            825.09933768,
            846.34103189,
        ];
        assert_eq!(upper.len(), 10);
        for (actual, expected) in upper.iter().zip(expected_u.iter()) {
            assert_approx_eq!(actual, expected);
        }
    }

    #[test]
    fn air_passengers_forecast_man() {
        let unfit = Unfit::new(ModelType {
            error: ErrorComponent::Multiplicative,
            trend: TrendComponent::Additive,
            season: SeasonalComponent::None,
        });
        let model = unfit.fit(&AP).unwrap();
        let forecasts = model.predict(10, 0.95);
        let expected_p = [
            436.15668239,
            440.31714837,
            444.47761434,
            448.63808031,
            452.79854629,
            456.95901226,
            461.11947823,
            465.27994421,
            469.44041018,
            473.60087615,
        ];
        assert_eq!(forecasts.point.len(), 10);
        for (actual, expected) in forecasts.point.iter().zip(expected_p.iter()) {
            assert_approx_eq!(actual, expected);
        }

        let expected_l = [
            345.14145884,
            310.62430297,
            284.42938026,
            262.42886479,
            243.03658151,
            225.44516176,
            209.1784846,
            193.92853297,
            179.48284058,
            165.68775958,
        ];
        let ForecastIntervals { lower, upper, .. } = forecasts.intervals.unwrap();
        assert_eq!(lower.len(), 10);
        for (actual, expected) in lower.iter().zip(expected_l.iter()) {
            assert_approx_eq!(actual, expected);
        }
        let expected_u = [
            527.17190595,
            570.00999376,
            604.52584842,
            634.84729584,
            662.56051106,
            688.47286276,
            713.06047187,
            736.63135545,
            759.39797978,
            781.51399273,
        ];
        assert_eq!(upper.len(), 10);
        for (actual, expected) in upper.iter().zip(expected_u.iter()) {
            assert_approx_eq!(actual, expected);
        }

        // For in-sample data, just check that the first 10 values match.
        let in_sample = model.predict_in_sample(0.95);
        let expected_p = [
            110.74681112,
            116.18804955,
            122.18817486,
            136.18835606,
            133.18933724,
            125.18861841,
            139.18739947,
            152.18838061,
            152.18926187,
            140.18884303,
        ];
        assert_eq!(in_sample.point.len(), AP.len());
        for (actual, expected) in in_sample.point.iter().zip(expected_p.iter()) {
            assert_approx_eq!(actual, expected);
        }

        let ForecastIntervals { lower, upper, .. } = in_sample.intervals.unwrap();
        let expected_l = [
            43.76306764,
            49.20430607,
            55.20443139,
            69.20461258,
            66.20559377,
            58.20487493,
            72.203656,
            85.20463713,
            85.20551839,
            73.20509956,
        ];
        assert_eq!(lower.len(), AP.len());
        for (actual, expected) in lower.iter().zip(expected_l.iter()) {
            assert_approx_eq!(actual, expected);
        }
        let expected_u = [
            177.73055459,
            183.17179302,
            189.17191834,
            203.17209954,
            200.17308072,
            192.17236188,
            206.17114295,
            219.17212409,
            219.17300535,
            207.17258651,
        ];
        assert_eq!(upper.len(), AP.len());
        for (actual, expected) in upper.iter().zip(expected_u.iter()) {
            assert_approx_eq!(actual, expected);
        }
    }

    #[test]
    fn predict_zero_horizon() {
        let unfit = Unfit::new(ModelType {
            error: ErrorComponent::Multiplicative,
            trend: TrendComponent::Additive,
            season: SeasonalComponent::None,
        });
        let model = unfit.fit(&AP).unwrap();
        let forecasts = model.predict(0, 0.95);
        assert!(forecasts.point.is_empty());
        let ForecastIntervals { lower, upper, .. } = forecasts.intervals.unwrap();
        assert!(lower.is_empty());
        assert!(upper.is_empty());
    }
}
