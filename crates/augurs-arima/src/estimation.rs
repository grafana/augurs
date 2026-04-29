//! ARIMA parameters estimation.
//!
//! Estimate an ARIMA model parameters through conditional sum-of-squares (CSS), maximum likelihood (ML) or a
//! combination of both (CSS-ML).

use crate::error::{Error, Result};
use crate::kalman::{kalman_filter, ArimaStateSpace};
use crate::types::{ArimaModel, ArimaOrder, EstimationMethod};
use crate::utils::{acf, diff, expand_poly, mean, seasonal_diff, variance};
use argmin::core::{CostFunction, Executor, Gradient, State};
use argmin::solver::linesearch::MoreThuenteLineSearch;
use argmin::solver::quasinewton::BFGS;
use nalgebra::{ComplexField, DMatrix, DVector};
use num_complex::Complex;
use std::ops::{Add, Div, Mul, Sub};

type BfgsState = argmin::core::IterState<Vec<f64>, Vec<f64>, (), Vec<Vec<f64>>, (), f64>;

/// Fit an (S)ARIMA model to a time series.
///
/// # Arguments
///
/// * `y: &[f64]` The time series data.
/// * `order: &ArimaOrder` An ArimaOrder struct with the order of the (S)ARIMA model.
/// * `method: EstimationMethod` The estimation method: CSS, ML or CSS-ML.
/// * `include_mean: bool` Should the stationary model include an intercept term.
/// * `include_drift: bool` Should the integrated model include an intercept term.
///
/// # Details
///
/// The hybrid approach CSS-ML runs the CSS estimation first to get a quick estimate of the parameters. Then, it runs
/// the ML estimation using the initial values obtained from the CSS estimation. Finally, it compares the log-likelihood
/// of the two models and returns the one with the highest log-likelihood.
///
/// # Returns
///
/// The estimated ARIMA model parameters.
///
/// # Examples
///
/// ```
/// use augurs_arima::estimation::fit_arima;
/// use augurs_arima::types::{ArimaOrder, EstimationMethod};
///
/// let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let order = ArimaOrder::new(1, 0, 1);
/// let method = EstimationMethod::CssMl;
///
/// let params = fit_arima(&y, &order, method, true, false);
/// println!("{:?}", params);
/// ```
pub fn fit_arima(
    y: &[f64],
    order: &ArimaOrder,
    method: EstimationMethod,
    include_mean: bool,
    include_drift: bool,
) -> Result<ArimaModel> {
    let n = y.len();

    // Check time series length against model order
    let mut required_n =
        order.p + order.d + order.q + order.sp + order.sd * order.period + order.sq;
    if include_drift || include_mean {
        required_n += 1;
    }
    if n < required_n {
        return Err(Error::SeriesTooShort {
            need: required_n,
            got: n,
        });
    };

    // Differencing
    let mut z = y.to_vec();
    if order.sd > 0 && order.period > 1 {
        z = seasonal_diff(&z, order.period, order.sd).unwrap();
    }
    if order.d > 0 {
        z = diff(&z, order.d).unwrap();
    }

    // compute mean
    let z_mean_init = if include_mean || include_drift {
        mean(&z)
    } else {
        0.0
    };

    // params
    let n_ar = order.p + order.sp;
    let n_ma = order.q + order.sq;
    let n_params = n_ar + n_ma;

    if n_params == 0 {
        let z_demeaned: Vec<f64> = z.iter().map(|&v| v - z_mean_init).collect();
        return fit_white_noise(
            y,
            &z_demeaned,
            order,
            z_mean_init,
            include_mean,
            include_drift,
        );
    }

    match method {
        EstimationMethod::Css => fit_css(y, &z, order, z_mean_init, include_mean, include_drift),
        EstimationMethod::Ml => fit_ml(y, &z, order, z_mean_init, include_mean, include_drift),
        EstimationMethod::CssMl => {
            let css_model = fit_css(y, &z, order, z_mean_init, include_mean, include_drift)?;

            let ar_unconstrained = inv_trans(&css_model.ar);
            let ma_inverted = ma_invert(&css_model.ma);
            let sar_unconstrained = inv_trans(&css_model.sar);
            let sma_inverted = ma_invert(&css_model.sma);

            let mut init: Vec<f64> = ar_unconstrained
                .iter()
                .chain(ma_inverted.iter())
                .chain(sar_unconstrained.iter())
                .chain(sma_inverted.iter())
                .copied()
                .collect();

            if include_mean || include_drift {
                init.push(css_model.intercept);
            }

            let options = FitMlOptions {
                include_mean,
                include_drift,
                transform_pars: true,
            };

            let ml_model = fit_ml_with_init(y, &z, order, z_mean_init, &init, options)?;

            let z_centered_css: Vec<f64> = z.iter().map(|&v| v - css_model.intercept).collect();
            let expanded_ar = expand_poly(&css_model.ar, &css_model.sar, order.period, true);
            let expanded_ma = expand_poly(&css_model.ma, &css_model.sma, order.period, false);
            let ss = ArimaStateSpace::new(&expanded_ar, &expanded_ma);

            let css_exact_log_lik = match kalman_filter(&z_centered_css, &ss, None) {
                Ok(out) => out.log_lik,
                Err(_) => -1e18,
            };

            if ml_model.log_lik >= css_exact_log_lik {
                Ok(ml_model)
            } else {
                Ok(css_model)
            }
        }
    }
}

fn fit_white_noise(
    y: &[f64],
    z: &[f64],
    order: &ArimaOrder,
    z_mean: f64,
    include_mean: bool,
    include_drift: bool,
) -> Result<ArimaModel> {
    let n = z.len();
    let sigma2 = variance(z);
    let nf = n as f64;
    let log_lik = -nf / 2.0 * ((2.0 * std::f64::consts::PI).ln() + 1.0 + sigma2.max(1e-10).ln());

    let fitted = compute_fitted_values(y, &[], &[], &[], &[], order, z_mean);
    let residuals: Vec<f64> = y.iter().zip(fitted.iter()).map(|(&a, &f)| a - f).collect();

    Ok(ArimaModel {
        order: *order,
        ar: vec![],
        ma: vec![],
        sar: vec![],
        sma: vec![],
        intercept: z_mean,
        include_mean,
        include_drift,
        sigma2,
        log_lik,
        fitted,
        residuals,
        y: y.to_vec(),
        lambda: None,
        method: EstimationMethod::Css,
    })
}

fn fit_css(
    y: &[f64],
    z: &[f64],
    order: &ArimaOrder,
    z_mean_init: f64,
    include_mean: bool,
    include_drift: bool,
) -> Result<ArimaModel> {
    let n_arma = order.p + order.q + order.sp + order.sq;
    let has_intercept = include_mean || include_drift;

    let mut init = vec![0.0; n_arma];
    if order.p > 0 && z.len() >= order.p + 10 {
        let ar_init = yule_walker_init(z, order.p).unwrap();
        init[..order.p].copy_from_slice(&ar_init);
    }
    let ma_start = order.p;
    let ma_end = order.p + order.q;
    for v in &mut init[ma_start..ma_end] {
        *v = 0.1;
    }
    let sma_start = order.p + order.q + order.sp;
    let sma_end = n_arma;
    for v in &mut init[sma_start..sma_end] {
        *v = 0.1;
    }

    if has_intercept {
        init.push(z_mean_init);
    }

    let problem = CssCostFunction {
        z: z.to_vec(),
        order: *order,
        include_mean: has_intercept,
    };

    let run_bfgs = |init_params: &[f64],
                    problem: &CssCostFunction|
     -> std::result::Result<Vec<f64>, argmin::core::Error> {
        let n = init_params.len();
        let linesearch = MoreThuenteLineSearch::new();
        let solver = BFGS::new(linesearch);
        let init_hessian: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                let mut row = vec![0.0; n];
                row[i] = 1.0;
                row
            })
            .collect();
        let result = Executor::new(problem.clone(), solver)
            .configure(|state: BfgsState| {
                state
                    .param(init_params.to_vec())
                    .inv_hessian(init_hessian)
                    .max_iters(100)
            })
            .run()?;
        Ok(result
            .state
            .get_best_param()
            .cloned()
            .unwrap_or_else(|| init_params.to_vec()))
    };

    let best = match run_bfgs(&init, &problem) {
        Ok(params) => params,
        Err(_) => {
            let mut alt_init = init.clone();
            for v in &mut alt_init[ma_start..ma_end] {
                *v = -*v;
            }
            for v in &mut alt_init[sma_start..sma_end] {
                *v = -*v;
            }
            match run_bfgs(&alt_init, &problem) {
                Ok(params) => params,
                Err(_) => init,
            }
        }
    };

    let intercept = if has_intercept {
        *best.last().unwrap_or(&z_mean_init)
    } else {
        z_mean_init
    };
    let arma_params = if has_intercept {
        &best[..n_arma]
    } else {
        &best
    };
    let (ar, ma, sar, sma) = unpack_params(arma_params, order);

    let z_centered: Vec<f64> = z.iter().map(|&v| v - intercept).collect();
    let residuals_z = arma_residuals(&z_centered, &ar, &ma, &sar, &sma, order.period);
    let n_eff = residuals_z.len();
    let sigma2 = residuals_z.iter().map(|&e| e * e).sum::<f64>() / n_eff as f64;
    let nf = n_eff as f64;
    let log_lik = -nf / 2.0 * ((2.0 * std::f64::consts::PI).ln() + 1.0 + sigma2.max(1e-10).ln());

    let fitted = compute_fitted_values(y, &ar, &ma, &sar, &sma, order, intercept);
    let residuals: Vec<f64> = y.iter().zip(fitted.iter()).map(|(&a, &f)| a - f).collect();

    Ok(ArimaModel {
        order: *order,
        ar,
        ma,
        sar,
        sma,
        intercept,
        include_mean,
        include_drift,
        sigma2,
        log_lik,
        fitted,
        residuals,
        y: y.to_vec(),
        lambda: None,
        method: EstimationMethod::Css,
    })
}

fn fit_ml(
    y: &[f64],
    z: &[f64],
    order: &ArimaOrder,
    z_mean_init: f64,
    include_mean: bool,
    include_drift: bool,
) -> Result<ArimaModel> {
    let n_arma = order.p + order.q + order.sp + order.sq;
    let mut init = vec![0.0; n_arma];
    if include_mean || include_drift {
        init.push(z_mean_init);
    }

    let options = FitMlOptions {
        include_mean,
        include_drift,
        transform_pars: true,
    };

    fit_ml_with_init(y, z, order, z_mean_init, &init, options)
}

fn fit_ml_with_init(
    y: &[f64],
    z: &[f64],
    order: &ArimaOrder,
    z_mean_init: f64,
    init: &[f64],
    options: FitMlOptions,
) -> Result<ArimaModel> {
    let has_intercept = options.include_mean || options.include_drift;

    let problem = MlCostFunction {
        z: z.to_vec(),
        order: *order,
        include_mean: has_intercept,
        transform_pars: options.transform_pars,
    };

    let linesearch = MoreThuenteLineSearch::new();
    let solver = BFGS::new(linesearch)
        .with_tolerance_cost(1e-8)
        .expect("valid tolerance");

    let n_params = init.len();
    let init_hessian: Vec<Vec<f64>> = (0..n_params)
        .map(|i| {
            let mut row = vec![0.0; n_params];
            row[i] = 1.0;
            row
        })
        .collect();

    let result = Executor::new(problem.clone(), solver)
        .configure(|state: BfgsState| {
            state
                .param(init.to_vec())
                .inv_hessian(init_hessian)
                .max_iters(100)
        })
        .run();

    let best = match &result {
        Ok(res) => res
            .state
            .get_best_param()
            .cloned()
            .unwrap_or_else(|| init.to_vec()),
        Err(_) => init.to_vec(),
    };

    let (ar, ma, sar, sma, intercept) = problem.to_constrained(&best);

    let ma = if options.transform_pars {
        ma_invert(&ma)
    } else {
        ma
    };
    let sma = if options.transform_pars {
        ma_invert(&sma)
    } else {
        sma
    };

    let intercept = if has_intercept {
        intercept
    } else {
        z_mean_init
    };

    let z_centered: Vec<f64> = z.iter().map(|&v| v - intercept).collect();
    let expanded_ar = expand_poly(&ar, &sar, order.period, true);
    let expanded_ma = expand_poly(&ma, &sma, order.period, false);
    let ss = ArimaStateSpace::new(&expanded_ar, &expanded_ma);
    let kf_output = kalman_filter(&z_centered, &ss, None)?;

    let fitted = compute_fitted_values(y, &ar, &ma, &sar, &sma, order, intercept);
    let residuals: Vec<f64> = y.iter().zip(fitted.iter()).map(|(&a, &f)| a - f).collect();

    Ok(ArimaModel {
        order: *order,
        ar,
        ma,
        sar,
        sma,
        intercept,
        include_mean: options.include_mean,
        include_drift: options.include_drift,
        sigma2: kf_output.sigma2,
        log_lik: kf_output.log_lik,
        fitted,
        residuals,
        y: y.to_vec(),
        lambda: None,
        method: EstimationMethod::Ml,
    })
}

struct FitMlOptions {
    include_mean: bool,
    include_drift: bool,
    transform_pars: bool,
}

#[derive(Debug, Clone)]
struct CssCostFunction {
    z: Vec<f64>,
    order: ArimaOrder,
    include_mean: bool,
}

impl CostFunction for CssCostFunction {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, p: &Self::Param) -> std::result::Result<f64, argmin::core::Error> {
        let n_arma = self.order.p + self.order.q + self.order.sp + self.order.sq;
        let (intercept, arma_p) = if self.include_mean {
            (*p.last().unwrap_or(&0.0), &p[..n_arma])
        } else {
            (0.0, p.as_slice())
        };
        let (ar, ma, sar, sma) = unpack_params(arma_p, &self.order);
        let z_centered: Vec<f64> = self.z.iter().map(|&v| v - intercept).collect();
        let res = arma_residuals(&z_centered, &ar, &ma, &sar, &sma, self.order.period);
        if res.is_empty() {
            return Ok(1e18);
        }
        let css: f64 = res.iter().map(|&e| e * e).sum();
        if css.is_finite() {
            Ok(css)
        } else {
            Ok(1e18)
        }
    }
}

impl Gradient for CssCostFunction {
    type Param = Vec<f64>;
    type Gradient = Vec<f64>;

    fn gradient(
        &self,
        p: &Self::Param,
    ) -> std::result::Result<Self::Gradient, argmin::core::Error> {
        let eps = 1e-6;
        let f0 = self.cost(p)?;
        let mut grad = vec![0.0; p.len()];
        for i in 0..p.len() {
            let mut p_plus = p.clone();
            p_plus[i] += eps;
            let f_plus = self.cost(&p_plus)?;
            grad[i] = (f_plus - f0) / eps;
        }
        Ok(grad)
    }
}

#[derive(Debug, Clone)]
struct MlCostFunction {
    z: Vec<f64>,
    order: ArimaOrder,
    include_mean: bool,
    transform_pars: bool,
}

impl MlCostFunction {
    fn to_constrained(&self, p: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, f64) {
        let n_arma = self.order.p + self.order.q + self.order.sp + self.order.sq;
        let (intercept, arma_p) = if self.include_mean {
            (*p.last().unwrap_or(&0.0), &p[..n_arma])
        } else {
            (0.0, p)
        };
        let (ar_raw, ma, sar_raw, sma) = unpack_params(arma_p, &self.order);
        let (ar, sar) = if self.transform_pars {
            (trans_pars(&ar_raw), trans_pars(&sar_raw))
        } else {
            (ar_raw, sar_raw)
        };
        (ar, ma, sar, sma, intercept)
    }
}

impl CostFunction for MlCostFunction {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, p: &Self::Param) -> std::result::Result<f64, argmin::core::Error> {
        let (ar, ma, sar, sma, intercept) = self.to_constrained(p);
        let z_centered: Vec<f64> = self.z.iter().map(|&v| v - intercept).collect();
        let expanded_ar = expand_poly(&ar, &sar, self.order.period, true);
        let expanded_ma = expand_poly(&ma, &sma, self.order.period, false);
        let ss = ArimaStateSpace::new(&expanded_ar, &expanded_ma);

        match kalman_filter(&z_centered, &ss, None) {
            Ok(output) if output.log_lik.is_finite() => {
                let n = output.innovations.len() as f64;
                let ssq: f64 = output
                    .innovations
                    .iter()
                    .zip(output.innovation_var.iter())
                    .map(|(&v, &f)| v * v / f)
                    .sum();
                let sumlog: f64 = output.innovation_var.iter().map(|&f| f.ln()).sum();
                let cost = 0.5 * ((ssq / n).max(1e-10).ln() + sumlog / n);
                if cost.is_finite() {
                    Ok(cost)
                } else {
                    Ok(1e18)
                }
            }
            _ => Ok(1e18),
        }
    }
}

impl Gradient for MlCostFunction {
    type Param = Vec<f64>;
    type Gradient = Vec<f64>;

    fn gradient(
        &self,
        p: &Self::Param,
    ) -> std::result::Result<Self::Gradient, argmin::core::Error> {
        let eps = 1e-6;
        let f0 = self.cost(p)?;
        let mut grad = vec![0.0; p.len()];
        for i in 0..p.len() {
            let mut p_plus = p.clone();
            p_plus[i] += eps;
            let f_plus = self.cost(&p_plus)?;
            grad[i] = (f_plus - f0) / eps;
        }
        Ok(grad)
    }
}

fn arma_residuals(
    z: &[f64],
    ar: &[f64],
    ma: &[f64],
    sar: &[f64],
    sma: &[f64],
    period: usize,
) -> Vec<f64> {
    let expanded_ar = expand_poly(ar, sar, period, true);
    let expanded_ma = expand_poly(ma, sma, period, false);
    let p = expanded_ar.len();
    let q = expanded_ma.len();
    let n = z.len();
    let start = p.max(q);
    if start >= n {
        return vec![];
    }
    let mut residuals = vec![0.0; n];
    for t in start..n {
        let ar_part: f64 = expanded_ar
            .iter()
            .enumerate()
            .map(|(i, &phi)| phi * z[t - 1 - i])
            .sum();
        let ma_part: f64 = expanded_ma
            .iter()
            .enumerate()
            .map(|(j, &theta)| theta * residuals[t - 1 - j])
            .sum();
        residuals[t] = z[t] - ar_part - ma_part;
    }
    residuals[start..].to_vec()
}

fn compute_fitted_values(
    y: &[f64],
    ar: &[f64],
    ma: &[f64],
    sar: &[f64],
    sma: &[f64],
    order: &ArimaOrder,
    intercept: f64,
) -> Vec<f64> {
    let n = y.len();
    let mut z = y.to_vec();
    if order.sd > 0 && order.period > 1 {
        z = seasonal_diff(&z, order.period, order.sd).unwrap();
    }
    if order.d > 0 {
        z = diff(&z, order.d).unwrap();
    }
    let z_centered: Vec<f64> = z.iter().map(|&v| v - intercept).collect();
    let res = arma_residuals(&z_centered, ar, ma, sar, sma, order.period);
    let diff_len = z.len();
    let offset = n - diff_len;
    let res_offset = diff_len - res.len();
    let mut fitted = y.to_vec();
    for (i, &e) in res.iter().enumerate() {
        let orig_idx = offset + res_offset + i;
        if orig_idx < n {
            fitted[orig_idx] = y[orig_idx] - e;
        }
    }
    fitted
}

fn polyroot(coeffs: &[f64]) -> Vec<Complex<f64>> {
    let n = coeffs.len() - 1;
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return vec![Complex::new(-coeffs[0] / coeffs[1], 0.0)];
    }
    let lead = coeffs[n];
    let norm: Vec<f64> = coeffs.iter().map(|&c| c / lead).collect();
    let mut roots: Vec<Complex<f64>> = (0..n)
        .map(|k| {
            let angle = 2.0 * std::f64::consts::PI * k as f64 / n as f64 + 0.4;
            Complex::new(angle.cos() * 0.9, angle.sin() * 0.9)
        })
        .collect();
    let eval = |z: Complex<f64>| -> Complex<f64> {
        let mut result = Complex::new(norm[n], 0.0);
        for i in (0..n).rev() {
            result = result.mul(z).add(Complex::new(norm[i], 0.0));
        }
        result
    };
    for _ in 0..200 {
        let mut max_change = 0.0f64;
        for i in 0..n {
            let zi = roots[i];
            let pz = eval(zi);
            let mut prod = Complex::new(1.0, 0.0);
            for (j, &root) in roots.iter().enumerate().take(n) {
                if j != i {
                    prod = prod.mul(zi.sub(root));
                }
            }
            let delta = pz.div(prod);
            roots[i] = zi.sub(delta);
            max_change = max_change.max(delta.modulus());
        }
        if max_change < 1e-12 {
            break;
        }
    }
    roots
}

fn ma_invert(ma: &[f64]) -> Vec<f64> {
    let q = ma.len();
    if q == 0 {
        return vec![];
    }
    let q0 = match ma.iter().rposition(|&v| v.abs() > 1e-15) {
        Some(i) => i + 1,
        None => return ma.to_vec(),
    };
    let mut poly = vec![1.0];
    poly.extend_from_slice(&ma[..q0]);
    let roots = polyroot(&poly);
    if !roots.iter().any(|r| r.modulus() < 1.0) {
        return ma.to_vec();
    }
    if q0 == 1 {
        let mut result = vec![0.0; q];
        result[0] = 1.0 / ma[0];
        return result;
    }
    let mut new_roots = roots;
    for r in &mut new_roots {
        if r.modulus() < 1.0 {
            *r = r.inv();
        }
    }
    let mut x: Vec<Complex<f64>> = vec![Complex::new(1.0, 0.0)];
    for r in &new_roots {
        let mut new_x = vec![Complex::new(0.0, 0.0); x.len() + 1];
        for (j, &v) in x.iter().enumerate() {
            new_x[j] = new_x[j].add(v);
        }
        for (j, &v) in x.iter().enumerate() {
            let divided = v.div(*r);
            new_x[j + 1] = new_x[j + 1].sub(divided);
        }
        x = new_x;
    }
    let mut result = vec![0.0; q];
    for i in 0..q0 {
        result[i] = x[i + 1].re;
    }
    result
}

fn trans_pars(raw: &[f64]) -> Vec<f64> {
    let p = raw.len();
    if p == 0 {
        return vec![];
    }
    let mut work: Vec<f64> = raw.iter().map(|&r| r.tanh()).collect();
    let mut result: Vec<f64> = work.clone();
    for j in 1..p {
        let a = result[j];
        for k in 0..j {
            work[k] -= a * result[j - k - 1];
        }
        result[..j].copy_from_slice(&work[..j]);
    }
    result
}

fn inv_trans(phi: &[f64]) -> Vec<f64> {
    let p = phi.len();
    if p == 0 {
        return vec![];
    }
    let mut work: Vec<f64> = phi.to_vec();
    let mut result: Vec<f64> = phi.to_vec();
    for j in (1..p).rev() {
        let a = result[j];
        let denom = 1.0 - a * a;
        if denom.abs() < 1e-15 {
            continue;
        }
        for k in 0..j {
            work[k] = (result[k] + a * result[j - k - 1]) / denom;
        }
        result[..j].copy_from_slice(&work[..j]);
    }
    result
        .iter()
        .map(|&v| v.clamp(-0.999999, 0.999999).atanh())
        .collect()
}

fn unpack_params(params: &[f64], order: &ArimaOrder) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut idx = 0;
    let ar = params[idx..idx + order.p].to_vec();
    idx += order.p;
    let ma = params[idx..idx + order.q].to_vec();
    idx += order.q;
    let sar = params[idx..idx + order.sp].to_vec();
    idx += order.sp;
    let sma = params[idx..idx + order.sq].to_vec();
    (ar, ma, sar, sma)
}

fn yule_walker_init(z: &[f64], p: usize) -> Result<Vec<f64>> {
    if p == 0 || z.len() < p + 1 {
        return Ok(vec![]);
    }
    let r = acf(z, p).unwrap_or_else(|_| vec![0.0; p + 1]);

    let r_matrix = DMatrix::from_fn(p, p, |i, j| {
        let index = (i as isize - j as isize).unsigned_abs();
        r[index]
    });

    let rhs = DVector::from_row_slice(&r[1..=p]);

    r_matrix
        .lu()
        .solve(&rhs)
        .map(|solution| solution.as_slice().to_vec())
        .ok_or(Error::MathError(String::from("Matrix is singular")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use augurs_testing::{
        assert_all_close, assert_approx_eq, assert_within_pct, data::AIR_PASSENGERS,
    };

    #[test]
    fn test_polyroot_1() {
        let coeffs = vec![-2.0, 1.0]; // x - 2 = 0 -> x = 2
        let roots = polyroot(&coeffs);
        assert_eq!(roots.len(), 1);
        assert_approx_eq!(roots[0].re, 2.0, 0.000001);
    }

    #[test]
    fn test_polyroot_2() {
        let coeffs = vec![6.0, -5.0, 1.0]; // $x^2 -5x + 6 = 0$ -> x \in {2,3}$
        let roots = polyroot(&coeffs);
        assert_eq!(roots.len(), 2);
        assert_approx_eq!(roots[0].re, 2.0, 0.000001);
        assert_approx_eq!(roots[1].re, 3.0, 0.000001)
    }

    #[test]
    fn test_ma_invert_simple() {
        let ma = vec![2.0]; // 1 + 2z, root at -0.5 (inside unit circle)
        let inverted = ma_invert(&ma);
        assert_approx_eq!(inverted[0], 0.5, 0.000001);
    }

    #[test]
    fn test_trans_back() {
        let p = vec![0.5, -0.3];
        let transformed = inv_trans(&p);
        println!("{:?}", transformed);
        let back = trans_pars(&transformed);
        println!("{:?}", back);
        assert_approx_eq!(back[0], p[0]);
        assert_approx_eq!(back[1], p[1]);
    }

    #[test]
    // Test if AR(p) coefficients matches R's stats::ar(AirPassengers, order.max = 2).
    // Results are 1.1656 & -0.2294.
    fn test_yule_walker() {
        let yule = yule_walker_init(AIR_PASSENGERS, 2).unwrap();
        assert_all_close(&yule, &[1.1656, -0.2294]);
    }

    #[test]
    // Test of Air Passengers fit matches R's forecast package auto.arima().
    // Result of auto.arima(AirPassengers) is ARIMA(2,1,1)(0,1,0)[12] with log-likelihood -504.92.
    fn air_passengers_fit() {
        let params = ArimaOrder::seasonal(2, 1, 1, 0, 1, 0, 12);
        let model = fit_arima(
            AIR_PASSENGERS,
            &params,
            EstimationMethod::CssMl,
            false,
            false,
        )
        .unwrap();

        assert_within_pct!(model.log_lik, -504.92, 0.01);
    }
}
