//! Power transformations, including Box-Cox and Yeo-Johnson.

use argmin::core::{CostFunction, Error, Executor};
use argmin::solver::brent::BrentOpt;

/// Returns the Box-Cox transformation of the given value.
/// Assumes x > 0.
pub(crate) fn box_cox(x: f64, lambda: f64) -> Result<f64, &'static str> {
    if x <= 0.0 {
        return Err("x must be greater than 0");
    }
    if lambda == 0.0 {
        Ok(x.ln())
    } else {
        Ok((x.powf(lambda) - 1.0) / lambda)
    }
}

/// Returns the Yeo-Johnson transformation of the given value.
pub(crate) fn yeo_johnson(x: f64, lambda: f64) -> Result<f64, &'static str> {
    if x.is_nan() || lambda.is_nan() {
        return Err("Input values must be valid numbers.");
    }

    if x >= 0.0 {
        if lambda == 0.0 {
            Ok((x + 1.0).ln())
        } else {
            Ok(((x + 1.0).powf(lambda) - 1.0) / lambda)
        }
    } else if lambda == 2.0 {
        Ok(-(-x + 1.0).ln())
    } else {
        Ok(-((-x + 1.0).powf(2.0 - lambda) - 1.0) / (2.0 - lambda))
    }
}

fn box_cox_log_likelihood(data: &[f64], lambda: f64) -> Result<f64, Error> {
    let n = data.len() as f64;
    if n == 0.0 {
        return Err(Error::msg("Data must not be empty"));
    }
    if data.iter().any(|&x| x <= 0.0) {
        return Err(Error::msg("All data must be greater than 0"));
    }
    let transformed_data: Result<Vec<f64>, _> = data.iter().map(|&x| box_cox(x, lambda)).collect();

    let transformed_data = match transformed_data {
        Ok(values) => values,
        Err(e) => return Err(Error::msg(e)),
    };
    let mean_transformed: f64 = transformed_data.iter().copied().sum::<f64>() / n;
    let variance: f64 = transformed_data
        .iter()
        .map(|&x| (x - mean_transformed).powi(2))
        .sum::<f64>()
        / n;

    // Avoid log(0) by ensuring variance is positive
    if variance <= 0.0 {
        return Err(Error::msg("Variance must be positive"));
    }
    let log_likelihood =
        -0.5 * n * variance.ln() + (lambda - 1.0) * data.iter().map(|&x| x.ln()).sum::<f64>();
    Ok(log_likelihood)
}

fn yeo_johnson_log_likelihood(data: &[f64], lambda: f64) -> Result<f64, Error> {
    let n = data.len() as f64;

    if n == 0.0 {
        return Err(Error::msg("Data array is empty"));
    }

    let transformed_data: Result<Vec<f64>, _> =
        data.iter().map(|&x| yeo_johnson(x, lambda)).collect();

    let transformed_data = match transformed_data {
        Ok(values) => values,
        Err(e) => return Err(Error::msg(e)),
    };

    let mean = transformed_data.iter().sum::<f64>() / n;

    let variance = transformed_data
        .iter()
        .map(|&x| (x - mean).powi(2))
        .sum::<f64>()
        / n;

    if variance <= 0.0 {
        return Err(Error::msg("Variance is non-positive"));
    }

    let log_sigma_squared = variance.ln();
    let log_likelihood = -n / 2.0 * log_sigma_squared;

    let additional_term: f64 = data
        .iter()
        .map(|&x| (x.signum() * (x.abs() + 1.0).ln()))
        .sum::<f64>()
        * (lambda - 1.0);

    Ok(log_likelihood + additional_term)
}

#[derive(Clone)]
struct BoxCoxProblem<'a> {
    data: &'a [f64],
}

impl CostFunction for BoxCoxProblem<'_> {
    type Param = f64;
    type Output = f64;

    // The goal is to minimize the negative log-likelihood
    fn cost(&self, lambda: &Self::Param) -> Result<Self::Output, Error> {
        box_cox_log_likelihood(self.data, *lambda).map(|ll| -ll)
    }
}

#[derive(Clone)]
struct YeoJohnsonProblem<'a> {
    data: &'a [f64],
}

impl CostFunction for YeoJohnsonProblem<'_> {
    type Param = f64;
    type Output = f64;

    // The goal is to minimize the negative log-likelihood
    fn cost(&self, lambda: &Self::Param) -> Result<Self::Output, Error> {
        yeo_johnson_log_likelihood(self.data, *lambda).map(|ll| -ll)
    }
}

struct OptimizationParams {
    initial_param: f64,
    lower_bound: f64,
    upper_bound: f64,
    max_iterations: u64,
}

impl Default for OptimizationParams {
    fn default() -> Self {
        Self {
            initial_param: 0.0,
            lower_bound: -2.0,
            upper_bound: 2.0,
            max_iterations: 1000,
        }
    }
}

fn optimize_lambda<T: CostFunction<Param = f64, Output = f64>>(
    cost: T,
    params: OptimizationParams,
) -> Result<f64, Error> {
    let solver = BrentOpt::new(params.lower_bound, params.upper_bound);
    let result = Executor::new(cost, solver)
        .configure(|state| {
            state
                .param(params.initial_param)
                .max_iters(params.max_iterations)
        })
        .run();

    result.and_then(|res| {
        res.state()
            .best_param
            .ok_or_else(|| Error::msg("No best parameter found"))
    })
}

/// Optimize the lambda parameter for the Box-Cox or Yeo-Johnson transformation
pub(crate) fn optimize_box_cox_lambda(data: &[f64]) -> Result<f64, Error> {
    // Use Box-Cox transformation
    let cost = BoxCoxProblem { data };
    let optimization_params = OptimizationParams::default();
    optimize_lambda(cost, optimization_params)
}

pub(crate) fn optimize_yeo_johnson_lambda(data: &[f64]) -> Result<f64, Error> {
    // Use Yeo-Johnson transformation
    let cost = YeoJohnsonProblem { data };
    let optimization_params = OptimizationParams::default();
    optimize_lambda(cost, optimization_params)
}

/// An iterator adapter that applies the Box-Cox transformation to each item.
#[derive(Clone, Debug)]
pub(crate) struct BoxCox<T> {
    inner: T,
    lambda: f64,
}

impl<T> Iterator for BoxCox<T>
where
    T: Iterator<Item = f64>,
{
    type Item = f64;
    fn next(&mut self) -> Option<Self::Item> {
        self.inner
            .next()
            .map(|x| box_cox(x, self.lambda).unwrap_or(f64::NAN))
    }
}

pub(crate) trait BoxCoxExt: Iterator<Item = f64> {
    fn box_cox(self, lambda: f64) -> BoxCox<Self>
    where
        Self: Sized,
    {
        BoxCox {
            inner: self,
            lambda,
        }
    }
}

impl<T> BoxCoxExt for T where T: Iterator<Item = f64> {}

/// Returns the inverse Box-Cox transformation of the given value.
fn inverse_box_cox(y: f64, lambda: f64) -> Result<f64, &'static str> {
    if lambda == 0.0 {
        Ok(y.exp())
    } else {
        let value = y * lambda + 1.0;
        if value <= 0.0 {
            Err("Invalid domain for inverse Box-Cox transformation")
        } else {
            Ok(value.powf(1.0 / lambda))
        }
    }
}

/// An iterator adapter that applies the inverse Box-Cox transformation to each item.
#[derive(Clone, Debug)]
pub(crate) struct InverseBoxCox<T> {
    inner: T,
    lambda: f64,
}

impl<T> Iterator for InverseBoxCox<T>
where
    T: Iterator<Item = f64>,
{
    type Item = f64;
    fn next(&mut self) -> Option<Self::Item> {
        self.inner
            .next()
            .map(|y| inverse_box_cox(y, self.lambda).unwrap_or(f64::NAN))
    }
}

pub(crate) trait InverseBoxCoxExt: Iterator<Item = f64> {
    fn inverse_box_cox(self, lambda: f64) -> InverseBoxCox<Self>
    where
        Self: Sized,
    {
        InverseBoxCox {
            inner: self,
            lambda,
        }
    }
}

impl<T> InverseBoxCoxExt for T where T: Iterator<Item = f64> {}

/// An iterator adapter that applies the Yeo-Johnson transformation to each item.
#[derive(Clone, Debug)]
pub(crate) struct YeoJohnson<T> {
    inner: T,
    lambda: f64,
}

impl<T> Iterator for YeoJohnson<T>
where
    T: Iterator<Item = f64>,
{
    type Item = f64;
    fn next(&mut self) -> Option<Self::Item> {
        self.inner
            .next()
            .map(|x| yeo_johnson(x, self.lambda).unwrap_or(f64::NAN))
    }
}

pub(crate) trait YeoJohnsonExt: Iterator<Item = f64> {
    fn yeo_johnson(self, lambda: f64) -> YeoJohnson<Self>
    where
        Self: Sized,
    {
        YeoJohnson {
            inner: self,
            lambda,
        }
    }
}

impl<T> YeoJohnsonExt for T where T: Iterator<Item = f64> {}

/// Returns the inverse Yeo-Johnson transformation of the given value.
fn inverse_yeo_johnson(y: f64, lambda: f64) -> f64 {
    const EPSILON: f64 = 1e-6;

    if y >= 0.0 && lambda.abs() < EPSILON {
        // For lambda close to 0 (positive values)
        (y.exp()) - 1.0
    } else if y >= 0.0 {
        // For positive values (lambda not close to 0)
        (y * lambda + 1.0).powf(1.0 / lambda) - 1.0
    } else if (lambda - 2.0).abs() < EPSILON {
        // For lambda close to 2 (negative values)
        -(-y.exp() - 1.0)
    } else {
        // For negative values (lambda not close to 2)
        -((-((2.0 - lambda) * y) + 1.0).powf(1.0 / (2.0 - lambda)) - 1.0)
    }
}

/// An iterator adapter that applies the inverse Yeo-Johnson transformation to each item.
#[derive(Clone, Debug)]
pub(crate) struct InverseYeoJohnson<T> {
    inner: T,
    lambda: f64,
}

impl<T> Iterator for InverseYeoJohnson<T>
where
    T: Iterator<Item = f64>,
{
    type Item = f64;
    fn next(&mut self) -> Option<Self::Item> {
        self.inner
            .next()
            .map(|y| inverse_yeo_johnson(y, self.lambda))
    }
}

pub(crate) trait InverseYeoJohnsonExt: Iterator<Item = f64> {
    fn inverse_yeo_johnson(self, lambda: f64) -> InverseYeoJohnson<Self>
    where
        Self: Sized,
    {
        InverseYeoJohnson {
            inner: self,
            lambda,
        }
    }
}

impl<T> InverseYeoJohnsonExt for T where T: Iterator<Item = f64> {}

/// A trait for types that can be used as the `lambda` parameter for the
/// `Transform::box_cox` method.
pub trait IntoBoxCoxLambda {
    fn into_box_cox_lambda(self) -> Result<f64, Error>;
}

impl IntoBoxCoxLambda for f64 {
    /// Use the given lambda parameter.
    fn into_box_cox_lambda(self) -> Result<f64, Error> {
        Ok(self)
    }
}

impl IntoBoxCoxLambda for &[f64] {
    /// Find the optimal Box-Cox lambda parameter using maximum likelihood estimation.
    fn into_box_cox_lambda(self) -> Result<f64, Error> {
        optimize_box_cox_lambda(self)
    }
}

/// A trait for types that can be used as the `lambda` parameter for the
/// `Transform::box_cox` method.
pub trait IntoYeoJohnsonLambda {
    fn into_yeo_johnson_lambda(self) -> Result<f64, Error>;
}

impl IntoYeoJohnsonLambda for f64 {
    /// Use the given lambda parameter.
    fn into_yeo_johnson_lambda(self) -> Result<f64, Error> {
        Ok(self)
    }
}

impl IntoYeoJohnsonLambda for &[f64] {
    /// Find the optimal Yeo-Johnson lambda parameter using maximum likelihood estimation.
    fn into_yeo_johnson_lambda(self) -> Result<f64, Error> {
        optimize_yeo_johnson_lambda(self)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use augurs_testing::{assert_all_close, assert_approx_eq};

    #[test]
    fn correct_optimal_box_cox_lambda() {
        let data = &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7];
        let got = optimize_box_cox_lambda(data);
        assert!(got.is_ok());
        let lambda = got.unwrap();
        assert_approx_eq!(lambda, 0.7123778635679304);
    }

    #[test]
    fn optimal_box_cox_lambda_lambda_empty_data() {
        let data = &[];
        let got = optimize_box_cox_lambda(data);
        assert!(got.is_err());
    }

    #[test]
    fn optimal_box_cox_lambda_non_positive_data() {
        let data = &[0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7];
        let got = optimize_box_cox_lambda(data);
        assert!(got.is_err());
    }

    #[test]
    fn correct_optimal_yeo_johnson_lambda() {
        let data = &[0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7];
        let got = optimize_yeo_johnson_lambda(data);
        assert!(got.is_ok());
        let lambda = got.unwrap();
        assert_approx_eq!(lambda, 1.7458442076987954);
    }

    #[test]
    fn test_box_cox_llf() {
        let data = &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7];
        let lambda = 1.0;
        let got = box_cox_log_likelihood(data, lambda);
        assert!(got.is_ok());
        let llf = got.unwrap();
        assert_approx_eq!(llf, 11.266065387038703);
    }

    #[test]
    fn test_box_cox_llf_non_positive() {
        let data = &[0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7];
        let lambda = 0.0;
        let got = box_cox_log_likelihood(data, lambda);
        assert!(got.is_err());
    }

    #[test]
    fn test_yeo_johnson_llf() {
        let data = &[0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7];
        let lambda = 1.0;
        let got = yeo_johnson_log_likelihood(data, lambda);
        assert!(got.is_ok());
        let llf = got.unwrap();
        assert_approx_eq!(llf, 10.499377905819307);
    }

    #[test]
    fn box_cox_test() {
        let data = vec![1.0, 2.0, 3.0];
        let lambda = 0.5;
        let expected = vec![0.0, 0.8284271247461903, 1.4641016151377544];
        let actual: Vec<_> = data.into_iter().box_cox(lambda).collect();
        assert_all_close(&expected, &actual);
    }

    #[test]
    fn inverse_box_cox_test() {
        let data = vec![0.0, 0.5_f64.ln(), 1.0_f64.ln()];
        let lambda = 0.5;
        let expected = vec![1.0, 0.426966072919605, 1.0];
        let actual: Vec<_> = data.into_iter().inverse_box_cox(lambda).collect();
        assert_all_close(&expected, &actual);
    }

    #[test]
    fn yeo_johnson_test() {
        let data = vec![-1.0, 0.0, 1.0];
        let lambda = 0.5;
        let expected = vec![-1.2189514164974602, 0.0, 0.8284271247461903];
        let actual: Vec<_> = data.into_iter().yeo_johnson(lambda).collect();
        assert_all_close(&expected, &actual);
    }

    #[test]
    fn inverse_yeo_johnson_test() {
        let data = vec![-1.2189514164974602, 0.0, 0.8284271247461903];
        let lambda = 0.5;
        let expected = vec![-1.0, 0.0, 1.0];
        let actual: Vec<_> = data.into_iter().inverse_yeo_johnson(lambda).collect();
        assert_all_close(&expected, &actual);
    }
}
