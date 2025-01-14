//! Power transformations, including Box-Cox and Yeo-Johnson.

use argmin::core::{CostFunction, Executor};
use argmin::solver::brent::BrentOpt;

use super::{Error, Transformer};

/// Returns the Box-Cox transformation of the given value.
/// Assumes x > 0.
fn box_cox(x: f64, lambda: f64) -> Result<f64, Error> {
    if x <= 0.0 {
        return Err(Error::NonPositiveData);
    }
    if x.is_nan() {
        return Ok(x);
    }
    if lambda == 0.0 {
        Ok(x.ln())
    } else {
        Ok((x.powf(lambda) - 1.0) / lambda)
    }
}

/// Returns the inverse Box-Cox transformation of the given value.
fn inverse_box_cox(y: f64, lambda: f64) -> Result<f64, Error> {
    if y.is_nan() {
        Ok(y)
    } else if lambda == 0.0 {
        Ok(y.exp())
    } else {
        let value = y * lambda + 1.0;
        if value <= 0.0 {
            Err(Error::InvalidDomain)
        } else {
            Ok(value.powf(1.0 / lambda))
        }
    }
}

fn box_cox_log_likelihood(data: &[f64], lambda: f64) -> Result<f64, Error> {
    let n = data.len() as f64;
    if n == 0.0 {
        return Err(Error::EmptyData);
    }
    if data.iter().any(|&x| x <= 0.0) {
        return Err(Error::NonPositiveData);
    }
    let transformed_data = data
        .iter()
        .map(|&x| box_cox(x, lambda))
        .collect::<Result<Vec<_>, _>>()?;

    let mean_transformed: f64 = transformed_data.iter().sum::<f64>() / n;
    let variance: f64 = transformed_data
        .iter()
        .map(|&x| (x - mean_transformed).powi(2))
        .sum::<f64>()
        / n;

    // Avoid log(0) by ensuring variance is positive
    if variance <= 0.0 {
        return Err(Error::VarianceNotPositive);
    }
    let log_likelihood =
        -0.5 * n * variance.ln() + (lambda - 1.0) * data.iter().map(|&x| x.ln()).sum::<f64>();
    Ok(log_likelihood)
}

#[derive(Clone)]
struct BoxCoxProblem<'a> {
    data: &'a [f64],
}

impl CostFunction for BoxCoxProblem<'_> {
    type Param = f64;
    type Output = f64;

    // The goal is to minimize the negative log-likelihood
    fn cost(&self, lambda: &Self::Param) -> Result<Self::Output, argmin::core::Error> {
        Ok(box_cox_log_likelihood(self.data, *lambda).map(|ll| -ll)?)
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

    result
        .map_err(Error::Optimize)
        .and_then(|res| res.state().best_param.ok_or_else(|| Error::NoBestParameter))
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

/// A transformer that applies the Box-Cox transformation to each item.
///
/// The Box-Cox transformation is defined as:
///
/// - if lambda == 0: x.ln()
/// - otherwise: (x^lambda - 1) / lambda
///
/// By default the optimal `lambda` parameter is found from the data in
/// `transform` using maximum likelihood estimation. If you want to use a
/// specific `lambda` value, you can use the `with_lambda` method.
///
/// Note that unlike the scikit-learn implementation, this transform does not
/// standardize the data after applying the transformation. This can be done
/// by using the [`StandardScaler`] transformer inside a [`Pipeline`].
///
/// [`StandardScaler`]: crate::transforms::StandardScaler
/// [`Pipeline`]: crate::transforms::Pipeline
#[derive(Clone, Debug)]
pub struct BoxCox {
    lambda: f64,
    ignore_nans: bool,
}

impl BoxCox {
    /// Create a new `BoxCox` transformer.
    pub fn new() -> Self {
        Self {
            lambda: f64::NAN,
            ignore_nans: false,
        }
    }

    /// Set the `lambda` parameter for the Box-Cox transformation.
    ///
    /// # Errors
    ///
    /// This function returns an error if the `lambda` parameter is NaN.
    pub fn with_lambda(mut self, lambda: f64) -> Result<Self, Error> {
        if !lambda.is_finite() {
            return Err(Error::InvalidLambda);
        }
        self.lambda = lambda;
        Ok(self)
    }

    /// Set whether to ignore NaN values when calculating the transform.
    ///
    /// If `true`, NaN values will be ignored when calculating the optimal
    /// lambda and simply passed through the transform.
    ///
    /// Defaults to `false`.
    pub fn ignore_nans(mut self, ignore_nans: bool) -> Self {
        self.ignore_nans = ignore_nans;
        self
    }
}

impl Default for BoxCox {
    fn default() -> Self {
        Self::new()
    }
}

impl Transformer for BoxCox {
    fn fit(&mut self, data: &[f64]) -> Result<(), Error> {
        // Avoid copying the data if we don't need to,
        // i.e. if we're not ignoring NaNs or if there are no NaNs.
        if !self.ignore_nans || !data.iter().any(|&x| x.is_nan()) {
            self.lambda = optimize_box_cox_lambda(data)?;
        } else {
            let data = data
                .iter()
                .copied()
                .filter(|&x| !x.is_nan())
                .collect::<Vec<_>>();
            if data.is_empty() {
                return Err(Error::EmptyData);
            }
            self.lambda = optimize_box_cox_lambda(&data)?;
        }
        Ok(())
    }

    fn transform(&self, data: &mut [f64]) -> Result<(), Error> {
        if self.lambda.is_nan() {
            return Err(Error::NotFitted);
        }
        for x in data.iter_mut() {
            *x = box_cox(*x, self.lambda)?;
        }
        Ok(())
    }

    fn inverse_transform(&self, data: &mut [f64]) -> Result<(), Error> {
        for x in data.iter_mut() {
            *x = inverse_box_cox(*x, self.lambda)?;
        }
        Ok(())
    }
}

/// Returns the Yeo-Johnson transformation of the given value.
fn yeo_johnson(x: f64, lambda: f64) -> Result<f64, Error> {
    if x.is_nan() {
        return Ok(x);
    }
    if !lambda.is_finite() {
        return Err(Error::InvalidLambda);
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

fn yeo_johnson_log_likelihood(data: &[f64], lambda: f64) -> Result<f64, Error> {
    let n = data.len() as f64;

    if n == 0.0 {
        return Err(Error::EmptyData);
    }

    let transformed_data = data
        .iter()
        .map(|&x| yeo_johnson(x, lambda))
        .collect::<Result<Vec<f64>, _>>()?;

    let mean = transformed_data.iter().sum::<f64>() / n;

    let variance = transformed_data
        .iter()
        .map(|&x| (x - mean).powi(2))
        .sum::<f64>()
        / n;

    if variance <= 0.0 {
        return Err(Error::VarianceNotPositive);
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
struct YeoJohnsonProblem<'a> {
    data: &'a [f64],
}

impl CostFunction for YeoJohnsonProblem<'_> {
    type Param = f64;
    type Output = f64;

    // The goal is to minimize the negative log-likelihood
    fn cost(&self, lambda: &Self::Param) -> Result<Self::Output, argmin::core::Error> {
        Ok(yeo_johnson_log_likelihood(self.data, *lambda).map(|ll| -ll)?)
    }
}

/// A transformer that applies the Yeo-Johnson transformation to each item.
///
/// The Yeo-Johnson transformation is a generalization of the Box-Cox transformation that
/// supports negative values. It is defined as:
///
/// - if lambda != 0 and x >= 0: ((x + 1)^lambda - 1) / lambda
/// - if lambda == 0 and x >= 0: (x + 1).ln()
/// - if lambda != 2 and x < 0:  ((-x + 1)^2 - 1) / 2
/// - if lambda == 2 and x < 0:  (-x + 1).ln()
///
/// By default the optimal `lambda` parameter is found from the data in
/// `transform` using maximum likelihood estimation. If you want to use a
/// specific `lambda` value, you can use the `with_lambda` method.
///
/// Note that unlike the scikit-learn implementation, this transform does not
/// standardize the data after applying the transformation. This can be done
/// by using the [`StandardScaler`] transformer inside a [`Pipeline`].
///
/// [`StandardScaler`]: crate::transforms::StandardScaler
/// [`Pipeline`]: crate::transforms::Pipeline
#[derive(Clone, Debug)]
pub struct YeoJohnson {
    lambda: f64,
    ignore_nans: bool,
}

impl YeoJohnson {
    /// Create a new `YeoJohnson` transformer.
    pub fn new() -> Self {
        Self {
            lambda: f64::NAN,
            ignore_nans: false,
        }
    }

    /// Set the `lambda` parameter for the Yeo-Johnson transformation.
    ///
    /// # Errors
    ///
    /// This function returns an error if the `lambda` parameter is NaN.
    pub fn with_lambda(mut self, lambda: f64) -> Result<Self, Error> {
        if !lambda.is_finite() {
            return Err(Error::InvalidLambda);
        }
        self.lambda = lambda;
        Ok(self)
    }

    /// Set whether to ignore NaN values when calculating the transform.
    ///
    /// If `true`, NaN values will be ignored when calculating the optimal
    /// lambda and simply passed through the transform.
    ///
    /// Defaults to `false`.
    pub fn ignore_nans(mut self, ignore_nans: bool) -> Self {
        self.ignore_nans = ignore_nans;
        self
    }
}

impl Default for YeoJohnson {
    fn default() -> Self {
        Self::new()
    }
}

impl Transformer for YeoJohnson {
    fn fit(&mut self, data: &[f64]) -> Result<(), Error> {
        // Avoid copying the data if we don't need to,
        // i.e. if we're not ignoring NaNs or if there are no NaNs.
        if !self.ignore_nans || !data.iter().any(|&x| x.is_nan()) {
            self.lambda = optimize_yeo_johnson_lambda(data)?;
        } else {
            let data = data
                .iter()
                .copied()
                .filter(|&x| !x.is_nan())
                .collect::<Vec<_>>();
            if data.is_empty() {
                return Err(Error::EmptyData);
            }
            self.lambda = optimize_yeo_johnson_lambda(&data)?;
        }
        Ok(())
    }

    fn transform(&self, data: &mut [f64]) -> Result<(), Error> {
        if self.lambda.is_nan() {
            return Err(Error::NotFitted);
        }
        for x in data.iter_mut() {
            *x = yeo_johnson(*x, self.lambda)?;
        }
        Ok(())
    }

    fn inverse_transform(&self, data: &mut [f64]) -> Result<(), Error> {
        for x in data.iter_mut() {
            *x = inverse_yeo_johnson(*x, self.lambda);
        }
        Ok(())
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
    fn box_cox_single() {
        assert_approx_eq!(box_cox(1.0, 0.5).unwrap(), 0.0);
        assert_approx_eq!(box_cox(2.0, 0.5).unwrap(), 0.8284271247461903);
        assert!(box_cox(f64::NAN, 0.5).unwrap().is_nan());
    }

    #[test]
    fn inverse_box_cox_single() {
        assert_approx_eq!(inverse_box_cox(0.0, 0.5).unwrap(), 1.0);
        assert_approx_eq!(inverse_box_cox(0.8284271247461903, 0.5).unwrap(), 2.0);
        assert!(inverse_box_cox(f64::NAN, 0.5).unwrap().is_nan());
    }

    #[test]
    fn box_cox_transform() {
        let mut data = vec![1.0, 2.0, 3.0];
        let lambda = 0.5;
        let box_cox = BoxCox::new().with_lambda(lambda).unwrap();
        let expected = vec![0.0, 0.8284271247461903, 1.4641016151377544];
        box_cox.transform(&mut data).unwrap();
        assert_all_close(&expected, &data);
    }

    #[test]
    fn box_cox_fit_transform_nans() {
        let mut data = vec![1.0, 2.0, f64::NAN, 3.0];
        let mut box_cox = BoxCox::new();
        assert!(box_cox.fit_transform(&mut data).is_err());
    }

    #[test]
    fn box_cox_transform_ignore_nans() {
        let mut data = vec![1.0, 2.0, f64::NAN, 3.0];
        let mut box_cox = BoxCox::new().ignore_nans(true);
        let expected = vec![0.0, 0.8284271247461903, f64::NAN, 1.4641016151377544];
        box_cox.fit_transform(&mut data).unwrap();
        assert_all_close(&expected, &data);
    }

    #[test]
    fn inverse_box_cox_transform() {
        let mut data = vec![0.0, 0.5_f64.ln(), 1.0_f64.ln()];
        let lambda = 0.5;
        let box_cox = BoxCox::new().with_lambda(lambda).unwrap();
        let expected = vec![1.0, 0.426966072919605, 1.0];
        box_cox.inverse_transform(&mut data).unwrap();
        assert_all_close(&expected, &data);
    }

    #[test]
    fn yeo_johnson_single() {
        assert_approx_eq!(yeo_johnson(1.0, 0.5).unwrap(), 0.8284271247461903);
        assert_approx_eq!(yeo_johnson(2.0, 0.5).unwrap(), 1.4641016151377544);
        assert!(yeo_johnson(f64::NAN, 0.5).unwrap().is_nan());
    }

    #[test]
    fn inverse_yeo_johnson_single() {
        assert_approx_eq!(inverse_yeo_johnson(0.8284271247461903, 0.5), 1.0);
        assert_approx_eq!(inverse_yeo_johnson(1.4641016151377544, 0.5), 2.0);
        assert!(inverse_yeo_johnson(f64::NAN, 0.5).is_nan());
    }

    #[test]
    fn yeo_johnson_transform() {
        let mut data = vec![-1.0, 0.0, 1.0];
        let lambda = 0.5;
        let yeo_johnson = YeoJohnson::new().with_lambda(lambda).unwrap();
        let expected = vec![-1.2189514164974602, 0.0, 0.8284271247461903];
        yeo_johnson.transform(&mut data).unwrap();
        assert_all_close(&expected, &data);
    }

    #[test]
    fn yeo_johnson_fit_transform_nans() {
        let mut data = vec![-1.0, 0.0, f64::NAN, 1.0];
        let mut yeo_johnson = YeoJohnson::new();
        assert!(yeo_johnson.fit_transform(&mut data).is_err());
    }

    #[test]
    fn yeo_johnson_fit_transform_ignore_nans() {
        let mut data = vec![-1.0, 0.0, f64::NAN, 1.0];
        let mut yeo_johnson = YeoJohnson::new().ignore_nans(true);
        let expected = vec![-1.0000010312156777, 0.0, f64::NAN, 0.9999989687856643];
        yeo_johnson.fit_transform(&mut data).unwrap();
        assert_all_close(&expected, &data);
    }

    #[test]
    fn inverse_yeo_johnson_transform() {
        let mut data = vec![-1.2189514164974602, 0.0, 0.8284271247461903];
        let lambda = 0.5;
        let yeo_johnson = YeoJohnson::new().with_lambda(lambda).unwrap();
        let expected = vec![-1.0, 0.0, 1.0];
        yeo_johnson.inverse_transform(&mut data).unwrap();
        assert_all_close(&expected, &data);
    }
}
