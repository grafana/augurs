use crate::transforms::box_cox;
use crate::transforms::yeo_johnson;
use argmin::core::*;
use argmin::solver::brent::BrentOpt;

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


fn optimize_lambda<T: CostFunction<Param = f64, Output = f64>>(
    cost: T,
    params: OptimizationParams,
) -> Result<f64, Error> {
    let solver = BrentOpt::new(params.lower_bound, params.upper_bound);
    let result = Executor::new(cost, solver)
        .configure(|state| state.param(params.initial_param).max_iters(params.max_iterations))
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
    let optimization_params = OptimizationParams {
        initial_param: 0.0,
        lower_bound: -2.0,
        upper_bound: 2.0,
        max_iterations: 1000,
    };
    optimize_lambda(cost, optimization_params)
}

pub(crate) fn optimize_yeo_johnson_lambda(data: &[f64]) -> Result<f64, Error> {
    // Use Yeo-Johnson transformation
    let cost = YeoJohnsonProblem { data };
    let optimization_params = OptimizationParams {
        initial_param: 0.0,
        lower_bound: -2.0,
        upper_bound: 2.0,
        max_iterations: 1000,
    };
    optimize_lambda(cost, optimization_params)
}

#[cfg(test)]
mod test {
    use super::*;
    use augurs_testing::assert_approx_eq;

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
}
