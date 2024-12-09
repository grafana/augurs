use crate::transforms::box_cox;
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
    let transformed_data: Vec<f64> = data.iter().map(|&x| box_cox(x, lambda)).collect();
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

#[derive(Clone)]
struct BoxCoxProblem<'a> {
    data: &'a [f64],
}

impl CostFunction for BoxCoxProblem<'_> {
    type Param = f64;
    type Output = f64;

    // The goal is to minimize the negative log-likelihood
    fn cost(&self, lambda: &Self::Param) -> Result<Self::Output, Error> {
        box_cox_log_likelihood(&self.data, *lambda).map(|ll| -ll)
    }
}

/// Optimize the lambda parameter for the Box-Cox transformation
pub (crate) fn optimize_lambda(data: &[f64]) -> Result<f64, Error> {
    let cost = BoxCoxProblem { data: data };
    let init_param = 0.5;
    let solver = BrentOpt::new(-2.0, 2.0);

    let result = Executor::new(cost, solver)
        .configure(|state| state.param(init_param).max_iters(100))
        .run();

    result
        .and_then(|res| {
            res.state()
                .best_param
                .ok_or_else(|| Error::msg("No best parameter found"))
            })
}

#[cfg(test)]
mod test {
    use super::*;
    use augurs_testing::assert_approx_eq;

    #[test]
    fn correct_optimal_lambda() {
        let data = &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7];
        let got = optimize_lambda(data);
        assert!(got.is_ok());
        let lambda = got.unwrap();
        assert_approx_eq!(lambda, 0.7123778635679304);
    }

    #[test]
    fn optimize_lambda_empty_data() {
        let data = &[];
        let got = optimize_lambda(data);
        assert!(got.is_err());
    }

    #[test]
    fn optimize_lambda_non_positive_data() {
        let data = &[0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7];
        let got = optimize_lambda(data);
        assert!(got.is_err());
    }

    #[test]
    fn test_boxcox_llf() {
        let data = &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7];
        let lambda = 1.0;
        let got = box_cox_log_likelihood(data, lambda);
        assert!(got.is_ok());
        let llf = got.unwrap();
        assert_approx_eq!(llf, 11.266065387038703);
    }

    #[test]
    fn test_boxcox_llf_non_positive() {
        let data = &[0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7];
        let lambda = 0.0;
        let got = box_cox_log_likelihood(data, lambda);
        assert!(got.is_err());
    }
}
