use crate::transforms::box_cox;
use argmin::core::*;
use argmin::solver::brent::BrentOpt;

fn box_cox_log_likelihood(data: &[f64], lambda: f64) -> f64 {
    let n = data.len() as f64;
    assert!(n > 0.0, "Data must not be empty");
    let transformed_data: Vec<f64> = data.iter().map(|&x| box_cox(x, lambda)).collect();
    let mean_transformed: f64 = transformed_data.iter().copied().sum::<f64>() / n;
    let variance: f64 = transformed_data
        .iter()
        .map(|&x| (x - mean_transformed).powi(2))
        .sum::<f64>()
        / n;

    // Avoid log(0) by ensuring variance is positive
    let log_likelihood =
        -0.5 * n * variance.ln() + (lambda - 1.0) * data.iter().map(|&x| x.ln()).sum::<f64>();
    log_likelihood
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
        Ok(-box_cox_log_likelihood(&self.data, *lambda))
    }
}

/// Optimize the lambda parameter for the Box-Cox transformation
pub fn optimize_lambda(data: &[f64]) -> f64 {
    let cost = BoxCoxProblem { data: data };
    let init_param = 0.5;
    let solver = BrentOpt::new(-2.0, 2.0);

    let result = Executor::new(cost, solver)
        .configure(|state| state.param(init_param).max_iters(100))
        .run();

    match result {
        Ok(result) => result.state().best_param.unwrap(),
        Err(error) => panic!("Optimization failed: {}", error),
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use augurs_testing::assert_approx_eq;

    #[test]
    fn correct_optimal_lambda() {
        let data = &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7];
        let got = optimize_lambda(data);
        assert_approx_eq!(got, 0.7123778635679304);
    }

    #[test]
    fn test_boxcox_llf() {
        let data = &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7];
        let lambda = 1.0;
        let got = box_cox_log_likelihood(data, lambda);
        assert_approx_eq!(got, 11.266065387038703);
    }
}
