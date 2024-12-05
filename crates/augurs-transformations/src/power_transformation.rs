use argmin::core::*;
use argmin::solver::brent::BrentRoot;
use crate::box_cox;

fn box_cox_log_likelihood(data: &[f64], lambda: f64) -> f64 {
    let n = data.len() as f64;
    let transformed_data: Vec<f64> = data.iter().map(|&x| box_cox(x, lambda)).collect();
    let mean_transformed: f64 = transformed_data.iter().copied().sum::<f64>() / n;
    let variance: f64 = transformed_data.iter().map(|&x| (x - mean_transformed).powi(2)).sum::<f64>() / n;
    
    // Avoid log(0) by ensuring variance is positive
    let log_likelihood = -0.5 * n * variance.ln() + (lambda - 1.0) * data.iter().map(|&x| x.ln()).sum::<f64>();
    log_likelihood
}

#[derive(Clone)]
struct BoxCoxProblem {
    data: Vec<f64>,
}

impl CostFunction for BoxCoxProblem {
    type Param = f64;
    type Output = f64;
    
    // The goal is to minimize the negative log-likelihood
    fn cost(&self, lambda: &Self::Param) -> Result<Self::Output, Error> {
        Ok(-box_cox_log_likelihood(&self.data, *lambda))
    }
}

/// Optimize the lambda parameter for the Box-Cox transformation
pub fn optimize_lambda(data: &[f64]) -> f64 {
    let cost = BoxCoxProblem { data: data.to_vec() };
    let init_param = 0.5;
    let solver = BrentRoot::new(-5.0, 5.0, 1e-10);

    let res = Executor::new(cost, solver)
        .configure(|state| state.param(init_param).max_iters(100))
        .run()
        .unwrap();
    return res.state.best_param.unwrap();
}