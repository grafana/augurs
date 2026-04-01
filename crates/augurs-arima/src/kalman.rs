//! Kalman filter for ARIMA models in state-space form.
//!
//! Puts an Arima(p, d, q) model into state-space form and runs the Kalman
//! filter to compute the exact log-likelihood.

use crate::error::{Error, Result};
use nalgebra::{DMatrix, DVector};

/// State-space representation of an ARMA model. It uses the Harvey representation defined as
/// $$
///   \begin{aligned}
///     \alpha_{t+1} &= T \alpha_t + R \epsilon_t && \text{(state transition)} \\\\
///     y_t &= Z^\top \alpha_t + \epsilon_t && \text{(observation)}
///   \end{aligned}
/// $$
///
/// where $r = max(p, q+1)$ is the state dimension.

#[derive(Debug, Clone)]
pub struct ArimaStateSpace {
    /// Transition matrix $T \in \mathbb{R}^{r \times r}$.
    pub transition: DMatrix<f64>,
    /// Observation vector $Z \in \mathbb{R}^{r}$.
    pub observation: DMatrix<f64>,
    /// Selection vector $R \in \mathbb{R}^{r}$.
    pub selection: DMatrix<f64>,
    /// State dimension.
    pub r: usize,
}

impl ArimaStateSpace {
    /// Build the state-space form from expanded AR and MA coefficients
    pub fn new(ar: &[f64], ma: &[f64]) -> Self {
        let p = ar.len();
        let q = ma.len();
        let r = p.max(q + 1);

        // Transition matrix T
        let mut transition = DMatrix::zeros(r, r);
        for i in 0..r {
            if i < p {
                transition[(i, 0)] = ar[i];
            }
            if i + 1 < r {
                transition[(i, i + 1)] = 1.0
            }
        }

        // Observation vector Z = [1, 0, ..., 0]'
        let mut observation = DMatrix::zeros(r, 1);
        observation[(0, 0)] = 1.0;

        // Selection vector R = [1, theta_1, theta_2, ..., theta_{r-1}]'
        let mut selection = DMatrix::zeros(r, 1);
        selection[(0, 0)] = 1.0;
        for i in 1..r {
            if i - 1 < q {
                selection[(i, 0)] = ma[i - 1];
            }
        }

        Self {
            transition,
            observation,
            selection,
            r,
        }
    }
}

/// Output of the Kalman filter.
#[derive(Debug, Clone)]
pub struct KalmanOutput {
    /// Log-likelihood.
    pub log_lik: f64,
    /// Innovation variance estimate $\sigma^2$.
    pub sigma2: f64,
    /// One-sted-ahead prediction errors.
    pub innovations: Vec<f64>,
    /// Innovation variances $F_t$.
    pub innovation_var: Vec<f64>,
    /// Final predicted state $a_{n+1|n}$
    pub final_state: DMatrix<f64>,
}

/// Compute the stationary covariance $P$ satisfying $P = TPT' + Q$.
///
/// Uses the vectorised Lyapunov equation, given by $vec(P) = (I − T \bigotimes T)^{-1} vec(Q)$.
fn stationary_covariance(t_mat: &DMatrix<f64>, q: &DMatrix<f64>) -> Option<DMatrix<f64>> {
    let r = t_mat.nrows();
    let r2 = r * r;

    // Build (I − TxT)
    let mut a = DMatrix::<f64>::identity(r2, r2);
    for i in 0..r {
        for j in 0..r {
            let row = i * r + j;
            for k in 0..r {
                for l in 0..r {
                    let col = k * r + l;
                    a[(row, col)] -= t_mat[(i, k)] * t_mat[(j, l)];
                }
            }
        }
    }

    // vec(Q)
    let q_vec = DVector::from_iterator(r2, q.iter().cloned());

    // Solve A * vec(P) = vec(Q)
    let p_vec = a.lu().solve(&q_vec)?;

    if p_vec.iter().any(|x| !x.is_finite()) {
        return None;
    }

    // Reshape vec(P) → P
    Some(DMatrix::from_iterator(r, r, p_vec.into_iter().cloned()))
}

/// Run the Kalman filter on a series using the ARIMA state-space form.
pub fn kalman_filter(y: &[f64], ss: &ArimaStateSpace, sigma2: Option<f64>) -> Result<KalmanOutput> {
    let n = y.len();
    let r = ss.r;

    if n < r {
        return Err(Error::NotEnoughData { need: ss.r, got: n });
    }

    let t_mat = &ss.transition;
    let z = &ss.observation;
    let sel = &ss.selection;

    // Precompute RR'
    let rrt = sel * sel.transpose();

    // Initial state: zero (r x 1)
    let mut a = DMatrix::zeros(r, 1);
    // Initial state covariance: stationary solution P = TPT' + RR'
    let mut p = stationary_covariance(t_mat, &rrt).unwrap_or_else(|| {
        // If non-stationary, fall back to diffuse initialization
        DMatrix::from_diagonal_element(r, r, 1e6)
    });

    let mut innovations = Vec::with_capacity(n);
    let mut innovation_var = Vec::with_capacity(n);
    let mut sum_log_f = 0.0;
    let mut ssq = 0.0;

    for obs in y {
        // Innovation: v_t = y_t - Z' a_{t|t-1}
        let za = (z.transpose() * &a)[(0, 0)];
        let v = obs - za;

        // Innovation variance: F_t = Z' P_{t|t-1} Z
        // Adding defense so it doesn't get zeroed and ensure it's invertible
        let f = (z.transpose() * &p * z)[(0, 0)].max(1e-10);

        innovations.push(v);
        innovation_var.push(f);

        sum_log_f += f.ln();
        ssq += v * v / f;

        // Kalman gain: K = PZ / F
        let k = (&p * z) / f;

        // State update: a_{t|t} = a_{t|t-1} + Kv
        let a_updated = &a + &k * v;

        // Covariance update: P_{t|t} = P - KZ'P
        let p_updated = &p - &k * (z.transpose() * &p);

        // Prediction: a_{t+1|t} = T a_{t|t}
        a = t_mat * &a_updated;

        // P_{t+1|t} = TP_{t|t} T' + RR'
        p = t_mat * &p_updated * t_mat.transpose() + &rrt;
    }

    let nf = n as f64;
    let estimated_sigma2 = sigma2.unwrap_or(ssq / nf);

    let log_lik = -nf / 2.0 * (2.0 * std::f64::consts::PI).ln()
        - nf / 2.0 * estimated_sigma2.max(1e-10).ln()
        - 0.5 * sum_log_f
        - ssq / (2.0 * estimated_sigma2.max(1e-10));

    Ok(KalmanOutput {
        log_lik,
        sigma2: estimated_sigma2,
        innovations,
        innovation_var,
        final_state: a,
    })
}

/// Produce h-step-ahead forecasts from a Kalman filter's final state.
pub fn kalman_forecast(ss: &ArimaStateSpace, final_state: &DMatrix<f64>, h: usize) -> Vec<f64> {
    let z = &ss.observation;
    let t_mat = &ss.transition;
    let mut a = final_state.clone();
    let mut forecasts = Vec::with_capacity(h);

    for _ in 0..h {
        let pred = (z.transpose() * &a)[(0, 0)];
        forecasts.push(pred);
        a = t_mat * &a;
    }

    forecasts
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_state_space_ar1() {
        let ss = ArimaStateSpace::new(&[0.8], &[]);
        assert_eq!(ss.r, 1);
        assert_relative_eq!(ss.transition[(0, 0)], 0.8);
        assert_relative_eq!(ss.observation[(0, 0)], 1.0);
        assert_relative_eq!(ss.selection[(0, 0)], 1.0);
    }

    #[test]
    fn test_state_space_arma11() {
        let ss = ArimaStateSpace::new(&[0.7], &[0.3]);
        assert_eq!(ss.r, 2);
        assert_relative_eq!(ss.transition[(0, 0)], 0.7);
        assert_relative_eq!(ss.transition[(0, 1)], 1.0);
        assert_relative_eq!(ss.transition[(1, 0)], 0.0);
        assert_relative_eq!(ss.selection[(0, 0)], 1.0);
        assert_relative_eq!(ss.selection[(1, 0)], 0.3);
    }

    #[test]
    fn test_kalman_filter_runs() {
        let y: Vec<f64> = (0..50).map(|i| i as f64 * 0.3).collect();
        let ss = ArimaStateSpace::new(&[0.4], &[]);
        let output = kalman_filter(&y, &ss, None).unwrap();
        let forecasts = kalman_forecast(&ss, &output.final_state, 12);

        assert_eq!(output.innovations.len(), 50);
        assert!(output.log_lik.is_finite());
        assert!(output.sigma2 > 0.0);
        assert_eq!(forecasts.len(), 12);
    }
}
