//! Structs and enums for ARIMA modeling.

use std::fmt;

/// Arima order specification: ARIMA(p,d,q)(P,D,Q)[m].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ArimaOrder {
    /// Non-seasonal AR order.
    pub p: usize,
    /// Non-seasonal differencing order.
    pub d: usize,
    /// Non-seasonal MA order.
    pub q: usize,
    /// Seasonal AR order.
    pub sp: usize,
    /// Seasonal differencing order.
    pub sd: usize,
    /// Seasonal MA order.
    pub sq: usize,
    /// Seasonal period (1 for non-seasonal).
    pub period: usize,
}

impl ArimaOrder {
    /// Create a non-seasonal ARIMA(p,d,q) order.
    pub fn new(p: usize, d: usize, q: usize) -> Self {
        Self {
            p,
            d,
            q,
            sp: 0,
            sd: 0,
            sq: 0,
            period: 1,
        }
    }

    /// Create a SARIMA(p,d,q)(P,D,Q)[m]
    pub fn seasonal(
        p: usize,
        d: usize,
        q: usize,
        sp: usize,
        sd: usize,
        sq: usize,
        period: usize,
    ) -> Self {
        Self {
            p,
            d,
            q,
            sp,
            sd,
            sq,
            period,
        }
    }

    /// Total number of AR coefficients (p + P).
    pub fn total_ar(&self) -> usize {
        self.p + self.sp
    }

    /// Total number of MA oefficients (q + Q).
    pub fn total_ma(&self) -> usize {
        self.q + self.sq
    }

    /// Check if there's any seasonal component.
    pub fn is_seasonal(&self) -> bool {
        self.period > 1 && (self.sp > 0 || self.d > 0 || self.sq > 0)
    }
}

impl fmt::Display for ArimaOrder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_seasonal() {
            write!(
                f,
                "ARIMA({},{},{})({},{},{})[{}]",
                self.p, self.d, self.q, self.sp, self.sd, self.sq, self.period
            )
        } else {
            write!(f, "ARIMA({},{},{})", self.p, self.d, self.q)
        }
    }
}

/// Estimation method for ARIMA models.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EstimationMethod {
    /// Conditional Sum of Squares.
    Css,
    /// Maximum Likelihood (exact) via Kalman filter.
    Ml,
    /// CSS for initial values, then ML refinement.
    CssMl,
}

/// A fitted ARIMA model.
#[derive(Debug, Clone)]
pub struct ArimaModel {
    /// Model order.
    pub order: ArimaOrder,
    /// AR coefficients (p).
    pub ar: Vec<f64>,
    /// MA coefficients (q).
    pub ma: Vec<f64>,
    /// Seasonal AR coefficients (P).
    pub sar: Vec<f64>,
    /// Seasonal MA coefficients (Q).
    pub sma: Vec<f64>,
    /// Mean (or drift, if integrated).
    pub intercept: f64,
    /// Whether a intercept term is included.
    pub include_mean: bool,
    /// Whether a drift term is included.
    pub include_drift: bool,
    /// Innovation variance $\sigma^2$.
    pub sigma2: f64,
    /// Log-likelihood.
    pub log_lik: f64,
    /// In-sample fitted values.
    pub fitted: Vec<f64>,
    /// In-sample residuals.
    pub residuals: Vec<f64>,
    /// The original series.
    pub y: Vec<f64>,
    /// Box-Cox lambda if applied.
    pub lambda: Option<f64>,
    /// Estimation method of choice.
    pub method: EstimationMethod,
}
