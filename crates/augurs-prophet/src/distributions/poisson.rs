use std::f64;

use rand::Rng;

use super::{consts, Error};

/// Implements the [Poisson](https://en.wikipedia.org/wiki/Poisson_distribution)
/// distribution.
#[derive(Debug, Copy, Clone, PartialEq)]
pub(crate) struct Poisson {
    lambda: f64,
}

impl Poisson {
    /// Constructs a new poisson distribution with a rate (λ)
    /// of `lambda`
    ///
    /// # Errors
    ///
    /// Returns an error if `lambda` is `NaN` or `lambda <= 0.0`
    pub(crate) fn new(lambda: f64) -> Result<Poisson, Error> {
        if lambda.is_nan() || lambda <= 0.0 {
            Err(Error)
        } else {
            Ok(Poisson { lambda })
        }
    }
}

impl ::rand::distributions::Distribution<f64> for Poisson {
    /// Generates one sample from the Poisson distribution either by
    /// Knuth's method if lambda < 30.0 or Rejection method PA by
    /// A. C. Atkinson from the Journal of the Royal Statistical Society
    /// Series C (Applied Statistics) Vol. 28 No. 1. (1979) pp. 29 - 35
    /// otherwise
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        sample_unchecked(rng, self.lambda)
    }
}

/// Generates one sample from the Poisson distribution either by
/// Knuth's method if lambda < 30.0 or Rejection method PA by
/// A. C. Atkinson from the Journal of the Royal Statistical Society
/// Series C (Applied Statistics) Vol. 28 No. 1. (1979) pp. 29 - 35
/// otherwise
fn sample_unchecked<R: Rng + ?Sized>(rng: &mut R, lambda: f64) -> f64 {
    if lambda < 30.0 {
        let limit = (-lambda).exp();
        let mut count = 0.0;
        let mut product: f64 = rng.gen();
        while product >= limit {
            count += 1.0;
            product *= rng.gen::<f64>();
        }
        count
    } else {
        let c = 0.767 - 3.36 / lambda;
        let beta = f64::consts::PI / (3.0 * lambda).sqrt();
        let alpha = beta * lambda;
        let k = c.ln() - lambda - beta.ln();

        loop {
            let u: f64 = rng.gen();
            let x = (alpha - ((1.0 - u) / u).ln()) / beta;
            let n = (x + 0.5).floor();
            if n < 0.0 {
                continue;
            }

            let v: f64 = rng.gen();
            let y = alpha - beta * x;
            let temp = 1.0 + y.exp();
            let lhs = y + (v / (temp * temp)).ln();
            let rhs = k + n * lambda.ln() - ln_factorial(n as u64);
            if lhs <= rhs {
                return n;
            }
        }
    }
}

/// Computes the logarithm of the gamma function
/// with an accuracy of 16 floating point digits.
/// The implementation is derived from
/// "An Analysis of the Lanczos Gamma Approximation",
/// Glendon Ralph Pugh, 2004 p. 116
fn ln_gamma(x: f64) -> f64 {
    if x < 0.5 {
        let s = consts::GAMMA_DK
            .iter()
            .enumerate()
            .skip(1)
            .fold(consts::GAMMA_DK[0], |s, t| s + t.1 / (t.0 as f64 - x));

        consts::LN_PI
            - (f64::consts::PI * x).sin().ln()
            - s.ln()
            - consts::LN_2_SQRT_E_OVER_PI
            - (0.5 - x) * ((0.5 - x + consts::GAMMA_R) / f64::consts::E).ln()
    } else {
        let s = consts::GAMMA_DK
            .iter()
            .enumerate()
            .skip(1)
            .fold(consts::GAMMA_DK[0], |s, t| s + t.1 / (x + t.0 as f64 - 1.0));

        s.ln()
            + consts::LN_2_SQRT_E_OVER_PI
            + (x - 0.5) * ((x - 0.5 + consts::GAMMA_R) / f64::consts::E).ln()
    }
}

/// Computes the logarithmic factorial function `x -> ln(x!)`
/// for `x >= 0`.
///
/// # Remarks
///
/// Returns `0.0` if `x <= 1`
fn ln_factorial(x: u64) -> f64 {
    let x = x as usize;
    FCACHE
        .get(x)
        .map_or_else(|| ln_gamma(x as f64 + 1.0), |&fac| fac.ln())
}

// Initialization for pre-computed cache of 171 factorial
// values 0!...170!
const FCACHE: [f64; consts::MAX_FACTORIAL + 1] = {
    let mut fcache = [1.0; consts::MAX_FACTORIAL + 1];

    // `const` only allow while loops
    let mut i = 1;
    while i < consts::MAX_FACTORIAL + 1 {
        fcache[i] = fcache[i - 1] * i as f64;
        i += 1;
    }

    fcache
};
