//! Exponential transformations, including log and logit.

use std::fmt;

use super::{Error, Transform};

// Logit and logistic functions.

/// Returns the logistic function of the given value.
fn logistic(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Returns the logit function of the given value.
fn logit(x: f64) -> f64 {
    (x / (1.0 - x)).ln()
}

/// The logit transform.
#[derive(Clone, Default)]
pub struct Logit {
    _priv: (),
}

impl Logit {
    /// Create a new logit transform.
    pub fn new() -> Self {
        Self::default()
    }
}

impl fmt::Debug for Logit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Logit").finish()
    }
}

impl Transform for Logit {
    fn transform(&mut self, data: &mut [f64]) -> Result<(), Error> {
        data.iter_mut().for_each(|x| *x = logit(*x));
        Ok(())
    }

    fn inverse_transform(&self, data: &mut [f64]) -> Result<(), Error> {
        data.iter_mut().for_each(|x| *x = logistic(*x));
        Ok(())
    }
}

/// The log transform.
#[derive(Clone, Default)]
pub struct Log {
    _priv: (),
}

impl Log {
    /// Create a new log transform.
    pub fn new() -> Self {
        Self::default()
    }
}

impl fmt::Debug for Log {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Log").finish()
    }
}

impl Transform for Log {
    fn transform(&mut self, data: &mut [f64]) -> Result<(), Error> {
        data.iter_mut().for_each(|x| *x = f64::ln(*x));
        Ok(())
    }

    fn inverse_transform(&self, data: &mut [f64]) -> Result<(), Error> {
        data.iter_mut().for_each(|x| *x = f64::exp(*x));
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use augurs_testing::{assert_all_close, assert_approx_eq};

    use super::*;

    #[test]
    fn test_logistic() {
        let x = 0.0;
        let expected = 0.5;
        let actual = logistic(x);
        assert_approx_eq!(expected, actual);
        let x = 1.0;
        let expected = 1.0 / (1.0 + (-1.0_f64).exp());
        let actual = logistic(x);
        assert_approx_eq!(expected, actual);
        let x = -1.0;
        let expected = 1.0 / (1.0 + 1.0_f64.exp());
        let actual = logistic(x);
        assert_approx_eq!(expected, actual);
    }

    #[test]
    fn test_logit() {
        let x = 0.5;
        let expected = 0.0;
        let actual = logit(x);
        assert_eq!(expected, actual);
        let x = 0.75;
        let expected = (0.75_f64 / (1.0 - 0.75)).ln();
        let actual = logit(x);
        assert_eq!(expected, actual);
        let x = 0.25;
        let expected = (0.25_f64 / (1.0 - 0.25)).ln();
        let actual = logit(x);
        assert_eq!(expected, actual);
    }

    #[test]
    fn logit_transform() {
        let mut data = vec![0.5, 0.75, 0.25];
        let expected = vec![
            0.0_f64,
            (0.75_f64 / (1.0 - 0.75)).ln(),
            (0.25_f64 / (1.0 - 0.25)).ln(),
        ];
        Logit::new()
            .transform(&mut data)
            .expect("failed to logit transform");
        assert_all_close(&expected, &data);
    }

    #[test]
    fn logit_inverse_transform() {
        let mut data = vec![0.0, 1.0, -1.0];
        let expected = vec![
            0.5_f64,
            1.0 / (1.0 + (-1.0_f64).exp()),
            1.0 / (1.0 + 1.0_f64.exp()),
        ];
        Logit::new()
            .inverse_transform(&mut data)
            .expect("failed to inverse logit transform");
        assert_all_close(&expected, &data);
    }

    #[test]
    fn log_transform() {
        let mut data = vec![1.0, 2.0, 3.0];
        let expected = vec![0.0_f64, 2.0_f64.ln(), 3.0_f64.ln()];
        Log::new()
            .transform(&mut data)
            .expect("failed to log transform");
        assert_all_close(&expected, &data);
    }

    #[test]
    fn log_inverse_transform() {
        let mut data = vec![0.0, 2.0_f64.ln(), 3.0_f64.ln()];
        let expected = vec![1.0, 2.0, 3.0];
        Log::new()
            .inverse_transform(&mut data)
            .expect("failed to inverse log transform");
        assert_all_close(&expected, &data);
    }
}
