//! Utilities for augurs-arima.
//!
//! - Take the n-th difference of a time series, including seasonal differencing.
//! - Expand the AR/MA backshift notation polynomial.
//! - Compute mean and variance of a slice.

use crate::error::*;

/// Apply differencing $d$ times.
pub fn diff(y: &[f64], d: usize) -> Result<Vec<f64>> {
    let n = y.len();
    if n <= d {
        return Err(Error::SeriesTooShort {
            need: d + 1,
            got: n,
        });
    }

    let mut y_vec = y.to_vec();
    for _ in 0..d {
        y_vec = y_vec.windows(2).map(|w| w[1] - w[0]).collect();
    }

    Ok(y_vec)
}

/// Apply seasonal diferencing of lag `period` $sd$ times.
pub fn seasonal_diff(y: &[f64], period: usize, sd: usize) -> Result<Vec<f64>> {
    let n = y.len();
    if n <= period * sd {
        return Err(Error::SeriesTooShort {
            need: period * sd + 1,
            got: n,
        });
    }

    let mut y_vec = y.to_vec();
    for _ in 0..sd {
        y_vec = y_vec
            .iter()
            .skip(period)
            .zip(y_vec.iter())
            .map(|(a, b)| a - b)
            .collect();
    }

    Ok(y_vec)
}

/// Compute the arithmetic mean of a slice.
pub fn mean(y: &[f64]) -> f64 {
    if y.is_empty() {
        return 0.0;
    }
    y.iter().sum::<f64>() / (y.len() as f64)
}

/// Compute the variance of a slice.
pub fn variance(y: &[f64]) -> f64 {
    if y.len() < 2 {
        return 0.0;
    };
    let m = mean(y);
    y.iter().map(|&v| (v - m).powi(2)).sum::<f64>() / (y.len() as f64)
}

/// Expand the AR/MA backshift notation polynomial.
pub fn expand_poly(non_seasonal: &[f64], seasonal: &[f64], period: usize, is_ar: bool) -> Vec<f64> {
    if seasonal.is_empty() {
        return non_seasonal.to_vec();
    }

    let sign = if is_ar { -1.0 } else { 1.0 };

    let mut ns_full = vec![0.0; non_seasonal.len() + 1];
    ns_full[0] = 1.0;
    for (i, &coef) in non_seasonal.iter().enumerate() {
        ns_full[i + 1] = sign * coef;
    }

    let s_max = seasonal.len() * period;
    let mut s_full = vec![0.0; s_max + 1];
    s_full[0] = 1.0;
    for (i, &coef) in seasonal.iter().enumerate() {
        s_full[(i + 1) * period] = sign * coef;
    }

    let prod_len = ns_full.len() + s_full.len() - 1;
    let mut prod = vec![0.0; prod_len];
    for (i, &a) in ns_full.iter().enumerate() {
        for (j, &b) in s_full.iter().enumerate() {
            prod[i + j] += a * b;
        }
    }

    prod[1..].iter().map(|&v| v * sign).collect()
}

/// Compute the sample autocorrelation function up to `max_lag`.
pub fn acf(y: &[f64], max_lag: usize) -> Result<Vec<f64>> {
    let n = y.len();
    if n < 2 {
        return Err(Error::SeriesTooShort { need: 2, got: n });
    }
    let m = mean(y);
    let var = variance(y);

    if var.abs() < f64::EPSILON {
        let mut r = vec![0.0; max_lag + 1];
        r[0] = 1.0;
        return Ok(r);
    }

    let mut r = Vec::with_capacity(max_lag + 1);
    for k in 0..=max_lag {
        if k >= n {
            r.push(0.0);
            continue;
        }
        let cov: f64 = y[..n - k]
            .iter()
            .zip(y[k..].iter())
            .map(|(&a, &b)| (a - m) * (b - m))
            .sum::<f64>()
            / n as f64;
        r.push(cov / var);
    }
    Ok(r)
}

/// Compute the sample partial autocorrelation function up to `max_lag`.
pub fn pacf(y: &[f64], max_lag: usize) -> Result<Vec<f64>> {
    let r = acf(y, max_lag)?;
    let mut phi = vec![0.0; max_lag + 1];
    if max_lag == 0 {
        return Ok(phi);
    }

    let mut prev = vec![0.0; max_lag + 1];
    phi[1] = r[1];
    prev[1] = r[1];

    for k in 2..=max_lag {
        let num: f64 = r[k] - (1..k).map(|j| prev[j] * r[k - j]).sum::<f64>();
        let den: f64 = 1.0 - (1..k).map(|j| prev[j] * r[j]).sum::<f64>();
        if den.abs() < f64::EPSILON {
            break;
        }
        phi[k] = num / den;

        let curr: Vec<f64> = (1..k).map(|j| prev[j] - phi[k] * prev[k - j]).collect();
        for (j, &v) in curr.iter().enumerate() {
            prev[j + 1] = v;
        }
        prev[k] = phi[k];
    }
    Ok(phi)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    // Diff (1) works.
    fn test_diff_1() {
        let y = [1.0, 3.0, 7.0, 14.0];
        let y_diff = diff(&y, 1);
        assert_eq!(y_diff.unwrap(), vec![2.0, 4.0, 7.0]);
    }

    #[test]
    // Diff (2) works.
    fn test_diff_2() {
        let y = [1.0, 3.0, 7.0, 14.0];
        let y_diff = diff(&y, 2);
        assert_eq!(y_diff.unwrap(), vec![2.0, 3.0]);
    }

    #[test]
    // Diff fails gracefully if series is too short.
    fn test_fail() {
        let y = [1.0, 3.0];
        let y_diff = diff(&y, 2);
        assert_eq!(y_diff, Err(Error::SeriesTooShort { need: 3, got: 2 }));
    }

    #[test]
    // Seasonal diff (1, two-period) works.
    fn test_seasonal_diff_1() {
        let y = [1.0, 3.0, 7.0, 14.0, 20.0, 13.0, 18.0, 4.0, 5.0];
        let y_diff = seasonal_diff(&y, 2, 1);
        assert_eq!(
            y_diff.unwrap(),
            vec![6.0, 11.0, 13.0, -1.0, -2.0, -9.0, -13.0]
        )
    }

    #[test]
    // Seasonal diff (1, three-period) works.
    fn test_seasonal_diff_2() {
        let y = [1.0, 3.0, 7.0, 14.0, 20.0, 13.0, 18.0, 4.0, 5.0];
        let y_diff = seasonal_diff(&y, 3, 1);
        assert_eq!(y_diff.unwrap(), vec![13.0, 17.0, 6.0, 4.0, -16.0, -8.0])
    }

    #[test]
    // Seasonal diff (2, two-period) works
    fn test_seasonal_diff_2_2() {
        let y = [1.0, 3.0, 7.0, 14.0, 20.0, 13.0, 18.0, 4.0, 5.0];
        let y_diff = seasonal_diff(&y, 3, 2);
        assert_eq!(y_diff.unwrap(), vec![-9.0, -33.0, -14.0])
    }

    #[test]
    // Fails gracefully if series is too short for seasonal diff.
    fn test_seasonal_diff_fail() {
        let y = [1.0, 3.0, 7.0, 14.0, 20.0, 13.0, 18.0, 4.0, 5.0];
        let y_diff = seasonal_diff(&y, 4, 3);
        assert_eq!(y_diff, Err(Error::SeriesTooShort { need: 13, got: 9 }))
    }

    #[test]
    // Polynomial expansion works.
    fn test_poly_expansion() {
        let non_seasonal = vec![0.7, -0.2];
        let seasonal = vec![0.5];
        let period = 12;
        let is_ar = true;

        assert_eq!(
            expand_poly(&non_seasonal, &seasonal, period, is_ar),
            vec![0.7, -0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, -0.35, 0.1]
        )
    }
}
