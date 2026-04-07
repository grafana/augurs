//! Utilities for augurs-arima.
//!
//! - Take the n-th difference of a time series, including seasonal differencing.

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
}
