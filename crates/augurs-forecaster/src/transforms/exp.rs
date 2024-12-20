//! Exponential transformations, including log and logit.

// Logit and logistic functions.

/// Returns the logistic function of the given value.
fn logistic(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Returns the logit function of the given value.
fn logit(x: f64) -> f64 {
    (x / (1.0 - x)).ln()
}

/// An iterator adapter that applies the logit function to each item.
#[derive(Clone, Debug)]
pub(crate) struct Logit<T> {
    inner: T,
}

impl<T> Iterator for Logit<T>
where
    T: Iterator<Item = f64>,
{
    type Item = f64;
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(logit)
    }
}

pub(crate) trait LogitExt: Iterator<Item = f64> {
    fn logit(self) -> Logit<Self>
    where
        Self: Sized,
    {
        Logit { inner: self }
    }
}

impl<T> LogitExt for T where T: Iterator<Item = f64> {}

/// An iterator adapter that applies the logistic function to each item.
#[derive(Clone, Debug)]
pub(crate) struct Logistic<T> {
    inner: T,
}

impl<T> Iterator for Logistic<T>
where
    T: Iterator<Item = f64>,
{
    type Item = f64;
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(logistic)
    }
}

pub(crate) trait LogisticExt: Iterator<Item = f64> {
    fn logistic(self) -> Logistic<Self>
    where
        Self: Sized,
    {
        Logistic { inner: self }
    }
}

impl<T> LogisticExt for T where T: Iterator<Item = f64> {}

/// An iterator adapter that applies the log function to each item.
#[derive(Clone, Debug)]
pub(crate) struct Log<T> {
    inner: T,
}

impl<T> Iterator for Log<T>
where
    T: Iterator<Item = f64>,
{
    type Item = f64;
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(f64::ln)
    }
}

pub(crate) trait LogExt: Iterator<Item = f64> {
    fn log(self) -> Log<Self>
    where
        Self: Sized,
    {
        Log { inner: self }
    }
}

impl<T> LogExt for T where T: Iterator<Item = f64> {}

/// An iterator adapter that applies the exponential function to each item.
#[derive(Clone, Debug)]
pub(crate) struct Exp<T> {
    inner: T,
}

impl<T> Iterator for Exp<T>
where
    T: Iterator<Item = f64>,
{
    type Item = f64;
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(f64::exp)
    }
}

pub(crate) trait ExpExt: Iterator<Item = f64> {
    fn exp(self) -> Exp<Self>
    where
        Self: Sized,
    {
        Exp { inner: self }
    }
}

impl<T> ExpExt for T where T: Iterator<Item = f64> {}

#[cfg(test)]
mod test {
    use augurs_testing::assert_approx_eq;

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
    fn logistic_transform() {
        let data = vec![0.0, 1.0, -1.0];
        let expected = vec![
            0.5_f64,
            1.0 / (1.0 + (-1.0_f64).exp()),
            1.0 / (1.0 + 1.0_f64.exp()),
        ];
        let actual: Vec<_> = data.into_iter().logistic().collect();
        assert_eq!(expected, actual);
    }

    #[test]
    fn logit_transform() {
        let data = vec![0.5, 0.75, 0.25];
        let expected = vec![
            0.0_f64,
            (0.75_f64 / (1.0 - 0.75)).ln(),
            (0.25_f64 / (1.0 - 0.25)).ln(),
        ];
        let actual: Vec<_> = data.into_iter().logit().collect();
        assert_eq!(expected, actual);
    }

    #[test]
    fn log_transform() {
        let data = vec![1.0, 2.0, 3.0];
        let expected = vec![0.0_f64, 2.0_f64.ln(), 3.0_f64.ln()];
        let actual: Vec<_> = data.into_iter().log().collect();
        assert_eq!(expected, actual);
    }
}
