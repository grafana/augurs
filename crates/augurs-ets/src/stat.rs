pub trait StatExt {
    fn mean(&self) -> f64;
    fn var(&self, ddof: usize) -> f64;
    fn std(&self, ddof: usize) -> f64 {
        self.var(ddof).sqrt()
    }
}

fn mean<T: AsRef<[f64]>>(x: T, ddof: usize) -> f64 {
    x.as_ref().iter().sum::<f64>() / (x.as_ref().len() - ddof) as f64
}

impl<T> StatExt for T
where
    T: AsRef<[f64]>,
{
    fn mean(&self) -> f64 {
        mean(self, 0)
    }

    fn var(&self, ddof: usize) -> f64 {
        let n = self.as_ref().len();
        assert!(
            ddof <= n,
            "`ddof` must not be greater than the length of the slice",
        );
        let dof = (n - ddof) as f64;
        let mut mean = 0.0;
        let mut sum_sq = 0.0;
        let mut i = 0;
        self.as_ref().iter().for_each(|&x| {
            let count = (i + 1) as f64;
            let delta = x - mean;
            mean += delta / count;
            sum_sq = (x - mean).mul_add(delta, sum_sq);
            i += 1;
        });
        sum_sq / dof
    }
}

#[cfg(test)]
mod test {
    use assert_approx_eq::assert_approx_eq;

    use super::StatExt;

    #[test]
    fn test_mean() {
        assert_eq!([3.0_f64, 5.0, 8.0, 1.0].mean(), 4.25);
    }

    #[test]
    fn test_var() {
        assert_eq!([3.0_f64, 5.0, 8.0, 1.0].var(0), 6.6875);
        assert_approx_eq!([3.0_f64, 5.0, 8.0, 1.0].var(1), 8.9166666666666);
    }
}
