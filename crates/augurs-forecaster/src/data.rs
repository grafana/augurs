/// Trait for data that can be used in the forecaster.
///
/// This trait is implemented for a number of types including slices, arrays, and
/// vectors. It is also implemented for references to these types.
pub trait Data {
    /// Return the data as a slice of `f64`.
    fn as_slice(&self) -> &[f64];
}

impl<const N: usize> Data for [f64; N] {
    fn as_slice(&self) -> &[f64] {
        self
    }
}

impl Data for &[f64] {
    fn as_slice(&self) -> &[f64] {
        self
    }
}

impl Data for Vec<f64> {
    fn as_slice(&self) -> &[f64] {
        self.as_slice()
    }
}

impl<T> Data for &T
where
    T: Data,
{
    fn as_slice(&self) -> &[f64] {
        (*self).as_slice()
    }
}
