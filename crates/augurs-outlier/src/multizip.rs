pub struct MultiZipped<I1, I2, T>
where
    I1: IntoIterator<Item = I2>,
    I2: Iterator<Item = T>,
{
    inner: I1::IntoIter,
}

impl<I1, I2, T> MultiZipped<I1, I2, T>
where
    I1: IntoIterator<Item = I2>,
    I2: Iterator<Item = T>,
{
    pub fn new(v: I1) -> Self {
        Self {
            inner: v.into_iter(),
        }
    }
}

impl<I1, I2, T> Iterator for MultiZipped<I1, I2, T>
where
    I1: IntoIterator<Item = I2>,
    I2: Iterator<Item = T>,
{
    type Item = Vec<T>;
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.by_ref().map(|mut i| i.next()).collect()
    }
}

trait MultiZip<I1, I2, T>
where
    I1: IntoIterator<Item = I2>,
    I2: Iterator<Item = T>,
{
    fn multi_zip(self) -> MultiZipped<I1, I2, T>;
}

impl<I1, I2, T> MultiZip<I1, I2, T> for I1
where
    I1: IntoIterator<Item = I2>,
    I2: Iterator<Item = T>,
{
    fn multi_zip(self) -> MultiZipped<I1, I2, T> {
        MultiZipped::new(self)
    }
}
