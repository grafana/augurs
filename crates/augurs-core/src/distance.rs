use std::{fmt, ops::Index};

/// An error that can occur when creating a `DistanceMatrix`.
#[derive(Debug)]
pub enum DistanceMatrixError {
    /// The input matrix is not square.
    InvalidDistanceMatrix,
}

impl fmt::Display for DistanceMatrixError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("invalid distance matrix")
    }
}

impl std::error::Error for DistanceMatrixError {}

/// A matrix representing the distances between pairs of items.
#[derive(Debug, Clone)]
pub struct DistanceMatrix {
    matrix: Vec<Vec<f64>>,
}

impl DistanceMatrix {
    /// Create a new `DistanceMatrix` from a square matrix.
    ///
    /// # Errors
    ///
    /// Returns an error if the input matrix is not square.
    pub fn try_from_square(matrix: Vec<Vec<f64>>) -> Result<Self, DistanceMatrixError> {
        if matrix.iter().all(|x| x.len() == matrix.len()) {
            Ok(Self { matrix })
        } else {
            Err(DistanceMatrixError::InvalidDistanceMatrix)
        }
    }

    /// Consumes the `DistanceMatrix` and returns the inner matrix.
    pub fn into_inner(self) -> Vec<Vec<f64>> {
        self.matrix
    }

    /// Returns an iterator over the rows of the matrix.
    pub fn iter(&self) -> DistanceMatrixIter<'_> {
        DistanceMatrixIter {
            iter: self.matrix.iter(),
        }
    }

    /// Returns the shape of the matrix.
    ///
    /// The first element is the number of rows and the second element
    /// is the number of columns.
    ///
    /// The matrix is square, so the number of rows is equal to the number of columns
    /// and the number of input series.
    pub fn shape(&self) -> (usize, usize) {
        (self.matrix.len(), self.matrix.len())
    }
}

impl Index<usize> for DistanceMatrix {
    type Output = [f64];
    fn index(&self, index: usize) -> &Self::Output {
        &self.matrix[index]
    }
}

impl Index<(usize, usize)> for DistanceMatrix {
    type Output = f64;
    fn index(&self, (i, j): (usize, usize)) -> &Self::Output {
        &self.matrix[i][j]
    }
}

impl IntoIterator for DistanceMatrix {
    type Item = Vec<f64>;
    type IntoIter = std::vec::IntoIter<Self::Item>;
    fn into_iter(self) -> Self::IntoIter {
        self.matrix.into_iter()
    }
}

/// An iterator over the rows of a `DistanceMatrix`.
#[derive(Debug)]
pub struct DistanceMatrixIter<'a> {
    iter: std::slice::Iter<'a, Vec<f64>>,
}

impl<'a> Iterator for DistanceMatrixIter<'a> {
    type Item = &'a Vec<f64>;
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }
}
