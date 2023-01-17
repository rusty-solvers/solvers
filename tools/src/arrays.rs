//! Containers to store multi-dimensional data

/// A two-dimensional rectangular array
pub struct Array2D<T> {
    data: Vec<T>,
    shape: (usize, usize),
}
impl<T> Array2D<T> {
    /// Create an array
    pub fn new(data: Vec<T>, shape: (usize, usize)) -> Self {
        Self {
            /// The data in the array, in row-major order
            data: data,
            /// The shape of the array
            shape: shape,
        }
    }

    /// Get an item from the array
    pub fn get(&self, index0: usize, index1: usize) -> &T {
        self.data.get(index0 * self.shape.1 + index1).unwrap()
    }
    /// Get a mutable item from the array
    pub fn get_mut(&mut self, index0: usize, index1: usize) -> &mut T {
        self.data.get_mut(index0 * self.shape.1 + index1).unwrap()
    }
    /// Get a row of the array
    pub fn row(&self, index: usize) -> &[T] {
        &self.data[index * self.shape.1..(index + 1) * self.shape.1]
    }
}

/// An adjacency list
///
/// An adjacency list stores two-dimensional data where each row may have a different number of items
pub struct AdjacencyList<T> {
    data: Vec<T>,
    offsets: Vec<usize>,
}
impl<T> AdjacencyList<T> {
    /// Create an adjacency list
    pub fn new(data: Vec<T>, offsets: Vec<usize>) -> Self {
        Self {
            data: data,
            offsets: offsets,
        }
    }
    /// Get an item from the adjacency list
    pub fn get(&self, index0: usize, index1: usize) -> &T {
        // TODO: check that self.offsets[index0] + index1 < self.offsets[index0 + 1]
        self.data.get(self.offsets[index0] + index1).unwrap()
    }
    /// Get a mutable item from the adjacency list
    pub fn get_mut(&mut self, index0: usize, index1: usize) -> &mut T {
        // TODO: check that self.offsets[index0] + index1 < self.offsets[index0 + 1]
        self.data.get_mut(self.offsets[index0] + index1).unwrap()
    }
    /// Get a row from the adjacency list
    pub fn row(&self, index: usize) -> &[T] {
        &self.data[self.offsets[index]..self.offsets[index + 1]]
    }
}

#[cfg(test)]
mod test {
    use crate::arrays::*;

    #[test]
    fn test_array_2d() {
        let mut arr = Array2D::new(vec![1, 2, 3, 4, 5, 6], (2, 3));
        assert_eq!(*arr.get(0, 0), 1);
        assert_eq!(*arr.get(0, 1), 2);
        assert_eq!(*arr.get(0, 2), 3);
        assert_eq!(*arr.get(1, 0), 4);
        assert_eq!(*arr.get(1, 1), 5);
        assert_eq!(*arr.get(1, 2), 6);

        let row1 = arr.row(1);
        assert_eq!(row1.len(), 3);
        assert_eq!(row1[0], 4);
        assert_eq!(row1[1], 5);
        assert_eq!(row1[2], 6);

        *arr.get_mut(1, 2) = 7;
        assert_eq!(*arr.get(1, 2), 7);

        let mut arr2 = Array2D::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3));
        assert_eq!(*arr2.get(0, 0), 1.0);
        assert_eq!(*arr2.get(0, 1), 2.0);
        assert_eq!(*arr2.get(0, 2), 3.0);
        assert_eq!(*arr2.get(1, 0), 4.0);
        assert_eq!(*arr2.get(1, 1), 5.0);
        assert_eq!(*arr2.get(1, 2), 6.0);

        let row1 = arr2.row(1);
        assert_eq!(row1.len(), 3);
        assert_eq!(row1[0], 4.0);
        assert_eq!(row1[1], 5.0);
        assert_eq!(row1[2], 6.0);

        *arr2.get_mut(1, 2) = 7.;
        assert_eq!(*arr2.get(1, 2), 7.0);
    }

    #[test]
    fn test_adjacency_list() {
        let mut arr = AdjacencyList::new(vec![1, 2, 3, 4, 5, 6], vec![0, 2, 3, 6]);
        assert_eq!(*arr.get(0, 0), 1);
        assert_eq!(*arr.get(0, 1), 2);
        assert_eq!(*arr.get(1, 0), 3);
        assert_eq!(*arr.get(2, 0), 4);
        assert_eq!(*arr.get(2, 1), 5);
        assert_eq!(*arr.get(2, 2), 6);

        let row1 = arr.row(1);
        assert_eq!(row1.len(), 1);
        assert_eq!(row1[0], 3);

        *arr.get_mut(2, 0) = 7;
        assert_eq!(*arr.get(2, 0), 7);
    }
}
