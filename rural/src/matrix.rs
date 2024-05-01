use core::fmt;
use rand_core::SeedableRng;
use rand_distr::Distribution;
use std::ops::{Index, IndexMut};

// TODO(toms): create custom Debug formatter that has sub-arrays
#[derive(Clone)]
pub struct Matrix {
    data: Box<[f32]>,
    shape: (usize, usize),
}

impl Matrix {
    pub fn new<const M: usize, const N: usize>(m: &[[f32; N]; M]) -> Self {
        Self {
            data: m.iter().flatten().copied().collect(),
            shape: (M, N),
        }
    }

    pub fn fill(m: usize, n: usize, value: f32) -> Self {
        Self {
            data: (0..m * n).map(|_| value).collect(),
            shape: (m, n),
        }
    }

    pub fn ones(m: usize, n: usize) -> Self {
        Self::fill(m, n, 1.)
    }

    pub fn zeroes(m: usize, n: usize) -> Self {
        Self::fill(m, n, 0.)
    }

    // TODO(toms): add support for passing rand-num-generator (in order to control with seed)
    pub fn random(m: usize, n: usize, seed: u64) -> Self {
        let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(seed);
        let dist = rand_distr::Uniform::new(-1., 1.);

        Self {
            data: (0..m * n).map(|_| dist.sample(&mut rng)).collect(),
            shape: (m, n),
        }
    }

    pub fn shape(&self) -> (usize, usize) {
        self.shape
    }
}

impl Index<(usize, usize)> for Matrix {
    type Output = f32;
    fn index(&self, (i, j): (usize, usize)) -> &Self::Output {
        let (m, n) = self.shape;
        debug_assert!(i < m && j < n);
        &self.data[i * n..][j]
    }
}
impl IndexMut<(usize, usize)> for Matrix {
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut Self::Output {
        let (m, n) = self.shape;
        debug_assert!(i < m && j < n);
        &mut self.data[i * n..][j]
    }
}

impl Index<usize> for Matrix {
    type Output = [f32];
    fn index(&self, i: usize) -> &Self::Output {
        let (m, n) = self.shape;
        debug_assert!(i < m);
        &self.data[i * n..][..n]
    }
}
impl IndexMut<usize> for Matrix {
    fn index_mut(&mut self, i: usize) -> &mut Self::Output {
        let (m, n) = self.shape;
        debug_assert!(i < m);
        &mut self.data[i * n..][..n]
    }
}

impl fmt::Debug for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut list = f.debug_list();

        let (m, n) = self.shape;
        for i in 0..m {
            let row = &self.data[i * n..][..n];
            list.entry(&row);
        }

        list.finish()
    }
}

impl PartialEq for Matrix {
    fn eq(&self, other: &Self) -> bool {
        self.shape == other.shape && self.data == other.data

        // for i in 0..self.shape.0 {
        //     for j in 0..self.shape.1 {
        //         if self[(i, j)] != other[(i, j)] {
        //             return false;
        //         }
        //     }
        // }
        //
        // true
    }
}

impl Matrix {
    // simple matrix multiplication (row-major order)
    pub fn matmul(&self, other: &Matrix) -> Matrix {
        debug_assert_eq!(self.shape.1, other.shape.0, "Matrix dimensions mismatch");

        let (m, n) = self.shape;
        let (_n, p) = other.shape;

        // TODO(toms): lift this allocation out?
        let mut result = Matrix::zeroes(m, p);

        for k in 0..n {
            for i in 0..m {
                for j in 0..p {
                    result[i][j] += self[i][k] * other[k][j];
                }
            }
        }

        result
    }

    pub fn add(&self, other: &Matrix) -> Matrix {
        debug_assert_eq!(self.shape, other.shape, "Matrix dimensions mismatch");

        let (m, n) = self.shape;

        let mut result = Matrix::zeroes(m, n);

        for i in 0..m {
            for j in 0..n {
                result[(i, j)] = self[(i, j)] + other[(i, j)];
            }
        }

        result
    }
}

fn indices(x: &[f32], c: usize) -> Box<[usize]> {
    let n = x.len();
    let mut indices: Vec<usize> = (0..n).collect();

    indices.sort_unstable_by(|&i, &j| {
        if x[i] < x[j] {
            std::cmp::Ordering::Less
        } else {
            std::cmp::Ordering::Greater
        }
    });

    indices.split_off(n - c).into_boxed_slice()
}

// TODO(toms): separate this into its own module?
impl Matrix {
    // 'random' matrix multiplication (row-major order)
    pub fn rand_matmul(&self, other: &Matrix, factor: f32) -> Matrix {
        let (m, n) = self.shape();
        let (_n, p) = other.shape();

        let a_col_norm: Box<[f32]> = (0..n)
            .map(|n| (0..m).map(|m| self[(m, n)].powi(2)).sum::<f32>().sqrt())
            .collect();

        let b_row_norm: Box<[f32]> = (0..n)
            .map(|n| (0..p).map(|p| other[(n, p)].powi(2)).sum::<f32>().sqrt())
            .collect();

        assert_eq!(a_col_norm.len(), n);
        assert_eq!(a_col_norm.len(), b_row_norm.len());

        let sum_norm = crate::math::inner_product(&a_col_norm, &b_row_norm, 0.);

        let prob: Box<[f32]> = (0..n)
            .into_iter()
            .map(|n| a_col_norm[n] * b_row_norm[n] / sum_norm)
            .collect();

        // TODO(toms): assert that elements in `prob` sum to 1.

        debug_assert!(0. <= factor && factor <= 1.);
        let c = ((n as f32 * factor).ceil() as usize).min(n);
        // println!("n={n} c={c}");

        // TODO(toms): lift this allocation out?
        let mut result = Matrix::zeroes(m, p);

        for &t in indices(&prob, c).into_iter() {
            let c = c as f32;
            let pt = 1. / (c * prob[t]);

            for i in 0..m {
                for j in 0..p {
                    result[(i, j)] += pt * self[(i, t)] * other[(t, j)];
                }
            }
        }

        result
    }
}
