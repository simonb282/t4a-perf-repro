//! Shared helpers: a deterministic generator, a random chain TreeTN and a
//! bitwise fingerprint of results.

use tensor4all_core::{DynIndex, IdxTensor, IndexLike};
use tensor4all_treetn::TreeTN;

pub type Error = Box<dyn std::error::Error>;

/// Deterministic 64-bit LCG, so every run sees the same inputs.
pub struct Lcg(u64);

impl Lcg {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }

    /// Uniform integer in `0..n`.
    pub fn below(&mut self, n: usize) -> usize {
        ((self.next_u64() >> 33) as usize) % n
    }

    /// Uniform float in `[-0.5, 0.5)`.
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    }
}

/// `n_points` random binary points of `n_sites` sites, column-major
/// (`[n_sites, n_points]`).
pub fn random_points(rng: &mut Lcg, n_sites: usize, n_points: usize) -> Vec<usize> {
    (0..n_sites * n_points).map(|_| rng.below(2)).collect()
}

/// A chain TreeTN of `n_sites` binary sites, all bonds of dimension `bond`,
/// random real entries. Node names are `0..n_sites`.
pub fn chain(
    n_sites: usize,
    bond: usize,
    seed: u64,
) -> Result<(Vec<DynIndex>, TreeTN<IdxTensor, usize>), Error> {
    let mut rng = Lcg::new(seed);
    let sites: Vec<DynIndex> = (0..n_sites).map(|_| DynIndex::new_dyn(2)).collect();
    let bonds: Vec<DynIndex> = (0..n_sites - 1).map(|_| DynIndex::new_dyn(bond)).collect();
    // Keeps the values of order one along the chain.
    let scale = 2.0 / (bond as f64).sqrt();
    let mut tensors = Vec::with_capacity(n_sites);
    for i in 0..n_sites {
        let mut indices = Vec::new();
        if i > 0 {
            indices.push(bonds[i - 1].clone());
        }
        indices.push(sites[i].clone());
        if i + 1 < n_sites {
            indices.push(bonds[i].clone());
        }
        let len: usize = indices.iter().map(|index| index.dim()).product();
        let data: Vec<f64> = (0..len).map(|_| scale * rng.uniform()).collect();
        tensors.push(IdxTensor::from_dense(indices, data)?);
    }
    let tree = TreeTN::from_tensors(tensors, (0..n_sites).collect())?;
    Ok((sites, tree))
}

/// FNV-1a over the bit patterns: equal fingerprints mean bitwise-equal results.
pub fn fingerprint(values: &[f64]) -> u64 {
    values.iter().fold(0xcbf2_9ce4_8422_2325u64, |hash, value| {
        (hash ^ value.to_bits()).wrapping_mul(0x0000_0100_0000_01b3)
    })
}
