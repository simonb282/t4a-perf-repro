//! `rrlu` (full L and U factors) of a full-rank Complex64 n x n matrix.
//!
//! The factor outputs are scanned for finiteness once per factor
//! (`extract_lu_from_factorized`). The fingerprint is FNV-1a over the bit
//! patterns of L and U, so two builds can be compared for identical results.

use std::time::Instant;

use num_complex::Complex64;
use rand::{Rng, SeedableRng};
use t4a_perf_repro::fingerprint;
use tensor4all_core::matrixlu::rrlu;
use tensor4all_tensorbackend::Matrix;

fn parts(m: &Matrix<Complex64>) -> Vec<f64> {
    m.as_col_major_slice().iter().flat_map(|z| [z.re, z.im]).collect()
}

fn main() {
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(11);
    for &n in &[64usize, 141, 282] {
        let data: Vec<Complex64> = (0..n * n)
            .map(|_| Complex64::new(rng.random::<f64>() - 0.5, rng.random::<f64>() - 0.5))
            .collect();
        let m = Matrix::from_col_major_vec(n, n, data);
        let reps = (2_000_000 / (n * n)).max(3);
        let mut best = f64::INFINITY;
        let mut print = 0u64;
        for _ in 0..3 {
            let start = Instant::now();
            for _ in 0..reps {
                std::hint::black_box(rrlu(&m, None).expect("rrlu"));
            }
            best = best.min(start.elapsed().as_secs_f64() / reps as f64);
        }
        let lu = rrlu(&m, None).expect("rrlu");
        let mut all = parts(&lu.left(true));
        all.extend(parts(&lu.right(true)));
        print ^= fingerprint(&all);
        println!(
            "  n = {n:3}: rrlu {:9.1} us, rank {}, fingerprint {print:016x}",
            1e6 * best,
            lu.npivots()
        );
    }
}
