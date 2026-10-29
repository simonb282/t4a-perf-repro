// Micro-benchmarks of three per-call costs in tensor4all-core / tensor4all-tensorbackend at rev 937f4738.
//  (1) finiteness validation of a complex matrix before rrLU: Scalar::abs_val (Complex64::norm = hypot) per entry,
//      as in matrixlu.rs:474-489 (validate_luci_matrix, two passes) + :871 (factorize_mut_in_place, third pass),
//      against one pass on the components that falls back to the exact |z| test only for a component above
//      f64::MAX / sqrt(2), so it rejects exactly the values the upstream passes reject.
//  (2) tensorbackend::mat_mul(&a, &b) against faer::linalg::matmul::matmul on borrowed column-major slices, and
//      against mat_mul_owned (the caller clones the operands, whose cost is printed) and Session::mat_mul on one
//      held session. mat_mul_owned (with the clones) and Session::mat_mul take the time of mat_mul, so the session
//      entry costs nothing measurable and mat_mul minus the clone time estimates the cost of the product itself.
//  (3) IdxTensor::dense_host_values (owned copy) against IdxTensor::with_dense_slice (borrow).
// Exits 1 if a fast variant disagrees with the upstream one beyond the stated tolerance.
use std::time::Instant;

use faer::linalg::matmul::matmul;
use faer::{Accum, MatMut, MatRef, Par};
use num_complex::Complex64;
use rand::{Rng, SeedableRng};
use tensor4all_core::index::DynIndex;
use tensor4all_core::{IdxTensor, Scalar};
use tensor4all_tensorbackend::{mat_mul, mat_mul_owned, CpuBackend, CpuExecutionContext, Matrix};

fn time<R>(reps: usize, mut f: impl FnMut() -> R) -> (f64, R) {
    let mut best = f64::INFINITY;
    let mut out = None;
    for _ in 0..3 {
        let t = Instant::now();
        for _ in 0..reps - 1 {
            std::hint::black_box(f());
        }
        out = Some(std::hint::black_box(f()));
        best = best.min(t.elapsed().as_secs_f64() / reps as f64);
    }
    (best, out.unwrap())
}

fn rand_c(rng: &mut impl Rng, n: usize) -> Vec<Complex64> {
    (0..n).map(|_| Complex64::new(rng.random::<f64>() - 0.5, rng.random::<f64>() - 0.5)).collect()
}

fn main() {
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(11);
    let mut ok = true;

    // ---------- (1) validation passes ----------
    println!("(1) finiteness validation of an n x n Complex64 matrix (rrLU input)");
    for &n in &[64usize, 141, 282] {
        let a = rand_c(&mut rng, n * n);
        let reps = 2_000_000 / (n * n) + 3;
        // upstream: is_nan pass, abs_val().is_finite() pass (validate_luci_matrix), abs_val pass (factorize_mut_in_place)
        let (t_up, r_up) = time(reps, || {
            let nan = a.iter().any(|&v| Scalar::is_nan(v));
            let inf = a.iter().any(|&v| !v.abs_val().is_finite());
            let inf2 = a.iter().any(|&v| !v.abs_val().is_finite());
            (nan, inf, inf2)
        });
        // one pass on the components; |z| overflows only if max(|re|,|im|) > MAX/sqrt(2), so test that bound too
        const BOUND: f64 = f64::MAX / std::f64::consts::SQRT_2;
        let (t_fast, r_fast) = time(reps, || {
            let bad = a.iter().any(|v| !(v.re.abs() <= BOUND && v.im.abs() <= BOUND) && !v.abs_val().is_finite());
            (bad, bad, bad)
        });
        ok &= r_up == (false, false, false) && r_fast == r_up;
        // the rank-revealing LU itself on the same matrix, for scale (rank capped at n)
        let m = Matrix::from_col_major_vec(n, n, a.clone());
        let (t_lu, _) = time((reps / 50).max(3), || tensor4all_core::matrixlu::rrlu_pivots(m.clone(), None).map(|p| p.rank).unwrap_or(0));
        println!(
            "  n = {n:3}: upstream 3 passes {:8.2} us | 1 component pass {:7.2} us | x{:.1} | rrlu_pivots (full rank, incl. the passes) {:9.1} us",
            1e6 * t_up, 1e6 * t_fast, t_up / t_fast, 1e6 * t_lu
        );
    }
    // Semantics of the bound: a value that hypot sends to infinity is caught, a finite one is not.
    let edge = [Complex64::new(1e308, 1e308), Complex64::new(f64::MAX / 2.0, 0.0), Complex64::new(f64::NAN, 0.0), Complex64::new(0.0, f64::INFINITY), Complex64::new(1.5e308, 1.5e308), Complex64::new(1.3e308, 0.0)];
    for z in edge {
        let up = !z.abs_val().is_finite();
        const BOUND: f64 = f64::MAX / std::f64::consts::SQRT_2;
        let fast = !(z.re.abs() <= BOUND && z.im.abs() <= BOUND) && !z.abs_val().is_finite();
        // the bound is conservative: it rejects only values within a factor sqrt(2) of f64::MAX that hypot keeps
        println!("  edge {z:e}: upstream rejects {up}, component test rejects {fast}");
        ok &= up == fast;
    }

    // ---------- (2) mat_mul ----------
    println!("(2) C = A B, Complex64, one thread: tensorbackend::mat_mul(&a,&b) vs faer matmul on borrowed slices");
    for &(m, k, n) in &[(1usize, 64usize, 128usize), (1, 141, 282), (8, 141, 282), (141, 282, 141)] {
        let a = rand_c(&mut rng, m * k);
        let b = rand_c(&mut rng, k * n);
        let am = Matrix::from_col_major_vec(m, k, a.clone());
        let bm = Matrix::from_col_major_vec(k, n, b.clone());
        let reps = (20_000_000 / (m * k * n + 2000)).max(5);
        let (t_up, c_up) = time(reps, || mat_mul(&am, &bm).unwrap());
        let (t_own, _) = time(reps, || mat_mul_owned(am.clone(), bm.clone()).unwrap());
        let (t_clone, _) = time(reps, || (am.clone(), bm.clone()));
        let ctx = CpuExecutionContext::from_backend(CpuBackend::with_threads(1).unwrap());
        let (t_sess, _) = ctx
            .with_concrete_session(|session| time(reps, || session.mat_mul(&am, &bm).unwrap()))
            .unwrap();
        println!("      mat_mul_owned(clone, clone) {:9.2} us (the two clones alone {:7.2} us) | session.mat_mul, one session {:9.2} us", 1e6 * t_own, 1e6 * t_clone, 1e6 * t_sess);
        let mut c = vec![Complex64::new(0.0, 0.0); m * n];
        let (t_f, _) = time(reps, || {
            matmul(
                MatMut::from_column_major_slice_mut(&mut c, m, n),
                Accum::Replace,
                MatRef::from_column_major_slice(&a, m, k),
                MatRef::from_column_major_slice(&b, k, n),
                Complex64::new(1.0, 0.0),
                Par::Seq,
            );
        });
        let up = c_up.as_col_major_slice();
        let scale = up.iter().map(|z| z.norm()).fold(0.0, f64::max);
        let maxdiff = up.iter().zip(&c).map(|(x, y)| (x - y).norm()).fold(0.0, f64::max) / scale;
        let bitwise = up.iter().zip(&c).filter(|(x, y)| x == y).count();
        ok &= maxdiff < 1e-14;
        println!(
            "  {m:3}x{k:3} * {k:3}x{n:3}: mat_mul {:9.2} us | faer borrowed {:9.2} us | x{:.1} | max rel diff {maxdiff:.1e}, bitwise equal {bitwise}/{}",
            1e6 * t_up, 1e6 * t_f, t_up / t_f, m * n
        );
    }

    // ---------- (3) dense_host_values ----------
    println!("(3) reading the payload of a chi x 2 x chi Complex64 core");
    for &chi in &[32usize, 64, 141] {
        let i = DynIndex::new_dyn(chi);
        let s = DynIndex::new_dyn(2);
        let j = DynIndex::new_dyn(chi);
        let data = rand_c(&mut rng, chi * 2 * chi);
        let t = IdxTensor::from_dense(vec![i, s, j], data.clone()).unwrap();
        let reps = (4_000_000 / (chi * chi)).max(5);
        let (t_own, s_own) = time(reps, || {
            let v = t.dense_host_values::<Complex64>().unwrap();
            v[v.len() / 2]
        });
        let (t_ref, s_ref) = time(reps, || t.with_dense_slice::<Complex64, _>(|v| v[v.len() / 2]).unwrap());
        ok &= s_own == s_ref && s_own == data[data.len() / 2];
        println!(
            "  chi = {chi:3} ({:6} KiB): dense_host_values {:8.2} us | with_dense_slice {:6.2} us | x{:.0}",
            chi * chi * 2 * 16 / 1024, 1e6 * t_own, 1e6 * t_ref, t_own / t_ref
        );
    }
    println!("all checks hold: {ok}");
    if !ok {
        std::process::exit(1);
    }
}
