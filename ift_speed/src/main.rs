//! Inverse quantics Fourier transform of a saved comb TreeTN, the way an application applies it: d branches of
//! r binary sites hanging from a dimension-1 junction (node names: site (level i, branch j) = i·d + j, junction r·d);
//! for each branch, canonicalize at its leaf, apply the transposed QFT MPO with a tree zip-up centred on the junction,
//! then one `truncate` of the whole tree towards the junction. Mode `readout` instead evaluates N pseudo-random points
//! with `TreeTNCachedEvaluator` centred on the junction.
//!
//! Prints phase wall times, peak RSS (VmHWM) and a fingerprint (FNV-1a over the bits of every output tensor, node
//! order, plus the bond dimensions).
//!
//! Usage: ift-speed ift <file.h5> <r> <d> <eps_zipup> <eps_final> [reps]
//!        ift-speed readout <file.h5> <r> <d> <n_points> [reps]
//! With feature `profile`, PPROF=<out.svg> writes a flame graph (pprof-rs, 199 Hz).

use std::time::Instant;

use anyhow::{Context, Result, anyhow};
use num_complex::Complex64;
use tensor4all_core::{ColMajorArrayRef, IdxTensor, SvdTruncationPolicy};
use tensor4all_quanticstransform::{FourierOptions, quantics_fourier_operator};
use tensor4all_treetn::{
    ApplyOptions, CachedEvaluatorOptions, CanonicalizationOptions, EvaluationHint, TreeTN,
    TreeTNCachedEvaluator, TruncationOptions, apply_linear_operator,
};

fn peak_rss_mib() -> f64 {
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    status
        .lines()
        .find(|l| l.starts_with("VmHWM:"))
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|kb| kb.parse::<f64>().ok())
        .map_or(f64::NAN, |kb| kb / 1024.0)
}

fn fnv(hash: &mut u64, word: u64) {
    for byte in word.to_le_bytes() {
        *hash ^= u64::from(byte);
        *hash = hash.wrapping_mul(0x100_0000_01B3);
    }
}

fn fingerprint(tn: &TreeTN<IdxTensor, usize>, n_nodes: usize) -> Result<(u64, usize)> {
    let mut hash = 0xCBF2_9CE4_8422_2325u64;
    let mut max_dim = 0;
    for v in 0..n_nodes {
        let node = tn.node_index(&v).ok_or_else(|| anyhow!("node {v}"))?;
        let t = tn.tensor(node).ok_or_else(|| anyhow!("tensor {v}"))?;
        for &dim in &t.dims() {
            fnv(&mut hash, dim as u64);
            max_dim = max_dim.max(dim);
        }
        for z in t.to_vec::<Complex64>()? {
            fnv(&mut hash, z.re.to_bits());
            fnv(&mut hash, z.im.to_bits());
        }
    }
    Ok((hash, max_dim))
}

fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

struct Profiler(#[cfg(feature = "profile")] Option<(pprof::ProfilerGuard<'static>, String)>);

impl Profiler {
    fn start() -> Self {
        #[cfg(feature = "profile")]
        {
            Self(std::env::var("PPROF").ok().map(|out| {
                let guard = pprof::ProfilerGuardBuilder::default()
                    .frequency(199)
                    .blocklist(&["vdso"])
                    .build()
                    .expect("pprof");
                (guard, out)
            }))
        }
        #[cfg(not(feature = "profile"))]
        Self()
    }

    fn finish(self) {
        #[cfg(feature = "profile")]
        if let Some((guard, out)) = self.0 {
            let report = guard.report().build().expect("pprof report");
            let file = std::fs::File::create(&out).expect("svg");
            report.flamegraph(file).expect("flamegraph");
            // Folded stacks too, for scripted shares.
            let mut folded = String::new();
            for (frames, count) in &report.data {
                let names: Vec<String> = frames
                    .frames
                    .iter()
                    .rev()
                    .flat_map(|f| f.iter().map(|s| format!("{}@{}", s.name(), s.lineno.unwrap_or(0))))
                    .collect();
                folded.push_str(&format!("{};{} {count}\n", frames.thread_name, names.join(";")));
            }
            std::fs::write(format!("{out}.folded"), folded).expect("folded");
        }
    }
}

fn ift(path: &str, r: usize, d: usize, eps_zipup: f64, eps_final: f64) -> Result<()> {
    let loaded = tensor4all_hdf5::load_treetn::<usize>(path, "state").context("load")?;
    ift_tree(loaded, r, d, eps_zipup, eps_final)
}

/// A low-rank real comb from TreeTCI: the comb_speed target (sum of `m` narrow 3D Lorentzians at fixed
/// pseudo-random centres; site (i, j) carries bit i of axis j, leaf = least significant), tolerance `tol`, seed 7.
fn lorentzian_comb(r: usize, m: usize, tol: f64) -> Result<TreeTN<IdxTensor, usize>> {
    use tensor4all_treetci::{
        DefaultProposer, GlobalIndexBatch, TreeTciEdge, TreeTciGraph, TreeTciOptions, crossinterpolate2,
    };
    let d = 3usize;
    let junction = r * d;
    let mut local_dims = vec![2usize; junction];
    local_dims.push(1);
    let mut edges = Vec::new();
    for j in 0..d {
        for i in 0..r - 1 {
            edges.push(TreeTciEdge::new(i * d + j, (i + 1) * d + j));
        }
        edges.push(TreeTciEdge::new((r - 1) * d + j, junction));
    }
    let graph = TreeTciGraph::new(junction + 1, &edges)?;
    let mut s = 12345u64;
    let mut unit = move || (splitmix(&mut s) >> 11) as f64 / (1u64 << 53) as f64;
    // (centre, 1/w², amplitude) per bump.
    let bumps: Vec<([f64; 3], f64, f64)> = (0..m)
        .map(|_| {
            let c = [0.2 + 0.6 * unit(), 0.2 + 0.6 * unit(), 0.2 + 0.6 * unit()];
            let w = 0.01 + 0.04 * unit();
            (c, 1.0 / (w * w), 0.5 + unit())
        })
        .collect();
    let scale = 1.0 / (1u64 << r) as f64;
    let f_point = |point: &[usize]| -> f64 {
        let mut x = [0u64; 3];
        for i in 0..r {
            for (j, xj) in x.iter_mut().enumerate() {
                *xj |= (point[i * d + j] as u64) << i;
            }
        }
        let x = x.map(|v| v as f64 * scale);
        bumps
            .iter()
            .map(|(c, inv_w2, a)| {
                let q = (x[0] - c[0]).powi(2) + (x[1] - c[1]).powi(2) + (x[2] - c[2]).powi(2);
                a / (1.0 + q * inv_w2)
            })
            .sum()
    };
    let mut pivot = vec![0usize; junction + 1];
    for (j, &c) in bumps[0].0.iter().enumerate() {
        let xi = (c / scale) as u64;
        for i in 0..r {
            pivot[i * d + j] = ((xi >> i) & 1) as usize;
        }
    }
    let evaluate = |batch: GlobalIndexBatch<'_>| -> anyhow::Result<Vec<f64>> {
        Ok(batch.data().chunks_exact(batch.n_sites()).map(f_point).collect())
    };
    let options = TreeTciOptions {
        tolerance: tol,
        max_iter: 20,
        seed: Some(7),
        evaluation_cache_bytes: Some(4 << 30),
        enable_global_pivots: true,
        scale_tolerance: true,
        ..TreeTciOptions::default()
    };
    let run = crossinterpolate2(evaluate, local_dims, graph, vec![pivot], options, Some(junction), &DefaultProposer, &mut ())?;
    eprintln!("tci rank {} err {:.3e}", run.ranks.iter().max().unwrap_or(&0), run.errors.last().unwrap_or(&f64::NAN));
    Ok(run.treetn)
}

fn ift_tree(loaded: TreeTN<IdxTensor, usize>, r: usize, d: usize, eps_zipup: f64, eps_final: f64) -> Result<()> {
    let t0 = Instant::now();
    let junction = r * d;
    let mut tn = loaded.canonicalize([junction], CanonicalizationOptions::default())?;
    let t_load = t0.elapsed().as_secs_f64();

    let op = quantics_fourier_operator(
        r,
        FourierOptions {
            sign: 1.0,
            tolerance: 1e-15,
            max_bond_dim: None,
            ..FourierOptions::forward()
        },
    )?
    .transpose();
    let options = ApplyOptions {
        center: Some(junction),
        ..ApplyOptions::zipup().with_svd_policy(SvdTruncationPolicy::new(eps_zipup))
    };
    let prof = Profiler::start();
    let (mut t_canon, mut t_prep, mut t_apply) = (0.0, 0.0, 0.0);
    let t_all = Instant::now();
    for axis in 0..d {
        let leaf = axis; // level 0 of branch `axis`
        let c = Instant::now();
        tn = tn.canonicalize([leaf], CanonicalizationOptions::forced())?;
        t_canon += c.elapsed().as_secs_f64();
        let c = Instant::now();
        // Operator node b acts on level r − 1 − b of the branch.
        let mapping: Vec<(usize, usize)> = (0..r).map(|b| (b, (r - 1 - b) * d + axis)).collect();
        let mut on_branch = op.clone().rename_nodes(&mapping)?;
        on_branch.align_to_state(&tn)?;
        t_prep += c.elapsed().as_secs_f64();
        let c = Instant::now();
        tn = apply_linear_operator(&on_branch, &tn, options.clone())?;
        t_apply += c.elapsed().as_secs_f64();
    }
    let t_zip = t_all.elapsed().as_secs_f64();
    let c = Instant::now();
    tn = tn.truncate(
        [junction],
        TruncationOptions::default().with_svd_policy(SvdTruncationPolicy::new(eps_final)),
    )?;
    let t_trunc = c.elapsed().as_secs_f64();
    prof.finish();
    let (hash, max_dim) = fingerprint(&tn, junction + 1)?;
    println!(
        "RESULT ift load {t_load:.2}s zipup {t_zip:.2}s (canon {t_canon:.2} prep {t_prep:.2} apply {t_apply:.2}) \
         truncate {t_trunc:.2}s total {:.2}s max_dim {max_dim} peak_rss {:.0} MiB fingerprint {hash:016x}",
        t_zip + t_trunc,
        peak_rss_mib()
    );
    Ok(())
}

/// A random complex comb: d branches of r binary sites, level i of branch j is node i·d + j (level 0 the leaf),
/// bond dimension min(2^(i+1), chi_j) above level i, a dimension-1 junction r·d. Site index first, then the bond
/// to the child (towards the leaf), then the bond to the parent: the layout of a tree canonicalized at the junction.
/// With `decay > 0`, entry b of every bond towards the parent is weighted by exp(−decay·b), so the bond spectra decay
/// geometrically and the comb compresses like a smooth state; `decay = 0` leaves the random entries unweighted.
fn synthetic_comb(r: usize, chis: &[usize], decay: f64, seed: u64) -> Result<TreeTN<IdxTensor, usize>> {
    use tensor4all_core::{DynIndex, IndexLike};
    let d = chis.len();
    let mut state = seed;
    let mut random = move || {
        let x = splitmix(&mut state);
        (x >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    };
    let bond_dim = |i: usize, j: usize| (1usize << (i + 1).min(40)).min(chis[j]);
    let bonds: Vec<Vec<DynIndex>> = (0..d)
        .map(|j| (0..r).map(|i| DynIndex::new_dyn(bond_dim(i, j))).collect())
        .collect();
    let mut tensors = Vec::with_capacity(r * d + 1);
    for i in 0..r {
        for branch in &bonds {
            let mut indices = vec![DynIndex::new_dyn(2)];
            if i > 0 {
                indices.push(branch[i - 1].clone());
            }
            indices.push(branch[i].clone());
            let len: usize = indices.iter().map(|x| x.dim()).product();
            // Column-major: the bond to the parent is the slowest index.
            let below = len / branch[i].dim();
            let data: Vec<Complex64> = (0..len)
                .map(|k| {
                    let z = Complex64::new(random(), random());
                    if decay > 0.0 { z * (-decay * (k / below) as f64).exp() } else { z }
                })
                .collect();
            tensors.push(IdxTensor::from_dense(indices, data)?);
        }
    }
    let mut indices = vec![DynIndex::new_dyn(1)];
    indices.extend(bonds.iter().map(|branch| branch[r - 1].clone()));
    let len: usize = indices.iter().map(|x| x.dim()).product();
    let data: Vec<Complex64> = (0..len).map(|_| Complex64::new(random(), random())).collect();
    tensors.push(IdxTensor::from_dense(indices, data)?);
    Ok(TreeTN::from_tensors(tensors, (0..=r * d).collect())?)
}

/// `n_lines` lines of `per_line` consecutive grid points along axis 0 (the other two axes fixed at random values per
/// line), each line one batch with the hint at the top of branch 0 (the node next to the junction), the way a caller
/// reads lines and planes. Site (i, j) carries bit i of axis j (leaf = least significant).
fn line_readout(path: &str, r: usize, d: usize, n_lines: usize, per_line: usize) -> Result<()> {
    let tn = tensor4all_hdf5::load_treetn::<usize>(path, "state").context("load")?;
    line_readout_tree(tn, r, d, n_lines, per_line)
}

fn line_readout_tree(
    tn: TreeTN<IdxTensor, usize>,
    r: usize,
    d: usize,
    n_lines: usize,
    per_line: usize,
) -> Result<()> {
    let junction = r * d;
    let mut sites = Vec::new();
    for v in 0..=junction {
        let space = tn.site_space(&v).ok_or_else(|| anyhow!("site space {v}"))?;
        sites.extend(space.iter().cloned());
    }
    let n_idx = sites.len();
    let mut ev = TreeTNCachedEvaluator::new(
        &tn,
        &sites,
        CachedEvaluatorOptions {
            center: Some(junction),
            max_batch_points: std::env::var("MAX_BATCH").ok().and_then(|s| s.parse().ok()),
            branch_slice_cache_max_bytes: std::env::var("SLICE_CAP")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(CachedEvaluatorOptions::<usize>::default().branch_slice_cache_max_bytes),
            // MSG_CAP=<bytes> overrides the per-edge message cache budget (default 8 MiB).
            message_cache_max_bytes: std::env::var("MSG_CAP")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(CachedEvaluatorOptions::<usize>::default().message_cache_max_bytes),
            ..CachedEvaluatorOptions::default()
        },
    )?;
    let mut state = 11u64;
    // HINT=leaf centres the lines at the leaf of branch 0 (low bits vary fastest), else at its top.
    let top = if std::env::var("HINT").as_deref() == Ok("leaf") { 0 } else { (r - 1) * d };
    let prof = Profiler::start();
    let c = Instant::now();
    let mut hash = 0xCBF2_9CE4_8422_2325u64;
    // PLANE=1: a plane as an application reads it, line by line: every line covers the same x range around the
    // grid centre, y steps by one per line, z is fixed.
    let plane = std::env::var_os("PLANE").is_some();
    let half = 1u64 << (r - 1);
    for line in 0..n_lines as u64 {
        let (fixed, start) = if plane {
            ([0u64, half - n_lines as u64 / 2 + line, half], half - per_line as u64 / 2)
        } else {
            (
                [0u64, splitmix(&mut state) >> (64 - r), splitmix(&mut state) >> (64 - r)],
                splitmix(&mut state) >> (64 - r),
            )
        };
        let mut columns = Vec::with_capacity(n_idx * per_line);
        for p in 0..per_line as u64 {
            let x = [(start + p) & ((1u64 << r) - 1), fixed[1], fixed[2]];
            for i in 0..r {
                for xj in x {
                    columns.push(((xj >> i) & 1) as usize);
                }
            }
            columns.push(0);
        }
        let shape = [n_idx, per_line];
        let batch = ColMajorArrayRef::new(&columns, &shape)?;
        for z in ev.evaluate_batched_typed::<Complex64>(batch, EvaluationHint::around(top))? {
            fnv(&mut hash, z.re.to_bits());
            fnv(&mut hash, z.im.to_bits());
        }
    }
    let t = c.elapsed().as_secs_f64();
    prof.finish();
    println!("RESULT lines {n_lines}x{per_line} eval {t:.3}s peak_rss {:.0} MiB fingerprint {hash:016x}", peak_rss_mib());
    Ok(())
}

fn readout(path: &str, r: usize, d: usize, n_points: usize) -> Result<()> {
    let tn = tensor4all_hdf5::load_treetn::<usize>(path, "state").context("load")?;
    readout_tree(tn, r, d, n_points)
}

fn readout_tree(tn: TreeTN<IdxTensor, usize>, r: usize, d: usize, n_points: usize) -> Result<()> {
    let junction = r * d;
    // NO_CANON=1 evaluates the tree as built (the canonicalizing QR is not bitwise across thread counts, #1162).
    let tn = if std::env::var_os("NO_CANON").is_some() {
        tn
    } else {
        tn.canonicalize([junction], CanonicalizationOptions::default())?
    };
    // Site indices: the branch nodes' in node order, then the junction's.
    let mut sites = Vec::new();
    for v in 0..=junction {
        let space = tn.site_space(&v).ok_or_else(|| anyhow!("site space {v}"))?;
        // One site index per node on this tree.
        sites.extend(space.iter().cloned());
    }
    let n_idx = sites.len();
    let jt = tn.tensor(tn.node_index(&junction).ok_or_else(|| anyhow!("junction"))?).ok_or_else(|| anyhow!("junction tensor"))?;
    eprintln!("junction dims {:?}", jt.dims());
    let mut state = 7u64;
    let mut columns = Vec::with_capacity(n_idx * n_points);
    for _ in 0..n_points {
        for k in 0..n_idx {
            columns.push(if k < junction { (splitmix(&mut state) & 1) as usize } else { 0 });
        }
    }
    let prof = Profiler::start();
    let c = Instant::now();
    let mut ev = TreeTNCachedEvaluator::new(
        &tn,
        &sites,
        CachedEvaluatorOptions {
            center: Some(junction),
            // SLICE_CAP=<bytes> overrides the branch-slice cache cap (default 32 MiB).
            branch_slice_cache_max_bytes: std::env::var("SLICE_CAP")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(CachedEvaluatorOptions::<usize>::default().branch_slice_cache_max_bytes),
            // MAX_BATCH=<points> overrides the per-chunk point limit (default: derived from the largest tensor).
            max_batch_points: std::env::var("MAX_BATCH").ok().and_then(|s| s.parse().ok()),
            ..CachedEvaluatorOptions::default()
        },
    )?;
    let t_new = c.elapsed().as_secs_f64();
    let c = Instant::now();
    let shape = [n_idx, n_points];
    let batch = ColMajorArrayRef::new(&columns, &shape)?;
    let values = ev.evaluate_batched_typed::<Complex64>(batch, EvaluationHint::default())?;
    let t_eval = c.elapsed().as_secs_f64();
    prof.finish();
    let mut hash = 0xCBF2_9CE4_8422_2325u64;
    for z in &values {
        fnv(&mut hash, z.re.to_bits());
        fnv(&mut hash, z.im.to_bits());
    }
    println!(
        "RESULT readout n {n_points} new {t_new:.3}s eval {t_eval:.3}s peak_rss {:.0} MiB fingerprint {hash:016x}",
        peak_rss_mib()
    );
    Ok(())
}

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let num = |i: usize| -> Result<f64> {
        a.get(i).ok_or_else(|| anyhow!("missing argument {i}"))?.parse::<f64>().context("number")
    };
    match a.get(1).map(String::as_str) {
        Some("ift") => {
            let reps = a.get(7).map_or(Ok(1), |s| s.parse()).context("reps")?;
            for _ in 0..reps {
                ift(&a[2], num(3)? as usize, num(4)? as usize, num(5)?, num(6)?)?;
            }
        }
        Some("readout") => {
            let reps = a.get(6).map_or(Ok(1), |s| s.parse()).context("reps")?;
            for _ in 0..reps {
                readout(&a[2], num(3)? as usize, num(4)? as usize, num(5)? as usize)?;
            }
        }
        Some("matmul") => {
            // matmul <m> <k> <n>: hash of tensor4all_tensorbackend::mat_mul of seeded complex matrices.
            use tensor4all_tensorbackend::{Matrix, mat_mul};
            let (m, k, n) = (num(2)? as usize, num(3)? as usize, num(4)? as usize);
            let mut state = 7u64;
            let mut random = || {
                let x = splitmix(&mut state);
                (x >> 11) as f64 / (1u64 << 53) as f64 - 0.5
            };
            let mut fill = |len: usize| -> Vec<Complex64> {
                (0..len).map(|_| Complex64::new(random(), random())).collect()
            };
            let a = Matrix::from_col_major_vec(m, k, fill(m * k));
            let b = Matrix::from_col_major_vec(k, n, fill(k * n));
            let c = mat_mul(&a, &b)?;
            let mut hash = 0xCBF2_9CE4_8422_2325u64;
            for z in c.as_col_major_slice() {
                fnv(&mut hash, z.re.to_bits());
                fnv(&mut hash, z.im.to_bits());
            }
            println!("RESULT matmul {m}x{k}x{n} fingerprint {hash:016x}");
        }
        Some("matmul-f64") => {
            use tensor4all_tensorbackend::{Matrix, mat_mul};
            let (m, k, n) = (num(2)? as usize, num(3)? as usize, num(4)? as usize);
            let mut state = 7u64;
            let mut fill = |len: usize| -> Vec<f64> {
                (0..len).map(|_| (splitmix(&mut state) >> 11) as f64 / (1u64 << 53) as f64 - 0.5).collect()
            };
            let a = Matrix::from_col_major_vec(m, k, fill(m * k));
            let b = Matrix::from_col_major_vec(k, n, fill(k * n));
            let c = mat_mul(&a, &b)?;
            let mut hash = 0xCBF2_9CE4_8422_2325u64;
            for x in c.as_col_major_slice() {
                fnv(&mut hash, x.to_bits());
            }
            println!("RESULT matmul-f64 {m}x{k}x{n} fingerprint {hash:016x}");
        }
        Some("contract") => {
            // contract <m> <k> <n>: the same complex product through IdxTensor contraction (the einsum route).
            use tensor4all_core::{DynIndex, TensorContractionLike};
            let (m, k, n) = (num(2)? as usize, num(3)? as usize, num(4)? as usize);
            let mut state = 7u64;
            let mut random = || {
                let x = splitmix(&mut state);
                (x >> 11) as f64 / (1u64 << 53) as f64 - 0.5
            };
            let mut fill = |len: usize| -> Vec<Complex64> {
                (0..len).map(|_| Complex64::new(random(), random())).collect()
            };
            let (i, kk, j) = (DynIndex::new_dyn(m), DynIndex::new_dyn(k), DynIndex::new_dyn(n));
            let a = IdxTensor::from_dense(vec![i, kk.clone()], fill(m * k))?;
            let b = IdxTensor::from_dense(vec![kk, j], fill(k * n))?;
            let c = a.contract_pair(&b)?;
            let mut hash = 0xCBF2_9CE4_8422_2325u64;
            for z in c.to_vec::<Complex64>()? {
                fnv(&mut hash, z.re.to_bits());
                fnv(&mut hash, z.im.to_bits());
            }
            println!("RESULT contract {m}x{k}x{n} fingerprint {hash:016x}");
        }
        Some("line-readout") => {
            // line-readout <file.h5> <r> <d> <n_lines> <per_line>
            line_readout(&a[2], num(3)? as usize, num(4)? as usize, num(5)? as usize, num(6)? as usize)?;
        }
        Some("synthetic-lines") => {
            // synthetic-lines <r> <chi0,chi1,chi2> <n_lines> <per_line>: line-readout on the random comb (not canonicalized).
            let r = num(2)? as usize;
            let chis: Vec<usize> = a
                .get(3)
                .ok_or_else(|| anyhow!("missing chis"))?
                .split(',')
                .map(|c| c.parse().context("chi"))
                .collect::<Result<_>>()?;
            let tn = synthetic_comb(r, &chis, 0.0, 7)?;
            line_readout_tree(tn, r, chis.len(), num(4)? as usize, num(5)? as usize)?;
        }
        Some("synthetic-ift") => {
            // synthetic-ift <r> <m> <tol>: TreeTCI of m Lorentzians, then the ift procedure (eps 1e-8 / 1e-9).
            let r = num(2)? as usize;
            let tn = lorentzian_comb(r, num(3)? as usize, num(4)?)?;
            ift_tree(tn, r, 3, 1e-8, 1e-9)?;
        }
        Some("synthetic-ift-random") => {
            // synthetic-ift-random <r> <chi0,chi1,chi2> <decay>: the random comb with bond spectra decaying as
            // exp(−decay·b), then the ift procedure (eps 1e-8 / 1e-9). PRETRUNC=<rtol> compresses it first.
            let r = num(2)? as usize;
            let chis: Vec<usize> = a
                .get(3)
                .ok_or_else(|| anyhow!("missing chis"))?
                .split(',')
                .map(str::parse)
                .collect::<Result<_, _>>()?;
            let mut tn = synthetic_comb(r, &chis, num(4)?, 7)?;
            // PRETRUNC=<rtol>: compress the comb once first, so that it enters the transform near its optimal
            // bonds, as the output of an approximation algorithm does.
            if let Ok(rtol) = std::env::var("PRETRUNC") {
                tn = tn.truncate(
                    [r * chis.len()],
                    TruncationOptions::default().with_svd_policy(SvdTruncationPolicy::new(rtol.parse()?)),
                )?;
            }
            ift_tree(tn, r, chis.len(), 1e-8, 1e-9)?;
        }
        Some("synthetic-readout") => {
            // synthetic-readout <r> <chi0,chi1,chi2> <n_points>
            let r = num(2)? as usize;
            let chis: Vec<usize> = a
                .get(3)
                .ok_or_else(|| anyhow!("missing chis"))?
                .split(',')
                .map(|c| c.parse().context("chi"))
                .collect::<Result<_>>()?;
            let tn = synthetic_comb(r, &chis, 0.0, 7)?;
            readout_tree(tn, r, chis.len(), num(4)? as usize)?;
        }
        _ => return Err(anyhow!("usage: ift-speed ift|readout <file.h5> <r> <d> ... | synthetic-readout <r> <chis> <n>")),
    }
    Ok(())
}
