// Standalone reproducers for tensor4all-rs @ 9222cbe (public API only).
//
//   comb tci   <r> <K> <rho> <memo> <reps>   treetci memo overhead (memo = none | <bytes>)
//   comb replay <r> <K> <rho> <reps>         replay of treetci's request stream through MultiIndexCache
//   comb eval  <r> <K> <reps>                TreeTNCachedEvaluator: 4096-point vs 28-point batches
//
// The tree is a comb: 3 branches of r nodes (local dim 2, bit i of axis j at node i*3 + j, node (r-1)*3 + j next to
// the junction) on a junction node 3r of local dim 1. The target is
//   f(n) = Σ_{k<K} ρ^k exp(i a_k · n / 2^r),
// a sum of K product terms, so every bond rank is ≤ K (exactly K for ρ = 1; at tolerance ε about log ε / log ρ).
// Its cost is ~3K complex multiply-adds per point from per-axis tables (~0.1-0.3 µs): a cheap target.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::time::{Duration, Instant};

use anyhow::Result;
use num_complex::Complex64 as C;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde_json::json;
use tensor4all_core::{ColMajorArrayRef, DynIndex, IndexLike, MultiIndexCache};
use tensor4all_treetci::{
    DefaultProposer, GlobalIndexBatch, MultiIndex, TreeTciEdge, TreeTciGraph, TreeTciOptions,
    crossinterpolate2,
};
use tensor4all_treetn::{CachedEvaluatorOptions, EvaluationHint, TreeTNCachedEvaluator};

const D: usize = 3;

struct Comb {
    r: usize,
}

impl Comb {
    fn n_nodes(&self) -> usize {
        self.r * D + 1
    }
    fn junction(&self) -> usize {
        self.r * D
    }
    fn local_dims(&self) -> Vec<usize> {
        (0..self.n_nodes()).map(|v| if v < self.junction() { 2 } else { 1 }).collect()
    }
    fn graph(&self) -> TreeTciGraph {
        let mut edges = Vec::new();
        for j in 0..D {
            for i in 0..self.r - 1 {
                edges.push(TreeTciEdge::new(i * D + j, (i + 1) * D + j));
            }
            edges.push(TreeTciEdge::new((self.r - 1) * D + j, self.junction()));
        }
        TreeTciGraph::new(self.n_nodes(), &edges).expect("comb graph")
    }
    /// The axis indices n_j of one point (column of the batch).
    fn axes(&self, point: &[usize]) -> [usize; D] {
        let mut n = [0usize; D];
        for i in 0..self.r {
            for (j, nj) in n.iter_mut().enumerate() {
                *nj |= point[i * D + j] << i;
            }
        }
        n
    }
    fn point(&self, n: [usize; D]) -> Vec<usize> {
        let mut p = vec![0usize; self.n_nodes()];
        for i in 0..self.r {
            for j in 0..D {
                p[i * D + j] = (n[j] >> i) & 1;
            }
        }
        p
    }
}

/// f(n) = Σ_k ρ^k Π_j tab[j][k][n_j], tab[j][k][n] = exp(i a_kj n / 2^r).
struct Target {
    k: usize,
    tab: Vec<Vec<C>>, // [j][n * k + term]
}

impl Target {
    fn new(r: usize, k: usize, rho: f64) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(1234);
        // Frequency range (env FREQ, default 100): sets how fast the singular values decay.
        let freq: f64 = std::env::var("FREQ").ok().and_then(|s| s.parse().ok()).unwrap_or(100.0);
        let a: Vec<[f64; D]> = (0..k)
            .map(|_| std::array::from_fn(|_| rng.random_range(-freq..freq)))
            .collect();
        let m = 1usize << r;
        let tab = (0..D)
            .map(|j| {
                let mut t = vec![C::new(0.0, 0.0); m * k];
                for n in 0..m {
                    let x = n as f64 / m as f64;
                    for term in 0..k {
                        // The weight ρ^k goes on axis 0.
                        let w = if j == 0 { rho.powi(term as i32) } else { 1.0 };
                        t[n * k + term] = C::from_polar(w, a[term][j] * x);
                    }
                }
                t
            })
            .collect();
        Self { k, tab }
    }
    fn eval(&self, n: [usize; D]) -> C {
        let (t0, t1, t2) = (
            &self.tab[0][n[0] * self.k..][..self.k],
            &self.tab[1][n[1] * self.k..][..self.k],
            &self.tab[2][n[2] * self.k..][..self.k],
        );
        let mut s = C::new(0.0, 0.0);
        for term in 0..self.k {
            s += t0[term] * t1[term] * t2[term];
        }
        s
    }
}

fn hash_bits(h: &mut DefaultHasher, v: &[C]) {
    for c in v {
        c.re.to_bits().hash(h);
        c.im.to_bits().hash(h);
    }
}

struct TciRun {
    treetn: tensor4all_treetn::TreeTN<tensor4all_core::IdxTensor, usize>,
    total: Duration,
    in_f: Duration,
    calls: usize,
    f_points: usize,
    out: serde_json::Value,
}

fn run_tci(
    comb: &Comb,
    target: &Target,
    memo: Option<usize>,
    mut record: Option<&mut Vec<Vec<u64>>>,
) -> Result<TciRun> {
    let mut in_f = Duration::ZERO;
    let mut calls = 0usize;
    let mut f_points = 0usize;
    let f = |batch: GlobalIndexBatch<'_>| -> Result<Vec<C>> {
        let clock = Instant::now();
        let ns = batch.n_sites();
        let values: Vec<C> = batch
            .data()
            .chunks_exact(ns)
            .map(|p| target.eval(comb.axes(p)))
            .collect();
        in_f += clock.elapsed();
        calls += 1;
        f_points += batch.n_points();
        if let Some(rec) = record.as_mut() {
            // 3r ≤ 63 bits: the point packed into a u64 (outside the timed part of f).
            rec.push(
                batch
                    .data()
                    .chunks_exact(ns)
                    .map(|p| p[..ns - 1].iter().enumerate().fold(0u64, |k, (s, &b)| k | ((b as u64) << s)))
                    .collect(),
            );
        }
        Ok(values)
    };
    let pivots: Vec<MultiIndex> = vec![
        comb.point([0, 0, 0]),
        comb.point([(1 << comb.r) - 1; D]),
    ];
    let options = TreeTciOptions {
        tolerance: 1e-8,
        max_iter: 30,
        max_bond_dim: None,
        seed: Some(11),
        evaluation_cache_bytes: memo,
        enable_global_pivots: true,
        scale_tolerance: true,
        ..TreeTciOptions::default()
    };
    let clock = Instant::now();
    let run = crossinterpolate2(
        f,
        comb.local_dims(),
        comb.graph(),
        pivots,
        options,
        Some(comb.junction()),
        &DefaultProposer,
    )?;
    let total = clock.elapsed();
    // Bitwise fingerprint of the result: every node tensor's data, in node order, plus ranks and errors.
    let mut h = DefaultHasher::new();
    for node in 0..comb.n_nodes() {
        let ix = run.treetn.node_index(&node).expect("node");
        let t = run.treetn.tensor(ix).expect("tensor");
        hash_bits(&mut h, &t.to_vec::<C>()?);
    }
    run.ranks.hash(&mut h);
    for e in &run.errors {
        e.to_bits().hash(&mut h);
    }
    let ev = &run.evaluation;
    let out = json!({
        "memo_bytes": memo,
        "total_s": total.as_secs_f64(),
        "in_f_s": in_f.as_secs_f64(),
        "own_s": (total - in_f).as_secs_f64(),
        "f_calls": calls,
        "requested": ev.requested_points,
        "evaluated": ev.evaluated_points,
        "cache_hits": ev.cache_hits,
        "cache_misses": ev.cache_misses,
        "dropped_inserts": ev.dropped_inserts,
        "ranks": run.ranks,
        "last_error": run.errors.last(),
        "termination": format!("{:?}", run.termination),
        "result_hash": format!("{:016x}", h.finish()),
    });
    Ok(TciRun { treetn: run.treetn, total, in_f, calls, f_points, out })
}

fn sites_of(comb: &Comb, tn: &tensor4all_treetn::TreeTN<tensor4all_core::IdxTensor, usize>) -> Vec<DynIndex> {
    let mut sites = Vec::new();
    for node in 0..comb.n_nodes() {
        let s: Vec<DynIndex> = tn.site_space(&node).map(|s| s.iter().cloned().collect()).unwrap_or_default();
        assert!(s.len() <= 1, "one site index per node");
        sites.extend(s);
    }
    sites
}

fn env() -> serde_json::Value {
    let read = |p: &str| std::fs::read_to_string(p).map(|s| s.trim().to_string()).unwrap_or_default();
    json!({
        "RAYON_NUM_THREADS": std::env::var("RAYON_NUM_THREADS").ok(),
        "loadavg": read("/proc/loadavg"),
        "governor": read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
        "cur_freq_khz": read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq"),
        "build": env!("CARGO_PKG_NAME"),
        "FREQ": std::env::var("FREQ").ok(),
    })
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let num = |i: usize| -> usize { args[i].parse().expect("integer argument") };
    match args.get(1).map(String::as_str) {
        Some("tci") => {
            let comb = Comb { r: num(2) };
            let target = Target::new(comb.r, num(3), args[4].parse()?);
            let memo = if args[5] == "none" { None } else { Some(num(5)) };
            for rep in 0..num(6) {
                let run = run_tci(&comb, &target, memo, None)?;
                let mut out = run.out;
                out["rep"] = json!(rep);
                out["r"] = json!(comb.r);
                out["K"] = json!(target.k);
                out["rho"] = json!(args[4]);
                out["env"] = env();
                let _ = (run.total, run.in_f, run.calls, run.f_points);
                println!("{out}");
            }
        }
        Some("replay") => {
            // Record treetci's request stream once (memo off, so f sees every requested point), then push the same
            // batches through MultiIndexCache::evaluate_batched, the memo treetci builds (evaluation.rs:220-283),
            // with a trivial evaluator. Only the evaluate_batched call is timed.
            let comb = Comb { r: num(2) };
            let target = Target::new(comb.r, num(3), args[4].parse()?);
            let mut rec = Vec::new();
            let run = run_tci(&comb, &target, None, Some(&mut rec))?;
            let n_req: usize = rec.iter().map(Vec::len).sum();
            let ns = comb.n_nodes();
            let unpack = |keys: &[u64]| -> Vec<usize> {
                let mut v = Vec::with_capacity(keys.len() * ns);
                for &k in keys {
                    for s in 0..ns - 1 {
                        v.push(((k >> s) & 1) as usize);
                    }
                    v.push(0);
                }
                v
            };
            let batches: Vec<Vec<usize>> = rec.iter().map(|b| unpack(b)).collect();
            let mut out = Vec::new();
            for rep in 0..num(5) {
                for limit in [1usize << 30, 0] {
                    let mut cache = MultiIndexCache::<C>::with_retained_byte_limit(&comb.local_dims(), limit)?;
                    let mut t = Duration::ZERO;
                    let mut misses = 0usize;
                    let mut h = DefaultHasher::new();
                    for b in &batches {
                        let shape = [ns, b.len() / ns];
                        let view = ColMajorArrayRef::new(b, &shape)?;
                        let clock = Instant::now();
                        let v = cache
                            .evaluate_batched(view, |m| {
                                misses += m.shape()[1];
                                Ok(vec![C::new(1.0, 0.0); m.shape()[1]])
                            })
                            .map_err(|e| anyhow::anyhow!("{e}"))?;
                        t += clock.elapsed();
                        hash_bits(&mut h, &v);
                    }
                    out.push(json!({
                        "what": "MultiIndexCache::evaluate_batched replay", "limit": limit, "rep": rep,
                        "requests": n_req, "batches": batches.len(), "misses": misses,
                        "s": t.as_secs_f64(), "ns_per_request": t.as_secs_f64() * 1e9 / n_req as f64,
                        "hash": format!("{:016x}", h.finish()), "env": env(),
                    }));
                }
                // Reference: the same memo with a u64 key and FxHash, no lock, pre-sized (what the patch aims at).
                let mut memo: rustc_hash::FxHashMap<u64, C> = rustc_hash::FxHashMap::default();
                memo.reserve(1 << 20);
                let mut t = Duration::ZERO;
                let mut misses = 0usize;
                for keys in &rec {
                    let clock = Instant::now();
                    let mut pending: rustc_hash::FxHashMap<u64, usize> = rustc_hash::FxHashMap::default();
                    pending.reserve(keys.len());
                    let mut slots = Vec::with_capacity(keys.len());
                    let mut miss_keys = Vec::new();
                    for &k in keys {
                        slots.push(match memo.get(&k) {
                            Some(v) => Ok(*v),
                            None => Err(*pending.entry(k).or_insert_with(|| {
                                miss_keys.push(k);
                                miss_keys.len() - 1
                            })),
                        });
                    }
                    misses += miss_keys.len();
                    let vals = vec![C::new(1.0, 0.0); miss_keys.len()];
                    for (k, v) in miss_keys.iter().zip(&vals) {
                        memo.insert(*k, *v);
                    }
                    let v: Vec<C> = slots.into_iter().map(|s| s.unwrap_or_else(|i| vals[i])).collect();
                    t += clock.elapsed();
                    std::hint::black_box(v);
                }
                out.push(json!({
                    "what": "reference FxHashMap<u64> memo, same stream", "rep": rep, "requests": n_req,
                    "misses": misses, "s": t.as_secs_f64(), "ns_per_request": t.as_secs_f64() * 1e9 / n_req as f64,
                }));
            }
            // Reference 2: the FxHashMap<u64> memo, but each key built from the batch's usize columns (bound check +
            // mixed-radix fold per site), as MultiIndexCache must, and no pre-reserve.
            for rep in 0..num(5) {
                let dims = comb.local_dims();
                let mut memo: rustc_hash::FxHashMap<u64, C> = rustc_hash::FxHashMap::default();
                let mut t = Duration::ZERO;
                for b in &batches {
                    let clock = Instant::now();
                    let np = b.len() / ns;
                    let mut pending: rustc_hash::FxHashMap<u64, usize> =
                        rustc_hash::FxHashMap::with_capacity_and_hasher(np, Default::default());
                    let mut slots = Vec::with_capacity(np);
                    let mut miss_keys = Vec::new();
                    for p in b.chunks_exact(ns) {
                        let mut k = 0u64;
                        let mut c = 1u64;
                        for (&v, &d) in p.iter().zip(&dims) {
                            anyhow::ensure!(v < d, "out of bounds");
                            k += c * v as u64;
                            c *= d as u64;
                        }
                        slots.push(match memo.get(&k) {
                            Some(v) => Ok(*v),
                            None => Err(*pending.entry(k).or_insert_with(|| {
                                miss_keys.push(k);
                                miss_keys.len() - 1
                            })),
                        });
                    }
                    let vals = vec![C::new(1.0, 0.0); miss_keys.len()];
                    memo.reserve(miss_keys.len());
                    for (k, v) in miss_keys.iter().zip(&vals) {
                        memo.insert(*k, *v);
                    }
                    let v: Vec<C> = slots.into_iter().map(|s| s.unwrap_or_else(|i| vals[i])).collect();
                    t += clock.elapsed();
                    std::hint::black_box(v);
                }
                out.push(json!({
                    "what": "reference FxHashMap<u64> memo, keys from usize columns, no pre-reserve", "rep": rep,
                    "requests": n_req, "s": t.as_secs_f64(), "ns_per_request": t.as_secs_f64() * 1e9 / n_req as f64,
                }));
            }
            println!("{}", json!({"tci": run.out, "replay": out}));
        }
        Some("eval") => {
            // A comb TreeTN from the TCI of reproducer 1 with K ≤ 12 terms (bonds ≤ K), then the evaluator.
            let comb = Comb { r: num(2) };
            let target = Target::new(comb.r, num(3), 1.0);
            let run = run_tci(&comb, &target, Some(1 << 30), None)?;
            let tn = &run.treetn;
            let sites = sites_of(&comb, tn);
            let ns = sites.len();
            assert_eq!(ns, comb.n_nodes(), "the junction carries a dim-1 site");
            let mut rng = ChaCha8Rng::seed_from_u64(99);
            let m = 1usize << comb.r;
            let total_points = 4096 * 100;
            let rand_points: Vec<[usize; D]> =
                (0..total_points).map(|_| std::array::from_fn(|_| rng.random_range(0..m))).collect();
            // 28-point lines along one axis (a typical small batch shape of a downstream reader).
            let line_points: Vec<[usize; D]> = (0..total_points / 28)
                .flat_map(|l| {
                    let axis = l % D;
                    let base: [usize; D] = std::array::from_fn(|_| rng.random_range(0..m));
                    let mut xs: Vec<usize> = (0..m).collect();
                    for i in 0..28 {
                        let j = rng.random_range(i..m);
                        xs.swap(i, j);
                    }
                    (0..28).map(move |i| {
                        let mut p = base;
                        p[axis] = xs[i];
                        p
                    }).collect::<Vec<_>>()
                })
                .collect();
            let cols = |pts: &[[usize; D]]| -> Vec<usize> { pts.iter().flat_map(|&n| comb.point(n)).collect() };
            let mut out = Vec::new();
            // Accuracy of the tree against f on 4096 points (sanity).
            {
                let mut ev = TreeTNCachedEvaluator::new(
                    tn,
                    &sites,
                    CachedEvaluatorOptions { center: Some(comb.junction()), ..CachedEvaluatorOptions::default() },
                )?;
                let c = cols(&rand_points[..4096]);
                let v: Vec<C> = ev.evaluate_batched_typed(ColMajorArrayRef::new(&c, &[ns, 4096])?, EvaluationHint::default())?;
                let err = v.iter().zip(&rand_points[..4096]).map(|(a, &n)| (a - target.eval(n)).norm()).fold(0.0, f64::max);
                out.push(json!({"what": "max |tree - f| on 4096 random points", "value": err, "ranks": run.out["ranks"]}));
            }
            for rep in 0..num(4) {
                for (name, pts, batch, hinted) in [
                    ("random, batches of 4096", &rand_points, 4096usize, false),
                    ("random, batches of 28", &rand_points, 28, false),
                    ("28-point lines, hint around the line's branch top", &line_points, 28, true),
                ] {
                    let mut ev = TreeTNCachedEvaluator::new(
                        tn,
                        &sites,
                        CachedEvaluatorOptions { center: Some(comb.junction()), ..CachedEvaluatorOptions::default() },
                    )?;
                    let c = cols(pts);
                    let n_pts = pts.len() / batch * batch;
                    let mut h = DefaultHasher::new();
                    let clock = Instant::now();
                    for (bi, chunk) in c[..n_pts * ns].chunks_exact(batch * ns).enumerate() {
                        let hint = if hinted {
                            // Top of the branch of the line's axis, as a reader that hints its batches may pick.
                            EvaluationHint::around((comb.r - 1) * D + (bi % D))
                        } else {
                            EvaluationHint::default()
                        };
                        let v: Vec<C> = ev.evaluate_batched_typed(ColMajorArrayRef::new(chunk, &[ns, batch])?, hint)?;
                        hash_bits(&mut h, &v);
                    }
                    let s = clock.elapsed().as_secs_f64();
                    out.push(json!({
                        "what": name, "rep": rep, "points": n_pts, "calls": n_pts / batch, "s": s,
                        "us_per_point": s * 1e6 / n_pts as f64, "hash": format!("{:016x}", h.finish()), "env": env(),
                    }));
                }
            }
            // Bond dimensions of the tree.
            let mut bonds = Vec::new();
            for a in 0..comb.junction() {
                let b = if a + D < comb.junction() { a + D } else { comb.junction() };
                bonds.push(tn.edge_between(&a, &b).and_then(|e| tn.bond_index(e)).map(|i| i.dim()).unwrap_or(0));
            }
            println!("{}", json!({"bonds_max": bonds.iter().max(), "tci": run.out, "eval": out}));
        }
        _ => eprintln!("usage: repro tci|replay|eval ..."),
    }
    Ok(())
}
