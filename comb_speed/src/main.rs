//! Speed reproducer: TreeTCI on a 3-branch quantics comb with a unit-dimension junction.
//!
//! Tree: `d` branches of `r` binary sites, node(i, j) = i·d + j, each branch top joined to a
//! junction site r·d of local dimension 1. Target: a sum of `m` narrow 3D Lorentzian bumps at fixed
//! pseudo-random centres (closed form, a few ns per point). At tolerance 1e-8: m = 6 → rank 65,
//! m = 20 → rank 126, m = 50 → rank 163 (7 iterations, 23 M distinct points).
//!
//! Usage: `cargo run --release -- [key=value ...]` (from `comb_speed/`)
//! keys: r, m, tol, iters, memo_mib, seed. Prints per-iteration rank/error/memo stats, the wall
//! time split into evaluator time and the rest, the peak RSS, and a fingerprint (FNV-1a over the
//! bits of every materialized tensor and of the error history) for bitwise comparisons.

use std::cell::Cell;
use std::time::{Duration, Instant};

use tensor4all_treetci::{
    crossinterpolate2, DefaultProposer, GlobalIndexBatch, TreeTciEdge, TreeTciGraph,
    TreeTciIteration, TreeTciObserver, TreeTciOptions,
};

struct Args {
    r: usize,
    m: usize,
    tol: f64,
    iters: usize,
    memo_mib: usize,
    seed: u64,
}

fn parse_args() -> Args {
    let mut a = Args {
        r: 20,
        m: 6,
        tol: 1e-8,
        iters: 20,
        memo_mib: 4096,
        seed: 7,
    };
    for arg in std::env::args().skip(1) {
        let (k, v) = arg.split_once('=').expect("arguments are key=value");
        match k {
            "r" => a.r = v.parse().unwrap(),
            "m" => a.m = v.parse().unwrap(),
            "tol" => a.tol = v.parse().unwrap(),
            "iters" => a.iters = v.parse().unwrap(),
            "memo_mib" => a.memo_mib = v.parse().unwrap(),
            "seed" => a.seed = v.parse().unwrap(),
            _ => panic!("unknown key {k}"),
        }
    }
    a
}

/// splitmix64, for fixed bump parameters independent of any crate's RNG.
fn splitmix(state: &mut u64) -> f64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
}

struct Bump {
    c: [f64; 3],
    inv_w2: f64,
    a: f64,
}

fn peak_rss_mib() -> f64 {
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    status
        .lines()
        .find(|l| l.starts_with("VmHWM:"))
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|kb| kb.parse::<f64>().ok())
        .map_or(f64::NAN, |kb| kb / 1024.0)
}

struct Printer {
    t0: Instant,
}

impl TreeTciObserver for Printer {
    fn iteration(&mut self, it: &TreeTciIteration<'_>) {
        let e = it.evaluation;
        eprintln!(
            "iter {:2} t {:7.1}s rank {:4} err {:.3e} requested {} evaluated {} hits {} misses {} \
             cached {} dropped {} rss {:.0} MiB",
            it.iteration,
            self.t0.elapsed().as_secs_f64(),
            it.max_bond_dim,
            it.error,
            e.requested_points,
            e.evaluated_points,
            e.cache_hits,
            e.cache_misses,
            e.cached_entries,
            e.dropped_inserts,
            peak_rss_mib(),
        );
    }
}

fn fnv(hash: &mut u64, word: u64) {
    for byte in word.to_le_bytes() {
        *hash ^= u64::from(byte);
        *hash = hash.wrapping_mul(0x100_0000_01B3);
    }
}

fn main() -> anyhow::Result<()> {
    let a = parse_args();
    let (r, d) = (a.r, 3usize);
    let junction = r * d;
    let n_sites = junction + 1;
    let mut local_dims = vec![2usize; junction];
    local_dims.push(1);
    let mut edges = Vec::new();
    for j in 0..d {
        for i in 0..r - 1 {
            edges.push(TreeTciEdge::new(i * d + j, (i + 1) * d + j));
        }
        edges.push(TreeTciEdge::new((r - 1) * d + j, junction));
    }
    let graph = TreeTciGraph::new(n_sites, &edges)?;

    let mut s = 12345u64;
    let bumps: Vec<Bump> = (0..a.m)
        .map(|_| {
            let c = [
                0.2 + 0.6 * splitmix(&mut s),
                0.2 + 0.6 * splitmix(&mut s),
                0.2 + 0.6 * splitmix(&mut s),
            ];
            let w = 0.01 + 0.04 * splitmix(&mut s);
            let a = 0.5 + splitmix(&mut s);
            Bump {
                c,
                inv_w2: 1.0 / (w * w),
                a,
            }
        })
        .collect();
    // Site (i, j) carries bit i of axis j, weight 2^i (leaf = least significant bit).
    let scale = 1.0 / (1u64 << r) as f64;
    let f_point = |point: &[usize]| -> f64 {
        let mut x = [0u64; 3];
        for i in 0..r {
            for (j, xj) in x.iter_mut().enumerate() {
                *xj |= (point[i * d + j] as u64) << i;
            }
        }
        let x = [
            x[0] as f64 * scale,
            x[1] as f64 * scale,
            x[2] as f64 * scale,
        ];
        bumps
            .iter()
            .map(|b| {
                let q = (x[0] - b.c[0]).powi(2) + (x[1] - b.c[1]).powi(2) + (x[2] - b.c[2]).powi(2);
                b.a / (1.0 + q * b.inv_w2)
            })
            .sum()
    };

    // One pivot at the first bump's centre.
    let mut pivot = vec![0usize; n_sites];
    for (j, &c) in bumps[0].c.iter().enumerate() {
        let xi = (c / scale) as u64;
        for i in 0..r {
            pivot[i * d + j] = ((xi >> i) & 1) as usize;
        }
    }

    let eval_time = Cell::new(Duration::ZERO);
    let calls = Cell::new(0usize);
    let evaluate = |batch: GlobalIndexBatch<'_>| -> anyhow::Result<Vec<f64>> {
        let t = Instant::now();
        let out = batch
            .data()
            .chunks_exact(batch.n_sites())
            .map(f_point)
            .collect();
        eval_time.set(eval_time.get() + t.elapsed());
        calls.set(calls.get() + 1);
        Ok(out)
    };
    let options = TreeTciOptions {
        tolerance: a.tol,
        max_iter: a.iters,
        seed: Some(a.seed),
        evaluation_cache_bytes: (a.memo_mib > 0).then_some(a.memo_mib << 20),
        enable_global_pivots: true,
        scale_tolerance: true,
        ..TreeTciOptions::default()
    };
    let t0 = Instant::now();
    let mut printer = Printer { t0 };
    let run = crossinterpolate2(
        evaluate,
        local_dims,
        graph,
        vec![pivot],
        options,
        Some(junction),
        &DefaultProposer,
        &mut printer,
    )?;
    let wall = t0.elapsed().as_secs_f64();
    let mut hash = 0xCBF2_9CE4_8422_2325u64;
    for e in &run.errors {
        fnv(&mut hash, e.to_bits());
    }
    for site in 0..n_sites {
        let node = run.treetn.node_index(&site).expect("site node");
        let tensor = run.treetn.tensor(node).expect("site tensor");
        for v in tensor.to_vec::<f64>()? {
            fnv(&mut hash, v.to_bits());
        }
    }
    let ev = run.evaluation;
    println!(
        "RESULT wall {wall:.2}s f {:.2}s calls {} rest {:.2}s iters {} rank {} err {:.3e} {:?} \
         requested {} evaluated {} cached {} dropped {} peak_rss {:.0} MiB fingerprint {hash:016x}",
        eval_time.get().as_secs_f64(),
        calls.get(),
        wall - eval_time.get().as_secs_f64(),
        run.ranks.len(),
        run.ranks.last().copied().unwrap_or(0),
        run.errors.last().copied().unwrap_or(f64::NAN),
        run.termination,
        ev.requested_points,
        ev.evaluated_points,
        ev.cached_entries,
        ev.dropped_inserts,
        peak_rss_mib(),
    );
    Ok(())
}
