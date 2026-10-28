//! Issue: MultiIndexCache::evaluate_batched cost per requested point.
//!
//! Sends the same request stream (4096-point batches drawn from a pool of
//! 20 000 distinct 30-site binary points, so most requests are hits, as in a
//! TCI) through `MultiIndexCache::evaluate_batched`, and through a minimal
//! baseline that does the same job with one key per point in a plain std
//! `HashMap` (same SipHash hasher, no lock). Also times calling f directly.

use std::collections::HashMap;
use std::time::Instant;

use t4a_perf_repro::{fingerprint, random_points, Error, Lcg};
use tensor4all_core::{ColMajorArrayRef, MultiIndexCache};

const N_SITES: usize = 30;
const POOL: usize = 20_000;
const BATCH: usize = 4096;
const BATCHES: usize = 500;

/// A cheap target, so the time measured is the memo's own.
fn f(point: &[usize]) -> f64 {
    point
        .iter()
        .enumerate()
        .map(|(site, &bit)| (bit * (site + 1)) as f64)
        .sum()
}

fn main() -> Result<(), Error> {
    let mut rng = Lcg::new(1);
    let pool = random_points(&mut rng, N_SITES, POOL);
    let mut requests = Vec::with_capacity(BATCH * BATCHES * N_SITES);
    for _ in 0..BATCH * BATCHES {
        let p = rng.below(POOL);
        requests.extend_from_slice(&pool[p * N_SITES..(p + 1) * N_SITES]);
    }
    let n_requests = BATCH * BATCHES;
    let shape = [N_SITES, BATCH];

    // 1. MultiIndexCache::evaluate_batched (default byte limit).
    let mut cache = MultiIndexCache::<f64>::new(&[2; N_SITES])?;
    let mut evaluated = 0usize;
    let mut memo = Vec::with_capacity(n_requests);
    let start = Instant::now();
    for batch in requests.chunks_exact(BATCH * N_SITES) {
        let values = cache.evaluate_batched(ColMajorArrayRef::new(batch, &shape)?, |misses| {
            evaluated += misses.shape()[1];
            Ok(misses.data().chunks_exact(N_SITES).map(f).collect())
        })?;
        memo.extend(values);
    }
    let memo_s = start.elapsed().as_secs_f64();

    // 2. Baseline: one key per point, one std HashMap lookup, no lock.
    let mut map: HashMap<u64, f64> = HashMap::new();
    let mut baseline = Vec::with_capacity(n_requests);
    let start = Instant::now();
    for point in requests.chunks_exact(N_SITES) {
        let key = point.iter().fold(0u64, |key, &bit| (key << 1) | bit as u64);
        baseline.push(*map.entry(key).or_insert_with(|| f(point)));
    }
    let baseline_s = start.elapsed().as_secs_f64();

    // 3. No memo: f on every request.
    let start = Instant::now();
    let direct: Vec<f64> = requests.chunks_exact(N_SITES).map(f).collect();
    let direct_s = start.elapsed().as_secs_f64();

    let ns = |seconds: f64| seconds * 1e9 / n_requests as f64;
    println!(
        "requests {n_requests}, evaluated {evaluated}, hits {}, misses {}",
        cache.hits(),
        cache.misses()
    );
    println!("MultiIndexCache::evaluate_batched  {:8.1} ns/request", ns(memo_s));
    println!("baseline (one key, one lookup)     {:8.1} ns/request", ns(baseline_s));
    println!("f directly, no memo                {:8.1} ns/request", ns(direct_s));
    println!(
        "fingerprints: memo {:016x}, baseline {:016x}, direct {:016x}",
        fingerprint(&memo),
        fingerprint(&baseline),
        fingerprint(&direct)
    );
    Ok(())
}
