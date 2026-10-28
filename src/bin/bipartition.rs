//! Issue: cost per entry of a candidate matrix through MultiIndexCache::evaluate_batched, against caller-side keys.
//!
//! treetci asks for the values of a matrix over a left set I and a right set J: entry (i, j) is the point
//! (i, j) of 30 binary sites (15 + 15). Here 200 matrices of 100 x 100 entries are drawn from pools of 300
//! left and 300 right multi-indices, so most entries are hits, as in a TCI. Three ways to serve them:
//!  (A) assemble the 30-site points and call `MultiIndexCache::evaluate_batched`;
//!  (B) assemble the points, one mixed-radix key per point, one lookup in a std `HashMap` (the baseline of `memo`);
//!  (C) never assemble a point for a hit: key(i, j) = key_left[i] * 2^15 + key_right[j] from per-side keys,
//!      one lookup in the same std `HashMap`; the point is assembled only for a miss.
//! (C) is caller-side code on the public API; it shows what a cache entry point that takes the keys of the
//! two sides could save, it is not an existing upstream API.

use std::collections::HashMap;
use std::time::Instant;

use t4a_perf_repro::{fingerprint, random_points, Error, Lcg};
use tensor4all_core::{ColMajorArrayRef, MultiIndexCache};

const HALF: usize = 15;
const N_SITES: usize = 2 * HALF;
const POOL: usize = 300;
const SIDE: usize = 100;
const MATRICES: usize = 200;

fn f(point: &[usize]) -> f64 {
    point
        .iter()
        .enumerate()
        .map(|(site, &bit)| (bit * (site + 1)) as f64)
        .sum()
}

fn key_of(bits: &[usize]) -> u64 {
    bits.iter().fold(0u64, |key, &bit| (key << 1) | bit as u64)
}

fn main() -> Result<(), Error> {
    let mut rng = Lcg::new(1);
    let left_pool = random_points(&mut rng, HALF, POOL);
    let right_pool = random_points(&mut rng, HALF, POOL);
    // per matrix: SIDE left and SIDE right pool positions
    let picks: Vec<(Vec<usize>, Vec<usize>)> = (0..MATRICES)
        .map(|_| {
            (
                (0..SIDE).map(|_| rng.below(POOL)).collect(),
                (0..SIDE).map(|_| rng.below(POOL)).collect(),
            )
        })
        .collect();
    let left = |p: usize| &left_pool[p * HALF..(p + 1) * HALF];
    let right = |p: usize| &right_pool[p * HALF..(p + 1) * HALF];
    let n_requests = MATRICES * SIDE * SIDE;
    let shape = [N_SITES, SIDE * SIDE];

    // (A) assemble points, MultiIndexCache::evaluate_batched
    let mut cache = MultiIndexCache::<f64>::new(&[2; N_SITES])?;
    let mut evaluated = 0usize;
    let mut out_a = Vec::with_capacity(n_requests);
    let start = Instant::now();
    let mut points = vec![0usize; N_SITES * SIDE * SIDE];
    for (li, ri) in &picks {
        let mut column = 0;
        for &r in ri {
            for &l in li {
                let dst = &mut points[column * N_SITES..(column + 1) * N_SITES];
                dst[..HALF].copy_from_slice(left(l));
                dst[HALF..].copy_from_slice(right(r));
                column += 1;
            }
        }
        let values = cache.evaluate_batched(ColMajorArrayRef::new(&points, &shape)?, |misses| {
            evaluated += misses.shape()[1];
            Ok(misses.data().chunks_exact(N_SITES).map(f).collect())
        })?;
        out_a.extend(values);
    }
    let a_s = start.elapsed().as_secs_f64();

    // (B) assemble points, one key per point, std HashMap
    let mut map: HashMap<u64, f64> = HashMap::new();
    let mut out_b = Vec::with_capacity(n_requests);
    let start = Instant::now();
    let mut point = [0usize; N_SITES];
    for (li, ri) in &picks {
        for &r in ri {
            for &l in li {
                point[..HALF].copy_from_slice(left(l));
                point[HALF..].copy_from_slice(right(r));
                out_b.push(*map.entry(key_of(&point)).or_insert_with(|| f(&point)));
            }
        }
    }
    let b_s = start.elapsed().as_secs_f64();

    // (C) keys from the two sides, a point only for a miss
    let left_keys: Vec<u64> = (0..POOL).map(|p| key_of(left(p))).collect();
    let right_keys: Vec<u64> = (0..POOL).map(|p| key_of(right(p))).collect();
    let mut map: HashMap<u64, f64> = HashMap::new();
    let mut out_c = Vec::with_capacity(n_requests);
    let start = Instant::now();
    for (li, ri) in &picks {
        for &r in ri {
            for &l in li {
                let key = (left_keys[l] << HALF) + right_keys[r];
                let value = *map.entry(key).or_insert_with(|| {
                    point[..HALF].copy_from_slice(left(l));
                    point[HALF..].copy_from_slice(right(r));
                    f(&point)
                });
                out_c.push(value);
            }
        }
    }
    let c_s = start.elapsed().as_secs_f64();

    let ns = |seconds: f64| seconds * 1e9 / n_requests as f64;
    println!(
        "requests {n_requests} in {MATRICES} matrices of {SIDE} x {SIDE}, evaluated {evaluated}, hits {}, misses {}",
        cache.hits(),
        cache.misses()
    );
    println!("(A) points + MultiIndexCache::evaluate_batched   {:7.1} ns/entry", ns(a_s));
    println!("(B) points + one key per point + std HashMap      {:7.1} ns/entry", ns(b_s));
    println!("(C) keys from the two sides + std HashMap          {:7.1} ns/entry", ns(c_s));
    println!(
        "fingerprints: A {:016x}, B {:016x}, C {:016x}",
        fingerprint(&out_a),
        fingerprint(&out_b),
        fingerprint(&out_c)
    );
    Ok(())
}
