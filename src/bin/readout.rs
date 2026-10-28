//! Issue: fixed cost per call of a one-point readout (the global-pivot walk
//! evaluates one point per call).
//!
//! The same 2000 random points of a random 30-site binary chain TreeTN are
//! read once as one batch and once one point per call, each with a fresh
//! TreeTNCachedEvaluator, at bonds 8 and 48.

use std::time::Instant;

use t4a_perf_repro::{chain, random_points, Error, Lcg};
use tensor4all_core::ColMajorArrayRef;
use tensor4all_treetn::{CachedEvaluatorOptions, EvaluationHint, TreeTNCachedEvaluator};

const N_SITES: usize = 30;
const N_POINTS: usize = 2000;

fn main() -> Result<(), Error> {
    for bond in [8usize, 48] {
        let (sites, tree) = chain(N_SITES, bond, 7)?;
        let points = random_points(&mut Lcg::new(11), N_SITES, N_POINTS);

        let mut evaluator =
            TreeTNCachedEvaluator::new(&tree, &sites, CachedEvaluatorOptions::default())?;
        let start = Instant::now();
        let batched: Vec<f64> = evaluator.evaluate_batched_typed(
            ColMajorArrayRef::new(points.as_slice(), &[N_SITES, N_POINTS])?,
            EvaluationHint::default(),
        )?;
        let batched_us = start.elapsed().as_secs_f64() * 1e6 / N_POINTS as f64;

        let mut evaluator =
            TreeTNCachedEvaluator::new(&tree, &sites, CachedEvaluatorOptions::default())?;
        let mut single = Vec::with_capacity(N_POINTS);
        let start = Instant::now();
        for point in points.chunks_exact(N_SITES) {
            let out: Vec<f64> = evaluator.evaluate_batched_typed(
                ColMajorArrayRef::new(point, &[N_SITES, 1])?,
                EvaluationHint::default(),
            )?;
            single.push(out[0]);
        }
        let single_us = start.elapsed().as_secs_f64() * 1e6 / N_POINTS as f64;

        let max_diff = batched
            .iter()
            .zip(&single)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        println!(
            "bond {bond:2}: one batch {batched_us:8.3} us/point, one point per call {single_us:8.3} us/point ({:.0}x), max |diff| {max_diff:.1e}",
            single_us / batched_us
        );
    }
    Ok(())
}
