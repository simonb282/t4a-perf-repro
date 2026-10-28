//! Issue: TreeTNCachedEvaluator::evaluate_batched_typed cost per point.
//!
//! A random chain TreeTN of 30 binary sites, bonds 4 and 12. One evaluator is
//! reused across calls, as a downstream reader does; 409 600 random points per
//! case, in batches of 4096 and of 28.

use std::time::Instant;

use t4a_perf_repro::{chain, fingerprint, random_points, Error, Lcg};
use tensor4all_core::ColMajorArrayRef;
use tensor4all_treetn::{CachedEvaluatorOptions, EvaluationHint, TreeTNCachedEvaluator};

const N_SITES: usize = 30;
const TOTAL: usize = 409_600;

fn main() -> Result<(), Error> {
    for bond in [4usize, 12] {
        let (sites, tree) = chain(N_SITES, bond, 7)?;
        let mut evaluator =
            TreeTNCachedEvaluator::new(&tree, &sites, CachedEvaluatorOptions::default())?;
        for batch in [4096usize, 28] {
            let calls = TOTAL / batch;
            let points = random_points(&mut Lcg::new(11), N_SITES, calls * batch);
            let shape = [N_SITES, batch];
            let mut values = Vec::with_capacity(calls * batch);
            let start = Instant::now();
            for chunk in points.chunks_exact(batch * N_SITES) {
                let out: Vec<f64> = evaluator.evaluate_batched_typed(
                    ColMajorArrayRef::new(chunk, &shape)?,
                    EvaluationHint::default(),
                )?;
                values.extend(out);
            }
            let us = start.elapsed().as_secs_f64() * 1e6 / (calls * batch) as f64;
            println!(
                "bond {bond:2}, {calls:6} calls of {batch:4} points: {us:7.3} us/point, fingerprint {:016x}",
                fingerprint(&values)
            );
        }
    }
    Ok(())
}
