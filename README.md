# t4a-perf-repro

Minimal timing reproducers for hot paths in
[tensor4all-rs](https://github.com/tensor4all/tensor4all-rs). They use only the
public API, pinned to `937f4738` (`main` of 2026-10-29; earlier commits of this
repository are pinned to `9222cbe`), and need nothing else.

| Binary | What it isolates | Issue |
|---|---|---|
| `memo` | `MultiIndexCache::evaluate_batched` per requested point, against a minimal one-key-one-lookup baseline and against no memo | tensor4all-rs#1020 |
| `evaluator` | `TreeTNCachedEvaluator::evaluate_batched_typed` per point, large and small batches | tensor4all-rs#1021 |
| `readout` | fixed cost per call: the same points read as one batch and one point per call | tensor4all-rs#1022 |
| `linalg` | three per-call costs below the evaluator: the finiteness passes before rrLU, `tensorbackend::mat_mul(&a, &b)` against faer on borrowed slices, `IdxTensor::dense_host_values` against `with_dense_slice` | tensor4all-rs issues for the three candidates (see the tracking issue #1027) |
| `rrlu_full` | `rrlu` (full L and U) of a Complex64 matrix, with a result fingerprint | tensor4all-rs#1051 |
| `thread_spin` | CPU time of idle pool workers: a parallel call (`mat_mul`) then a serial phase, on the process-wide backend, and on a bare `threadteam::Pool` (0.2.0) with the idle spin and the spin budget given, and with a `CpuWaitPolicy` (feature `wait-policy`); prints wall, CPU and their ratio (not a timing of the library, a cost of waiting) | the issue on the pool's wait policy |
| `bipartition` | cost per entry of candidate matrices (points assembled, then `MultiIndexCache::evaluate_batched`) against caller-side keys built from the two sides of the matrix | tensor4all-rs#1026 |
| `comb` | end-to-end check: a full `crossinterpolate2` on a comb tree (three binary branches on a site-free junction, target a sum of K exponentials); `replay` sends treetci's recorded request stream through `MultiIndexCache`; `eval` times the evaluator on the result | all of the above |

## Run

```sh
cargo build --release
RAYON_NUM_THREADS=1 taskset -c 2 ./target/release/memo
RAYON_NUM_THREADS=1 taskset -c 2 ./target/release/evaluator
RAYON_NUM_THREADS=1 taskset -c 2 ./target/release/readout
RAYON_NUM_THREADS=1 taskset -c 2 ./target/release/bipartition
# thread_spin is run WITHOUT taskset, on an otherwise idle machine, once per thread count
for n in 1 2 4 8; do RAYON_NUM_THREADS=$n ./target/release/thread_spin backend; done
./target/release/thread_spin team 4 200; ./target/release/thread_spin team 4 200 10000   # threadteam 0.2.0: default spin budget, then 10000 polls
# with a tensor4all-rs that has CpuWaitPolicy (uncomment the [patch], then): cargo build --release --features wait-policy
RAYON_NUM_THREADS=8 ./target/release/thread_spin backend passive   # also: active, or a poll count
RAYON_NUM_THREADS=1 taskset -c 2 ./target/release/linalg    # exits 1 if a fast variant disagrees beyond its stated tolerance

# end to end (arguments: see the header of src/bin/comb.rs)
RAYON_NUM_THREADS=1 taskset -c 2 ./target/release/comb tci 10 60 1.0 1073741824 5
RAYON_NUM_THREADS=1 taskset -c 2 ./target/release/comb tci 10 60 1.0 none 5
RAYON_NUM_THREADS=1 taskset -c 2 ./target/release/comb replay 10 60 1.0 5
RAYON_NUM_THREADS=1 taskset -c 2 ./target/release/comb eval 10 4 3
```

`comb` prints one JSON line per repetition, with times, counts, the
environment (load, governor, frequency) and `result_hash`.

Inputs are deterministic (fixed seeds). Each binary prints times per point and
a fingerprint of the results (FNV-1a over the bit patterns): equal fingerprints
mean bitwise-equal results.

## Compare a patched tensor4all-rs

1. Check out tensor4all-rs at `937f4738` next to this directory and apply a
   patch from `patches/`, inside the checkout:
   `git apply ../t4a-perf-repro/patches/memo.diff` (memo) or
   `.../evaluator.diff` (evaluator). The patches are proofs of concept and
   add a `rustc-hash` dependency to the patched crate.
2. Uncomment the `[patch."https://github.com/tensor4all/tensor4all-rs"]` block
   in `Cargo.toml` and point it at that checkout (keep all listed crates, so that only one copy
   of each is built).
3. Rebuild and run: the times should drop and the fingerprints must not change.

## Measurement notes

The numbers quoted in the issues are indicative laptop measurements (Intel
i5-1155G7, one thread, medians of interleaved repetitions). Please re-measure
under `benchmarks/2026-10-08-treetci-memo-protocol.md`; the three minimal binaries take no
arguments (`comb` takes them on the command line), so they fit that
protocol's paired runs directly.
