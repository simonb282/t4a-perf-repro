# evaluator_keys

Time per point of `TreeTNCachedEvaluator::evaluate_batched_typed` (one call per batch) on recorded batches, against a minimal per-point reference reader, for tensor4all-rs#1475. Public API only, pinned to tensor4all-rs `0d2389512f19be432b63c5a7f6fb48387a9f1536`.

## Data

The trees and batches are not in git. Download `evaluator-keys-data.tar.gz` (118 MB, 229 MB unpacked) from the release of this repository and unpack it next to this directory:

```sh
sha256sum -c evaluator-keys-data.tar.gz.sha256   # f7f53c321a9f29fb1986c61c5faef9c2d07ff9b4c7154042edbf32e998bb8c1e
tar -xzf evaluator-keys-data.tar.gz
```

`evaluator-keys-data/` holds one directory per case and `trees/`:

- `trees/*.h5`: complex `TreeTN<IdxTensor, usize>` saved by `tensor4all_hdf5::save_treetn` (group `state`). Each is a three-branch comb: three chains of binary sites joined at a degree-3 node that carries a dimension-1 site. The `a_` to `d_` trees have 13 or 15 sites per chain and bonds 16–24; `e_bond468_tree.h5` has 20 sites per chain and bonds up to 468.
- `<case>/trees.txt`: the trees the case reads, one path per line, relative to `evaluator-keys-data/`. Every batch is read from every tree.
- `<case>/batch_NNNN.bin`: the points as u8 site values, one column of `n_sites` bytes per point, the sites in ascending node order.
- `<case>/hints.txt` (optional): per batch, the node passed as `EvaluationHint::around` (`-`: no hint). For `a_`–`c_` (batches recorded from `TreeTCI2` runs, each a candidate matrix or the memo's misses of one) it is the node next to the batch's edge. For `d_`/`e_` (planes, lines, scattered points) the batches are the runs along one axis that a line/plane wrapper sends to the evaluator, with that wrapper's hint.
- `<case>/whole` (marker): the reference reads all the case's batches as one batch (the figure workloads).

## Run

```sh
cargo build --release
RAYON_NUM_THREADS=1 taskset -c 2 ./target/release/evaluator-keys ../evaluator-keys-data            # all cases, ~4 min
RAYON_NUM_THREADS=1 taskset -c 2 ./target/release/evaluator-keys ../evaluator-keys-data a_         # cases by prefix
HINT=0 ...      # ignore hints.txt
REPS=3 ...      # minimum over 3 repetitions, fresh evaluators each time
cargo build --release --features diagnostics   # also prints tensor4all-treetn's contraction counters per case
```

µs per point is the time for all the case's trees divided by its points. The evaluator is built once per tree and case, centred on the junction, and kept across the case's batches. The reference reader contracts each chain from its leaf once per distinct site string on that chain, then the junction with the three chain vectors, holding the partial products shared by consecutive points. `max rel` is max |evaluator − reference| / max |reference| (`max|v|`).

## Expected output

Laptop, one thread, on battery (powersave governor), 4 min 13 s, peak RSS 367 MB:

```text
case                                  points batches  bond evaluator µs/pt reference µs/pt   max rel    max|v|
a_bond16_tci_memo                    1188998      40    16           13.25            0.87   3.0e-15   9.71e-3
b_bond16_tci_nomemo                  1084948      40    16           12.68            1.73   3.2e-15   9.71e-3
c_bond24_tci                          262608      20    24           13.06            2.37   1.8e-15   3.53e-3
d_bond24_line_4096_block                4096       1    24            4.33           11.60   1.5e-15  3.50e-10
d_bond24_line_4096_strided              4096       1    24            3.02           11.61   2.3e-15  3.50e-10
d_bond24_plane_256x256_block           65536     256    24            5.84            0.27   2.3e-15  3.50e-10
d_bond24_plane_256x256_strided         65536     256    24            6.21            0.27   1.0e-15  3.49e-10
d_bond24_scattered_20000               20000       1    24           12.66           24.48   2.3e-15   7.65e-5
e_bond468_line_4096_block               4096       1   468           14.40         2157.73   1.1e-14   3.75e-2
e_bond468_line_4096_strided             4096       1   468           81.95         2143.06   8.5e-16   3.75e-2
e_bond468_plane_256x256_block          65536     256   468           15.18           47.19   7.0e-15   3.75e-2
e_bond468_plane_256x256_strided        65536     256   468            2.93           19.39   8.5e-16   3.75e-2
e_bond468_scattered_20000              20000       1   468         3987.72         5488.60     0.0e0  1.97e-11
```

Absolute times vary by about ±30 % between runs on this machine; compare the two columns of a row. Without hints (`HINT=0`), `a_bond16_tci_memo` takes 19.6 µs/pt.
