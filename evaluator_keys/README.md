# evaluator_keys

Time per point of `TreeTNCachedEvaluator::evaluate_batched_typed` (one call per batch) on recorded batches, against a minimal per-point reference reader, for tensor4all-rs#1475; the bond-64 data is for tensor4all-rs#1481. Public API only, pinned to tensor4all-rs `0d2389512f19be432b63c5a7f6fb48387a9f1536`.

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

## Bond-64 data (tensor4all-rs#1481)

A second archive, `evaluator-bond64-data.tar.gz` (36 MB, 146 MB unpacked), from the release of this repository, unpacked next to this directory like the first:

```sh
sha256sum -c evaluator-bond64-data.tar.gz.sha256   # be5896feec0c9dee0a41d85e3d1ba5b35f00f1a7fdc56864d0a995787cf28066
tar -xzf evaluator-bond64-data.tar.gz
RAYON_NUM_THREADS=1 taskset -c 2 ./target/release/evaluator-keys ../evaluator-bond64-data    # ~10 min
```

The batches of one late `TreeTCI2` call (memo on) on three comb trees with 15 binary sites per chain and bonds up to 64 (`trees/bond64_tree_{0,1,2}.h5`), in the same layout as above. Each case's `batches.txt` maps its batch files to the request number in the recorded call and to the replay index quoted in tensor4all-rs#1481. `hints.txt` holds, per batch, the node of the edge recovered from the batch's points (the edge with the fewest distinct left × right key pairs), not an edge reported by `TreeTCI2`.

- `f_bond64_slow_leaf_edge`: the 15 batches that are slow with the hint (#1481). For all of them the recovered edge is at the leaf end of a chain, with one or two left keys × all points, so the hint is a leaf-side node.
- `g_bond64_large_sample`: every 10th of the call's other 980 batches of ≥ 5 000 points, in request order.
- `h_bond64_small_sample`: every 20th of its 12 170 batches of < 5 000 points.

Laptop, one thread, on battery (powersave governor), 10 min 27 s, peak RSS 565 MB:

```text
case                                  points batches  bond evaluator µs/pt reference µs/pt   max rel    max|v|
f_bond64_slow_leaf_edge               385519      15    64          979.92            6.95   3.7e-15   8.60e-4
g_bond64_large_sample                2182736      98    64           52.55           46.85   3.1e-15   1.09e-3
h_bond64_small_sample                  29607     609    64           64.94          891.39   2.6e-15   8.98e-4
```
