//! `TreeTNCachedEvaluator::evaluate_batched_typed` per point on recorded batches (one call per batch), against a minimal
//! per-point reference reader, on three-branch comb trees (tensor4all-rs#1475).
//!
//! Usage: `evaluator-keys <data dir> [case prefix]`. Each case is a directory of `<data dir>` holding `trees.txt` (one
//! HDF5 file per line, relative to `<data dir>`, saved by `tensor4all_hdf5::save_treetn` in group `state`) and
//! `batch_NNNN.bin` (u8 site values, one column per point, the sites in ascending node order). Every batch is read from
//! every tree of its case; the µs per point are the time for all the case's trees divided by the points.
//! An optional `hints.txt` gives, per batch, the node to centre the evaluation on (`EvaluationHint::around`): the node
//! next to the tree edge the batch is a candidate matrix of; `HINT=0` evaluates without hints.
//! `REPS=k` (default 1): the minimum over k repetitions, each with fresh evaluators.
//!
//! The reference reader: the tree is a degree-3 junction with three chains. Per batch, each chain is contracted from
//! its leaf to the junction once per distinct site-value string on that chain; the junction is contracted with the
//! first chain's vector once per run of that vector (points sorted by their first two chains), with the second once
//! per distinct pair, and with the third once per point.

use std::io::Write;
use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result, bail, ensure};
use num_complex::Complex64 as C;
use rustc_hash::FxHashMap;
use tensor4all_core::{ColMajorArrayRef, DynIndex, IdxTensor, IndexLike};
use tensor4all_treetn::{CachedEvaluatorOptions, EvaluationHint, TreeTN, TreeTNCachedEvaluator};

type Tree = TreeTN<IdxTensor, usize>;

/// The sites in ascending node order (one per node that has one).
fn sites(tree: &Tree) -> Result<Vec<DynIndex>> {
    let mut nodes = tree.node_names();
    nodes.sort();
    nodes
        .iter()
        .filter_map(|v| tree.site_space(v).filter(|s| !s.is_empty()))
        .map(|s| {
            ensure!(s.len() == 1, "a node with {} site indices", s.len());
            s.iter().next().cloned().context("empty site space")
        })
        .collect()
}

fn bond(tree: &Tree, a: usize, b: usize) -> Option<DynIndex> {
    tree.edge_between(&a, &b).and_then(|e| tree.bond_index(e)).cloned()
}

fn entries(tree: &Tree, v: usize, order: &[DynIndex]) -> Result<Vec<C>> {
    let t = tree.node_index(&v).and_then(|i| tree.tensor(i)).context("no tensor")?.permute_indices(order)?;
    Ok(match t.to_vec::<C>() {
        Ok(values) => values,
        Err(_) => t.to_vec::<f64>()?.into_iter().map(C::from).collect(),
    })
}

/// One chain, top (next to the junction) first: per node its core (down, site, up; down fastest), dims, and the
/// column of its site in a batch.
struct Chain {
    cores: Vec<Vec<C>>,
    down: Vec<usize>,
    up: Vec<usize>,
    site_dim: Vec<usize>,
    column: Vec<usize>,
}

struct Reference {
    chains: Vec<Chain>,
    junction: Vec<C>,
}

impl Reference {
    fn new(tree: &Tree) -> Result<Self> {
        let mut nodes = tree.node_names();
        nodes.sort();
        let column_of: FxHashMap<usize, usize> = nodes
            .iter()
            .filter(|v| tree.site_space(v).is_some_and(|s| !s.is_empty()))
            .enumerate()
            .map(|(c, &v)| (v, c))
            .collect();
        let neighbours = |v: usize| -> Vec<usize> { nodes.iter().copied().filter(|&w| w != v && bond(tree, v, w).is_some()).collect() };
        let junctions: Vec<usize> = nodes.iter().copied().filter(|&v| neighbours(v).len() == 3).collect();
        let [junction] = junctions[..] else { bail!("the reference reader needs exactly one degree-3 node, found {}", junctions.len()) };
        let site = |v: usize| tree.site_space(&v).and_then(|s| s.iter().next().cloned());
        let mut chains = Vec::new();
        let mut top_bonds = Vec::new();
        for top in neighbours(junction) {
            let mut path = vec![top];
            let mut prev = junction;
            while let Some(&next) = neighbours(*path.last().context("empty path")?).iter().find(|&&w| w != prev) {
                prev = *path.last().context("empty path")?;
                ensure!(neighbours(prev).len() <= 2, "the junction's subtrees must be chains");
                path.push(next);
            }
            let mut chain = Chain { cores: Vec::new(), down: Vec::new(), up: Vec::new(), site_dim: Vec::new(), column: Vec::new() };
            for (k, &v) in path.iter().enumerate() {
                let up = bond(tree, v, if k == 0 { junction } else { path[k - 1] }).context("no bond")?;
                let s = site(v).context("a chain node without a site")?;
                let mut order = Vec::new();
                match path.get(k + 1) {
                    Some(&below) => {
                        let down = bond(tree, v, below).context("no bond")?;
                        chain.down.push(down.dim());
                        order.push(down);
                    }
                    None => chain.down.push(1),
                }
                chain.site_dim.push(s.dim());
                chain.up.push(up.dim());
                chain.column.push(column_of[&v]);
                order.push(s);
                if k == 0 {
                    top_bonds.push(up.clone());
                }
                order.push(up);
                chain.cores.push(entries(tree, v, &order)?);
            }
            chains.push(chain);
        }
        let mut order = top_bonds;
        order.extend(site(junction));
        Ok(Self { chains, junction: entries(tree, junction, &order)? })
    }

    /// The chain's vector on its top bond at the site values of `point`.
    fn top(&self, c: usize, point: &[u8]) -> Vec<C> {
        let chain = &self.chains[c];
        (0..chain.cores.len()).rev().fold(vec![C::new(1.0, 0.0)], |env, k| {
            let (d, s) = (chain.down[k], chain.site_dim[k]);
            let sigma = usize::from(point[chain.column[k]]);
            let core = &chain.cores[k];
            (0..chain.up[k])
                .map(|p| {
                    let base = d * (sigma + s * p);
                    env.iter().zip(&core[base..base + d]).map(|(e, t)| e * t).sum()
                })
                .collect()
        })
    }

    fn read(&self, columns: &[u8], n_sites: usize) -> Vec<C> {
        let points: Vec<&[u8]> = columns.chunks_exact(n_sites).collect();
        // Per chain: the distinct site-value strings (as byte strings) and each point's slot.
        let tops: Vec<(Vec<Vec<C>>, Vec<usize>)> = (0..3)
            .map(|c| {
                let mut slot_of: FxHashMap<Vec<u8>, usize> = FxHashMap::default();
                let mut envs = Vec::new();
                let slots = points
                    .iter()
                    .map(|p| {
                        let key: Vec<u8> = self.chains[c].column.iter().map(|&i| p[i]).collect();
                        *slot_of.entry(key).or_insert_with(|| {
                            envs.push(self.top(c, p));
                            envs.len() - 1
                        })
                    })
                    .collect();
                (envs, slots)
            })
            .collect();
        let lead = |v: &[C], e: &[C]| -> Vec<C> { v.chunks_exact(e.len()).map(|ch| ch.iter().zip(e).map(|(a, b)| a * b).sum()).collect() };
        let mut order: Vec<usize> = (0..points.len()).collect();
        order.sort_by_key(|&i| (tops[0].1[i], tops[1].1[i]));
        let mut out = vec![C::new(0.0, 0.0); points.len()];
        let mut held: Option<(usize, Vec<C>)> = None;
        for group in order.chunk_by(|&a, &b| (tops[0].1[a], tops[1].1[a]) == (tops[0].1[b], tops[1].1[b])) {
            let first = group[0];
            let s0 = tops[0].1[first];
            let v = match held.take() {
                Some((s, v)) if s == s0 => v,
                _ => lead(&self.junction, &tops[0].0[s0]),
            };
            let w = lead(&v, &tops[1].0[tops[1].1[first]]);
            held = Some((s0, v));
            for &i in group {
                out[i] = w.iter().zip(&tops[2].0[tops[2].1[i]]).map(|(a, b)| a * b).sum();
            }
        }
        out
    }
}

fn junction_of(tree: &Tree) -> Result<usize> {
    let nodes = tree.node_names();
    nodes
        .iter()
        .copied()
        .find(|&v| nodes.iter().filter(|&&w| w != v && bond(tree, v, w).is_some()).count() == 3)
        .context("no degree-3 node")
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let root = Path::new(args.get(1).context("usage: evaluator-keys <data dir> [case prefix]")?);
    let filter = args.get(2).cloned().unwrap_or_default();
    let use_hints = std::env::var("HINT").map_or(true, |v| v != "0");
    let reps: usize = std::env::var("REPS").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
    let mut cases: Vec<_> = std::fs::read_dir(root)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.join("trees.txt").exists())
        .filter(|p| p.file_name().is_some_and(|n| n.to_string_lossy().starts_with(&filter)))
        .collect();
    cases.sort();
    println!("{:34} {:>9} {:>7} {:>5} {:>15} {:>15} {:>9} {:>9}", "case", "points", "batches", "bond", "evaluator µs/pt", "reference µs/pt", "max rel", "max|v|");
    for case in cases {
        let trees: Vec<Tree> = std::fs::read_to_string(case.join("trees.txt"))?
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| {
                let path = root.join(l.trim());
                tensor4all_hdf5::load_treetn::<usize>(&path.to_string_lossy(), "state").with_context(|| format!("load {}", path.display()))
            })
            .collect::<Result<_>>()?;
        let n_sites = sites(&trees[0])?.len();
        let mut names: Vec<_> = std::fs::read_dir(&case)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.file_name().is_some_and(|n| n.to_string_lossy().starts_with("batch_")))
            .collect();
        names.sort();
        let batches: Vec<Vec<u8>> = names.iter().map(std::fs::read).collect::<Result<_, _>>()?;
        ensure!(batches.iter().all(|b| b.len() % n_sites == 0), "a batch is not a whole number of {n_sites}-site points");
        // A file `whole` in the case: the reference reads all its batches as one (the figure workloads, whose batches
        // are the runs that a line/plane wrapper sends to the evaluator).
        let whole = case.join("whole").exists();
        let points: usize = batches.iter().map(|b| b.len() / n_sites).sum();
        let bond = trees.iter().flat_map(|t| t.link_dims()).max().unwrap_or(0);
        // Optional `hints.txt`: per batch, the node next to the edge the batch is a candidate matrix of (HINT=0 ignores it).
        let hints: Vec<Option<usize>> = match std::fs::read_to_string(case.join("hints.txt")) {
            Ok(text) if use_hints => text.lines().map(|l| l.trim().parse().ok()).collect(),
            _ => vec![None; batches.len()],
        };
        ensure!(hints.len() == batches.len(), "hints.txt has {} lines for {} batches", hints.len(), batches.len());
        #[cfg(feature = "diagnostics")]
        tensor4all_treetn::diagnostics::reset_contraction();
        let (mut t_eval, mut t_ref, mut worst, mut scale) = (f64::INFINITY, f64::INFINITY, 0.0f64, 0.0f64);
        for _ in 0..reps {
            let clock = Instant::now();
            let mut values: Vec<Vec<C>> = Vec::new();
            for tree in &trees {
                let options = CachedEvaluatorOptions { center: Some(junction_of(tree)?), ..CachedEvaluatorOptions::default() };
                let mut evaluator = TreeTNCachedEvaluator::new(tree, &sites(tree)?, options)?;
                for (b, hint) in batches.iter().zip(&hints) {
                    let hint = hint.map_or_else(EvaluationHint::default, EvaluationHint::around);
                    let columns: Vec<usize> = b.iter().map(|&s| usize::from(s)).collect();
                    let shape = [n_sites, b.len() / n_sites];
                    values.push(evaluator.evaluate_batched_typed::<C>(ColMajorArrayRef::new(&columns, &shape)?, hint)?);
                }
            }
            t_eval = t_eval.min(clock.elapsed().as_secs_f64());
            let clock = Instant::now();
            let mut reference: Vec<Vec<C>> = Vec::new();
            for tree in &trees {
                let reader = Reference::new(tree)?;
                if whole {
                    let all = reader.read(&batches.concat(), n_sites);
                    let mut rest = all.as_slice();
                    for b in &batches {
                        let (head, tail) = rest.split_at(b.len() / n_sites);
                        reference.push(head.to_vec());
                        rest = tail;
                    }
                } else {
                    reference.extend(batches.iter().map(|b| reader.read(b, n_sites)));
                }
            }
            t_ref = t_ref.min(clock.elapsed().as_secs_f64());
            for (x, y) in values.iter().zip(&reference) {
                for (a, b) in x.iter().zip(y) {
                    worst = worst.max((a - b).norm());
                    scale = scale.max(b.norm());
                }
            }
        }
        let us = |s: f64| 1e6 * s / points as f64;
        println!(
            "{:34} {points:>9} {:>7} {bond:>5} {:>15.2} {:>15.2} {:>9.1e} {:>9.2e}",
            case.file_name().map_or(String::new(), |n| n.to_string_lossy().into_owned()),
            batches.len(),
            us(t_eval),
            us(t_ref),
            worst / scale.max(f64::MIN_POSITIVE),
            scale
        );
        #[cfg(feature = "diagnostics")]
        println!("    {}", tensor4all_treetn::diagnostics::contraction_summary());
        std::io::stdout().flush()?;
    }
    Ok(())
}
