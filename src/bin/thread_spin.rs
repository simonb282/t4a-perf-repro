//! CPU time of idle pool workers: a short parallel call, then a serial phase.
//!
//! `thread_spin backend`: the process-wide CPU backend of tensor4all-tensorbackend (sized from `RAYON_NUM_THREADS`),
//! one 512 x 512 Complex64 `mat_mul` and then 20 ms of serial work, for 3 s. The other threads do nothing in the
//! serial phase, so with sleeping workers the CPU time is about the wall time.
//!
//! `thread_spin backend <active|passive|N>` (feature `wait-policy`, tensor4all-rs with `CpuWaitPolicy`): the same with
//! `set_default_cpu_wait_policy` called first; `N` is `CpuWaitPolicy::SpinPolls(N)`.
//!
//! `thread_spin team <threads> <idle_spin_ms> [spin_polls]`: the same loop on a bare `threadteam::Pool` (0.2.0), with
//! the pool's idle spin (default 200 ms) and optionally its spin budget (`spin_polls`, default per architecture).
//!
//! Prints wall time, CPU time (utime + stime of the process) and their ratio.

use std::time::{Duration, Instant};

use num_complex::Complex64;
use tensor4all_tensorbackend::{Matrix, mat_mul, with_default_backend};

const RUN: Duration = Duration::from_secs(3);
const SERIAL: Duration = Duration::from_millis(20);

fn cpu_seconds() -> f64 {
    let stat = std::fs::read_to_string("/proc/self/stat").unwrap_or_default();
    let after_name = stat.rsplit(')').next().unwrap_or("");
    let fields: Vec<f64> = after_name
        .split_whitespace()
        .map(|f| f.parse().unwrap_or(0.0))
        .collect();
    // utime and stime are fields 14 and 15 of the line, 12 and 13 after the `)` of the name.
    (fields.get(11).copied().unwrap_or(0.0) + fields.get(12).copied().unwrap_or(0.0)) / 100.0
}

/// A dependent chain of floating-point operations that takes `d` on one thread.
fn serial_work(d: Duration) -> f64 {
    let start = Instant::now();
    let mut x = 1.0_f64;
    while start.elapsed() < d {
        for _ in 0..1000 {
            x = std::hint::black_box(x * 1.000_000_1 + 1e-9);
        }
    }
    x
}

fn report(label: &str, wall: f64, cpu0: f64) {
    let cpu = cpu_seconds() - cpu0;
    println!("{label}: wall {wall:5.2} s, cpu {cpu:5.2} s, cpu/wall {:4.2}", cpu / wall);
}

#[cfg(feature = "wait-policy")]
fn set_policy(arg: &str) -> String {
    use tensor4all_tensorbackend::{CpuWaitPolicy, set_default_cpu_wait_policy};
    let policy = match arg {
        "active" => CpuWaitPolicy::Active,
        "passive" => CpuWaitPolicy::Passive,
        polls => CpuWaitPolicy::SpinPolls(polls.parse().expect("active, passive or a poll count")),
    };
    set_default_cpu_wait_policy(policy).expect("first set");
    format!("{policy:?}")
}

fn backend(policy: &str) {
    let n = 512;
    let data: Vec<Complex64> = (0..n * n)
        .map(|i| Complex64::new((i % 7) as f64 - 3.0, (i % 5) as f64 - 2.0))
        .collect();
    let a = Matrix::from_col_major_vec(n, n, data);
    let threads = with_default_backend(|b| b.num_threads());
    let policy = policy.to_owned();
    let (cpu0, start) = (cpu_seconds(), Instant::now());
    let mut calls = 0;
    while start.elapsed() < RUN {
        std::hint::black_box(mat_mul(&a, &a).expect("mat_mul"));
        std::hint::black_box(serial_work(SERIAL));
        calls += 1;
    }
    report(
        &format!("backend {policy}, {threads} threads, {calls} x (mat_mul + 20 ms serial)"),
        start.elapsed().as_secs_f64(),
        cpu0,
    );
}

fn team(threads: usize, idle_spin_ms: u64, spin_polls: Option<u32>) {
    let mut builder = threadteam::Pool::builder()
        .threads(threads)
        .idle_spin(Duration::from_millis(idle_spin_ms))
        .acquire_timeout(Duration::from_secs(10));
    if let Some(polls) = spin_polls {
        builder = builder.spin_polls(polls);
    }
    let pool = builder.build().expect("pool");
    let polls = pool.spin_polls();
    let mut exec = threadteam::Exec::new(&pool, threads).expect("exec");
    let (cpu0, start) = (cpu_seconds(), Instant::now());
    let mut calls = 0;
    let outcome = exec.team(|mut team| {
        while start.elapsed() < RUN {
            // A parallel section of negligible work, then a serial phase of the leader.
            team.reborrow().run(|_i, _n| {
                std::hint::black_box(serial_work(Duration::from_micros(200)));
            });
            std::hint::black_box(serial_work(SERIAL));
            calls += 1;
        }
    });
    outcome.expect("team scope");
    report(
        &format!("team, {threads} threads, idle_spin {idle_spin_ms} ms, spin_polls {polls}, {calls} x (section + 20 ms serial)"),
        start.elapsed().as_secs_f64(),
        cpu0,
    );
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.iter().map(String::as_str).collect::<Vec<_>>().as_slice() {
        ["backend"] => backend("default"),
        #[cfg(feature = "wait-policy")]
        ["backend", policy] => {
            let label = set_policy(policy);
            backend(&label)
        }
        ["team", threads, spin] => team(
            threads.parse().expect("threads"),
            spin.parse().expect("idle_spin_ms"),
            None,
        ),
        ["team", threads, spin, polls] => team(
            threads.parse().expect("threads"),
            spin.parse().expect("idle_spin_ms"),
            Some(polls.parse().expect("spin_polls")),
        ),
        _ => eprintln!(
            "usage: thread_spin backend [active|passive|N] | thread_spin team <threads> <idle_spin_ms> [spin_polls]"
        ),
    }
}
