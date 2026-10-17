// Copyright 2026 Entalpic
//! Fork-safe rayon thread pool.
//!
//! `fork()` copies a process's memory but only the calling thread, so a child
//! inherits rayon's global pool marked as running with none of its worker
//! threads. The first parallel call in the child then queues work that nothing
//! executes and blocks forever. This bites PyTorch DataLoader workers, which
//! are forked by default on Linux after the parent has often read a batch.
//!
//! All of atompack's parallel work runs through [`install`], which uses a pool
//! owned by the current process and builds a fresh one after a fork.

use std::sync::{Arc, Mutex};

use rayon::{ThreadPool, ThreadPoolBuilder};

/// The pool and the id of the process that built it.
static POOL: Mutex<Option<(u32, Arc<ThreadPool>)>> = Mutex::new(None);

/// Run `op` on atompack's thread pool, so parallel iterators inside it use
/// that pool instead of rayon's global one.
///
/// The pool honors `RAYON_NUM_THREADS` like rayon's global pool does.
pub fn install<OP, R>(op: OP) -> R
where
    OP: FnOnce() -> R + Send,
    R: Send,
{
    // Already on a pool worker (nested call): stay on the current pool.
    if rayon::current_thread_index().is_some() {
        return op();
    }
    match pool() {
        Some(pool) => pool.install(op),
        // Could not spawn threads (e.g. a thread limit): run on this thread.
        None => op(),
    }
}

fn pool() -> Option<Arc<ThreadPool>> {
    let pid = std::process::id();
    let mut slot = POOL.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((owner, pool)) = slot.as_ref()
        && *owner == pid
    {
        return Some(Arc::clone(pool));
    }
    // A pool built by a parent process has no threads here. Dropping it would
    // try to signal those threads through state they may have left locked, so
    // leak it instead.
    if let Some(stale) = slot.take() {
        std::mem::forget(stale);
    }
    let pool = Arc::new(
        ThreadPoolBuilder::new()
            .thread_name(|i| format!("atompack-{i}"))
            .build()
            .ok()?,
    );
    *slot = Some((pid, Arc::clone(&pool)));
    Some(pool)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rayon::prelude::*;

    #[test]
    fn install_runs_parallel_work_on_atompack_pool() {
        let names: Vec<String> = install(|| {
            (0..64)
                .into_par_iter()
                .map(|_| std::thread::current().name().unwrap_or("").to_string())
                .collect()
        });
        assert!(names.iter().all(|name| name.starts_with("atompack-")));
    }

    #[test]
    fn nested_install_does_not_deadlock() {
        let sum: u64 = install(|| (0..8u64).into_par_iter().map(|i| install(|| i * 2)).sum());
        assert_eq!(sum, 56);
    }
}
