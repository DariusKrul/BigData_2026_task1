"""
Task 4: Performance Benchmarking
==================================
Runs two experiments using the cached Phase 1 data:

1. SPEEDUP TEST:
   Run the parallel analysis with 1, 2, 4, 6, 8, 10, 12 workers.
   Plot: number of workers (x) vs speedup S = T1/Tn (y)

2. CHUNK SIZE TEST:
   Run the parallel analysis with different batch sizes.
   Plot: batch size (x) vs execution time (y)

Both tests reuse the cached vessel_data from cache_phase1.py, so they
run in seconds instead of minutes.

Usage:
    python benchmark.py

Requires: cached_vessel_data.pkl (created by cache_phase1.py)
"""

import pickle
import time
import multiprocessing

import matplotlib
matplotlib.use("Agg")  # Don't open GUI windows, just save to PNG
import matplotlib.pyplot as plt

from main_pipeline import analyze_batch_full
from parallel_processor import create_worker_batches
from anomaly_detection import detect_ship_transfers


def load_cache(cache_file="cached_vessel_data.pkl"):
    """Load the pre-computed vessel data from disk."""
    print(f"Loading cache from {cache_file}...")
    start = time.time()
    with open(cache_file, "rb") as f:
        cache = pickle.load(f)
    elapsed = time.time() - start
    print(f"  Loaded in {elapsed:.2f} seconds")
    print(f"  Vessels: {len(cache['vessel_data']):,}")
    print(f"  Source files: {cache['source_files']}")
    return cache["vessel_data"], cache["vessel_meta"]


def run_phase3_sequential(vessel_data):
    """Run Phase 3 (Anomaly B) sequentially."""
    start = time.time()
    detect_ship_transfers(vessel_data, parallel=False)
    return time.time() - start


def run_phase3_parallel(vessel_data, num_workers):
    """Run Phase 3 (Anomaly B) with N parallel workers."""
    start = time.time()
    detect_ship_transfers(vessel_data, parallel=True, num_workers=num_workers)
    return time.time() - start


def run_phase2_with_workers(vessel_data, vessel_meta, num_workers, batch_size=50):
    """
    Run Phase 2 (per-vessel analysis) with a specific number of workers.
    Used by the chunk-size test.
    """
    batches = create_worker_batches(vessel_data, vessel_meta, batch_size=batch_size)
    start = time.time()
    with multiprocessing.Pool(processes=num_workers) as pool:
        for batch_result in pool.imap_unordered(analyze_batch_full, batches):
            pass
    return time.time() - start


# ============================================================
# EXPERIMENT 1: SPEEDUP TEST
# ============================================================

def run_speedup_test(vessel_data, vessel_meta):
    """
    Measure speedup of Phase 3 (Anomaly B) as workers increase.

    Phase 3 is the heavy workload in our pipeline — the spatial pair
    scanning across time buckets. This is where parallelism matters most.

    Speedup formula: S(N) = T(1) / T(N)
    where T(1) is sequential Phase 3 time
    and   T(N) is parallel Phase 3 time with N workers
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: SPEEDUP TEST (Phase 3 — Anomaly B)")
    print("=" * 60)

    worker_counts = [1, 2, 4, 6, 8, 10, 12]
    max_cores = multiprocessing.cpu_count()
    worker_counts = [w for w in worker_counts if w <= max_cores]

    # Sequential baseline
    print("\nMeasuring sequential baseline (T1)...")
    seq_times = []
    for run in range(3):
        t = run_phase3_sequential(vessel_data)
        seq_times.append(t)
        print(f"  Run {run + 1}: {t:.3f}s")
    seq_baseline = sorted(seq_times)[1]  # median
    print(f"Sequential baseline (median): {seq_baseline:.3f}s")

    # Parallel runs
    results = {}
    for n_workers in worker_counts:
        print(f"\nTesting {n_workers} worker(s)...")
        times = []
        for run in range(3):
            t = run_phase3_parallel(vessel_data, n_workers)
            times.append(t)
            print(f"  Run {run + 1}: {t:.3f}s")
        median_time = sorted(times)[1]
        speedup = seq_baseline / median_time
        results[n_workers] = {"time": median_time, "speedup": speedup}
        print(f"  Median: {median_time:.3f}s, Speedup: {speedup:.2f}x")

    # Summary table
    print("\n" + "-" * 60)
    print(f"{'Workers':<10}{'Time (s)':<15}{'Speedup':<15}{'Efficiency':<15}")
    print("-" * 60)
    for n in worker_counts:
        r = results[n]
        efficiency = r["speedup"] / n * 100
        print(f"{n:<10}{r['time']:<15.3f}{r['speedup']:<15.2f}{efficiency:<15.1f}%")

    generate_speedup_graph(worker_counts, results, seq_baseline)
    return results, seq_baseline


def generate_speedup_graph(worker_counts, results, seq_baseline):
    """Plot actual speedup vs ideal (linear) speedup."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    speedups = [results[n]["speedup"] for n in worker_counts]
    ideal = worker_counts

    ax1.plot(worker_counts, ideal, "k--", label="Ideal (linear)", linewidth=1.5)
    ax1.plot(worker_counts, speedups, "o-", color="#2E86AB",
             label="Actual", linewidth=2, markersize=8)
    ax1.set_xlabel("Number of Workers")
    ax1.set_ylabel("Speedup Factor S = T(1) / T(N)")
    ax1.set_title("Parallel Speedup (Phase 3 — Anomaly B)")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(worker_counts)

    times = [results[n]["time"] for n in worker_counts]
    ax2.plot(worker_counts, times, "o-", color="#A23B72",
             linewidth=2, markersize=8)
    ax2.axhline(y=seq_baseline, color="gray", linestyle=":",
                label=f"Sequential = {seq_baseline:.2f}s")
    ax2.set_xlabel("Number of Workers")
    ax2.set_ylabel("Execution Time (seconds)")
    ax2.set_title("Phase 3 Execution Time vs Workers")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(worker_counts)

    plt.tight_layout()
    plt.savefig("benchmark_speedup.png", dpi=150, bbox_inches="tight")
    print(f"\n  Graph saved: benchmark_speedup.png")
    plt.close()


# ============================================================
# EXPERIMENT 2: CHUNK SIZE TEST
# ============================================================

def run_chunk_size_test(vessel_data, vessel_meta):
    """
    Measure execution time with different batch sizes (Phase 2).

    Phase 2 distributes vessels across workers in batches. This test
    finds the optimal batch size — too small means high per-batch
    overhead, too large means uneven load balancing.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: CHUNK SIZE TEST (Phase 2 — per-vessel analysis)")
    print("=" * 60)

    num_workers = multiprocessing.cpu_count()
    chunk_sizes = [10, 25, 50, 100, 200, 500]
    print(f"Using {num_workers} workers")

    results = {}
    for batch_size in chunk_sizes:
        print(f"\nTesting batch_size = {batch_size}...")
        times = []
        for run in range(3):
            t = run_phase2_with_workers(vessel_data, vessel_meta,
                                        num_workers, batch_size=batch_size)
            times.append(t)
            print(f"  Run {run + 1}: {t:.3f}s")
        median_time = sorted(times)[1]

        batches = create_worker_batches(vessel_data, vessel_meta, batch_size)
        num_batches = len(batches)

        results[batch_size] = {"time": median_time, "num_batches": num_batches}
        print(f"  Median: {median_time:.3f}s, Batches: {num_batches}")

    print("\n" + "-" * 60)
    print(f"{'Batch Size':<15}{'# Batches':<15}{'Time (s)':<15}")
    print("-" * 60)
    for bs in chunk_sizes:
        r = results[bs]
        print(f"{bs:<15}{r['num_batches']:<15}{r['time']:<15.3f}")

    generate_chunk_graph(chunk_sizes, results)
    return results


def generate_chunk_graph(chunk_sizes, results):
    """Plot execution time vs batch size."""
    fig, ax = plt.subplots(figsize=(9, 5))

    times = [results[bs]["time"] for bs in chunk_sizes]

    ax.plot(chunk_sizes, times, "o-", color="#F18F01",
            linewidth=2, markersize=10)
    ax.set_xlabel("Batch Size (vessels per worker batch)")
    ax.set_ylabel("Execution Time (seconds)")
    ax.set_title("Chunk Size Optimization")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(chunk_sizes)
    ax.set_xticklabels([str(x) for x in chunk_sizes])

    # Mark the minimum
    min_idx = times.index(min(times))
    best_size = chunk_sizes[min_idx]
    ax.annotate(f"Best: {best_size}\n({times[min_idx]:.3f}s)",
                xy=(best_size, times[min_idx]),
                xytext=(10, 20), textcoords="offset points",
                bbox=dict(boxstyle="round", facecolor="lightyellow"),
                arrowprops=dict(arrowstyle="->", color="black"))

    plt.tight_layout()
    plt.savefig("benchmark_chunks.png", dpi=150, bbox_inches="tight")
    print(f"\n  Graph saved: benchmark_chunks.png")
    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("TASK 4: PERFORMANCE BENCHMARKING")
    print("=" * 60)
    print(f"CPU cores available: {multiprocessing.cpu_count()}")

    vessel_data, vessel_meta = load_cache()

    speedup_results, seq_baseline = run_speedup_test(vessel_data, vessel_meta)
    chunk_results = run_chunk_size_test(vessel_data, vessel_meta)

    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - benchmark_speedup.png")
    print("  - benchmark_chunks.png")
    print("\nNext step: run memory profiling with mprof (see next message)")


if __name__ == "__main__":
    main()
