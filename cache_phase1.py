"""
Cache Phase 1 Results
======================
Phase 1 (reading and grouping the CSV files) takes ~10 minutes.
Running it repeatedly for benchmarks would waste hours.

This script runs Phase 1 ONCE and saves the result to a pickle file.
The benchmark script can then load the pickle in seconds and focus
purely on testing Phase 2 (analysis) and Phase 3 (Anomaly B) performance.

Usage:
    python cache_phase1.py aisdk-2025-03-07.csv aisdk-2025-03-08.csv
    # Creates: cached_vessel_data.pkl

Then benchmark.py can load from cached_vessel_data.pkl instantly.
"""

import sys
import time
import pickle

from parallel_processor import group_by_mmsi, group_by_mmsi_multi


def main():
    if len(sys.argv) < 2:
        print("Usage: python cache_phase1.py <csv_file> [csv_file2 ...]")
        sys.exit(1)

    filepaths = sys.argv[1:]
    print(f"Caching Phase 1 for {len(filepaths)} file(s):")
    for fp in filepaths:
        print(f"  - {fp}")

    start = time.time()

    if len(filepaths) == 1:
        vessel_data, vessel_meta = group_by_mmsi(filepaths[0])
    else:
        vessel_data, vessel_meta = group_by_mmsi_multi(filepaths)

    elapsed = time.time() - start
    print(f"\nPhase 1 completed in {elapsed:.2f} seconds")

    # Convert defaultdict to plain dict for pickling
    vessel_data_plain = dict(vessel_data)

    # Save to disk
    cache_file = "cached_vessel_data.pkl"
    print(f"\nSaving to {cache_file}...")

    with open(cache_file, "wb") as f:
        pickle.dump({
            "vessel_data": vessel_data_plain,
            "vessel_meta": vessel_meta,
            "source_files": filepaths,
            "phase1_time": elapsed,
        }, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Report file size
    import os
    size_mb = os.path.getsize(cache_file) / (1024 * 1024)
    print(f"Cache file saved: {cache_file} ({size_mb:.1f} MB)")
    print(f"\nYou can now run benchmark.py — it will load this in seconds.")


if __name__ == "__main__":
    main()
