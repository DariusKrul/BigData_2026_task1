"""
Task 2: Parallel Processing Pipeline for AIS Data
===================================================
This module builds on the partitioner (Task 1) to process AIS data
across multiple CPU cores using Python's multiprocessing library.

Architecture:
    Phase 1 (Group):  Stream through file, group rows by MMSI
    Phase 2 (Parallel Analyze): Distribute ship groups across workers

Key concepts:
    - multiprocessing.Pool for worker management
    - Map-Reduce pattern: group (map) then analyze (reduce)
    - Tuples instead of dicts for memory efficiency
    - Chronological sorting per vessel
"""

import csv
import sys
import time
import multiprocessing
from collections import defaultdict
from datetime import datetime

# Import our partitioner from Task 1
from partitioner import (
    parse_row, is_valid_mmsi, is_valid_position,
    NON_VESSEL_TYPES, safe_float, parse_timestamp,
    COL_TIMESTAMP, COL_TYPE_OF_MOBILE, COL_MMSI,
    COL_LATITUDE, COL_LONGITUDE, COL_SOG, COL_DRAUGHT,
    COL_NAME, COL_SHIP_TYPE
)


# ============================================================
# PHASE 1: GROUP ROWS BY MMSI (Memory-Efficient)
# ============================================================

def floor_to_bucket(ts_epoch, interval_seconds=120):
    """
    Round an epoch timestamp DOWN to the nearest time bucket.

    Example with 2-minute (120-second) intervals:
        10:00:00 -> 10:00:00  (already on boundary)
        10:00:45 -> 10:00:00  (rounded down)
        10:01:30 -> 10:00:00  (still same bucket)
        10:02:01 -> 10:02:00  (new bucket)

    This is how downsampling works: all timestamps within the same
    2-minute window get the same bucket value. We only keep the first
    point per bucket per vessel, reducing ~20 pings to 1.

    Args:
        ts_epoch: Timestamp as epoch float (seconds since 1970)
        interval_seconds: Bucket size in seconds (default: 120 = 2 minutes)

    Returns:
        The bucket's start time as epoch float
    """
    return ts_epoch - (ts_epoch % interval_seconds)


def group_by_mmsi(filepath, downsample_interval=120):
    """
    Stream through the CSV file and group all clean rows by MMSI,
    with downsampling to reduce data volume.

    Instead of storing full dictionaries, we store compact tuples:
        (ts_epoch, lat, lon, sog, draught)

    Why tuples?
    - A dict like {"timestamp": ..., "lat": ..., "lon": ...} stores
      the key strings for EVERY row. With 21 million rows, those
      repeated strings eat up gigabytes.
    - A tuple (ts_epoch, lat, lon, sog, draught) stores only the
      values. Python tuples are also stored more compactly in memory.

    Why epoch floats instead of datetime objects?
    - A float is 8 bytes. A datetime object is much larger.
    - When multiprocessing sends data to workers, it "pickles"
      (serializes) the data. Floats pickle cheaply; datetimes don't.
    - This was the fix for the MemoryError we hit on the first attempt.

    Downsampling:
    - Ships can ping every few seconds, producing thousands of redundant
      rows. We keep only one point per 2-minute window per vessel.
    - This reduces ~21 million rows to ~1.3 million while preserving
      the vessel's trajectory for anomaly detection.
    - For detecting 4-hour gaps or 2-hour loitering, 2-minute resolution
      is more than sufficient.

    We also separately track ship metadata (name, ship_type) since
    these don't change per-ping — we only need to store them once
    per MMSI.

    Args:
        filepath: Path to the AIS CSV file
        downsample_interval: Seconds per time bucket (default: 120 = 2 min)

    Returns:
        vessel_data: dict mapping MMSI -> list of tuples
                     Each tuple: (ts_epoch, lat, lon, sog, draught)
                     ts_epoch is seconds since 1970-01-01 (float)
        vessel_meta: dict mapping MMSI -> (name, ship_type)
    """
    vessel_data = defaultdict(list)
    vessel_meta = {}
    # Track the last time bucket stored for each vessel.
    # Key: MMSI, Value: the bucket start epoch of the last stored point.
    last_bucket = {}
    rows_processed = 0
    rows_filtered = 0
    rows_downsampled = 0

    print(f"Phase 1: Grouping rows by MMSI from {filepath}")
    print(f"  Downsampling interval: {downsample_interval} seconds")

    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # Skip header

        for raw_row in reader:
            rows_processed += 1

            if rows_processed % 2_000_000 == 0:
                print(f"  ...read {rows_processed:,} rows, "
                      f"{len(vessel_data):,} vessels found, "
                      f"{rows_downsampled:,} downsampled away")

            # --- Quick filtering (inlined for speed) ---
            if len(raw_row) < 19:
                rows_filtered += 1
                continue

            mobile_type = raw_row[COL_TYPE_OF_MOBILE].strip()
            if mobile_type in NON_VESSEL_TYPES:
                rows_filtered += 1
                continue

            mmsi = raw_row[COL_MMSI].strip()
            if not is_valid_mmsi(mmsi):
                rows_filtered += 1
                continue

            timestamp = parse_timestamp(raw_row[COL_TIMESTAMP].strip())
            if timestamp is None:
                rows_filtered += 1
                continue

            # Convert datetime to epoch float (seconds since 1970).
            # Why? Floats are 8 bytes each and pickle instantly.
            # datetime objects are much larger and slow to serialize
            # when sent to worker processes via multiprocessing.
            ts_epoch = timestamp.timestamp()

            # --- Downsampling ---
            # Round timestamp to nearest 2-minute bucket. If we already
            # stored a point for this vessel in this bucket, skip it.
            bucket = floor_to_bucket(ts_epoch, downsample_interval)
            if last_bucket.get(mmsi) == bucket:
                rows_downsampled += 1
                continue
            last_bucket[mmsi] = bucket

            lat = safe_float(raw_row[COL_LATITUDE].strip())
            lon = safe_float(raw_row[COL_LONGITUDE].strip())
            if not is_valid_position(lat, lon):
                rows_filtered += 1
                continue

            sog = safe_float(raw_row[COL_SOG].strip())
            draught = safe_float(raw_row[COL_DRAUGHT].strip())

            # --- Store as compact tuple ---
            # Tuple index:  0=ts_epoch, 1=lat, 2=lon, 3=sog, 4=draught
            # Using epoch float instead of datetime for memory efficiency
            vessel_data[mmsi].append((ts_epoch, lat, lon, sog, draught))

            # --- Store metadata once per MMSI ---
            if mmsi not in vessel_meta:
                name = raw_row[COL_NAME].strip()
                ship_type = raw_row[COL_SHIP_TYPE].strip()
                vessel_meta[mmsi] = (name, ship_type)

    print(f"  Phase 1 complete: {rows_processed:,} rows read, "
          f"{rows_filtered:,} filtered, "
          f"{rows_downsampled:,} downsampled, "
          f"{len(vessel_data):,} valid vessels")

    # Count total points kept for reporting
    total_kept = sum(len(points) for points in vessel_data.values())
    print(f"  Points kept after downsampling: {total_kept:,} "
          f"({total_kept/rows_processed*100:.1f}% of original)")

    return vessel_data, vessel_meta


def group_by_mmsi_multi(filepaths, downsample_interval=120):
    """
    Process multiple AIS files and merge their data into one dataset.

    For each file, we call group_by_mmsi() which returns its own
    vessel_data and vessel_meta. We then merge them into a combined
    dataset where each vessel's points span across all files.

    Why merge instead of process each file independently?
    - Some anomalies span midnight (e.g., a ship goes dark at 22:00
      on day 1 and reappears at 02:00 on day 2 — that's ONE incident)
    - DFSI is a per-vessel score that should account for the vessel's
      behavior across the entire observation window
    - A single combined report is easier to analyze than two separate ones

    The merge is straightforward because epoch timestamps are absolute:
    we just append each file's per-vessel points to the combined list.
    Sorting (done later in the analysis phase) handles ordering.

    Args:
        filepaths: List of CSV file paths to process
        downsample_interval: Seconds per time bucket for downsampling

    Returns:
        combined_vessel_data: dict mapping MMSI -> combined list of tuples
        combined_vessel_meta: dict mapping MMSI -> (name, ship_type)
    """
    from collections import defaultdict

    combined_vessel_data = defaultdict(list)
    combined_vessel_meta = {}

    for i, filepath in enumerate(filepaths, start=1):
        print(f"\n--- File {i}/{len(filepaths)}: {filepath} ---")
        vessel_data, vessel_meta = group_by_mmsi(filepath, downsample_interval)

        # Merge: extend each vessel's combined list with this file's points
        for mmsi, points in vessel_data.items():
            combined_vessel_data[mmsi].extend(points)

        # Metadata: keep first non-empty values seen
        for mmsi, meta in vessel_meta.items():
            if mmsi not in combined_vessel_meta:
                combined_vessel_meta[mmsi] = meta

    total_vessels = len(combined_vessel_data)
    total_points = sum(len(p) for p in combined_vessel_data.values())
    print(f"\n--- Combined dataset ---")
    print(f"  Total unique vessels:       {total_vessels:,}")
    print(f"  Total downsampled points:   {total_points:,}")

    return combined_vessel_data, combined_vessel_meta


# ============================================================
# PHASE 2: PARALLEL ANALYSIS
# ============================================================

def analyze_vessel(args):
    """
    Analyze a single vessel's AIS data for anomalies.

    This function runs INSIDE a worker process. Each worker receives
    one vessel's data (one MMSI) and performs:
    1. Chronological sorting (critical! the file isn't guaranteed sorted)
    2. Anomaly detection (A, B, C, D) — implemented in Task 3

    For now (Task 2), we set up the framework and do the sorting.
    The actual anomaly logic will be added in Task 3.

    Args:
        args: A tuple of (mmsi, data_points, metadata)
              data_points: list of (timestamp, lat, lon, sog, draught)
              metadata: (name, ship_type)

    Returns:
        A dictionary with the vessel's analysis results
    """
    mmsi, data_points, metadata = args
    name, ship_type = metadata

    # --- Step 1: Sort chronologically ---
    # This is critical. AIS data in the file is roughly time-ordered
    # but not guaranteed. We sort by epoch timestamp (index 0 of tuple).
    data_points.sort(key=lambda x: x[0])

    # --- Step 2: Placeholder for anomaly detection (Task 3) ---
    # For now, we return basic statistics about the vessel
    num_pings = len(data_points)
    first_seen = data_points[0][0]   # Earliest epoch timestamp
    last_seen = data_points[-1][0]   # Latest epoch timestamp
    time_span = (last_seen - first_seen) / 3600  # Seconds -> Hours

    # Count how many pings have draught data
    draught_count = sum(1 for p in data_points if p[4] is not None)

    # Convert epoch back to readable string for display
    from datetime import datetime
    first_seen_str = datetime.fromtimestamp(first_seen).strftime("%Y-%m-%d %H:%M:%S")
    last_seen_str = datetime.fromtimestamp(last_seen).strftime("%Y-%m-%d %H:%M:%S")

    result = {
        "mmsi": mmsi,
        "name": name,
        "ship_type": ship_type,
        "num_pings": num_pings,
        "first_seen": first_seen_str,
        "last_seen": last_seen_str,
        "time_span_hours": round(time_span, 2),
        "has_draught_data": draught_count > 0,
        "draught_ping_count": draught_count,
        # Anomaly results will be added in Task 3:
        "anomaly_a_gaps": [],
        "anomaly_b_transfers": [],
        "anomaly_c_draft_changes": [],
        "anomaly_d_teleports": [],
        "dfsi": 0.0,
    }

    return result


def create_worker_batches(vessel_data, vessel_meta, batch_size=50):
    """
    Group multiple vessels into batches for worker processes.

    Why batches? If we send one vessel at a time to a worker, the
    overhead of starting/communicating with the worker can exceed
    the actual work time for small vessels. By batching ~50 vessels
    per worker call, we reduce this overhead.

    Args:
        vessel_data: dict mapping MMSI -> list of data tuples
        vessel_meta: dict mapping MMSI -> (name, ship_type)
        batch_size: Number of vessels per batch

    Returns:
        A list of lists, where each inner list contains
        (mmsi, data_points, metadata) tuples
    """
    all_vessels = []
    for mmsi, data_points in vessel_data.items():
        metadata = vessel_meta.get(mmsi, ("Unknown", "Unknown"))
        all_vessels.append((mmsi, data_points, metadata))

    # Sort by number of pings (descending) so large vessels are
    # spread across batches evenly
    all_vessels.sort(key=lambda x: len(x[1]), reverse=True)

    batches = []
    for i in range(0, len(all_vessels), batch_size):
        batches.append(all_vessels[i:i + batch_size])

    return batches


def analyze_batch(batch):
    """
    Analyze a batch of vessels. This is what each worker process runs.

    Args:
        batch: List of (mmsi, data_points, metadata) tuples

    Returns:
        List of result dictionaries, one per vessel
    """
    results = []
    for vessel_args in batch:
        result = analyze_vessel(vessel_args)
        results.append(result)
    return results


def run_parallel(vessel_data, vessel_meta, num_workers=None):
    """
    Distribute vessel analysis across multiple CPU cores.

    Uses multiprocessing.Pool which:
    1. Creates N worker processes (one per CPU core by default)
    2. Sends batches of vessels to workers via pool.map()
    3. Collects results from all workers
    4. Returns combined results

    Args:
        vessel_data: dict mapping MMSI -> data tuples
        vessel_meta: dict mapping MMSI -> metadata
        num_workers: Number of CPU cores to use (None = all available)

    Returns:
        List of all vessel analysis results
    """
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()

    print(f"\nPhase 2: Parallel analysis with {num_workers} workers")
    print(f"  Total vessels to analyze: {len(vessel_data):,}")

    # Create batches
    batches = create_worker_batches(vessel_data, vessel_meta, batch_size=50)
    print(f"  Created {len(batches)} batches")

    # Run in parallel using imap instead of map.
    # pool.map queues ALL batches at once (huge memory spike during pickling).
    # pool.imap sends batches one at a time as workers become free,
    # so only a few batches are in transit at any moment.
    start_time = time.time()

    all_results = []
    with multiprocessing.Pool(processes=num_workers) as pool:
        for batch_result in pool.imap_unordered(analyze_batch, batches):
            all_results.extend(batch_result)

    elapsed = time.time() - start_time

    print(f"  Parallel analysis complete in {elapsed:.2f} seconds")
    print(f"  Processed {len(all_results):,} vessels")

    return all_results, elapsed


def run_sequential(vessel_data, vessel_meta):
    """
    Run the same analysis on a single core (for speedup comparison).

    This is identical to run_parallel but doesn't use multiprocessing.
    We need this for Task 4's speedup calculation: S = T_seq / T_par

    Args:
        vessel_data: dict mapping MMSI -> data tuples
        vessel_meta: dict mapping MMSI -> metadata

    Returns:
        List of all vessel analysis results
    """
    print(f"\nSequential analysis (single core)")
    print(f"  Total vessels to analyze: {len(vessel_data):,}")

    start_time = time.time()

    all_results = []
    for mmsi, data_points in vessel_data.items():
        metadata = vessel_meta.get(mmsi, ("Unknown", "Unknown"))
        result = analyze_vessel((mmsi, data_points, metadata))
        all_results.append(result)

    elapsed = time.time() - start_time

    print(f"  Sequential analysis complete in {elapsed:.2f} seconds")
    print(f"  Processed {len(all_results):,} vessels")

    return all_results, elapsed


# ============================================================
# MAIN: TEST THE PIPELINE
# ============================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python parallel_processor.py <path_to_csv_file>")
        sys.exit(1)

    filepath = sys.argv[1]
    print(f"{'='*60}")
    print(f"AIS Parallel Processing Pipeline")
    print(f"File: {filepath}")
    print(f"Available CPU cores: {multiprocessing.cpu_count()}")
    print(f"{'='*60}")

    # Phase 1: Group by MMSI
    phase1_start = time.time()
    vessel_data, vessel_meta = group_by_mmsi(filepath)
    phase1_time = time.time() - phase1_start
    print(f"  Phase 1 took {phase1_time:.2f} seconds")

    # Phase 2a: Sequential (for comparison)
    seq_results, seq_time = run_sequential(vessel_data, vessel_meta)

    # Phase 2b: Parallel
    par_results, par_time = run_parallel(vessel_data, vessel_meta)

    # Speedup calculation
    speedup = seq_time / par_time if par_time > 0 else 0
    print(f"\n{'='*60}")
    print(f"PERFORMANCE COMPARISON")
    print(f"{'='*60}")
    print(f"  Phase 1 (grouping):    {phase1_time:.2f}s")
    print(f"  Sequential analysis:   {seq_time:.2f}s")
    print(f"  Parallel analysis:     {par_time:.2f}s")
    print(f"  Speedup factor:        {speedup:.2f}x")
    print(f"  CPU cores used:        {multiprocessing.cpu_count()}")

    # Show some sample results
    print(f"\nSample results (first 5 vessels with most pings):")
    sorted_results = sorted(par_results,
                            key=lambda x: x["num_pings"],
                            reverse=True)
    for r in sorted_results[:5]:
        print(f"  MMSI: {r['mmsi']}, Name: {r['name'] or 'N/A'}, "
              f"Pings: {r['num_pings']:,}, "
              f"Span: {r['time_span_hours']:.1f}h, "
              f"Draught data: {r['has_draught_data']}")
