"""
Main Pipeline: Shadow Fleet Detection
======================================
This script ties everything together:
  1. Streams the AIS file and groups by MMSI (Task 1 + downsampling)
  2. Runs anomaly detection in parallel across CPU cores (Task 2)
  3. Applies Anomalies A, C, D per vessel + Anomaly B across vessels (Task 3)
  4. Calculates DFSI for each vessel
  5. Writes a master incident report to CSV
  6. Reports top suspicious vessels

Usage:
    python main_pipeline.py aisdk-2025-03-07.csv
    python main_pipeline.py aisdk-2025-03-07.csv output_name.csv
"""

import sys
import time
import csv
import multiprocessing
from datetime import datetime

from parallel_processor import (
    group_by_mmsi, group_by_mmsi_multi, create_worker_batches
)
from anomaly_detection import (
    detect_going_dark,
    detect_ship_transfers,
    detect_draft_changes,
    detect_teleportation,
    calculate_dfsi,
)


# ============================================================
# PER-VESSEL ANOMALY RUNNER (used inside worker processes)
# ============================================================

def analyze_vessel_full(args):
    """
    Run Anomalies A, C, D on a single vessel, compute its DFSI.

    This runs INSIDE worker processes. Note that Anomaly B is NOT
    run here — it needs cross-vessel data and must run in the main
    process after all vessels are grouped.

    Args:
        args: tuple of (mmsi, data_points, metadata)
              data_points: list of (ts_epoch, lat, lon, sog, draught)
              metadata: (name, ship_type)

    Returns:
        Dictionary with all per-vessel results
    """
    mmsi, data_points, metadata = args
    name, ship_type = metadata

    # Sort chronologically (critical for all anomaly detectors)
    data_points.sort(key=lambda x: x[0])

    # Run the three per-vessel anomalies
    anomaly_a = detect_going_dark(mmsi, data_points)
    anomaly_c = detect_draft_changes(mmsi, data_points)
    anomaly_d = detect_teleportation(mmsi, data_points)

    # Calculate DFSI for this vessel (Anomaly B added later)
    dfsi = calculate_dfsi(anomaly_a, anomaly_c, anomaly_d)

    return {
        "mmsi": mmsi,
        "name": name,
        "ship_type": ship_type,
        "num_pings": len(data_points),
        "anomaly_a": anomaly_a,
        "anomaly_c": anomaly_c,
        "anomaly_d": anomaly_d,
        "dfsi": dfsi,
    }


def analyze_batch_full(batch):
    """Worker function: analyze a batch of vessels."""
    return [analyze_vessel_full(args) for args in batch]


def run_parallel_analysis(vessel_data, vessel_meta, num_workers=None):
    """
    Run per-vessel anomaly detection across multiple CPU cores.

    Returns:
        (list of vessel result dicts, elapsed seconds)
    """
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()

    print(f"\nPhase 2a: Parallel analysis with {num_workers} workers")
    batches = create_worker_batches(vessel_data, vessel_meta, batch_size=50)
    print(f"  Created {len(batches)} batches of vessels")

    start = time.time()
    all_results = []
    with multiprocessing.Pool(processes=num_workers) as pool:
        for batch_result in pool.imap_unordered(analyze_batch_full, batches):
            all_results.extend(batch_result)
    elapsed = time.time() - start

    print(f"  Parallel analysis complete in {elapsed:.2f} seconds")
    return all_results, elapsed


def run_sequential_analysis(vessel_data, vessel_meta):
    """
    Run per-vessel anomaly detection on a single core (for speedup comparison).

    Returns:
        (list of vessel result dicts, elapsed seconds)
    """
    print(f"\nPhase 2b: Sequential analysis (single core)")
    start = time.time()

    all_results = []
    for mmsi, data_points in vessel_data.items():
        metadata = vessel_meta.get(mmsi, ("Unknown", "Unknown"))
        # Copy data_points so sort doesn't affect parallel run's list
        result = analyze_vessel_full((mmsi, list(data_points), metadata))
        all_results.append(result)

    elapsed = time.time() - start
    print(f"  Sequential analysis complete in {elapsed:.2f} seconds")
    return all_results, elapsed


# ============================================================
# MASTER INCIDENT REPORT
# ============================================================

def build_incident_records(vessel_results, anomaly_b_incidents):
    """
    Flatten all incidents into a single unified list of records.

    Each record has a common set of columns so they can be written
    to one master CSV, even though different anomaly types have
    different details. Anomaly-specific fields go in 'details'.

    Args:
        vessel_results: List of per-vessel results from parallel analysis
        anomaly_b_incidents: List of cross-vessel transfer incidents

    Returns:
        List of flat incident dictionaries
    """
    records = []

    # --- Per-vessel incidents (A, C, D) ---
    for vr in vessel_results:
        mmsi = vr["mmsi"]
        name = vr["name"]
        ship_type = vr["ship_type"]

        # Anomaly A: Going Dark
        for inc in vr["anomaly_a"]:
            records.append({
                "anomaly_type": "A",
                "mmsi": mmsi,
                "mmsi_other": "",
                "name": name,
                "ship_type": ship_type,
                "start_time": inc["disappear_time"],
                "end_time": inc["reappear_time"],
                "lat_start": inc["disappear_lat"],
                "lon_start": inc["disappear_lon"],
                "lat_end": inc["reappear_lat"],
                "lon_end": inc["reappear_lon"],
                "duration_hours": inc["gap_hours"],
                "distance_nm": inc["distance_nm"],
                "implied_speed_knots": inc["implied_speed_knots"],
                "draught_before": "",
                "draught_after": "",
                "draught_change_percent": "",
                "description": inc["description"],
            })

        # Anomaly C: Draft Changes
        for inc in vr["anomaly_c"]:
            records.append({
                "anomaly_type": "C",
                "mmsi": mmsi,
                "mmsi_other": "",
                "name": name,
                "ship_type": ship_type,
                "start_time": inc["disappear_time"],
                "end_time": inc["reappear_time"],
                "lat_start": inc["disappear_lat"],
                "lon_start": inc["disappear_lon"],
                "lat_end": inc["reappear_lat"],
                "lon_end": inc["reappear_lon"],
                "duration_hours": inc["gap_hours"],
                "distance_nm": inc["distance_nm"],
                "implied_speed_knots": "",
                "draught_before": inc["draught_before"],
                "draught_after": inc["draught_after"],
                "draught_change_percent": inc["draught_change_percent"],
                "description": inc["description"],
            })

        # Anomaly D: Teleportation
        for inc in vr["anomaly_d"]:
            records.append({
                "anomaly_type": "D",
                "mmsi": mmsi,
                "mmsi_other": "",
                "name": name,
                "ship_type": ship_type,
                "start_time": inc["time_1"],
                "end_time": inc["time_2"],
                "lat_start": inc["lat_1"],
                "lon_start": inc["lon_1"],
                "lat_end": inc["lat_2"],
                "lon_end": inc["lon_2"],
                "duration_hours": round(inc["time_diff_minutes"] / 60, 3),
                "distance_nm": inc["distance_nm"],
                "implied_speed_knots": inc["implied_speed_knots"],
                "draught_before": "",
                "draught_after": "",
                "draught_change_percent": "",
                "description": inc["description"],
            })

    # --- Cross-vessel incidents (B) ---
    for inc in anomaly_b_incidents:
        records.append({
            "anomaly_type": "B",
            "mmsi": inc["mmsi"],
            "mmsi_other": inc["mmsi_b"],
            "name": "",
            "ship_type": "",
            "start_time": inc["start_time"],
            "end_time": inc["end_time"],
            "lat_start": inc["vessel_a_lat"],
            "lon_start": inc["vessel_a_lon"],
            "lat_end": inc["vessel_b_lat"],
            "lon_end": inc["vessel_b_lon"],
            "duration_hours": inc["duration_hours"],
            "distance_nm": inc["min_distance_nm"],
            "implied_speed_knots": "",
            "draught_before": "",
            "draught_after": "",
            "draught_change_percent": "",
            "description": inc["description"],
        })

    return records


def write_incident_csv(records, filepath):
    """
    Write the master incident report to CSV.

    Args:
        records: List of incident dictionaries
        filepath: Output CSV path
    """
    if not records:
        print(f"  No incidents to write.")
        return

    fieldnames = [
        "anomaly_type", "mmsi", "mmsi_other", "name", "ship_type",
        "start_time", "end_time",
        "lat_start", "lon_start", "lat_end", "lon_end",
        "duration_hours", "distance_nm", "implied_speed_knots",
        "draught_before", "draught_after", "draught_change_percent",
        "description",
    ]

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"  Wrote {len(records)} incidents to {filepath}")


def write_dfsi_ranking(vessel_results, filepath):
    """
    Write the DFSI ranking CSV — one row per vessel with its total score.
    """
    # Only keep vessels with any anomalies or non-zero DFSI
    flagged = [v for v in vessel_results
               if v["dfsi"] > 0
               or v["anomaly_a"] or v["anomaly_c"] or v["anomaly_d"]]
    flagged.sort(key=lambda x: x["dfsi"], reverse=True)

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "rank", "mmsi", "name", "ship_type", "dfsi",
            "anomaly_a_count", "anomaly_c_count", "anomaly_d_count",
            "num_pings"
        ])
        for rank, v in enumerate(flagged, start=1):
            writer.writerow([
                rank, v["mmsi"], v["name"], v["ship_type"], v["dfsi"],
                len(v["anomaly_a"]), len(v["anomaly_c"]), len(v["anomaly_d"]),
                v["num_pings"],
            ])

    print(f"  Wrote {len(flagged)} flagged vessels to {filepath}")
    return flagged


# ============================================================
# MAIN
# ============================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python main_pipeline.py <csv_file> [csv_file2 ...] [--output PREFIX]")
        print("Examples:")
        print("  python main_pipeline.py aisdk-2025-03-07.csv")
        print("  python main_pipeline.py aisdk-2025-03-07.csv aisdk-2025-03-08.csv --output combined")
        sys.exit(1)

    # Parse arguments: collect all .csv paths and look for --output
    args = sys.argv[1:]
    output_prefix = "results"
    filepaths = []
    i = 0
    while i < len(args):
        if args[i] == "--output" and i + 1 < len(args):
            output_prefix = args[i + 1]
            i += 2
        else:
            filepaths.append(args[i])
            i += 1

    print(f"{'='*60}")
    print(f"SHADOW FLEET DETECTION PIPELINE")
    print(f"Input files ({len(filepaths)}):")
    for fp in filepaths:
        print(f"  - {fp}")
    print(f"Output prefix: {output_prefix}")
    print(f"CPU cores: {multiprocessing.cpu_count()}")
    print(f"{'='*60}")

    total_start = time.time()

    # --- Phase 1: Stream, filter, downsample, group by MMSI ---
    phase1_start = time.time()
    if len(filepaths) == 1:
        vessel_data, vessel_meta = group_by_mmsi(filepaths[0])
    else:
        vessel_data, vessel_meta = group_by_mmsi_multi(filepaths)
    phase1_time = time.time() - phase1_start
    print(f"  Phase 1 took {phase1_time:.2f} seconds")

    # --- Phase 2a: Sequential analysis (for speedup comparison) ---
    seq_results, seq_time = run_sequential_analysis(vessel_data, vessel_meta)

    # --- Phase 2b: Parallel analysis (real run) ---
    par_results, par_time = run_parallel_analysis(vessel_data, vessel_meta)

    # --- Phase 3: Cross-vessel Anomaly B (parallelized) ---
    print(f"\nPhase 3: Anomaly B (cross-vessel loitering/transfers)")
    phase_b_start = time.time()
    anomaly_b_incidents = detect_ship_transfers(vessel_data, parallel=True)
    phase_b_time = time.time() - phase_b_start
    print(f"  Phase 3 took {phase_b_time:.2f} seconds")

    # --- Phase 5: Write output files ---
    print(f"\nPhase 5: Writing output files")
    incident_records = build_incident_records(par_results, anomaly_b_incidents)
    incidents_csv = f"{output_prefix}_incidents.csv"
    dfsi_csv = f"{output_prefix}_dfsi_ranking.csv"

    write_incident_csv(incident_records, incidents_csv)
    flagged = write_dfsi_ranking(par_results, dfsi_csv)

    # --- Summary ---
    total_time = time.time() - total_start
    speedup = seq_time / par_time if par_time > 0 else 0

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"  Phase 1 (group):          {phase1_time:.2f}s")
    print(f"  Phase 2a (sequential):    {seq_time:.2f}s")
    print(f"  Phase 2b (parallel):      {par_time:.2f}s")
    print(f"  Phase 3 (anomaly B):      {phase_b_time:.2f}s")
    print(f"  TOTAL:                    {total_time:.2f}s")
    print(f"  Speedup (S = Tseq/Tpar):  {speedup:.2f}x")

    # Incident counts by type
    a_count = sum(len(v["anomaly_a"]) for v in par_results)
    c_count = sum(len(v["anomaly_c"]) for v in par_results)
    d_count = sum(len(v["anomaly_d"]) for v in par_results)
    b_count = len(anomaly_b_incidents)

    print(f"\n  Anomaly A (Going Dark):           {a_count}")
    print(f"  Anomaly B (Transfers):            {b_count}")
    print(f"  Anomaly C (Draft Changes):        {c_count}")
    print(f"  Anomaly D (Identity Cloning):     {d_count}")
    print(f"  Total incidents:                  {a_count + b_count + c_count + d_count}")
    print(f"  Vessels flagged with DFSI > 0:    {len(flagged)}")

    # Top 10 most suspicious vessels
    print(f"\n  TOP 10 VESSELS BY DFSI:")
    print(f"  {'Rank':<6}{'MMSI':<12}{'DFSI':<10}{'A':<5}{'C':<5}{'D':<5}{'Name'}")
    for rank, v in enumerate(flagged[:10], start=1):
        print(f"  {rank:<6}{v['mmsi']:<12}{v['dfsi']:<10}"
              f"{len(v['anomaly_a']):<5}{len(v['anomaly_c']):<5}"
              f"{len(v['anomaly_d']):<5}{v['name']}")


if __name__ == "__main__":
    main()
