"""
Diagnostic: Why does Anomaly C return 0 incidents?

Possible causes:
1. No vessels report draught at all
2. Vessels that report draught never have > 2-hour gaps
3. Vessels with draught + gaps never have > 5% draught change
4. A bug in our detection logic

This script answers each question by streaming the file once.
"""

import sys
from collections import defaultdict
from parallel_processor import group_by_mmsi


def diagnose_anomaly_c(filepath):
    print(f"Loading data from {filepath}...")
    vessel_data, vessel_meta = group_by_mmsi(filepath)

    # Question 1: How many vessels report ANY draught data?
    vessels_with_draught = 0
    vessels_with_multiple_draughts = 0
    total_draught_pings = 0
    unique_draught_values_per_vessel = defaultdict(set)

    for mmsi, points in vessel_data.items():
        draught_points = [p for p in points if p[4] is not None and p[4] > 0]
        if draught_points:
            vessels_with_draught += 1
            total_draught_pings += len(draught_points)
            for p in draught_points:
                unique_draught_values_per_vessel[mmsi].add(round(p[4], 2))
            if len(unique_draught_values_per_vessel[mmsi]) >= 2:
                vessels_with_multiple_draughts += 1

    print(f"\n--- Question 1: Draught reporting ---")
    print(f"  Total vessels:                       {len(vessel_data):,}")
    print(f"  Vessels reporting any draught:       {vessels_with_draught:,}")
    print(f"  Vessels reporting >=2 unique values: {vessels_with_multiple_draughts:,}")
    print(f"  Total pings with draught:            {total_draught_pings:,}")

    if vessels_with_draught == 0:
        print("\nCONCLUSION: No vessel reports draught at all. Anomaly C cannot fire.")
        return

    # Question 2: For vessels with multiple draught values,
    # do any have a gap > 2 hours BETWEEN draught-reporting pings?
    print(f"\n--- Question 2: Gaps between draught pings ---")

    GAP_THRESHOLD = 2 * 3600  # 2 hours
    DRAFT_CHANGE_PERCENT = 5.0

    vessels_with_gaps = 0
    vessels_with_changes = 0
    vessels_with_changes_during_gaps = 0
    sample_changes = []

    for mmsi, points in vessel_data.items():
        # Sort chronologically
        points = sorted(points, key=lambda x: x[0])

        # Walk through consecutive pairs
        had_gap = False
        had_change = False
        had_both = False

        for i in range(len(points) - 1):
            ts1, lat1, lon1, sog1, draught1 = points[i]
            ts2, lat2, lon2, sog2, draught2 = points[i + 1]

            if draught1 is None or draught2 is None:
                continue
            if draught1 <= 0 or draught2 <= 0:
                continue

            gap = ts2 - ts1
            change_pct = abs(draught2 - draught1) / draught1 * 100

            if gap > GAP_THRESHOLD:
                had_gap = True
            if change_pct > DRAFT_CHANGE_PERCENT:
                had_change = True
            if gap > GAP_THRESHOLD and change_pct > DRAFT_CHANGE_PERCENT:
                had_both = True
                if len(sample_changes) < 5:
                    sample_changes.append({
                        "mmsi": mmsi,
                        "gap_hours": gap / 3600,
                        "draught_before": draught1,
                        "draught_after": draught2,
                        "change_pct": change_pct,
                    })

        if had_gap:
            vessels_with_gaps += 1
        if had_change:
            vessels_with_changes += 1
        if had_both:
            vessels_with_changes_during_gaps += 1

    print(f"  Vessels with draught + gap > 2h:           {vessels_with_gaps:,}")
    print(f"  Vessels with draught change > 5%:          {vessels_with_changes:,}")
    print(f"  Vessels with BOTH (would trigger Anom C):  {vessels_with_changes_during_gaps:,}")

    if sample_changes:
        print(f"\n  Sample matches (would be Anomaly C):")
        for s in sample_changes:
            print(f"    MMSI {s['mmsi']}: gap {s['gap_hours']:.1f}h, "
                  f"draught {s['draught_before']:.2f} -> {s['draught_after']:.2f} "
                  f"({s['change_pct']:.1f}%)")
    else:
        print(f"\n  No vessel has draught + gap + change all aligned.")

    # Question 3: How "stable" is draught when it IS reported?
    # Do values change at all, or do ships report constant draught?
    print(f"\n--- Question 3: Draught stability ---")
    constant_draught_vessels = sum(
        1 for mmsi, vals in unique_draught_values_per_vessel.items()
        if len(vals) == 1)
    multi_value = sum(
        1 for mmsi, vals in unique_draught_values_per_vessel.items()
        if len(vals) > 1)
    print(f"  Vessels with constant draught (1 value):     {constant_draught_vessels:,}")
    print(f"  Vessels with multiple draught values:        {multi_value:,}")

    if multi_value > 0:
        max_unique = max(len(v) for v in unique_draught_values_per_vessel.values())
        print(f"  Max unique draught values for one vessel:    {max_unique}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python diagnose_anomaly_c.py <csv_file>")
        sys.exit(1)
    diagnose_anomaly_c(sys.argv[1])
