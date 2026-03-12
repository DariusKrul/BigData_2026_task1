"""
AIS Data Exploration Script
============================
This script explores the AIS CSV files WITHOUT loading them into memory.
It reads line-by-line (streaming) to stay memory-friendly.

Usage:
    python explore_data.py path/to/your_file.csv
"""

import csv
import sys
from collections import Counter


def peek_at_data(filepath, num_rows=5):
    """
    Read and display the first few rows of the CSV.

    Why line-by-line? Because the file is 3-4 GB. We only want to see
    a handful of rows, so we open the file, grab a few lines, and stop.
    The 'with' statement ensures the file is properly closed after.
    """
    print(f"\n{'=' * 60}")
    print(f"PEEKING AT FIRST {num_rows} ROWS")
    print(f"{'=' * 60}")

    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)  # First line is always the column names

        print(f"\nColumn names ({len(header)} columns):")
        for i, col in enumerate(header):
            print(f"  [{i}] {col}")

        print(f"\nFirst {num_rows} data rows:")
        for row_num, row in enumerate(reader):
            if row_num >= num_rows:
                break  # Stop after we've seen enough — don't read the whole file!
            print(f"\n  --- Row {row_num + 1} ---")
            for i, col in enumerate(header):
                print(f"  {col}: {row[i] if i < len(row) else 'MISSING'}")


def count_and_analyze(filepath):
    """
    Stream through the ENTIRE file, counting rows and tracking MMSI frequency.

    KEY CONCEPT: We never store the actual rows. We only keep:
      - A running count (one integer)
      - A Counter dict for MMSIs (one entry per unique MMSI)

    Even with 20 million rows, we only store maybe 50,000 unique MMSIs
    in memory — that's tiny compared to the full dataset.
    """
    print(f"\n{'=' * 60}")
    print(f"FULL FILE ANALYSIS (streaming — this may take a few minutes)")
    print(f"{'=' * 60}")

    mmsi_counts = Counter()  # Tracks how many rows each MMSI has
    total_rows = 0
    empty_rows = 0
    bad_rows = 0  # Rows that can't be parsed properly

    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip the header

        # Find which column index corresponds to MMSI
        # The header from your file: # Timestamp, Type of mobile, MMSI, ...
        # Note: the first column name starts with "# " which is a quirk
        try:
            mmsi_index = header.index("MMSI")
        except ValueError:
            # Sometimes the header has extra spaces or the # prefix
            # Let's find it by checking each column
            mmsi_index = None
            for i, col in enumerate(header):
                if "MMSI" in col.upper():
                    mmsi_index = i
                    break
            if mmsi_index is None:
                print("ERROR: Could not find MMSI column!")
                return

        print(f"MMSI is in column index: {mmsi_index}")
        print("Streaming through file...")

        for row in reader:
            total_rows += 1

            # Progress indicator every 1 million rows
            if total_rows % 1_000_000 == 0:
                print(f"  ...processed {total_rows:,} rows so far")

            # Handle rows that might be malformed
            if len(row) <= mmsi_index:
                bad_rows += 1
                continue

            mmsi = row[mmsi_index].strip()

            if mmsi == "":
                empty_rows += 1
                continue

            mmsi_counts[mmsi] += 1

    # --- Print Results ---
    print(f"\nTotal data rows: {total_rows:,}")
    print(f"Rows with empty MMSI: {empty_rows:,}")
    print(f"Malformed rows: {bad_rows:,}")
    print(f"Unique MMSI values: {len(mmsi_counts):,}")

    # Show the top 20 most frequent MMSIs — this reveals dirty data
    print(f"\nTop 20 most frequent MMSIs:")
    print(f"  {'MMSI':<15} {'Count':>12}  {'Note'}")
    print(f"  {'-' * 50}")

    # Known default/invalid MMSIs to flag
    known_bad = {
        "000000000", "111111111", "123456789",
        "999999999", "222222222", "333333333",
        "888888888", "666666666"
    }

    for mmsi, count in mmsi_counts.most_common(20):
        note = ""
        if mmsi in known_bad:
            note = "<-- DEFAULT/INVALID"
        elif mmsi.startswith("0000"):
            note = "<-- SUSPICIOUS (starts with 0000)"
        elif len(set(mmsi)) == 1:
            note = "<-- ALL SAME DIGIT"
        print(f"  {mmsi:<15} {count:>12,}  {note}")

    # Show distribution: how many MMSIs have very few vs very many pings
    print(f"\nMMSI frequency distribution:")
    brackets = [1, 10, 100, 1000, 10000, 100000, 1000000]
    for i in range(len(brackets) - 1):
        low, high = brackets[i], brackets[i + 1]
        count_in_range = sum(1 for c in mmsi_counts.values() if low <= c < high)
        print(f"  {low:>8,} - {high:>8,} rows: {count_in_range:,} MMSIs")
    high_count = sum(1 for c in mmsi_counts.values() if c >= brackets[-1])
    if high_count > 0:
        print(f"  {brackets[-1]:>8,}+          rows: {high_count:,} MMSIs")


def main():
    if len(sys.argv) < 2:
        print("Usage: python explore_data.py <path_to_csv_file>")
        print("Example: python explore_data.py aisdk-2025-03-07.csv")
        sys.exit(1)

    filepath = sys.argv[1]
    print(f"Exploring file: {filepath}")

    # Step 1: Quick peek (reads only 5 rows — instant)
    peek_at_data(filepath, num_rows=5)

    # Step 2: Full analysis (reads entire file — takes a few minutes)
    count_and_analyze(filepath)

    print(f"\n{'=' * 60}")
    print("Exploration complete!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()