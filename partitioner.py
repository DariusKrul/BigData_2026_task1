"""
Task 1: Low-Memory Streaming Partitioner for AIS Data
======================================================
This module reads a multi-gigabyte AIS CSV file line-by-line,
filters out dirty/invalid data, parses each row into a compact
dictionary, and yields chunks of clean rows.

It NEVER loads the entire file into memory.

Key concepts:
- Generator functions (yield) for memory-efficient streaming
- Data validation and filtering at the source
- Chunked output for parallel processing (Task 2 will consume these chunks)
"""

import csv
from datetime import datetime


# ============================================================
# DIRTY DATA CONFIGURATION
# ============================================================

# Known default/invalid MMSI numbers from unconfigured transponders
INVALID_MMSIS = {
    "000000000", "111111111", "222222222", "333333333",
    "444444444", "555555555", "666666666", "777777777",
    "888888888", "999999999", "123456789", "012345678",
}

# MMSI prefixes that are NOT actual vessels (per ITU-R M.585 standard):
#   001 - MOB (Man Overboard) devices
#   111 - SAR (Search and Rescue) aircraft
#   970 - AIS-SART (Search and Rescue Transmitters)
#   972 - Man Overboard devices
#   974 - EPIRB (emergency position-indicating radio beacons)
#   99  - Aids to Navigation (99xxxxxxx)
# These produce AIS signals but they aren't ships — they'd pollute our
# anomaly detection (aircraft "teleport" naturally, AIS-SART devices
# ping from wherever they were activated, etc.).
NON_VESSEL_MMSI_PREFIXES = (
    "001",  # MOB devices
    "111",  # SAR aircraft
    "970",  # AIS-SART
    "972",  # MOB devices
    "974",  # EPIRB
    "99",   # Aids to Navigation
)

# Types of mobile that are NOT vessels we want to analyze
NON_VESSEL_TYPES = {
    "Base Station",
}

# Column indices (based on our exploration of the data)
# Header: # Timestamp, Type of mobile, MMSI, Latitude, Longitude,
#         Navigational status, ROT, SOG, COG, Heading, IMO, Callsign,
#         Name, Ship type, Cargo type, Width, Length,
#         Type of position fixing device, Draught, Destination, ETA,
#         Data source type, A, B, C, D
COL_TIMESTAMP = 0
COL_TYPE_OF_MOBILE = 1
COL_MMSI = 2
COL_LATITUDE = 3
COL_LONGITUDE = 4
COL_SOG = 7
COL_DRAUGHT = 18
COL_NAME = 12
COL_SHIP_TYPE = 13


def safe_float(value, default=None):
    """
    Safely convert a string to float.

    Many fields in AIS data are empty strings. Calling float("") would crash.
    This function returns a default value (None) if the conversion fails.

    Args:
        value: The string to convert
        default: What to return if conversion fails (default: None)

    Returns:
        The float value, or the default if conversion fails
    """
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def parse_timestamp(ts_string):
    """
    Parse the AIS timestamp string into a Python datetime object.

    The format from the Danish AIS data is: DD/MM/YYYY HH:MM:SS
    Example: "07/03/2025 00:00:00"

    Args:
        ts_string: The timestamp string from the CSV

    Returns:
        A datetime object, or None if parsing fails
    """
    try:
        return datetime.strptime(ts_string.strip(), "%d/%m/%Y %H:%M:%S")
    except (ValueError, AttributeError):
        return None


def is_valid_mmsi(mmsi):
    """
    Check if an MMSI is a valid vessel identifier.

    Rules:
    1. Must be exactly 9 digits long (shorter = base stations, aids)
    2. Must not be in the known-invalid list (000000000, etc.)
    3. Must be all digits (no letters or special characters)
    4. Must not start with a non-vessel prefix (SAR aircraft, EPIRB, etc.)

    Args:
        mmsi: The MMSI string from the CSV

    Returns:
        True if the MMSI is valid, False otherwise
    """
    mmsi = mmsi.strip()

    # Must be exactly 9 characters
    if len(mmsi) != 9:
        return False

    # Must be all digits
    if not mmsi.isdigit():
        return False

    # Must not be a known default/invalid MMSI
    if mmsi in INVALID_MMSIS:
        return False

    # Must not start with a non-vessel prefix
    # (SAR aircraft, AIS-SART, EPIRB, aids to navigation, etc.)
    for prefix in NON_VESSEL_MMSI_PREFIXES:
        if mmsi.startswith(prefix):
            return False

    return True


def is_valid_position(lat, lon):
    """
    Check if coordinates are physically possible.

    Valid ranges:
    - Latitude:  -90 to +90
    - Longitude: -180 to +180

    We also reject the exact default values (91.0, 0.0) which indicate
    "no position available" in AIS.

    Args:
        lat: Latitude as a float (or None)
        lon: Longitude as a float (or None)

    Returns:
        True if position is valid, False otherwise
    """
    if lat is None or lon is None:
        return False

    if lat < -90.0 or lat > 90.0:
        return False

    if lon < -180.0 or lon > 180.0:
        return False

    return True


def parse_row(row):
    """
    Parse a raw CSV row into a clean, compact dictionary.

    We only extract the fields we actually need for anomaly detection.
    This keeps memory usage low — no point storing 26 columns when we
    only use 7 or 8 of them.

    Args:
        row: A list of strings from csv.reader

    Returns:
        A dictionary with parsed values, or None if the row is invalid
    """
    # Check we have enough columns
    if len(row) < 19:
        return None

    # --- Extract raw values ---
    mmsi = row[COL_MMSI].strip()
    mobile_type = row[COL_TYPE_OF_MOBILE].strip()
    ts_string = row[COL_TIMESTAMP].strip()
    lat_str = row[COL_LATITUDE].strip()
    lon_str = row[COL_LONGITUDE].strip()
    sog_str = row[COL_SOG].strip()
    draught_str = row[COL_DRAUGHT].strip()
    name = row[COL_NAME].strip()
    ship_type = row[COL_SHIP_TYPE].strip()

    # --- Apply dirty data filters ---

    # Filter 1: Non-vessel types (base stations, etc.)
    if mobile_type in NON_VESSEL_TYPES:
        return None

    # Filter 2: Invalid MMSI
    if not is_valid_mmsi(mmsi):
        return None

    # Filter 3: Parse and validate timestamp
    timestamp = parse_timestamp(ts_string)
    if timestamp is None:
        return None

    # Filter 4: Parse and validate coordinates
    lat = safe_float(lat_str)
    lon = safe_float(lon_str)
    if not is_valid_position(lat, lon):
        return None

    # --- Parse optional numeric fields ---
    sog = safe_float(sog_str)        # Speed over ground (knots)
    draught = safe_float(draught_str) # Ship's depth in water (meters)

    # --- Return a compact dictionary with only what we need ---
    return {
        "mmsi": mmsi,
        "timestamp": timestamp,
        "lat": lat,
        "lon": lon,
        "sog": sog,
        "draught": draught,
        "name": name,
        "ship_type": ship_type,
    }


def stream_clean_rows(filepath):
    """
    Generator that streams clean, parsed rows from a CSV file.

    This is the core building block. It:
    1. Opens the file
    2. Reads one line at a time (never loads the whole file)
    3. Parses and validates each line
    4. Yields only clean, valid rows

    Usage:
        for row in stream_clean_rows("data.csv"):
            print(row["mmsi"], row["lat"], row["lon"])

    Args:
        filepath: Path to the AIS CSV file

    Yields:
        Dictionaries containing parsed, validated AIS data
    """
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header row

        for raw_row in reader:
            parsed = parse_row(raw_row)
            if parsed is not None:
                yield parsed


def stream_chunks(filepath, chunk_size=50000):
    """
    Generator that yields chunks (batches) of clean rows.

    Instead of yielding one row at a time, this collects rows into
    a list of `chunk_size` rows, then yields the whole list at once.

    Why chunks instead of single rows?
    - Sending one row at a time to a worker process is very slow
      (each send has overhead for inter-process communication)
    - Sending a big chunk amortizes that overhead
    - Think of it like shipping: sending 50,000 packages in one truck
      is faster than sending 50,000 individual deliveries

    Args:
        filepath: Path to the AIS CSV file
        chunk_size: Number of rows per chunk (default: 50,000)

    Yields:
        Lists of parsed row dictionaries, each list up to chunk_size long
    """
    chunk = []

    for row in stream_clean_rows(filepath):
        chunk.append(row)

        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []  # Start a fresh chunk (old one can be garbage collected)

    # Don't forget the last partial chunk!
    if chunk:
        yield chunk


# ============================================================
# TESTING / DEMONSTRATION
# ============================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python partitioner.py <path_to_csv_file>")
        sys.exit(1)

    filepath = sys.argv[1]
    print(f"Testing partitioner on: {filepath}")

    # --- Test 1: Stream a few individual clean rows ---
    print(f"\nTest 1: First 3 clean rows")
    print("-" * 40)
    count = 0
    for row in stream_clean_rows(filepath):
        print(f"  MMSI: {row['mmsi']}, Time: {row['timestamp']}, "
              f"Lat: {row['lat']:.4f}, Lon: {row['lon']:.4f}, "
              f"SOG: {row['sog']}, Draught: {row['draught']}")
        count += 1
        if count >= 3:
            break

    # --- Test 2: Stream chunks and count totals ---
    print(f"\nTest 2: Streaming full file in chunks of 50,000")
    print("-" * 40)

    total_clean_rows = 0
    chunk_count = 0
    unique_mmsis = set()

    for chunk in stream_chunks(filepath, chunk_size=50000):
        chunk_count += 1
        total_clean_rows += len(chunk)

        # Track unique MMSIs we've seen
        for row in chunk:
            unique_mmsis.add(row["mmsi"])

        # Progress indicator
        if chunk_count % 50 == 0:
            print(f"  ...processed {chunk_count} chunks, "
                  f"{total_clean_rows:,} clean rows so far")

    print(f"\nResults:")
    print(f"  Total chunks produced: {chunk_count}")
    print(f"  Total clean rows: {total_clean_rows:,}")
    print(f"  Unique valid vessel MMSIs: {len(unique_mmsis):,}")
    print(f"  Rows filtered out: calculated when compared to explore_data results")
