"""
Task 3: Shadow Fleet Anomaly Detection
========================================
This module implements the four anomaly detection algorithms:
    A: "Going Dark" — AIS gaps > 4 hours with movement
    B: Loitering & Transfers — two ships close together at low speed
    C: Draft Changes at Sea — draught changes during AIS blackout
    D: Identity Cloning — same MMSI pinging from impossible locations

And the DFSI (Dark Fleet Suspicion Index) calculation.

Each anomaly detector takes a vessel's sorted data points and returns
a list of detected incidents.
"""

import math
from datetime import datetime


# ============================================================
# DISTANCE CALCULATION: THE HAVERSINE FORMULA
# ============================================================

# Earth's radius in nautical miles (1 nm = 1.852 km, Earth radius = 6371 km)
EARTH_RADIUS_NM = 3440.065

def haversine_nm(lat1, lon1, lat2, lon2):
    """
    Calculate the distance between two points on Earth in nautical miles.

    The Haversine formula works by:
    1. Converting lat/lon from degrees to radians (math functions need radians)
    2. Computing the differences in lat and lon
    3. Applying the haversine function: hav(θ) = sin²(θ/2)
    4. Combining the results accounting for Earth's curvature
    5. Multiplying by Earth's radius to get actual distance

    Why nautical miles?
    - Maritime speed (knots) is defined as nautical miles per hour
    - The assignment uses knots and nautical miles
    - 1 nautical mile = 1.852 kilometers

    Args:
        lat1, lon1: First point (degrees)
        lat2, lon2: Second point (degrees)

    Returns:
        Distance in nautical miles
    """
    # Step 1: Convert degrees to radians
    lat1_r = math.radians(lat1)
    lat2_r = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    # Step 2: Apply haversine formula
    # The 'a' value represents the square of half the chord length
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2)

    # Step 3: Calculate angular distance in radians
    # atan2 is more numerically stable than asin for this purpose
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # Step 4: Multiply by Earth's radius to get distance
    return EARTH_RADIUS_NM * c


# ============================================================
# ANOMALY A: "GOING DARK" — AIS Gaps with Movement
# ============================================================

# Configuration thresholds
GAP_THRESHOLD_HOURS = 4.0       # Minimum gap to be suspicious
GAP_THRESHOLD_SECONDS = GAP_THRESHOLD_HOURS * 3600  # Same in seconds
MOVEMENT_THRESHOLD_NM = 1.0     # Minimum distance to count as "moved"


def detect_going_dark(mmsi, data_points):
    """
    Detect AIS gaps > 4 hours where the vessel moved during the gap.

    Algorithm:
    1. Data points are already sorted chronologically
    2. Walk through consecutive pairs of points
    3. For each pair, calculate the time gap
    4. If gap > 4 hours, calculate distance between the two points
    5. If distance > 1 nm, the ship was moving with AIS off — flag it

    Args:
        mmsi: The vessel's MMSI identifier
        data_points: List of tuples (ts_epoch, lat, lon, sog, draught)
                     MUST be sorted by ts_epoch ascending

    Returns:
        List of incident dictionaries, one per detected gap
    """
    incidents = []

    # We need at least 2 points to find a gap
    if len(data_points) < 2:
        return incidents

    for i in range(len(data_points) - 1):
        # Current point (where the ship "disappears")
        ts1, lat1, lon1, sog1, draught1 = data_points[i]
        # Next point (where the ship "reappears")
        ts2, lat2, lon2, sog2, draught2 = data_points[i + 1]

        # Calculate time gap in seconds
        gap_seconds = ts2 - ts1

        # Check if gap exceeds threshold
        if gap_seconds <= GAP_THRESHOLD_SECONDS:
            continue

        # Gap is long enough — now check if the ship moved
        distance_nm = haversine_nm(lat1, lon1, lat2, lon2)

        if distance_nm <= MOVEMENT_THRESHOLD_NM:
            continue  # Ship stayed put (anchored), not suspicious

        # --- This is a "Going Dark" incident ---
        gap_hours = gap_seconds / 3600

        # Calculate implied speed during the gap
        # (how fast would the ship need to travel in a straight line?)
        implied_speed_knots = distance_nm / gap_hours if gap_hours > 0 else 0

        incident = {
            "anomaly_type": "A",
            "mmsi": mmsi,
            "description": "Going Dark — AIS gap with movement",
            # When and where the ship disappeared
            "disappear_time": datetime.fromtimestamp(ts1).strftime(
                "%Y-%m-%d %H:%M:%S"),
            "disappear_lat": lat1,
            "disappear_lon": lon1,
            # When and where it reappeared
            "reappear_time": datetime.fromtimestamp(ts2).strftime(
                "%Y-%m-%d %H:%M:%S"),
            "reappear_lat": lat2,
            "reappear_lon": lon2,
            # Gap details
            "gap_hours": round(gap_hours, 2),
            "distance_nm": round(distance_nm, 2),
            "implied_speed_knots": round(implied_speed_knots, 2),
        }

        incidents.append(incident)

    return incidents


# ============================================================
# ANOMALY B: LOITERING & SHIP-TO-SHIP TRANSFERS
# ============================================================

# Configuration thresholds
PROXIMITY_THRESHOLD_NM = 0.27   # 500 meters ≈ 0.27 nautical miles
LOITER_SPEED_THRESHOLD = 1.0    # SOG < 1 knot = essentially stationary
LOITER_DURATION_HOURS = 2.0     # Must be close together for > 2 hours
LOITER_DURATION_SECONDS = LOITER_DURATION_HOURS * 3600
TIME_BUCKET_SECONDS = 120       # Match our 2-minute downsampling interval

# Port-exclusion filters
MAX_ENCOUNTER_HOURS = 12.0      # Longer than this = moored in port, not STS
MAX_ENCOUNTER_SECONDS = MAX_ENCOUNTER_HOURS * 3600
CLUSTER_RADIUS_NM = 2.0         # Count nearby stationary vessels within this
MAX_CLUSTER_SIZE = 2            # If > N stationary vessels nearby, it's a port
                                # (Lowered from 5 — small harbors have clusters
                                # of only 3-5 boats, which slipped through)
MAX_PARTNERS_PER_VESSEL = 2     # If a vessel has encounters with more than
                                # this many different partners, it's moored
                                # in a harbor, not doing a real STS transfer

# Spatial grid cell size (degrees). Two vessels within 500m must share a
# cell OR be in neighboring cells. At ~0.01 degrees ≈ 1.1 km, this works
# as a coarse bin.
GRID_DEG = 0.01


def _scan_buckets_for_pairs(bucket_items):
    """
    Worker function: scan a batch of time buckets for nearby vessel pairs.

    This is the hot loop of Anomaly B. Because each time bucket is
    independent (comparisons only happen between vessels in the same
    2-minute window), we can run many buckets in parallel.

    Args:
        bucket_items: list of (bucket_epoch, vessels_list) tuples,
                      where vessels_list is [(mmsi, lat, lon), ...]

    Returns:
        list of (pair_key, encounter_record) tuples to be merged later.
        encounter_record = (bucket, lat_a, lon_a, lat_b, lon_b, dist, cluster_count)
    """
    results = []

    for bucket, vessels in bucket_items:
        # Put each vessel into its grid cell for this bucket
        grid = {}
        for v in vessels:
            _, lat, lon = v
            cell = (int(lat / GRID_DEG), int(lon / GRID_DEG))
            if cell not in grid:
                grid[cell] = []
            grid[cell].append(v)

        # For each vessel, only look at its own cell + 8 neighbors
        for i in range(len(vessels)):
            mmsi_a, lat_a, lon_a = vessels[i]
            cell_a = (int(lat_a / GRID_DEG), int(lon_a / GRID_DEG))

            # Collect candidates from this cell + 8 neighbors
            candidates = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    neighbor_cell = (cell_a[0] + dx, cell_a[1] + dy)
                    if neighbor_cell in grid:
                        candidates.extend(grid[neighbor_cell])

            # Count how many stationary vessels are within cluster radius
            cluster_count = 0
            for cand in candidates:
                mmsi_c, lat_c, lon_c = cand
                if mmsi_c == mmsi_a:
                    continue
                d = haversine_nm(lat_a, lon_a, lat_c, lon_c)
                if d <= CLUSTER_RADIUS_NM:
                    cluster_count += 1

            # Find close pairs (mmsi ordering avoids duplicates)
            for cand in candidates:
                mmsi_b, lat_b, lon_b = cand
                if mmsi_b <= mmsi_a:
                    continue

                dist = haversine_nm(lat_a, lon_a, lat_b, lon_b)
                if dist > PROXIMITY_THRESHOLD_NM:
                    continue

                pair_key = (mmsi_a, mmsi_b)
                encounter = (bucket, lat_a, lon_a, lat_b, lon_b,
                             dist, cluster_count)
                results.append((pair_key, encounter))

    return results


def detect_ship_transfers(vessel_data, parallel=False, num_workers=None):
    """
    Detect pairs of vessels loitering near each other (potential transfers).

    This is the only anomaly that compares ACROSS vessels rather than
    analyzing a single vessel's timeline.

    Core algorithm:
    1. Build a time-indexed structure: for each 2-minute time bucket,
       list all vessels that are nearly stationary (SOG < 1 knot).
    2. Within each bucket, use a SPATIAL GRID to find close pairs
       efficiently (avoids O(n²) comparisons in crowded buckets).
       [THIS STEP IS PARALLELIZABLE — each bucket is independent]
    3. Track encounters over time and find sustained ones (> 2 hours).
    4. Apply PORT-EXCLUSION FILTERS to remove false positives.

    Port-exclusion logic:
    - Ports contain many stationary vessels clustered together.
    - A real STS transfer is usually two ships meeting in open water.
    - We count how many other stationary vessels are within 2 nm of
      the pair during their encounter. If > MAX_CLUSTER_SIZE, treat as port.
    - We also reject encounters that last > 12 hours (moored vessels
      typically stay for the entire data window; real STS is 2-8 hrs).

    Args:
        vessel_data: dict mapping MMSI -> list of (ts_epoch, lat, lon, sog, draught)
        parallel: If True, parallelize the spatial scan (step 2)
        num_workers: Number of workers for parallel mode (default: all CPUs)

    Returns:
        List of incident dictionaries for detected transfers
    """
    # --- Step 1: Build time-bucketed index of stationary vessels ---
    print("    Anomaly B: Building time-bucketed index of stationary vessels...")

    time_buckets = {}
    for mmsi, points in vessel_data.items():
        for ts_epoch, lat, lon, sog, draught in points:
            if sog is None or sog >= LOITER_SPEED_THRESHOLD:
                continue
            bucket = ts_epoch - (ts_epoch % TIME_BUCKET_SECONDS)
            if bucket not in time_buckets:
                time_buckets[bucket] = []
            time_buckets[bucket].append((mmsi, lat, lon))

    print(f"    Built {len(time_buckets):,} time buckets with stationary vessels")

    # --- Step 2: Find close pairs (sequential or parallel) ---
    encounter_times = {}

    if parallel:
        import multiprocessing
        if num_workers is None:
            num_workers = multiprocessing.cpu_count()

        print(f"    Anomaly B: Scanning (parallel, {num_workers} workers)...")

        # Split buckets into roughly-equal batches for workers
        bucket_items = list(time_buckets.items())
        # Sort by workload (vessels^2 is roughly how much work each bucket
        # takes) so we distribute heavy buckets evenly rather than clumping
        bucket_items.sort(key=lambda x: len(x[1]) ** 2, reverse=True)

        # Round-robin distribution: bucket i goes to worker (i % num_workers)
        # This naturally load-balances because heavy buckets are spread out
        worker_batches = [[] for _ in range(num_workers)]
        for i, item in enumerate(bucket_items):
            worker_batches[i % num_workers].append(item)

        with multiprocessing.Pool(processes=num_workers) as pool:
            all_results = pool.map(_scan_buckets_for_pairs, worker_batches)

        # Merge results from all workers into encounter_times
        for worker_result in all_results:
            for pair_key, encounter in worker_result:
                if pair_key not in encounter_times:
                    encounter_times[pair_key] = []
                encounter_times[pair_key].append(encounter)
    else:
        print("    Anomaly B: Scanning for nearby vessel pairs (spatial grid)...")

        # Sequential: process buckets one at a time using the same worker function
        bucket_items = list(time_buckets.items())
        results = _scan_buckets_for_pairs(bucket_items)
        for pair_key, encounter in results:
            if pair_key not in encounter_times:
                encounter_times[pair_key] = []
            encounter_times[pair_key].append(encounter)

    print(f"    Found {len(encounter_times):,} vessel pairs with close encounters")

    # --- Step 3: Find sustained encounters (> 2 hours) with filters ---
    incidents = []
    filtered_cluster = 0
    filtered_duration = 0
    MAX_GAP_BETWEEN_BUCKETS = 600  # 10 minutes tolerance

    for (mmsi_a, mmsi_b), encounters in encounter_times.items():
        encounters.sort(key=lambda x: x[0])

        run_start = 0
        for k in range(1, len(encounters)):
            time_gap = encounters[k][0] - encounters[k - 1][0]

            if time_gap > MAX_GAP_BETWEEN_BUCKETS:
                run_duration = encounters[k - 1][0] - encounters[run_start][0]
                if run_duration >= LOITER_DURATION_SECONDS:
                    result = _check_and_add_transfer(
                        incidents, mmsi_a, mmsi_b,
                        encounters[run_start:k])
                    if result == "cluster":
                        filtered_cluster += 1
                    elif result == "duration":
                        filtered_duration += 1
                run_start = k

        # Final run
        if len(encounters) > 0:
            run_duration = encounters[-1][0] - encounters[run_start][0]
            if run_duration >= LOITER_DURATION_SECONDS:
                result = _check_and_add_transfer(
                    incidents, mmsi_a, mmsi_b,
                    encounters[run_start:])
                if result == "cluster":
                    filtered_cluster += 1
                elif result == "duration":
                    filtered_duration += 1

    print(f"    Filtered out {filtered_cluster:,} port/cluster encounters")
    print(f"    Filtered out {filtered_duration:,} overly-long encounters")
    print(f"    Before partner filter: {len(incidents)} candidate incidents")

    # --- Step 4: Repeat-partner filter ---
    # Count how many different partners each MMSI has across all
    # surviving incidents. If a vessel has many partners, it's
    # parked in a harbor surrounded by other boats, not doing
    # a real STS transfer with one specific partner.
    partner_count = {}
    for inc in incidents:
        a = inc["mmsi"]
        b = inc["mmsi_b"]
        partner_count[a] = partner_count.get(a, set())
        partner_count[a].add(b)
        partner_count[b] = partner_count.get(b, set())
        partner_count[b].add(a)

    # Identify vessels with too many partners
    promiscuous_vessels = {
        mmsi for mmsi, partners in partner_count.items()
        if len(partners) > MAX_PARTNERS_PER_VESSEL
    }

    # Keep only incidents where NEITHER vessel is promiscuous
    filtered_incidents = [
        inc for inc in incidents
        if inc["mmsi"] not in promiscuous_vessels
        and inc["mmsi_b"] not in promiscuous_vessels
    ]

    partner_filtered = len(incidents) - len(filtered_incidents)
    print(f"    Filtered out {partner_filtered:,} incidents from vessels with too many partners")
    print(f"    Anomaly B: Detected {len(filtered_incidents)} genuine transfer events")
    return filtered_incidents


def _check_and_add_transfer(incidents, mmsi_a, mmsi_b, run_encounters):
    """
    Apply port-exclusion filters and record the incident if it passes.

    Returns:
        "added" if incident was recorded,
        "duration" if filtered for being too long (moored),
        "cluster" if filtered for being in a crowded area (port)
    """
    first_enc = run_encounters[0]
    last_enc = run_encounters[-1]
    duration_seconds = last_enc[0] - first_enc[0]

    # Filter 1: Too-long encounters are port moorings, not STS transfers
    if duration_seconds > MAX_ENCOUNTER_SECONDS:
        return "duration"

    # Filter 2: High cluster density = port area
    # Take the average cluster count across the encounter.
    avg_cluster = sum(enc[6] for enc in run_encounters) / len(run_encounters)
    if avg_cluster > MAX_CLUSTER_SIZE:
        return "cluster"

    # Passes both filters — record as a real transfer
    _add_transfer_incident(incidents, mmsi_a, mmsi_b, first_enc, last_enc)
    return "added"


def _add_transfer_incident(incidents, mmsi_a, mmsi_b, first_enc, last_enc):
    """Helper to create a transfer incident dictionary."""
    duration_hours = (last_enc[0] - first_enc[0]) / 3600

    incident = {
        "anomaly_type": "B",
        "mmsi": mmsi_a,         # First vessel
        "mmsi_b": mmsi_b,       # Second vessel
        "description": "Loitering & Transfer — two vessels stationary and close",
        "start_time": datetime.fromtimestamp(first_enc[0]).strftime(
            "%Y-%m-%d %H:%M:%S"),
        "end_time": datetime.fromtimestamp(last_enc[0]).strftime(
            "%Y-%m-%d %H:%M:%S"),
        "duration_hours": round(duration_hours, 2),
        "vessel_a_lat": first_enc[1],
        "vessel_a_lon": first_enc[2],
        "vessel_b_lat": first_enc[3],
        "vessel_b_lon": first_enc[4],
        "min_distance_nm": round(first_enc[5], 4),
    }

    incidents.append(incident)


# ============================================================
# ANOMALY C: DRAFT CHANGES AT SEA
# ============================================================

# Configuration thresholds
DRAFT_GAP_THRESHOLD_HOURS = 2.0     # AIS blackout must be > 2 hours
DRAFT_GAP_THRESHOLD_SECONDS = DRAFT_GAP_THRESHOLD_HOURS * 3600
DRAFT_CHANGE_PERCENT = 5.0          # Draught must change by > 5%


def detect_draft_changes(mmsi, data_points):
    """
    Detect suspicious draught changes during AIS blackouts.

    If a ship's draught changes significantly while its AIS was off,
    it suggests cargo was loaded or unloaded outside of port — illegally.

    Algorithm:
    1. Walk through consecutive pairs of points (already time-sorted)
    2. Find gaps > 2 hours where BOTH points have draught data
    3. Calculate the percentage change in draught
    4. If change > 5%, flag it

    Why percentage rather than absolute change?
    - A 0.5m change on a small boat (draught 3m) is ~17% — significant.
    - A 0.5m change on a supertanker (draught 20m) is ~2.5% — normal
      variation from fuel consumption, ballast adjustment, etc.
    - Percentage normalizes across vessel sizes.

    Note: Many vessels don't report draught at all. This anomaly only
    applies to the subset that does. A vessel with no draught data
    will simply return an empty list — that's expected.

    Args:
        mmsi: The vessel's MMSI identifier
        data_points: List of tuples (ts_epoch, lat, lon, sog, draught)
                     MUST be sorted by ts_epoch ascending

    Returns:
        List of incident dictionaries for detected draft changes
    """
    incidents = []

    if len(data_points) < 2:
        return incidents

    for i in range(len(data_points) - 1):
        ts1, lat1, lon1, sog1, draught1 = data_points[i]
        ts2, lat2, lon2, sog2, draught2 = data_points[i + 1]

        # Both points must have draught data
        if draught1 is None or draught2 is None:
            continue

        # Draught must be a positive, reasonable value
        # (avoid division by zero and nonsense values)
        if draught1 <= 0 or draught2 <= 0:
            continue

        # Check for AIS gap > 2 hours
        gap_seconds = ts2 - ts1
        if gap_seconds <= DRAFT_GAP_THRESHOLD_SECONDS:
            continue

        # Calculate percentage change in draught
        # We use the earlier draught as the baseline
        change_percent = abs(draught2 - draught1) / draught1 * 100

        if change_percent <= DRAFT_CHANGE_PERCENT:
            continue

        # --- This is a suspicious draft change ---
        gap_hours = gap_seconds / 3600
        distance_nm = haversine_nm(lat1, lon1, lat2, lon2)

        incident = {
            "anomaly_type": "C",
            "mmsi": mmsi,
            "description": "Draft Change at Sea — draught changed during AIS blackout",
            "disappear_time": datetime.fromtimestamp(ts1).strftime(
                "%Y-%m-%d %H:%M:%S"),
            "disappear_lat": lat1,
            "disappear_lon": lon1,
            "reappear_time": datetime.fromtimestamp(ts2).strftime(
                "%Y-%m-%d %H:%M:%S"),
            "reappear_lat": lat2,
            "reappear_lon": lon2,
            "gap_hours": round(gap_hours, 2),
            "distance_nm": round(distance_nm, 2),
            "draught_before": draught1,
            "draught_after": draught2,
            "draught_change_percent": round(change_percent, 2),
        }

        incidents.append(incident)

    return incidents


# ============================================================
# ANOMALY D: IDENTITY CLONING / "TELEPORTATION"
# ============================================================

# Configuration thresholds
TELEPORT_SPEED_THRESHOLD = 60.0   # Knots — no commercial ship goes this fast
MIN_TIME_FOR_TELEPORT = 120       # Seconds — minimum gap to avoid GPS jitter
                                  # false positives (matches our 2-min downsample)
MAX_TELEPORTS_PER_VESSEL = 5      # If we see more than this, it's data
                                  # corruption / broken transponder, not cloning.
                                  # Real cloning = a few jumps. Corruption = many.


def detect_teleportation(mmsi, data_points):
    """
    Detect impossible "teleportation" suggesting identity cloning.

    If two consecutive AIS pings from the same MMSI imply a speed
    greater than 60 knots, it's physically impossible. This means
    two different physical ships are broadcasting the same MMSI —
    one is using a stolen/cloned identity.

    Why 60 knots?
    - Fast container ships: ~24 knots
    - Fast ferries: ~40 knots
    - Military frigates: ~30-35 knots
    - 60 knots leaves a huge margin — anything above this is not
      a real single vessel

    Why a minimum time gap?
    - GPS coordinates have small errors (±10 meters typically)
    - If two pings are only seconds apart, a 10-meter GPS error
      could compute as a very high speed
    - Requiring at least 2 minutes between pings eliminates this
    - Our downsampled data already has ~2 min spacing, so this
      mainly guards against edge cases

    Why a max-teleports filter?
    - Real identity cloning involves two physical ships, each moving
      normally but sharing the same MMSI. The timeline shows a few
      dramatic jumps between their locations — typically 1-3 per day.
    - A vessel with 30+ teleport events has a broken transponder
      or corrupted GPS data, not a cloned identity. Including these
      would blow up the DFSI score with noise.

    Args:
        mmsi: The vessel's MMSI identifier
        data_points: List of tuples (ts_epoch, lat, lon, sog, draught)
                     MUST be sorted by ts_epoch ascending

    Returns:
        List of incident dictionaries for detected teleportation events.
        Returns empty list if too many teleports detected (corrupted data).
    """
    incidents = []

    if len(data_points) < 2:
        return incidents

    for i in range(len(data_points) - 1):
        ts1, lat1, lon1, sog1, draught1 = data_points[i]
        ts2, lat2, lon2, sog2, draught2 = data_points[i + 1]

        # Calculate time difference in seconds
        time_diff = ts2 - ts1

        # Skip if time gap is too small (GPS jitter risk)
        if time_diff < MIN_TIME_FOR_TELEPORT:
            continue

        # Calculate distance
        distance_nm = haversine_nm(lat1, lon1, lat2, lon2)

        # Calculate implied speed in knots (nm per hour)
        time_hours = time_diff / 3600
        implied_speed = distance_nm / time_hours if time_hours > 0 else 0

        if implied_speed <= TELEPORT_SPEED_THRESHOLD:
            continue

        # --- This is an impossible jump — identity cloning ---
        incident = {
            "anomaly_type": "D",
            "mmsi": mmsi,
            "description": "Identity Cloning — impossible speed between pings",
            "time_1": datetime.fromtimestamp(ts1).strftime(
                "%Y-%m-%d %H:%M:%S"),
            "lat_1": lat1,
            "lon_1": lon1,
            "time_2": datetime.fromtimestamp(ts2).strftime(
                "%Y-%m-%d %H:%M:%S"),
            "lat_2": lat2,
            "lon_2": lon2,
            "time_diff_minutes": round(time_diff / 60, 2),
            "distance_nm": round(distance_nm, 2),
            "implied_speed_knots": round(implied_speed, 2),
        }

        incidents.append(incident)

    # Post-filter: if too many teleport events, this is almost certainly
    # a broken transponder or corrupted GPS data, not a real cloning case.
    # Return empty list — we don't trust any of them.
    if len(incidents) > MAX_TELEPORTS_PER_VESSEL:
        return []

    return incidents


# ============================================================
# DFSI CALCULATION (Dark Fleet Suspicion Index)
# ============================================================

def calculate_dfsi(anomaly_a_incidents, anomaly_c_incidents,
                   anomaly_d_incidents):
    """
    Calculate the Dark Fleet Suspicion Index for a vessel.

    Formula from the assignment:
        DFSI = (max_gap_hours / 2)
             + (total_impossible_distance_nm / 10)
             + (C * 15)

    Where:
        max_gap_hours = longest "Going Dark" gap (Anomaly A) in hours
        total_impossible_distance_nm = sum of all teleportation distances
                                       (Anomaly D) in nautical miles
        C = number of illicit draft changes detected (Anomaly C)

    The formula weighs each anomaly type differently:
    - A long AIS gap contributes moderately (divided by 2)
    - Teleportation distance contributes mildly (divided by 10)
    - Draft changes are the strongest signal (multiplied by 15),
      because a draught change at sea is very strong evidence of
      illegal cargo operations

    Args:
        anomaly_a_incidents: List of Anomaly A incidents for this vessel
        anomaly_c_incidents: List of Anomaly C incidents for this vessel
        anomaly_d_incidents: List of Anomaly D incidents for this vessel

    Returns:
        DFSI score as a float (0.0 if no anomalies detected)
    """
    # Term 1: max gap hours / 2
    if anomaly_a_incidents:
        max_gap = max(inc["gap_hours"] for inc in anomaly_a_incidents)
    else:
        max_gap = 0.0

    # Term 2: total impossible distance / 10
    if anomaly_d_incidents:
        total_impossible_dist = sum(
            inc["distance_nm"] for inc in anomaly_d_incidents)
    else:
        total_impossible_dist = 0.0

    # Term 3: count of draft changes * 15
    c_count = len(anomaly_c_incidents)

    dfsi = (max_gap / 2) + (total_impossible_dist / 10) + (c_count * 15)

    return round(dfsi, 2)


# ============================================================
# TESTING
# ============================================================

if __name__ == "__main__":
    print("Testing Haversine formula:")
    print("  Copenhagen to Malmö (known: ~24 nm):")
    d = haversine_nm(55.6761, 12.5683, 55.6050, 13.0038)
    print(f"    Calculated: {d:.2f} nm")

    print("\n  Copenhagen to Gothenburg (known: ~135 nm):")
    d = haversine_nm(55.6761, 12.5683, 57.7089, 11.9746)
    print(f"    Calculated: {d:.2f} nm")

    print("\nTesting Anomaly A detection:")
    # Create fake data: a ship with a 6-hour gap where it moved 50 nm
    from datetime import datetime
    fake_data = [
        # Point 1: 08:00, Copenhagen
        (datetime(2025, 3, 7, 8, 0, 0).timestamp(),
         55.6761, 12.5683, 12.0, 5.0),
        # Point 2: 14:00 (6 hours later), Gothenburg — ship moved!
        (datetime(2025, 3, 7, 14, 0, 0).timestamp(),
         57.7089, 11.9746, 10.0, 5.0),
        # Point 3: 14:30 (30 min later), still near Gothenburg — normal gap
        (datetime(2025, 3, 7, 14, 30, 0).timestamp(),
         57.7100, 11.9800, 8.0, 5.0),
    ]

    incidents = detect_going_dark("123456789", fake_data)
    print(f"  Detected {len(incidents)} incident(s):")
    for inc in incidents:
        print(f"    Gap: {inc['gap_hours']}h, "
              f"Distance: {inc['distance_nm']}nm, "
              f"Implied speed: {inc['implied_speed_knots']}kn")
        print(f"    From: ({inc['disappear_lat']}, {inc['disappear_lon']}) "
              f"at {inc['disappear_time']}")
        print(f"    To:   ({inc['reappear_lat']}, {inc['reappear_lon']}) "
              f"at {inc['reappear_time']}")
