# Maritime Shadow Fleet Detection

Parallel big data pipeline for detecting illicit "shadow fleet" vessel behavior in Danish Maritime Authority AIS (Automatic Identification System) data. This project processes gigabyte-scale vessel tracking datasets to identify four categories of suspicious activity and compute a Dark Fleet Suspicion Index (DFSI) for each vessel.

**Course:** Big Data Analysis, Vilnius University (MSc Data Science)
**Assignment:** Assignment 1 — Shadow Fleet Detection with Parallel Computing

---

## Overview

Shadow fleet vessels manipulate or disable their AIS transponders to evade sanctions, conduct illegal fishing, or perform illicit ship-to-ship cargo transfers. This pipeline detects four anomaly types:

| Anomaly | Description | Rule |
|---------|-------------|------|
| **A — Going Dark** | AIS gap where the ship moved during the blackout | Gap > 4 hours AND distance > 1 nm |
| **B — Loitering & Transfers** | Two vessels close together at very low speed for extended time | < 500 m apart, SOG < 1 kn, for > 2 hours |
| **C — Draft Changes at Sea** | Draught changed during an AIS blackout | Gap > 2 hours AND draught change > 5% |
| **D — Identity Cloning** | Same MMSI pinging from physically impossible locations | Implied speed > 60 knots |

Each detected anomaly contributes to the vessel's DFSI score:

```
DFSI = (max_gap_hours / 2) + (total_impossible_distance_nm / 10) + (draft_change_count × 15)
```

---

## Architecture

The pipeline is built around four key principles:

1. **Streaming, not loading** — files are read line-by-line via Python generators. `pandas.read_csv()` is never used for full-file loading.
2. **Dirty data at the source** — filters are applied during streaming to drop base stations, non-vessel MMSI ranges (SAR aircraft, EPIRB, etc.), and invalid coordinates before anything reaches analysis.
3. **Downsampling** — each vessel is reduced to one point per 2-minute window (~6% of original data) while preserving the trajectory detail needed for anomaly detection.
4. **Parallelism where it pays** — per-vessel analysis and the cross-vessel spatial scan are distributed across CPU cores using `multiprocessing.Pool`.

### Pipeline Phases

```
CSV files (2 × ~3.5 GB)
      │
      ▼
┌─────────────────────────────┐
│  Phase 1: Stream & Group    │  ~500s   (disk-bound, single pass)
│  - line-by-line read        │
│  - filter dirty data        │
│  - downsample to 2-min      │
│  - group by MMSI            │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  Phase 2: Per-vessel        │  ~2s (parallel, 12 cores)
│  - sort chronologically     │
│  - detect Anomalies A, C, D │
│  - compute DFSI             │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  Phase 3: Cross-vessel      │  ~25s (parallel, 12 cores)
│  - build time-bucket index  │
│  - spatial grid pair scan   │
│  - port-exclusion filters   │
│  - sustained encounter runs │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  Phase 5: Write Reports     │  <1s
│  - combined_incidents.csv   │
│  - combined_dfsi_ranking    │
└─────────────────────────────┘
```

### Dirty Data Handling

The dataset contains several categories of non-vessel or corrupted records that must be removed before analysis:

- **Base stations** — shore-based AIS receivers that broadcast their own position
- **Short MMSIs** — maritime infrastructure like aids-to-navigation use non-9-digit IDs
- **ITU-reserved MMSI prefixes** — `001` (MOB), `111` (SAR aircraft), `970/972/974` (EPIRB/SART), `99` (aids to navigation)
- **Default invalid MMSIs** — `000000000`, `111111111`, `123456789`, etc.
- **Out-of-range coordinates** — latitude outside ±90° or the placeholder `91.0, 0.0`
- **Corrupted GPS** — vessels with >5 teleportation events are flagged as broken data, not real cloning

Without these filters, a single default MMSI group could contain millions of rows from unrelated transmitters and crash workers with memory overload.

---

## Project Structure

```
.
├── partitioner.py           # Task 1: low-memory streaming + dirty data filters
├── parallel_processor.py    # Task 2: MMSI grouping, parallel Pool infrastructure
├── anomaly_detection.py     # Task 3: Anomalies A-D, DFSI, Haversine distance
├── main_pipeline.py         # Orchestrates the full pipeline, writes reports
├── cache_phase1.py          # One-time cache for fast benchmarking
├── benchmark.py             # Task 4: speedup + chunk-size tests, generates graphs
├── plot_memory.py           # Task 4: polished memory profile visualization
├── diagnose_anomaly_c.py    # Diagnostic script for investigating Anomaly C
├── explore_data.py          # Initial data exploration
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

---

## Requirements

- Python 3.10+
- Packages listed in `requirements.txt`:
  - `matplotlib` (for graphs)
  - `memory_profiler` (for memory profiling)

Install:
```bash
pip install -r requirements.txt
```

---

## Usage

### 1. Run the Full Pipeline

Process one or two days of AIS data:

```bash
# Single file
python main_pipeline.py aisdk-2025-03-07.csv --output march07

# Two files combined (recommended — catches anomalies spanning midnight)
python main_pipeline.py aisdk-2025-03-07.csv aisdk-2025-03-08.csv --output combined
```

Output files:
- `combined_incidents.csv` — master list of every detected anomaly
- `combined_dfsi_ranking.csv` — vessels ranked by DFSI score

### 2. Benchmark Performance (Task 4)

Step 2.1: Cache Phase 1 results (runs the slow file-reading once):

```bash
python cache_phase1.py aisdk-2025-03-07.csv aisdk-2025-03-08.csv
```

Step 2.2: Run the benchmarks (uses the cache, takes ~5 minutes):

```bash
python benchmark.py
```

Generates:
- `benchmark_speedup.png` — speedup curve across 1-12 workers
- `benchmark_chunks.png` — optimal batch size analysis

### 3. Memory Profiling

```bash
mprof run python main_pipeline.py aisdk-2025-03-07.csv aisdk-2025-03-08.csv --output combined
python plot_memory.py
```

Generates `memory_profile_polished.png` with phase annotations.

---

## Key Results

**Dataset:** Danish Maritime Authority AIS, March 7–8, 2025
- Input: 41.3 million rows (~7.4 GB combined)
- After filtering/downsampling: 2.6 million points, 4,204 unique vessels

**Detection Summary:**

| Anomaly | Count | Notes |
|---------|------:|-------|
| A (Going Dark) | 417 | Most common |
| B (Transfers) | 365 | After port-exclusion filters |
| C (Draft Changes) | 1 | Rare but high-confidence |
| D (Identity Cloning) | 55 | After corruption filters |
| **Total** | **838** | |

**Performance:**

| Metric | Value |
|--------|------:|
| Total pipeline runtime | ~530 seconds |
| Phase 1 (disk-bound) | ~500 s |
| Phase 3 sequential | 58.8 s |
| Phase 3 parallel (12 workers) | 25.1 s |
| **Phase 3 speedup** | **2.34×** |
| Peak total memory | 3.55 GB |
| Memory per process | ~279 MB (< 1 GB limit ✓) |

**Highest-confidence finding:** Marshall Islands–flagged tanker (MMSI `538007841`) sat stationary at the Port of Stenungsund (58.086° N, 11.804° E — a Swedish petrochemical tanker terminal handling fuel oil, naphtha, ethylene, propane, and LPG) for 20 hours with AIS off. Draught went from 7.3 m to 9.2 m (+26%), consistent with covert cargo loading at a working commercial terminal. This was the only Anomaly C detection across 41 million rows.

---

## Notes on Methodology

- **Why downsampling?** Raw AIS pings arrive every few seconds. For detecting 4-hour gaps or 2-hour loitering, 2-minute resolution is more than sufficient and reduces the analysis workload by ~94%.
- **Why epoch floats for timestamps?** Python `datetime` objects are expensive to pickle when sending data to worker processes. Converting to epoch seconds (float) reduced per-row memory and eliminated a `MemoryError` during the initial parallel run.
- **Why port-exclusion filters for Anomaly B?** The naive implementation returned 15,000+ "transfers" — nearly all were fishing boats moored in small Danish harbors. We filter encounters by cluster density, total duration (>12h suggests mooring), and partner multiplicity to get realistic results.
- **Why Anomaly C returned only one detection:** Draught is a manually-entered field in AIS. Analysis of both days showed 312 vessels with >2h gaps AND 121 vessels with >5% draught changes, but these are almost entirely disjoint sets. Diligent crews that update draught do not typically turn off AIS; the single detection across 41M rows is genuine and high-confidence.

---

## Academic Integrity

This project was developed as part of the Big Data Analysis course at Vilnius University. The lecturer provided a reference preprocessing script that demonstrates chunked CSV reading and 2-minute downsampling via `concurrent.futures`. I studied it, adopted the downsampling concept, and implemented my own version inside `parallel_processor.py` — preserving the draught field (which the reference script discards) that is required for Anomaly C detection.
