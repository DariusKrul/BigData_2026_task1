"""
Polished Memory Profile Plot
==============================
Reads the raw .dat file produced by `mprof run` and generates a
cleaner graph with phase annotations for the presentation.

Usage:
    # After you've run: mprof run python main_pipeline.py ...
    python plot_memory.py
    # Or specify a particular .dat file:
    python plot_memory.py mprofile_20260410_171138.dat
"""

import sys
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def find_latest_dat_file():
    """Find the most recent mprofile_*.dat file in the current directory."""
    files = glob.glob("mprofile_*.dat")
    if not files:
        return None
    # Most recent by modification time
    return max(files, key=os.path.getmtime)


def parse_mprof_dat(filepath):
    """
    Parse an mprof .dat file.

    The format is simple text, one sample per line:
        CMDLINE python ...
        MEM 25.23 1728400123.45
        MEM 26.01 1728400123.55
        ...

    Each MEM line has: "MEM <memory_in_mib> <unix_timestamp>"

    Returns (times, memories) where times are relative seconds from start.
    """
    times = []
    memories = []
    start_time = None

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("MEM"):
                continue

            parts = line.split()
            if len(parts) < 3:
                continue

            mem_mib = float(parts[1])
            ts = float(parts[2])

            if start_time is None:
                start_time = ts

            times.append(ts - start_time)
            memories.append(mem_mib)

    return times, memories


def plot_memory_profile(dat_file, output="memory_profile_polished.png"):
    """Generate a polished memory usage plot from an mprof .dat file."""
    print(f"Reading {dat_file}...")
    times, memories = parse_mprof_dat(dat_file)

    if not times:
        print("ERROR: No memory samples found in .dat file")
        return

    peak_mem = max(memories)
    final_mem = memories[-1]
    total_time = times[-1]
    print(f"  Samples: {len(times)}")
    print(f"  Duration: {total_time:.1f} seconds")
    print(f"  Peak memory: {peak_mem:.1f} MiB")

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(14, 6))

    # Main memory curve
    ax.plot(times, memories, color="#1f77b4", linewidth=1.5, zorder=3)
    ax.fill_between(times, memories, alpha=0.2, color="#1f77b4", zorder=2)

    # --- Phase boundaries (rough estimates from terminal output) ---
    # These are approximations. Phase 1 dominates, the rest is quick.
    # The peak memory at the end tells us where Phase 3 happens.
    phase1_end = total_time - 30  # Last ~30s is phase 2+3+5
    phase2_end = total_time - 25  # Phase 2 is quick
    phase3_end = total_time - 2   # Phase 3 is the bulk of the tail
    # phase5 is the final 1-2 seconds

    # Phase shaded regions (pastel colors)
    phase_colors = {
        "Phase 1 (Stream + Group)": ("#e3f2fd", 0, phase1_end),
        "Phase 2 (Per-vessel)":     ("#fff3e0", phase1_end, phase2_end),
        "Phase 3 (Anomaly B)":      ("#fce4ec", phase2_end, phase3_end),
        "Phase 5 (Write)":          ("#e8f5e9", phase3_end, total_time),
    }

    y_max = peak_mem * 1.15
    for label, (color, start, end) in phase_colors.items():
        ax.axvspan(start, end, alpha=0.35, color=color, zorder=1)
        # Add phase label in middle of span
        mid = (start + end) / 2
        ax.text(mid, y_max * 0.95, label,
                ha="center", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor="white", edgecolor="gray", alpha=0.8))

    # --- Horizontal reference lines ---
    # 1 GB per core limit for a single process
    ax.axhline(y=1024, color="red", linestyle="--",
               linewidth=1.2, alpha=0.6,
               label="1 GB per-core limit (per process)")

    # Peak memory annotation
    peak_idx = memories.index(peak_mem)
    peak_time = times[peak_idx]
    ax.annotate(f"Peak: {peak_mem:.0f} MiB\n({peak_mem/1024:.2f} GB total,\n~{peak_mem/13:.0f} MiB per process)",
                xy=(peak_time, peak_mem),
                xytext=(peak_time - total_time * 0.25, peak_mem * 0.85),
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.4",
                          facecolor="lightyellow", edgecolor="black"),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.2))

    # --- Styling ---
    ax.set_xlabel("Time (seconds)", fontsize=11)
    ax.set_ylabel("Memory Used (MiB)", fontsize=11)
    ax.set_title("Shadow Fleet Pipeline — Memory Profile\n"
                 "2-day AIS data (41M rows, 4,204 vessels)",
                 fontsize=12, pad=15)
    ax.set_xlim(0, total_time * 1.02)
    ax.set_ylim(0, y_max)
    ax.grid(True, alpha=0.3, zorder=0)
    ax.legend(loc="upper left", framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {output}")
    plt.close()


def main():
    if len(sys.argv) > 1:
        dat_file = sys.argv[1]
    else:
        dat_file = find_latest_dat_file()
        if dat_file is None:
            print("ERROR: No mprofile_*.dat files found in current directory.")
            print("Run `mprof run python main_pipeline.py ...` first.")
            sys.exit(1)
        print(f"Using latest .dat file: {dat_file}")

    plot_memory_profile(dat_file)


if __name__ == "__main__":
    main()
