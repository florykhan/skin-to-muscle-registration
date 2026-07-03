"""
metrics_utils.py
================
Measure and export what a cleanup/smoothing step actually did.

For the research goal of moving from manual cleanup toward a repeatable
algorithm, it helps to quantify each edit: how far vertices moved, where the
biggest changes were, and how a "before" state compares to an "after" state.

These functions work on plain ``[[x, y, z], ...]`` vertex lists (as produced by
:func:`mesh_utils.get_mesh_vertices`), so they can compare:

* the mesh before vs. after a smoothing pass, or
* two saved snapshots (e.g. t32 vs. a new cleanup version).

Displacement summaries are printed and can be exported to CSV for plotting
alongside the existing ``simulation_metrics.csv`` workflow.
"""

import csv
import math

from mesh_utils import vec_sub, vec_length


def per_vertex_displacement(before, after, indices=None):
    """Return a list of ``(index, distance)`` moved between two vertex sets.

    If ``indices`` is given, only those vertices are measured. The two inputs
    must share vertex ordering (same topology).
    """
    n = min(len(before), len(after))
    idx_iter = range(n) if indices is None else [i for i in indices if 0 <= i < n]
    return [(i, vec_length(vec_sub(after[i], before[i]))) for i in idx_iter]


def displacement_stats(before, after, indices=None):
    """Summary statistics of per-vertex movement between two vertex sets."""
    disp = per_vertex_displacement(before, after, indices=indices)
    if not disp:
        return {"count": 0, "mean": 0.0, "max": 0.0, "rms": 0.0,
                "max_index": -1, "moved": 0}

    dists = [d for _, d in disp]
    count = len(dists)
    mean = sum(dists) / count
    rms = math.sqrt(sum(d * d for d in dists) / count)
    max_index, max_val = max(disp, key=lambda p: p[1])
    moved = sum(1 for d in dists if d > 1e-6)
    return {
        "count": count,
        "moved": moved,
        "mean": mean,
        "max": max_val,
        "max_index": max_index,
        "rms": rms,
    }


def print_displacement_report(before, after, indices=None, label="cleanup"):
    """Print a short human-readable report of a change. Returns the stats dict."""
    s = displacement_stats(before, after, indices=indices)
    print("[metrics] {0}: {1} verts considered, {2} moved | "
          "mean={3:.4f} rms={4:.4f} max={5:.4f} (vtx {6})".format(
              label, s["count"], s["moved"], s["mean"], s["rms"],
              s["max"], s["max_index"]))
    return s


def export_displacement_csv(before, after, path, indices=None):
    """Write per-vertex displacement to a CSV (``vertex_index, displacement``)."""
    disp = per_vertex_displacement(before, after, indices=indices)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["vertex_index", "displacement"])
        for i, d in disp:
            writer.writerow([i, "{0:.6f}".format(d)])
    print("[metrics] wrote {0} rows -> {1}".format(len(disp), path))
    return path


def export_stats_csv(stats, path, extra=None):
    """Append a one-row summary (from :func:`displacement_stats`) to a CSV.

    ``extra`` may be a dict of additional columns (e.g. region name, iteration,
    smoothing strength) to record alongside the stats.
    """
    row = dict(stats)
    if extra:
        row.update(extra)
    fieldnames = list(row.keys())

    import os
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print("[metrics] appended summary -> {0}".format(path))
    return path
