"""
cleanup_pipeline.py
===================
Closed-loop iterative artifact cleanup for the skin-to-muscle registration
project:

    DETECT (M5) -> CORRECT (localized smoothing) -> MEASURE (displacement) ->
    RE-DETECT (M5) -> ADAPT the region -> REPEAT until convergence.

This module is ORCHESTRATION ONLY. It reuses, without reimplementing:

* detection  -- :func:`artifact_detection.detect_unified_artifacts` (the final
  M5 detector fusing M3 multi-scale geometry evidence and M4 anatomy-derived SDF
  reference evidence). M5 stays DETECTION-ONLY here; this module never changes
  detector thresholds between cycles.
* smoothing  -- :func:`smoothing_utils.smooth_mesh_region` (region-aware Taubin /
  Laplacian; unselected neighbours are frozen references so the patch blends in).
* metrics    -- :mod:`metrics_utils` displacement statistics.
* mesh I/O   -- :mod:`mesh_utils` (positions/selection) and
  :func:`maya_io.duplicate_mesh` (one pre-cleanup backup).

Design decisions (see the research brief):

* One OUTER CLEANUP CYCLE = ``smoothing_iterations_per_cycle`` localized smoothing
  passes followed by exactly ONE M5 re-detection. M5 (its M4/SDF stage in
  particular) is expensive, and a single smoothing iteration changes the M5
  selection only marginally, so re-detecting after every smoothing pass would be
  wasteful. Default cycle = 3 smoothing iterations -> 1 re-detection.
* The detected index set is FIXED within a cycle. After re-detection the NEW set
  becomes the smoothing region for the next cycle (the detector adapts to the
  modified surface). We do NOT accumulate all historically-selected vertices into
  the smoothing region; ``ever_selected`` is tracked only for diagnostics.
* Convergence is decided from SEVERAL signals, never from "selection count == 0"
  or "count decreased" alone (percentile detectors keep selecting a top fraction
  even on a clean mesh, and smoothing one artifact can expose a neighbour).

Limitation: M4's SDF backend is UNSIGNED closest-surface distance, so this is
ITERATIVE ANATOMY-AWARE ARTIFACT CLEANUP, not "automatic penetration
correction". Penetration/collision correctness must be evaluated separately.

The module imports cleanly outside Maya (Maya access is isolated in the helper
modules and lazily imported for undo chunks), so the control/convergence/logging
logic can be unit-tested without a running Maya session.
"""

import csv
import datetime
import json
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import artifact_detection
import mesh_utils
import metrics_utils
import smoothing_utils

try:
    import maya_io
except ImportError:  # pragma: no cover - maya_io present in this project
    maya_io = None


_STOP_NO_REGION = "no detected vertices"
_STOP_TINY_DISP = "geometry change below threshold"
_STOP_STABLE = "artifact selection stabilized"
_STOP_MAX_CYCLES = "reached max cycles"
_STOP_MAX_CUM_DISP = "maximum cumulative displacement exceeded"
_STOP_NEW_EXPLODE = "new detection fraction exceeded threshold"
_STOP_DRY_RUN = "dry run (no geometry modified)"
_BACKUP_SUFFIX = "_precleanup"


# =============================================================================
# SMALL LAZY-MAYA HELPERS (undo chunking; safe no-ops outside Maya)
# =============================================================================

def _maya_cmds():
    """Return ``maya.cmds`` if inside Maya, else ``None`` (keeps this testable)."""
    try:
        import maya.cmds as cmds
        return cmds
    except ImportError:
        return None


def _open_undo_chunk(name):
    cmds = _maya_cmds()
    if cmds is not None:
        try:
            cmds.undoInfo(openChunk=True, chunkName=name)
        except Exception:
            pass


def _close_undo_chunk():
    cmds = _maya_cmds()
    if cmds is not None:
        try:
            cmds.undoInfo(closeChunk=True)
        except Exception:
            pass


# =============================================================================
# INTERNAL PRIMITIVES (thin wrappers around reused helpers)
# =============================================================================

def _run_m5(skin_mesh: str,
            anatomical_meshes: Optional[Sequence[str]],
            target_offset: float,
            cfg: Dict[str, Any],
            ) -> Tuple[List[int], Dict[str, Any], float]:
    """Run the M5 detector ONCE with the FIXED detector configuration.

    Returns ``(indices, m5_report, seconds)``. ``select=False`` so intermediate
    re-detections never disturb the viewport selection. Detector parameters are
    taken verbatim from ``cfg`` on every call so detection stays reproducible.
    """
    t0 = time.time()
    indices, report = artifact_detection.detect_unified_artifacts(
        skin_mesh,
        anatomical_meshes,
        target_offset,
        fusion_mode=cfg["fusion_mode"],
        m3_percentile=cfg["m3_percentile"],
        m4_percentile=cfg["m4_percentile"],
        final_growth_rings=cfg["final_growth_rings"],
        select=False,
        **cfg["extra_m5_kwargs"],
    )
    return indices, report, time.time() - t0


def _m5_diagnostics(report: Dict[str, Any]) -> Dict[str, int]:
    """Pull the M3/M4/overlap breakdown out of an M5 report (research data)."""
    m3 = report.get("m3", {}) or {}
    m4 = report.get("m4", {}) or {}
    return {
        "m3_count": int(m3.get("count", 0)),
        "m4_count": int(m4.get("count", 0)),
        "m3_m4_overlap_count": int(report.get("overlap_count", 0)),
        "m3_only_count": int(report.get("m3_only_count", 0)),
        "m4_only_count": int(report.get("m4_only_count", 0)),
        "final_m5_count": int(report.get("final_count", len(report.get("final_indices", [])))),
    }


def _compare_selections(previous_indices: Sequence[int],
                        new_indices: Sequence[int],
                        ) -> Dict[str, Any]:
    """Set comparison between the previous and re-detected selections.

    ``jaccard`` is defined as ``|intersection| / |union|``; an empty union
    (nothing detected before or after) is treated as fully stable (1.0). Fractions
    guard against zero denominators.
    """
    prev_set = set(previous_indices)
    new_set = set(new_indices)
    resolved = prev_set - new_set
    remaining = prev_set & new_set
    newly = new_set - prev_set
    union = prev_set | new_set

    selected_before = len(prev_set)
    selected_after = len(new_set)
    jaccard = (len(remaining) / len(union)) if union else 1.0
    resolved_fraction = (len(resolved) / selected_before) if selected_before else 0.0
    new_fraction = (len(newly) / selected_after) if selected_after else 0.0

    return {
        "selected_before": selected_before,
        "selected_after": selected_after,
        "resolved_count": len(resolved),
        "remaining_count": len(remaining),
        "overlap_count": len(remaining),
        "newly_detected_count": len(newly),
        "union_count": len(union),
        "jaccard": jaccard,
        "resolved_fraction": resolved_fraction,
        "new_fraction": new_fraction,
    }


def _disp(before: Sequence[Sequence[float]],
          after: Sequence[Sequence[float]],
          indices: Optional[Sequence[int]] = None,
          ) -> Dict[str, Any]:
    """Displacement stats via metrics_utils (reused; no duplicated math)."""
    return metrics_utils.displacement_stats(before, after, indices=indices)


def _outside_indices(num_verts: int, region: Sequence[int]) -> List[int]:
    """Indices NOT in ``region`` (for outside-region displacement diagnostics)."""
    region_set = set(region)
    return [i for i in range(num_verts) if i not in region_set]


# =============================================================================
# LOGGING
# =============================================================================

_CSV_COLUMNS = [
    "cycle",
    "selected_before", "selected_after",
    "resolved_count", "remaining_count", "newly_detected_count",
    "resolved_fraction", "new_fraction", "jaccard",
    "m3_count", "m4_count", "m3_m4_overlap_count", "m3_only_count", "m4_only_count",
    "cycle_mean_displacement", "cycle_rms_displacement", "cycle_max_displacement",
    "cumulative_mean_displacement", "cumulative_rms_displacement",
    "cumulative_max_displacement",
    "elapsed_seconds", "stop_reason",
]


def _unique_path(path: str) -> str:
    """Return ``path`` if free, else ``<stem>_vNN<ext>`` (never overwrite)."""
    if not os.path.exists(path):
        return path
    stem, ext = os.path.splitext(path)
    n = 2
    while os.path.exists("{0}_v{1:02d}{2}".format(stem, n, ext)):
        n += 1
    return "{0}_v{1:02d}{2}".format(stem, n, ext)


def _write_json(report: Dict[str, Any], path: str) -> Optional[str]:
    path = _unique_path(path)
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print("[cleanup] wrote JSON log -> {0}".format(path))
        return path
    except (OSError, TypeError) as exc:
        print("[cleanup] WARNING: could not write JSON log ({0})".format(exc))
        return None


def _write_csv(csv_rows: List[Dict[str, Any]], path: str) -> Optional[str]:
    path = _unique_path(path)
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
            writer.writeheader()
            for row in csv_rows:
                writer.writerow({k: row.get(k, "") for k in _CSV_COLUMNS})
        print("[cleanup] wrote CSV log  -> {0}".format(path))
        return path
    except OSError as exc:
        print("[cleanup] WARNING: could not write CSV log ({0})".format(exc))
        return None


def _csv_row_from_cycle(record: Dict[str, Any], stop_reason: str = "") -> Dict[str, Any]:
    sel = record["selection"]
    m5 = record["m5"]
    cyc = record["cycle_displacement"]
    cum = record["cumulative_displacement"]
    return {
        "cycle": record["cycle"],
        "selected_before": sel["before"],
        "selected_after": sel["after"],
        "resolved_count": sel["resolved"],
        "remaining_count": sel["remaining"],
        "newly_detected_count": sel["new"],
        "resolved_fraction": round(sel["resolved_fraction"], 6),
        "new_fraction": round(sel["new_fraction"], 6),
        "jaccard": round(sel["jaccard"], 6),
        "m3_count": m5["m3_count"],
        "m4_count": m5["m4_count"],
        "m3_m4_overlap_count": m5["m3_m4_overlap"],
        "m3_only_count": m5["m3_only"],
        "m4_only_count": m5["m4_only"],
        "cycle_mean_displacement": round(cyc["mean"], 8),
        "cycle_rms_displacement": round(cyc["rms"], 8),
        "cycle_max_displacement": round(cyc["max"], 8),
        "cumulative_mean_displacement": round(cum["mean"], 8),
        "cumulative_rms_displacement": round(cum["rms"], 8),
        "cumulative_max_displacement": round(cum["max"], 8),
        "elapsed_seconds": round(record["timing"]["cycle_seconds"], 4),
        "stop_reason": stop_reason,
    }


# =============================================================================
# VALIDATION
# =============================================================================

def _validate_config(target_offset,
                     fusion_mode,
                     m3_percentile,
                     m4_percentile,
                     final_growth_rings,
                     smoothing_method,
                     smoothing_strength,
                     smoothing_iterations_per_cycle,
                     max_cycles,
                     min_cycles,
                     convergence_patience,
                     min_relative_selection_change,
                     stable_jaccard_threshold,
                     min_cycle_mean_displacement):
    """Raise a descriptive ``ValueError`` for any invalid configuration."""
    if target_offset is None:
        raise ValueError(
            "target_offset is REQUIRED and is not guessed. Run "
            "summarize_skin_anatomy_distances() (d98) / "
            "artifact_detection.summarize_skin_sdf_values() first, then pass e.g. "
            "the observed median distance as target_offset.")
    if fusion_mode not in ("union", "intersection", "m3_only", "m4_only"):
        raise ValueError("fusion_mode must be one of union/intersection/m3_only/"
                         "m4_only, got '{0}'".format(fusion_mode))
    for name, pct in (("m3_percentile", m3_percentile), ("m4_percentile", m4_percentile)):
        if not (0.0 <= pct <= 100.0):
            raise ValueError("{0} must be in [0, 100], got {1}".format(name, pct))
    if final_growth_rings < 0:
        raise ValueError("final_growth_rings must be >= 0, got {0}".format(final_growth_rings))
    if smoothing_method not in ("taubin", "laplacian"):
        raise ValueError("smoothing_method must be 'taubin' or 'laplacian', got "
                         "'{0}'".format(smoothing_method))
    if not (0.0 < smoothing_strength <= 1.0):
        raise ValueError("smoothing_strength must be in (0, 1], got {0}".format(smoothing_strength))
    if smoothing_iterations_per_cycle < 1:
        raise ValueError("smoothing_iterations_per_cycle must be >= 1, got {0}".format(
            smoothing_iterations_per_cycle))
    if max_cycles < 1:
        raise ValueError("max_cycles must be >= 1, got {0}".format(max_cycles))
    if min_cycles < 0 or min_cycles > max_cycles:
        raise ValueError("min_cycles must be in [0, max_cycles], got {0}".format(min_cycles))
    if convergence_patience < 1:
        raise ValueError("convergence_patience must be >= 1, got {0}".format(convergence_patience))
    if not (0.0 <= min_relative_selection_change <= 1.0):
        raise ValueError("min_relative_selection_change must be in [0, 1], got {0}".format(
            min_relative_selection_change))
    if not (0.0 <= stable_jaccard_threshold <= 1.0):
        raise ValueError("stable_jaccard_threshold must be in [0, 1], got {0}".format(
            stable_jaccard_threshold))
    if min_cycle_mean_displacement < 0.0:
        raise ValueError("min_cycle_mean_displacement must be >= 0, got {0}".format(
            min_cycle_mean_displacement))


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def run_iterative_m5_cleanup(skin_mesh: str,
                             anatomical_meshes: Optional[Sequence[str]],
                             target_offset: float,
                             # --- M5 detector parameters (FIXED across cycles) -
                             fusion_mode: str = "union",
                             m3_percentile: float = 97.5,
                             m4_percentile: float = 96.0,
                             final_growth_rings: int = 1,
                             # --- smoothing ------------------------------------
                             smoothing_method: str = "taubin",
                             smoothing_strength: float = 0.3,
                             smoothing_iterations_per_cycle: int = 3,
                             # --- iterative process ----------------------------
                             max_cycles: int = 10,
                             min_cycles: int = 1,
                             # --- convergence ----------------------------------
                             convergence_patience: int = 2,
                             min_relative_selection_change: float = 0.02,
                             stable_jaccard_threshold: float = 0.98,
                             min_cycle_mean_displacement: float = 1e-5,
                             # --- safety ---------------------------------------
                             max_cumulative_displacement: Optional[float] = None,
                             stop_if_new_detection_fraction_exceeds: Optional[float] = None,
                             create_backup: bool = True,
                             # --- output ---------------------------------------
                             log_path: Optional[str] = None,
                             save_json: bool = True,
                             save_csv: bool = True,
                             select_final: bool = True,
                             verbose: bool = True,
                             apply: bool = True,
                             # --- advanced M5 passthrough (kept fixed) ---------
                             m5_detector_kwargs: Optional[Dict[str, Any]] = None,
                             ) -> Dict[str, Any]:
    """Run the closed-loop M5 detect -> smooth -> re-detect cleanup to convergence.

    One OUTER CYCLE = ``smoothing_iterations_per_cycle`` localized smoothing passes
    on the CURRENT M5 region, then ONE M5 re-detection on the modified mesh. The
    re-detected set becomes the region for the next cycle. Detector parameters are
    held FIXED across all cycles so observed changes are attributable to smoothing,
    not to detector tuning.

    Parameters
    ----------
    skin_mesh, anatomical_meshes, target_offset:
        Passed to the M5 detector. ``target_offset`` is REQUIRED (choose it from
        the scene scale via ``summarize_skin_anatomy_distances()``); ``None``
        raises ``ValueError`` -- it is never guessed.
    fusion_mode, m3_percentile, m4_percentile, final_growth_rings:
        FIXED M5 detector configuration (defaults are the current preferred
        experiment: union / 97.5 / 96.0 / 1). Never auto-tuned between cycles.
    smoothing_method, smoothing_strength, smoothing_iterations_per_cycle:
        Localized smoother settings (reused ``smoothing_utils.smooth_mesh_region``).
    max_cycles, min_cycles:
        Hard cap and minimum outer cycles. ``max_cycles`` is an absolute safety
        limit -- the loop is never unbounded.
    convergence_patience, min_relative_selection_change, stable_jaccard_threshold,
    min_cycle_mean_displacement:
        Multi-signal convergence knobs (see the stop conditions below).
    max_cumulative_displacement, stop_if_new_detection_fraction_exceeds:
        Optional safety limits (``None`` = disabled; no unit-dependent default).
    create_backup:
        Duplicate the skin mesh once (``<skin_mesh>_precleanup``) before the first
        smoothing cycle, reusing ``maya_io.duplicate_mesh``. Skipped if a backup
        with that name already exists (never blindly overwrites).
    log_path, save_json, save_csv:
        Log directory (default ``cleanup_logs/``) and which artefacts to write.
        Existing files are never overwritten silently.
    select_final:
        Select the final M5 region in Maya at the end (for inspection).
    apply:
        ``True`` runs the full loop. ``False`` is a DRY RUN: validate config, run
        the baseline (cycle-0) detection, print the plan, and change NOTHING (no
        backup, no smoothing). No fake future results are simulated.

    Stop conditions (whichever fires first; soft ones respect ``min_cycles``):
        1. no detected vertices after re-detection;
        2. cycle mean displacement (over the region) below
           ``min_cycle_mean_displacement`` for ``convergence_patience`` cycles;
        3. ``jaccard >= stable_jaccard_threshold`` AND relative selection-count
           change ``<= min_relative_selection_change`` for ``convergence_patience``
           cycles;
        4. ``max_cycles`` reached (hard limit);
        (safety) cumulative max displacement exceeds ``max_cumulative_displacement``;
        (safety) newly-detected fraction exceeds
        ``stop_if_new_detection_fraction_exceeds``.

    The selection count is NOT assumed to decrease monotonically: smoothing one
    artifact can expose a neighbour, so ``resolved`` / ``remaining`` /
    ``newly_detected`` are logged separately and a rising count is not a failure.

    Returns
    -------
    dict
        The full run report (configuration, baseline, per-cycle records, stop
        reason, final indices, final M5 report, cumulative displacement, runtime,
        and any log paths written).
    """
    run_start = time.time()
    extra_m5 = dict(m5_detector_kwargs) if m5_detector_kwargs else {}

    _validate_config(
        target_offset, fusion_mode, m3_percentile, m4_percentile, final_growth_rings,
        smoothing_method, smoothing_strength, smoothing_iterations_per_cycle,
        max_cycles, min_cycles, convergence_patience, min_relative_selection_change,
        stable_jaccard_threshold, min_cycle_mean_displacement)

    cfg = {
        "fusion_mode": fusion_mode,
        "m3_percentile": m3_percentile,
        "m4_percentile": m4_percentile,
        "final_growth_rings": final_growth_rings,
        "extra_m5_kwargs": extra_m5,
    }

    configuration = {
        "skin_mesh": skin_mesh,
        "anatomical_mesh_count": len(anatomical_meshes) if anatomical_meshes else 0,
        "target_offset": target_offset,
        "fusion_mode": fusion_mode,
        "m3_percentile": m3_percentile,
        "m4_percentile": m4_percentile,
        "final_growth_rings": final_growth_rings,
        "smoothing_method": smoothing_method,
        "smoothing_strength": smoothing_strength,
        "smoothing_iterations_per_cycle": smoothing_iterations_per_cycle,
        "max_cycles": max_cycles,
        "min_cycles": min_cycles,
        "convergence_patience": convergence_patience,
        "min_relative_selection_change": min_relative_selection_change,
        "stable_jaccard_threshold": stable_jaccard_threshold,
        "min_cycle_mean_displacement": min_cycle_mean_displacement,
        "max_cumulative_displacement": max_cumulative_displacement,
        "stop_if_new_detection_fraction_exceeds": stop_if_new_detection_fraction_exceeds,
        "create_backup": create_backup,
        "apply": apply,
        "extra_m5_kwargs": extra_m5,
    }

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report: Dict[str, Any] = {
        "configuration": configuration,
        "timestamp": timestamp,
        "dry_run": (not apply),
        "baseline": {},
        "cycles": [],
        "stop_reason": None,
        "completed_cycles": 0,
        "final_indices": [],
        "final_selected_count": 0,
        "final_m5_report": {},
        "ever_selected_count": 0,
        "cumulative_displacement": {},
        "runtime_seconds": 0.0,
        "log_files": {},
    }

    if not mesh_utils.mesh_exists(skin_mesh):
        raise ValueError("skin_mesh '{0}' does not exist in the scene".format(skin_mesh))

    original_positions = mesh_utils.get_mesh_vertices(skin_mesh)
    if not original_positions:
        raise ValueError("skin_mesh '{0}' has no readable vertices".format(skin_mesh))
    num_verts = len(original_positions)

    # --- CYCLE 0: baseline detection (no smoothing) --------------------------
    if verbose:
        print("\n" + "=" * 50)
        print("M5 ITERATIVE CLEANUP  (baseline / cycle 0)")
        print("=" * 50)
        print("mesh: {0} ({1} verts) | target_offset={2}".format(
            skin_mesh, num_verts, target_offset))
        print("detector: fusion={0} m3p={1} m4p={2} growth={3} (FIXED across cycles)".format(
            fusion_mode, m3_percentile, m4_percentile, final_growth_rings))

    baseline_indices, baseline_report, baseline_seconds = _run_m5(
        skin_mesh, anatomical_meshes, target_offset, cfg)
    baseline_diag = _m5_diagnostics(baseline_report)
    report["baseline"] = {
        "selected": len(baseline_indices),
        "detection_seconds": baseline_seconds,
        "m5": baseline_diag,
        "warnings": list(baseline_report.get("warnings", [])),
    }
    if verbose:
        print("baseline M5 selected: {0}  (M3={1} M4={2} overlap={3})".format(
            len(baseline_indices), baseline_diag["m3_count"],
            baseline_diag["m4_count"], baseline_diag["m3_m4_overlap_count"]))

    csv_rows: List[Dict[str, Any]] = [{
        "cycle": 0,
        "selected_before": len(baseline_indices),
        "selected_after": len(baseline_indices),
        "resolved_count": 0, "remaining_count": len(baseline_indices),
        "newly_detected_count": 0,
        "resolved_fraction": 0.0, "new_fraction": 0.0, "jaccard": 1.0,
        "m3_count": baseline_diag["m3_count"], "m4_count": baseline_diag["m4_count"],
        "m3_m4_overlap_count": baseline_diag["m3_m4_overlap_count"],
        "m3_only_count": baseline_diag["m3_only_count"],
        "m4_only_count": baseline_diag["m4_only_count"],
        "cycle_mean_displacement": 0.0, "cycle_rms_displacement": 0.0,
        "cycle_max_displacement": 0.0,
        "cumulative_mean_displacement": 0.0, "cumulative_rms_displacement": 0.0,
        "cumulative_max_displacement": 0.0,
        "elapsed_seconds": round(baseline_seconds, 4), "stop_reason": "",
    }]

    # --- DRY RUN: validate + plan only, change nothing -----------------------
    if not apply:
        report["stop_reason"] = _STOP_DRY_RUN
        report["final_indices"] = sorted(baseline_indices)
        report["final_selected_count"] = len(baseline_indices)
        report["final_m5_report"] = baseline_report
        report["ever_selected_count"] = len(set(baseline_indices))
        report["runtime_seconds"] = time.time() - run_start
        if verbose:
            print("\n[DRY RUN] configuration validated; baseline detection done.")
            print("[DRY RUN] would run up to {0} cycle(s) of {1}x '{2}' smoothing "
                  "(strength {3}) on the current M5 region, re-detecting once per "
                  "cycle.".format(max_cycles, smoothing_iterations_per_cycle,
                                  smoothing_method, smoothing_strength))
            print("[DRY RUN] no backup made, no geometry modified.")
        if select_final and baseline_indices:
            mesh_utils.select_vertices(skin_mesh, sorted(baseline_indices), replace=True)
        _maybe_write_logs(report, csv_rows, log_path, save_json, save_csv, timestamp)
        return report

    # --- one-time pre-cleanup backup (only if there is something to clean) ----
    if create_backup and baseline_indices:
        backup_name = skin_mesh + _BACKUP_SUFFIX
        if mesh_utils.mesh_exists(backup_name):
            if verbose:
                print("[cleanup] backup '{0}' already exists; keeping it".format(backup_name))
            report["baseline"]["backup"] = backup_name
        elif maya_io is not None:
            made = maya_io.duplicate_mesh(skin_mesh, suffix=_BACKUP_SUFFIX)
            report["baseline"]["backup"] = made
        else:
            report["baseline"]["backup"] = None

    # --- iterative cleanup cycles --------------------------------------------
    current_indices = list(baseline_indices)
    current_report = baseline_report
    ever_selected = set(baseline_indices)
    consecutive_tiny = 0
    consecutive_stable = 0
    stop_reason: Optional[str] = None
    failed_cycle: Optional[int] = None

    for k in range(1, max_cycles + 1):
        cycle_t0 = time.time()
        previous_indices = list(current_indices)
        previous_report = current_report

        if not previous_indices:
            stop_reason = _STOP_NO_REGION
            if verbose:
                print("\n[cleanup] no detected region to smooth; stopping.")
            break

        if verbose:
            print("\n" + "-" * 50)
            print("M5 CLEANUP CYCLE {0}".format(k))
            print("-" * 50)
            print("Selected before: {0}".format(len(previous_indices)))

        cycle_failed = False
        fail_exc = None
        _open_undo_chunk("m5_cleanup_cycle_{0}".format(k))
        try:
            # B. smooth the FIXED current region (N internal iterations, no re-detect)
            smooth_t0 = time.time()
            before, after = smoothing_utils.smooth_mesh_region(
                skin_mesh,
                indices=previous_indices,
                strength=smoothing_strength,
                iterations=smoothing_iterations_per_cycle,
                method=smoothing_method,
                apply=True,
            )
            smoothing_seconds = time.time() - smooth_t0
            if not before:
                raise RuntimeError("smoothing returned no vertices for '{0}'".format(skin_mesh))

            # D. per-cycle displacement: region / outside / whole
            outside = _outside_indices(len(before), previous_indices)
            cyc_region = _disp(before, after, previous_indices)
            cyc_outside = _disp(before, after, outside)
            cyc_whole = _disp(before, after)

            # E. cumulative displacement: original -> current (whole + ever-touched)
            cum_whole = _disp(original_positions, after)
            cum_touched = _disp(original_positions, after, sorted(ever_selected))

            # F. re-detect M5 on the modified mesh (ONCE per cycle)
            new_indices, new_report, detection_seconds = _run_m5(
                skin_mesh, anatomical_meshes, target_offset, cfg)

            # G. compare previous vs new selection
            cmp = _compare_selections(previous_indices, new_indices)
        except Exception as exc:  # noqa: BLE001 - report, don't swallow
            cycle_failed = True
            fail_exc = exc
            import traceback
            print("[cleanup] EXCEPTION in cycle {0}:".format(k))
            traceback.print_exc()
        finally:
            _close_undo_chunk()

        if cycle_failed:
            failed_cycle = k
            stop_reason = "exception in cycle {0}: {1}".format(k, fail_exc)
            break

        # advance the region for the NEXT cycle (detector adapts to new surface)
        current_indices = new_indices
        current_report = new_report
        ever_selected |= set(new_indices)

        # H. M5 diagnostics for this re-detection
        diag = _m5_diagnostics(new_report)

        # --- convergence signals (this cycle) --------------------------------
        cycle_mean_disp = cyc_region["mean"]
        tiny = cycle_mean_disp < min_cycle_mean_displacement
        consecutive_tiny = consecutive_tiny + 1 if tiny else 0

        rel_change = (abs(cmp["selected_after"] - cmp["selected_before"])
                      / cmp["selected_before"]) if cmp["selected_before"] else 0.0
        stable = (cmp["jaccard"] >= stable_jaccard_threshold
                  and rel_change <= min_relative_selection_change)
        consecutive_stable = consecutive_stable + 1 if stable else 0

        cum_exceeded = (max_cumulative_displacement is not None
                        and cum_whole["max"] > max_cumulative_displacement)
        new_explode = (stop_if_new_detection_fraction_exceeds is not None
                       and cmp["new_fraction"] > stop_if_new_detection_fraction_exceeds)

        # --- decide (first match wins; soft stops respect min_cycles) --------
        if len(new_indices) == 0:
            stop_reason = _STOP_NO_REGION
        elif cum_exceeded:
            stop_reason = _STOP_MAX_CUM_DISP
        elif new_explode:
            stop_reason = _STOP_NEW_EXPLODE
        elif k >= min_cycles and consecutive_tiny >= convergence_patience:
            stop_reason = _STOP_TINY_DISP
        elif k >= min_cycles and consecutive_stable >= convergence_patience:
            stop_reason = _STOP_STABLE
        elif k >= max_cycles:
            stop_reason = _STOP_MAX_CYCLES

        cycle_seconds = time.time() - cycle_t0
        record = {
            "cycle": k,
            "smoothing": {
                "method": smoothing_method,
                "strength": smoothing_strength,
                "iterations": smoothing_iterations_per_cycle,
            },
            "selection": {
                "before": cmp["selected_before"],
                "after": cmp["selected_after"],
                "resolved": cmp["resolved_count"],
                "remaining": cmp["remaining_count"],
                "new": cmp["newly_detected_count"],
                "overlap": cmp["overlap_count"],
                "union": cmp["union_count"],
                "jaccard": cmp["jaccard"],
                "resolved_fraction": cmp["resolved_fraction"],
                "new_fraction": cmp["new_fraction"],
            },
            "m5": {
                "m3_count": diag["m3_count"],
                "m4_count": diag["m4_count"],
                "m3_m4_overlap": diag["m3_m4_overlap_count"],
                "m3_only": diag["m3_only_count"],
                "m4_only": diag["m4_only_count"],
                "final": diag["final_m5_count"],
            },
            "cycle_displacement": {
                "mean": cyc_region["mean"],
                "rms": cyc_region["rms"],
                "max": cyc_region["max"],
                "region": cyc_region,
                "outside": cyc_outside,
                "whole": cyc_whole,
            },
            "cumulative_displacement": {
                "mean": cum_whole["mean"],
                "rms": cum_whole["rms"],
                "max": cum_whole["max"],
                "whole": cum_whole,
                "touched": cum_touched,
            },
            "timing": {
                "smoothing_seconds": smoothing_seconds,
                "detection_seconds": detection_seconds,
                "cycle_seconds": cycle_seconds,
            },
            "stop_checks": {
                "tiny_displacement": bool(tiny),
                "stable_selection": bool(stable),
                "max_displacement_exceeded": bool(cum_exceeded),
                "new_detection_exploded": bool(new_explode),
                "consecutive_tiny": consecutive_tiny,
                "consecutive_stable": consecutive_stable,
            },
            "continue": stop_reason is None,
        }
        report["cycles"].append(record)
        csv_rows.append(_csv_row_from_cycle(record, stop_reason or ""))

        if verbose:
            _print_cycle(record, stop_reason)

        if stop_reason is not None:
            break

    if stop_reason is None:
        stop_reason = _STOP_MAX_CYCLES

    # --- finalize ------------------------------------------------------------
    final_cum = report["cycles"][-1]["cumulative_displacement"] if report["cycles"] else {}
    report["stop_reason"] = stop_reason
    report["completed_cycles"] = len(report["cycles"])
    report["failed_cycle"] = failed_cycle
    report["final_indices"] = sorted(current_indices)
    report["final_selected_count"] = len(current_indices)
    report["final_m5_report"] = current_report
    report["ever_selected_count"] = len(ever_selected)
    report["cumulative_displacement"] = final_cum
    report["runtime_seconds"] = time.time() - run_start

    if select_final and current_indices:
        mesh_utils.select_vertices(skin_mesh, sorted(current_indices), replace=True)

    if verbose:
        _print_summary(report)

    _maybe_write_logs(report, csv_rows, log_path, save_json, save_csv, timestamp)
    return report


# =============================================================================
# CONSOLE OUTPUT
# =============================================================================

def _print_cycle(record: Dict[str, Any], stop_reason: Optional[str]) -> None:
    sel = record["selection"]
    m5 = record["m5"]
    cyc = record["cycle_displacement"]
    cum = record["cumulative_displacement"]
    sm = record["smoothing"]
    print("\nSmoothing:")
    print("  method: {0}".format(sm["method"]))
    print("  iterations: {0}".format(sm["iterations"]))
    print("  strength: {0:.2f}".format(sm["strength"]))
    print("Displacement (region):")
    print("  mean: {0:.6f}".format(cyc["mean"]))
    print("  rms:  {0:.6f}".format(cyc["rms"]))
    print("  max:  {0:.6f}".format(cyc["max"]))
    print("  cumulative max (whole mesh): {0:.6f}".format(cum["max"]))
    print("Re-detection:")
    print("  selected after: {0}".format(sel["after"]))
    print("  resolved: {0}".format(sel["resolved"]))
    print("  remaining: {0}".format(sel["remaining"]))
    print("  new: {0}".format(sel["new"]))
    print("  Jaccard: {0:.4f}".format(sel["jaccard"]))
    print("M3: {0}   M4: {1}   M3n_M4: {2}".format(
        m5["m3_count"], m5["m4_count"], m5["m3_m4_overlap"]))
    print("Continue: {0}{1}".format(
        "yes" if stop_reason is None else "no",
        "" if stop_reason is None else "  (stop: {0})".format(stop_reason)))


def _print_summary(report: Dict[str, Any]) -> None:
    cum = report.get("cumulative_displacement", {}) or {}
    print("\n" + "=" * 50)
    print("M5 ITERATIVE CLEANUP COMPLETE")
    print("=" * 50)
    print("cycles: {0}".format(report["completed_cycles"]))
    print("stop reason: {0}".format(report["stop_reason"]))
    print("initial selected: {0}".format(report["baseline"].get("selected", 0)))
    print("final selected: {0}".format(report["final_selected_count"]))
    print("ever-selected (touched) vertices: {0}".format(report["ever_selected_count"]))
    print("cumulative max displacement: {0:.6f}".format(cum.get("max", 0.0)))
    print("runtime: {0:.2f}s".format(report["runtime_seconds"]))


# =============================================================================
# LOG DISPATCH
# =============================================================================

def _maybe_write_logs(report: Dict[str, Any],
                      csv_rows: List[Dict[str, Any]],
                      log_path: Optional[str],
                      save_json: bool,
                      save_csv: bool,
                      timestamp: str) -> None:
    if not (save_json or save_csv):
        return
    log_dir = log_path if log_path else "cleanup_logs"
    base = "m5_cleanup_{0}".format(timestamp)
    if save_json:
        p = _write_json(report, os.path.join(log_dir, base + ".json"))
        if p:
            report["log_files"]["json"] = p
    if save_csv:
        p = _write_csv(csv_rows, os.path.join(log_dir, base + ".csv"))
        if p:
            report["log_files"]["csv"] = p
