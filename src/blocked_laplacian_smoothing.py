"""
blocked_laplacian_smoothing.py
================================
EXPERIMENTAL pipeline that tests one specific idea: keep the exact pure
Jacobi Laplacian smoothing from :mod:`pure_laplacian_smoothing` (which was
proven to visibly flatten bumps -- see that module's docstring), and add the
SMALLEST possible mechanism that stops a local region once it actually
crosses into anatomy:

    smooth freely
    -> detect where the PROPOSED step creates a real anatomy crossing
    -> roll ONLY that local region back to its last safe position
    -> freeze it (permanently, for the rest of this run)
    -> keep smoothing everywhere else

This is deliberately NOT another optimization solver. There is no repair
target, no harmonic displacement field, no multi-ring remediation patch.
A frozen vertex's correction is always exactly::

    violating_vertex_position = previous_iteration_safe_position

Completely separate module
---------------------------
Does not modify and does not import from :mod:`final_cleanup_solver`. Does
not modify :mod:`pure_laplacian_smoothing` -- it REUSES that module's pure
helpers (``jacobi_laplacian_step``, ``build_falloff_weights``,
``select_rough_vertices``, ``ring_distances``, ``roughness_percentile_summary``)
by import, so the smoothing/selection/falloff behavior is *identical* to the
already-visually-verified pure experiment, not a hand-copied near-duplicate
that could quietly drift. The only new behavior added here is the
freeze-on-intersection mechanism itself.

Reuses ONLY
-----------
* :mod:`mesh_utils` -- read/write Maya vertices, adjacency, topology.
* :mod:`pure_laplacian_smoothing` -- the proven Jacobi step + region
  selection/falloff (imported, not reimplemented).
* :func:`artifact_detection.compute_laplacian_scores` -- automatic region
  selection and before/after roughness reporting, same as the pure module.
* :mod:`anatomy_constraint` -- the EXISTING true skin/anatomy triangle
  intersection detector (``analyze_skin_anatomy_intersections``) and its
  face-to-vertex mapping (``intersection_vertices_from_report``), used
  strictly as a READ-ONLY authoritative test of whether a proposed skin
  state creates a forbidden crossing. The detector itself is not modified,
  not reimplemented, and not weakened -- this module calls it exactly the
  way :mod:`final_cleanup_solver` already does (see its
  ``_scoped_intersection_core``, which this module's ``_scan_intersections``
  mirrors as a fresh, independent, public-API-only call -- it does not import
  that private helper).
* :func:`maya_io.duplicate_mesh` -- the same optional pre-edit backup every
  other stage in this project uses.

What this module deliberately does NOT contain
------------------------------------------------
Taubin force, shape force, rest-reference smoothing, M4, M5 correction
force, roughness-targeted per-vertex weighting (beyond the falloff already
in the pure module's region selection), trust-region clamping, minimum-
clearance floors, SDF attraction, closest-point projection, adaptive
damping, best-state optimization, oscillation logic, or broad multi-ring
repair. A vertex is either smoothing freely, or frozen at its last safe
position. Nothing else.

Nothing here modifies M1-M5, artifact_detection.py, anatomy_constraint.py,
final_cleanup_solver.py, smoothing_utils.py, pure_laplacian_smoothing.py, or
the intersection detector.
"""

from __future__ import print_function

import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import mesh_utils
import artifact_detection
import anatomy_constraint
import pure_laplacian_smoothing as pls

try:
    import maya_io
except ImportError:  # pragma: no cover - maya_io present in this project
    maya_io = None


DEFAULT_ROUGHNESS_PERCENTILE = 75.0
DEFAULT_GROWTH_RINGS = 2
DEFAULT_STRENGTH = 0.35
DEFAULT_ITERATIONS = 10
DEFAULT_FREEZE_RINGS = 1
DEFAULT_BOUNDARY_BUFFER_RINGS = 1
DEFAULT_INTERSECTION_TOLERANCE = 1e-6
DEFAULT_REPORT_INTERVAL = 1
_BACKUP_SUFFIX = "_pre_blocked_laplacian"


# =============================================================================
# 1. Intersection scan -- thin, read-only use of the EXISTING detector
# =============================================================================

def _scan_intersections(skin_mesh: str,
                        positions: List[List[float]],
                        scope: Sequence[int],
                        backend: Any,
                        neighbors: List[List[int]],
                        skin_topology: Dict[str, Any],
                        boundary_buffer_rings: int,
                        intersection_tolerance: float,
                        ) -> Tuple[List[int], Dict[str, Any]]:
    """Real triangle/triangle intersection scan of a PROPOSED (in-memory,
    not-yet-written) ``positions`` array, scoped to ``scope``. Returns
    ``(offending_vertex_indices, report)``, where ``offending_vertex_indices``
    is ``intersecting_skin_faces`` mapped to vertex ids and clipped to
    ``scope`` -- i.e. exactly the existing detector's own outputs, not a
    reimplementation of triangle/triangle testing. Eye/mouth/nostril/neck
    boundary OPENINGS are excluded exactly as everywhere else in this project
    (``boundary_buffer_rings``, the detector's own established behavior) --
    this module does not touch that logic.
    """
    report = anatomy_constraint.analyze_skin_anatomy_intersections(
        skin_mesh, skin_indices=scope, backend=backend, positions=positions,
        neighbors=neighbors, skin_topology=skin_topology,
        boundary_buffer_rings=boundary_buffer_rings,
        intersection_tolerance=intersection_tolerance,
        detailed=False, verbose=False)
    offending = anatomy_constraint.intersection_vertices_from_report(
        report, skin_topology, allowed=set(scope))
    return offending, report


def freeze_footprint(offending: Sequence[int],
                     neighbors: List[List[int]],
                     freeze_rings: int,
                     active_set: Set[int],
                     ) -> Set[int]:
    """Vertices to freeze given the ``offending`` (intersecting) vertex set:
    ``freeze_rings=0`` freezes exactly ``offending``; ``freeze_rings=N>0``
    also grows ``N`` topological rings outward from it (plain adjacency BFS,
    reusing :func:`pure_laplacian_smoothing.ring_distances`). Always clipped
    to ``active_set`` -- a vertex outside the smoothing region (already fixed
    boundary/inactive) is never added to the freeze set, since it was never
    going to move anyway. Pure function, no Maya access -- unit-tested in
    isolation from the real intersection detector.
    """
    if not offending:
        return set()
    if freeze_rings > 0:
        footprint = set(pls.ring_distances(offending, neighbors, freeze_rings).keys())
    else:
        footprint = set(int(i) for i in offending)
    return footprint & set(active_set)


# =============================================================================
# 2. Main entry point
# =============================================================================

def run_blocked_laplacian_smoothing(skin_mesh: str,
                                    anatomy_backend: Any,
                                    indices: Optional[Sequence[int]] = None,
                                    roughness_percentile: float = DEFAULT_ROUGHNESS_PERCENTILE,
                                    growth_rings: int = DEFAULT_GROWTH_RINGS,
                                    strength: float = DEFAULT_STRENGTH,
                                    iterations: int = DEFAULT_ITERATIONS,
                                    freeze_rings: int = DEFAULT_FREEZE_RINGS,
                                    protect_boundary: bool = True,
                                    boundary_buffer_rings: int = DEFAULT_BOUNDARY_BUFFER_RINGS,
                                    intersection_tolerance: float = DEFAULT_INTERSECTION_TOLERANCE,
                                    apply: bool = True,
                                    create_backup: bool = True,
                                    backup_suffix: str = _BACKUP_SUFFIX,
                                    report_interval: int = DEFAULT_REPORT_INTERVAL,
                                    verbose: bool = True,
                                    ) -> Dict[str, Any]:
    """EXPERIMENTAL: pure Jacobi Laplacian smoothing (identical operator to
    :func:`pure_laplacian_smoothing.run_pure_laplacian_smoothing`) that
    locally freezes a region the moment its PROPOSED step would create a real
    anatomy crossing, rolling that region back to its last safe position.
    Nothing else resists the smoothing -- no clearance floor, no SDF
    attraction, no repair target. See the module docstring for the full
    design.

    Parameters
    ----------
    skin_mesh:
        Mesh to smooth (read fresh from Maya; no reliance on any pre-existing
        Python globals -- works in a brand new Maya session).
    anatomy_backend:
        A backend exposing ``.anatomy_surface_cache()`` (e.g.
        ``anatomy_constraint.MayaAnatomyBackend`` -- build it once with
        ``_make_anatomy_backend`` in the d98 wrapper and pass it in here;
        anatomy never moves during this run, so the cache is built once and
        reused for every iteration). Required -- without it there is nothing
        to test proposals against.
    indices, roughness_percentile, growth_rings, protect_boundary:
        Same region-selection/falloff behavior as
        :func:`pure_laplacian_smoothing.run_pure_laplacian_smoothing`
        (literally reused, not reimplemented).
    strength:
        Direct multiplier on the Jacobi step for still-movable vertices --
        NOT damped, clamped, or adaptively reduced. Identical meaning to the
        pure module. A frozen vertex's effective step is permanently 0 for
        the rest of the run; nothing else is weakened because safety exists.
    freeze_rings:
        When a proposed step creates a real intersection, the vertices of
        the offending skin faces (clipped to the active region) are always
        frozen. ``freeze_rings=0`` freezes exactly those vertices;
        ``freeze_rings=1`` (default) also freezes their immediate
        topological neighbors, to avoid an isolated single-vertex freeze
        seam.
    boundary_buffer_rings, intersection_tolerance:
        Passed straight through to the existing detector; same defaults used
        elsewhere in this project (:data:`DEFAULT_BOUNDARY_BUFFER_RINGS` = 1,
        :data:`DEFAULT_INTERSECTION_TOLERANCE` = 1e-6).
    apply:
        If False, compute everything in memory and do not write or back up
        the scene.

    Returns
    -------
    dict
        See the module-level field list in the class docstring analogue
        below (``selected_count``, ``active_count``, ``blocked_indices``,
        per-iteration blocking/intersection logs, roughness before/after,
        displacement stats, ``final_forbidden_intersection_count``, etc).
    """
    t0 = time.time()

    if not mesh_utils.mesh_exists(skin_mesh):
        raise ValueError("blocked_laplacian_smoothing: skin_mesh '{0}' does not exist".format(skin_mesh))
    if anatomy_backend is None:
        raise ValueError("blocked_laplacian_smoothing: anatomy_backend is required "
                         "(nothing to test proposals against without it)")
    if iterations < 1:
        raise ValueError("iterations must be >= 1, got {0}".format(iterations))
    if growth_rings < 0:
        raise ValueError("growth_rings must be >= 0, got {0}".format(growth_rings))
    if freeze_rings < 0:
        raise ValueError("freeze_rings must be >= 0, got {0}".format(freeze_rings))

    positions0 = mesh_utils.get_mesh_vertices(skin_mesh)
    if not positions0:
        raise ValueError("blocked_laplacian_smoothing: skin_mesh '{0}' has no readable "
                         "vertices".format(skin_mesh))
    neighbors = mesh_utils.get_vertex_neighbors(skin_mesh)
    n = len(positions0)
    mesh_fn = mesh_utils.get_mesh_fn(skin_mesh)
    skin_topology = mesh_utils.get_triangle_topology(mesh_fn)

    boundary: Set[int] = set()
    if protect_boundary:
        try:
            boundary = set(mesh_utils.get_boundary_vertices(skin_mesh) or [])
        except Exception:
            boundary = set()

    # -- region selection: literally the pure module's own logic -------------
    scores: Optional[Dict[int, float]] = None
    if indices is not None:
        core = sorted((set(int(i) for i in indices if 0 <= int(i) < n)) - boundary)
        selection_source = "caller"
    else:
        scores = artifact_detection.compute_laplacian_scores(skin_mesh)
        core = sorted(set(pls.select_rough_vertices(scores, roughness_percentile)) - boundary)
        selection_source = "auto_percentile"

    if not core:
        if verbose:
            print("[blocked_laplacian] no vertices selected; nothing to do")
        return {"skin_mesh": skin_mesh, "selected_count": 0, "active_count": 0,
               "selection_source": selection_source, "selected_indices": [],
               "active_indices": [], "blocked_indices": [], "iterations_requested": iterations,
               "iterations_completed": 0, "dry_run": not apply,
               "runtime_seconds": time.time() - t0}

    weights = pls.build_falloff_weights(core, neighbors, growth_rings, exclude=boundary)
    active = sorted(weights.keys())
    active_set = set(active)
    scaled_weights = {i: w * float(strength) for i, w in weights.items()}

    if scores is None:
        scores = artifact_detection.compute_laplacian_scores(skin_mesh, indices=core)
    roughness_pct_before = pls.roughness_percentile_summary(
        [scores[i] for i in core if i in scores])

    # -- baseline validity check (the whole design assumes we start valid) --
    _baseline_offending, baseline_report = _scan_intersections(
        skin_mesh, positions0, active, anatomy_backend, neighbors, skin_topology,
        boundary_buffer_rings, intersection_tolerance)
    initial_intersection_count = baseline_report.get("intersecting_skin_face_count", 0)
    if initial_intersection_count > 0 and verbose:
        print("[blocked_laplacian] WARNING: starting mesh already has {0} forbidden "
             "intersecting face(s) in the active region BEFORE any smoothing. "
             "Rollback-to-previous-position only prevents making things WORSE -- it "
             "cannot fix a baseline that was already invalid. Not auto-repairing "
             "the baseline (by design).".format(initial_intersection_count))

    if create_backup and apply and maya_io is not None:
        backup_name = skin_mesh + backup_suffix
        if not mesh_utils.mesh_exists(backup_name):
            maya_io.duplicate_mesh(skin_mesh, suffix=backup_suffix)
        elif verbose:
            print("[blocked_laplacian] backup '{0}' already exists; keeping it".format(backup_name))

    if verbose:
        print("[blocked_laplacian] EXPERIMENTAL -- pure Laplacian smoothing + local "
             "freeze-on-intersection. No repair target, no clearance floor.")
        print("[blocked_laplacian] selected={0} ({1}) active incl. falloff={2} "
             "(boundary-protected={3}) strength={4} growth_rings={5} freeze_rings={6} "
             "iterations={7} initial_intersections={8}".format(
                 len(core), selection_source, len(active), protect_boundary,
                 strength, growth_rings, freeze_rings, iterations,
                 initial_intersection_count))

    x = [list(p) for p in positions0]  # last known-SAFE committed state
    blocked: Set[int] = set()
    newly_blocked_per_iteration: List[int] = []
    cumulative_blocked_per_iteration: List[int] = []
    intersections_before_rollback: List[int] = []
    intersections_after_rollback: List[int] = []
    iterations_completed = 0
    stopped_early = False
    stop_reason = None

    for it in range(1, int(iterations) + 1):
        movable = [i for i in active if i not in blocked]
        if not movable:
            if verbose:
                print("  [iter {0:3d}] every active vertex is blocked; nothing left to "
                     "move -- stopping early".format(it))
            stopped_early = True
            stop_reason = "all active vertices blocked before requested iterations completed"
            break

        positions_before = [list(p) for p in x]
        iter_weights = {i: scaled_weights[i] for i in movable}
        proposal = pls.jacobi_laplacian_step(x, neighbors, iter_weights)

        offending, report = _scan_intersections(
            skin_mesh, proposal, active, anatomy_backend, neighbors, skin_topology,
            boundary_buffer_rings, intersection_tolerance)
        proposed_count = report.get("intersecting_skin_face_count", 0)

        newly_blocked: Set[int] = set()
        after_count = 0
        if offending:
            # Tier 1: roll back the offending faces' own vertices (+ freeze_rings halo).
            footprint = freeze_footprint(offending, neighbors, freeze_rings, active_set)
            for i in footprint:
                proposal[i] = list(positions_before[i])
            newly_blocked |= (footprint - blocked)
            blocked |= footprint

            offending2, report2 = _scan_intersections(
                skin_mesh, proposal, active, anatomy_backend, neighbors, skin_topology,
                boundary_buffer_rings, intersection_tolerance)
            after_count = report2.get("intersecting_skin_face_count", 0)

            if after_count > 0:
                # Tier 2 (spec-mandated smallest fallback): also roll back the
                # immediate 1-ring of whatever is STILL offending. No broad
                # repair, no anatomy-derived target -- still just "restore to
                # the previous safe position."
                footprint2 = freeze_footprint(offending2, neighbors, 1, active_set)
                for i in footprint2:
                    proposal[i] = list(positions_before[i])
                newly_blocked |= (footprint2 - blocked)
                blocked |= footprint2

                offending3, report3 = _scan_intersections(
                    skin_mesh, proposal, active, anatomy_backend, neighbors, skin_topology,
                    boundary_buffer_rings, intersection_tolerance)
                after_count = report3.get("intersecting_skin_face_count", 0)

                if after_count > 0:
                    if verbose:
                        print("  [iter {0:3d}] !!! could not restore a valid state with "
                             "the minimal rollback fallback ({1} face(s) still "
                             "intersecting) -- STOPPING the experiment rather than "
                             "invoking broad repair. Last known-safe state is kept "
                             "(this iteration's proposal is discarded).".format(it, after_count))
                    stopped_early = True
                    stop_reason = ("rollback + 1-ring fallback could not clear a real "
                                   "intersection at iteration {0} ({1} face(s) remained)"
                                   .format(it, after_count))
                    break

        x = proposal
        iterations_completed = it
        newly_blocked_per_iteration.append(len(newly_blocked))
        cumulative_blocked_per_iteration.append(len(blocked))
        intersections_before_rollback.append(proposed_count)
        intersections_after_rollback.append(after_count if offending else proposed_count)

        if verbose and (it == 1 or it % max(1, report_interval) == 0
                       or it == int(iterations) or newly_blocked):
            live = pls.roughness_percentile_summary(
                list(pls._laplacian_magnitudes(x, neighbors, core).values()))
            line = "  [iter {0:3d}] roughness={1:.5f}  proposed intersections={2}  " \
                  "newly blocked={3}  total blocked={4}".format(
                      it, live["mean"], proposed_count, len(newly_blocked), len(blocked))
            if offending:
                line += "  after rollback intersections={0}".format(after_count)
            print(line)

    if apply:
        mesh_utils.set_mesh_vertices(skin_mesh, x)

    roughness_pct_after = pls.roughness_percentile_summary(
        list(pls._laplacian_magnitudes(x, neighbors, core).values()))

    disp = [mesh_utils.vec_length(mesh_utils.vec_sub(x[i], positions0[i])) for i in active]
    mean_disp = (sum(disp) / len(disp)) if disp else 0.0
    max_disp = max(disp) if disp else 0.0
    max_disp_idx = active[disp.index(max_disp)] if disp else -1

    _final_offending, final_report = _scan_intersections(
        skin_mesh, x, active, anatomy_backend, neighbors, skin_topology,
        boundary_buffer_rings, intersection_tolerance)
    final_intersection_count = final_report.get("intersecting_skin_face_count", 0)

    result: Dict[str, Any] = {
        "skin_mesh": skin_mesh,
        "selected_count": len(core),
        "active_count": len(active),
        "selection_source": selection_source,
        "selected_indices": core,
        "active_indices": active,
        "blocked_indices": sorted(blocked),
        "blocked_count": len(blocked),
        "blocked_pct_of_active": (100.0 * len(blocked) / len(active)) if active else 0.0,
        "iterations_requested": int(iterations),
        "iterations_completed": iterations_completed,
        "stopped_early": stopped_early,
        "stop_reason": stop_reason,
        "strength": float(strength),
        "growth_rings": int(growth_rings),
        "freeze_rings": int(freeze_rings),
        "roughness_percentile": float(roughness_percentile),
        "protect_boundary": bool(protect_boundary),
        "dry_run": not apply,
        "initial_intersection_count": initial_intersection_count,
        "final_forbidden_intersection_count": final_intersection_count,
        "newly_blocked_per_iteration": newly_blocked_per_iteration,
        "cumulative_blocked_per_iteration": cumulative_blocked_per_iteration,
        "intersections_before_rollback_per_iteration": intersections_before_rollback,
        "intersections_after_rollback_per_iteration": intersections_after_rollback,
        "roughness": {
            "before_mean": roughness_pct_before["mean"],
            "after_mean": roughness_pct_after["mean"],
            "percentiles_before": roughness_pct_before,
            "percentiles_after": roughness_pct_after,
        },
        "mean_displacement": mean_disp,
        "max_displacement": max_disp,
        "max_displacement_vertex": max_disp_idx,
        "runtime_seconds": time.time() - t0,
    }

    if verbose:
        print_blocked_laplacian_report(result)
    return result


def print_blocked_laplacian_report(result: Dict[str, Any]) -> None:
    """Human-readable summary of a :func:`run_blocked_laplacian_smoothing` result."""
    print("\n" + "=" * 60)
    print("BLOCKED LAPLACIAN SMOOTHING -- pure smoothing + local freeze-on-intersection")
    print("=" * 60)
    print("selected: {0} ({1})   active incl. falloff: {2}   boundary-protected: {3}".format(
        result.get("selected_count"), result.get("selection_source"),
        result.get("active_count"), result.get("protect_boundary")))
    print("strength={0}  growth_rings={1}  freeze_rings={2}  iterations requested={3} "
         "completed={4}".format(
             result.get("strength"), result.get("growth_rings"), result.get("freeze_rings"),
             result.get("iterations_requested"), result.get("iterations_completed")))
    if result.get("stopped_early"):
        print("STOPPED EARLY: {0}".format(result.get("stop_reason")))
    if result.get("dry_run"):
        print("(dry run -- scene NOT modified)")
    print("initial forbidden intersections (active region, before any smoothing): {0}".format(
        result.get("initial_intersection_count")))
    print("blocked vertices: {0} ({1:.1f}% of active region)".format(
        result.get("blocked_count"), result.get("blocked_pct_of_active", 0.0)))
    rough = result.get("roughness") or {}
    print("roughness (mean): {0:.5f} -> {1:.5f}".format(
        rough.get("before_mean", 0.0), rough.get("after_mean", 0.0)))
    pb = rough.get("percentiles_before") or {}
    pa = rough.get("percentiles_after") or {}
    print("  before  p50={0:.5f}  p90={1:.5f}  p95={2:.5f}  p99={3:.5f}  max={4:.5f}".format(
        pb.get("p50", 0.0), pb.get("p90", 0.0), pb.get("p95", 0.0),
        pb.get("p99", 0.0), pb.get("max", 0.0)))
    print("  after   p50={0:.5f}  p90={1:.5f}  p95={2:.5f}  p99={3:.5f}  max={4:.5f}".format(
        pa.get("p50", 0.0), pa.get("p90", 0.0), pa.get("p95", 0.0),
        pa.get("p99", 0.0), pa.get("max", 0.0)))
    print("displacement from STARTING mesh: mean={0:.5f}  max={1:.5f} (vertex {2})".format(
        result.get("mean_displacement", 0.0), result.get("max_displacement", 0.0),
        result.get("max_displacement_vertex", -1)))
    print("FINAL forbidden intersection count (active region): {0}".format(
        result.get("final_forbidden_intersection_count")))
    print("runtime: {0:.2f}s".format(result.get("runtime_seconds", 0.0)))
    print("=" * 60)
