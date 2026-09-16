"""
micro_clearance_push.py
=========================
A tiny, ONE-SHOT, post-processing correction for the FINISHED
blocked-Laplacian result (see :mod:`blocked_laplacian_smoothing`). It is NOT
a smoothing pass and NOT a repair pipeline. It answers a single narrow
question after the surface already looks good: in the very small number of
closed-skin regions where internal anatomy is still visible/too close, push
just that tiny local skin patch outward, once, to a small clearance floor.

    good smooth skin
        -> detect forbidden skin/anatomy crossings (the EXISTING true
           triangle/triangle detector, same eye/mouth/opening exclusions)
        -> push only the offending ("core") vertices outward to
           target_clearance, using the EXISTING closest-point/outward
           machinery (anatomy_constraint.enforce_anatomy_clearance)
        -> blend that per-vertex correction ONE topology ring outward at
           half weight, so there is no hard seam (no independent anatomy
           re-projection of the ring -- purely a spatial blend of the core
           correction vectors)
        -> re-run the true detector once, report, STOP

Completely separate module
---------------------------
Does not modify and does not import :mod:`pure_laplacian_smoothing`,
:mod:`blocked_laplacian_smoothing`, :mod:`final_cleanup_solver`, M1-M5, or
the intersection detector itself. Does not perform Laplacian smoothing, does
not construct a harmonic repair patch, does not iterate a region-wide
collision-correction loop, and does not attract the whole skin toward
anatomy -- it touches only the faces that are ACTUALLY crossing, plus their
immediate 1-ring, exactly once.

Reuses ONLY
-----------
* :mod:`mesh_utils` -- read/write Maya vertices, adjacency, topology,
  normals, and ``get_boundary_vertices`` (a pure topology fact).
* :mod:`anatomy_constraint` -- the EXISTING true skin/anatomy triangle
  intersection detector (``analyze_skin_anatomy_intersections`` +
  ``intersection_vertices_from_report``, same eye/mouth/opening exclusion
  behavior as everywhere else in this project) AND the existing exact
  closest-point / outward-direction clearance projection
  (``enforce_anatomy_clearance``) -- the SAME per-vertex machinery
  :mod:`final_cleanup_solver` already reuses for its own clearance
  projection. Neither is reimplemented or modified here.
* :func:`maya_io.duplicate_mesh` -- the same optional pre-edit backup every
  other stage in this project uses.

Nothing here modifies M1-M5, artifact_detection.py, anatomy_constraint.py,
final_cleanup_solver.py, smoothing_utils.py, pure_laplacian_smoothing.py,
blocked_laplacian_smoothing.py, or the intersection detector.
"""

from __future__ import print_function

import time
from typing import Any, Dict, List, Optional, Sequence, Set

import mesh_utils
import anatomy_constraint

try:
    import maya_io
except ImportError:  # pragma: no cover - maya_io present in this project
    maya_io = None


DEFAULT_TARGET_CLEARANCE = 0.15
DEFAULT_BLEND_RINGS = 1
DEFAULT_BOUNDARY_BUFFER_RINGS = 1
DEFAULT_INTERSECTION_TOLERANCE = 1e-6
DEFAULT_MAX_CONSTRAINT_ITERATIONS = anatomy_constraint.DEFAULT_MAX_CONSTRAINT_ITERATIONS
DEFAULT_CLEARANCE_TOLERANCE = anatomy_constraint.DEFAULT_CLEARANCE_TOLERANCE
_BACKUP_SUFFIX = "_pre_micro_clearance_push"


# =============================================================================
# 1. PURE HELPERS (no Maya access; unit-tested in isolation)
# =============================================================================

def ring_distances(seeds: Sequence[int],
                   neighbors: List[List[int]],
                   max_rings: int,
                   ) -> Dict[int, int]:
    """Plain adjacency BFS ring distance (0 = a seed itself) from ``seeds``,
    out to ``max_rings``. Standalone (this module does not import the
    Laplacian-experiment modules' equivalent -- see module docstring)."""
    dist: Dict[int, int] = {int(i): 0 for i in seeds}
    frontier: Set[int] = set(dist.keys())
    for r in range(1, max(0, int(max_rings)) + 1):
        nxt: Set[int] = set()
        for u in frontier:
            if 0 <= u < len(neighbors):
                for w in neighbors[u]:
                    if w not in dist:
                        dist[w] = r
                        nxt.add(w)
        if not nxt:
            break
        frontier = nxt
    return dist


def blend_corrections(core_corrections: Dict[int, List[float]],
                      neighbors: List[List[int]],
                      blend_rings: int,
                      exclude: Optional[Set[int]] = None,
                      ) -> Dict[int, List[float]]:
    """Spatially blend a per-vertex correction-displacement field OUTWARD by
    ``blend_rings`` topological rings, WITHOUT re-projecting the halo onto
    anatomy at all -- purely a diffusion of the core's own correction
    vectors, so there is no hard seam at the edge of the pushed region.

    Core vertices (the keys of ``core_corrections``) always keep their own
    exact correction (weight 1.0). A ring-``r`` halo vertex's blended
    correction is ``0.5 * mean(blended correction of its already-blended
    neighbors one ring closer to the core)`` -- ring 1 is therefore exactly
    ``0.5 * mean(corrections of its CORE neighbors)`` (the table requested:
    core=1.0, ring1=0.5), and any further ring (if ``blend_rings`` > 1)
    continues the same halving pattern outward. Vertices in ``exclude`` (e.g.
    true topological mesh-boundary vertices) never receive a blended value.
    """
    excl = set(exclude or [])
    blended: Dict[int, List[float]] = {
        i: list(v) for i, v in core_corrections.items() if i not in excl
    }
    seen: Set[int] = set(blended.keys())
    frontier: Set[int] = set(blended.keys())
    for _ring in range(1, max(0, int(blend_rings)) + 1):
        nxt: Dict[int, List[float]] = {}
        for j in range(len(neighbors)):
            if j in seen or j in excl:
                continue
            contributing = [blended[k] for k in neighbors[j] if k in frontier]
            if not contributing:
                continue
            avg = mesh_utils.vec_mean(contributing)
            nxt[j] = [0.5 * c for c in avg]
        if not nxt:
            break
        blended.update(nxt)
        seen |= set(nxt.keys())
        frontier = set(nxt.keys())
    return blended


# =============================================================================
# 2. Intersection scan -- thin, read-only use of the EXISTING detector
# =============================================================================

def _scan_intersections(skin_mesh: str,
                        positions: List[List[float]],
                        scope: Optional[Sequence[int]],
                        backend: Any,
                        neighbors: List[List[int]],
                        skin_topology: Dict[str, Any],
                        boundary_buffer_rings: int,
                        intersection_tolerance: float,
                        ):
    """Real triangle/triangle intersection scan of the given ``positions``,
    scoped to ``scope`` (``None`` scans the whole mesh). Returns
    ``(offending_vertex_indices, report)`` -- the existing detector's own
    outputs, not a reimplementation. Eye/mouth/nostril/neck boundary OPENINGS
    are excluded exactly as everywhere else in this project
    (``boundary_buffer_rings``); this module does not touch that logic.
    """
    report = anatomy_constraint.analyze_skin_anatomy_intersections(
        skin_mesh, skin_indices=scope, backend=backend, positions=positions,
        neighbors=neighbors, skin_topology=skin_topology,
        boundary_buffer_rings=boundary_buffer_rings,
        intersection_tolerance=intersection_tolerance,
        detailed=False, verbose=False)
    allowed = set(scope) if scope is not None else None
    offending = anatomy_constraint.intersection_vertices_from_report(
        report, skin_topology, allowed=allowed)
    return offending, report


# =============================================================================
# 3. Main entry point
# =============================================================================

def run_micro_clearance_push(skin_mesh: str,
                             anatomy_backend: Any,
                             indices: Optional[Sequence[int]] = None,
                             target_clearance: float = DEFAULT_TARGET_CLEARANCE,
                             blend_rings: int = DEFAULT_BLEND_RINGS,
                             protect_boundary: bool = True,
                             boundary_buffer_rings: int = DEFAULT_BOUNDARY_BUFFER_RINGS,
                             intersection_tolerance: float = DEFAULT_INTERSECTION_TOLERANCE,
                             max_constraint_iterations: int = DEFAULT_MAX_CONSTRAINT_ITERATIONS,
                             clearance_tolerance: float = DEFAULT_CLEARANCE_TOLERANCE,
                             apply: bool = True,
                             create_backup: bool = True,
                             backup_suffix: str = _BACKUP_SUFFIX,
                             verbose: bool = True,
                             ) -> Dict[str, Any]:
    """ONE-SHOT local outward clearance push on the small remaining forbidden
    skin/anatomy region. See the module docstring for the full design --
    this is explicitly NOT another smoothing pass or repair pipeline.

    Parameters
    ----------
    skin_mesh:
        Mesh to correct (read fresh from Maya; no reliance on any
        pre-existing Python globals -- works in a brand new Maya session).
    anatomy_backend:
        A backend exposing ``.anatomy_surface_cache()`` (for the
        intersection scan) and ``.exact_closest()``/``.is_closed()``/
        ``.point_inside()`` (for the clearance push itself) -- i.e. a real
        ``anatomy_constraint.MayaAnatomyBackend``. Build it once (e.g. via
        the d98 wrapper's ``_make_anatomy_backend``) and pass it in.
    indices:
        If given, the intersection scan is scoped to these vertices (e.g.
        reuse ``cleanup_result["active_indices"]``). If omitted (the usual
        case for a final pass), the WHOLE mesh is scanned -- this runs once,
        so the extra cost is acceptable.
    target_clearance:
        The clearance floor every offending ("core") vertex is pushed out
        to, via the EXISTING ``anatomy_constraint.enforce_anatomy_clearance``
        (exact closest-point + outward-direction, iteratively re-queried --
        the same machinery :mod:`final_cleanup_solver` already uses for its
        own clearance projection). A core vertex already at or beyond this
        clearance is left untouched (zero correction).
    blend_rings:
        How many topological rings beyond the core to blend the correction
        into (no independent anatomy re-projection there -- see
        :func:`blend_corrections`). Default 1, per spec.
    apply:
        If False, compute everything in memory and do not write or back up
        the scene.

    Returns
    -------
    dict
        ``core_indices`` (vertices directly pushed to the clearance floor),
        ``halo_indices`` (ring-1..``blend_rings`` vertices that received a
        blended correction only, not a direct push), ``blended_indices``
        (the union of both -- everything that moved at all),
        ``initial_forbidden_face_count``, ``final_forbidden_face_count``,
        ``corrected_core_vertex_count``, ``blended_vertex_count`` (halo-only,
        i.e. additional to the core), ``mean_correction_displacement``,
        ``max_correction_displacement``, ``max_correction_vertex``.
    """
    t0 = time.time()

    if not mesh_utils.mesh_exists(skin_mesh):
        raise ValueError("micro_clearance_push: skin_mesh '{0}' does not exist".format(skin_mesh))
    if anatomy_backend is None:
        raise ValueError("micro_clearance_push: anatomy_backend is required")
    if blend_rings < 0:
        raise ValueError("blend_rings must be >= 0, got {0}".format(blend_rings))

    positions0 = mesh_utils.get_mesh_vertices(skin_mesh)
    if not positions0:
        raise ValueError("micro_clearance_push: skin_mesh '{0}' has no readable "
                         "vertices".format(skin_mesh))
    neighbors = mesh_utils.get_vertex_neighbors(skin_mesh)
    mesh_fn = mesh_utils.get_mesh_fn(skin_mesh)
    skin_topology = mesh_utils.get_triangle_topology(mesh_fn)
    try:
        normals = mesh_utils.get_vertex_normals(skin_mesh)
    except Exception:
        normals = None

    boundary: Set[int] = set()
    if protect_boundary:
        try:
            boundary = set(mesh_utils.get_boundary_vertices(skin_mesh) or [])
        except Exception:
            boundary = set()

    scope = None if indices is None else sorted(
        set(int(i) for i in indices if 0 <= int(i) < len(positions0)))

    core, initial_report = _scan_intersections(
        skin_mesh, positions0, scope, anatomy_backend, neighbors, skin_topology,
        boundary_buffer_rings, intersection_tolerance)
    core = sorted(set(core) - boundary)
    initial_forbidden_face_count = initial_report.get("intersecting_skin_face_count", 0)

    if verbose:
        print("[micro_clearance_push] ONE-SHOT outward clearance push -- no smoothing, "
             "no repair patch, no iteration.")
        print("[micro_clearance_push] initial forbidden faces={0}  offending core "
             "vertices={1}  target_clearance={2}  blend_rings={3}".format(
                 initial_forbidden_face_count, len(core), target_clearance, blend_rings))

    if not core:
        if verbose:
            print("[micro_clearance_push] nothing forbidden found; no correction needed")
        return {
            "skin_mesh": skin_mesh, "core_indices": [], "halo_indices": [],
            "blended_indices": [], "initial_forbidden_face_count": initial_forbidden_face_count,
            "final_forbidden_face_count": initial_forbidden_face_count,
            "corrected_core_vertex_count": 0, "blended_vertex_count": 0,
            "mean_correction_displacement": 0.0, "max_correction_displacement": 0.0,
            "max_correction_vertex": -1, "target_clearance": float(target_clearance),
            "blend_rings": int(blend_rings), "dry_run": not apply,
            "runtime_seconds": time.time() - t0,
        }

    # -- 1 shot: push each offending core vertex out to target_clearance ----
    # No pre-check against is_clearance_satisfied() here: a genuinely
    # crossing vertex can have a large UNSIGNED exact_closest distance while
    # still being on the wrong (inside) side of a closed anatomy mesh, so a
    # plain distance-only pre-check would wrongly call it "already safe".
    # enforce_anatomy_clearance() itself makes the correct resolved check
    # (distance AND not-inside) as the first action of its own loop, and is
    # a true no-op (zero corrections) for a vertex that is already fine.
    x = [list(p) for p in positions0]
    core_corrections: Dict[int, List[float]] = {}
    for i in core:
        nrm = normals[i] if (normals and 0 <= i < len(normals)) else None
        sol = anatomy_constraint.enforce_anatomy_clearance(
            x[i], anatomy_backend, target_clearance,
            max_constraint_iterations=max_constraint_iterations,
            clearance_tolerance=clearance_tolerance, skin_normal=nrm)
        new_p = sol["position"]
        correction = mesh_utils.vec_sub(new_p, x[i])
        if mesh_utils.vec_length(correction) > 1e-12:
            core_corrections[i] = correction

    # -- blend that correction field ONE ring outward, no re-projection -----
    blended = blend_corrections(core_corrections, neighbors, blend_rings, exclude=boundary)
    for i, corr in blended.items():
        x[i] = [x[i][k] + corr[k] for k in range(3)]

    halo_indices = sorted(set(blended.keys()) - set(core_corrections.keys()))
    blended_indices = sorted(blended.keys())

    if create_backup and apply and maya_io is not None:
        backup_name = skin_mesh + backup_suffix
        if not mesh_utils.mesh_exists(backup_name):
            maya_io.duplicate_mesh(skin_mesh, suffix=backup_suffix)
        elif verbose:
            print("[micro_clearance_push] backup '{0}' already exists; keeping it".format(
                backup_name))

    if apply:
        mesh_utils.set_mesh_vertices(skin_mesh, x)

    _final_offending, final_report = _scan_intersections(
        skin_mesh, x, scope, anatomy_backend, neighbors, skin_topology,
        boundary_buffer_rings, intersection_tolerance)
    final_forbidden_face_count = final_report.get("intersecting_skin_face_count", 0)

    disp = [mesh_utils.vec_length(mesh_utils.vec_sub(x[i], positions0[i])) for i in blended_indices]
    mean_disp = (sum(disp) / len(disp)) if disp else 0.0
    max_disp = max(disp) if disp else 0.0
    max_disp_idx = blended_indices[disp.index(max_disp)] if disp else -1

    if verbose and final_forbidden_face_count > 0:
        print("[micro_clearance_push] NOTE: {0} forbidden face(s) remain after the "
             "one-shot push. This module performs the correction ONCE by design and "
             "does not loop or escalate -- re-run manually if needed, or investigate "
             "why a single exact-clearance push did not clear it.".format(
                 final_forbidden_face_count))

    result: Dict[str, Any] = {
        "skin_mesh": skin_mesh,
        "core_indices": sorted(core_corrections.keys()),
        "halo_indices": halo_indices,
        "blended_indices": blended_indices,
        "initial_forbidden_face_count": initial_forbidden_face_count,
        "final_forbidden_face_count": final_forbidden_face_count,
        "corrected_core_vertex_count": len(core_corrections),
        "blended_vertex_count": len(halo_indices),
        "mean_correction_displacement": mean_disp,
        "max_correction_displacement": max_disp,
        "max_correction_vertex": max_disp_idx,
        "target_clearance": float(target_clearance),
        "blend_rings": int(blend_rings),
        "dry_run": not apply,
        "runtime_seconds": time.time() - t0,
    }

    if verbose:
        print_micro_clearance_report(result)
    return result


def print_micro_clearance_report(result: Dict[str, Any]) -> None:
    """Human-readable summary of a :func:`run_micro_clearance_push` result."""
    print("\n" + "=" * 60)
    print("MICRO CLEARANCE PUSH -- one-shot local outward correction")
    print("=" * 60)
    print("target_clearance={0}  blend_rings={1}".format(
        result.get("target_clearance"), result.get("blend_rings")))
    if result.get("dry_run"):
        print("(dry run -- scene NOT modified)")
    print("initial forbidden face count: {0}".format(result.get("initial_forbidden_face_count")))
    print("corrected core vertex count: {0}".format(result.get("corrected_core_vertex_count")))
    print("blended vertex count (halo only): {0}".format(result.get("blended_vertex_count")))
    print("correction displacement: mean={0:.5f}  max={1:.5f} (vertex {2})".format(
        result.get("mean_correction_displacement", 0.0),
        result.get("max_correction_displacement", 0.0),
        result.get("max_correction_vertex", -1)))
    print("final forbidden face count: {0}".format(result.get("final_forbidden_face_count")))
    print("runtime: {0:.2f}s".format(result.get("runtime_seconds", 0.0)))
    print("=" * 60)
