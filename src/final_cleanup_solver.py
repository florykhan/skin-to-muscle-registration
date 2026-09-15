"""
final_cleanup_solver.py
========================
Unified, automatic, iterative cleanup solver for post-registration skin.

This module is the FINAL correction stage the M1-M5 detection research and the
V2 anatomy-supported intersection repair were building toward. It is
orchestration + a small amount of new force/region logic; it does NOT
reimplement anything that already exists and is considered correct:

* detection            -- :func:`artifact_detection.detect_unified_artifacts` (M5,
  UNCHANGED). M3/M4 provenance (m3_only / m4_only / overlap) is read from the M5
  report and used to choose a correction force per vertex; detection thresholds
  are never touched here.
* anatomy-derived target -- :func:`artifact_detection.compute_sdf_target_positions`
  (the SAME SDF projection M4 already uses) supplies the "gentle attraction
  toward anatomy" signal for m4_only / overlap vertices. Computed ONCE per run
  (anatomy does not move), not re-solved every outer iteration.
* exact clearance       -- :func:`anatomy_constraint.enforce_anatomy_clearance` /
  :func:`anatomy_constraint.compute_clearance_floors` (baseline-aware; a vertex
  that already sat closer than the nominal floor is never forced outward).
* the ground-truth safety test -- :func:`anatomy_constraint.analyze_skin_anatomy_intersections`
  (real triangle/triangle skin-vs-anatomy intersection; UNCHANGED).
* repair                -- :func:`anatomy_constraint.resolve_skin_anatomy_intersections`
  (the broad, anatomy-supported, harmonic-displacement-field patch repair; V1
  per-vertex normal pushing is NOT used here). This function already partitions
  a repair request into topology-connected components internally and solves
  each with its own local step size, which is exactly the "no single global
  alpha" property this solver requires -- reused as-is, not rebuilt.
* mesh I/O / metrics    -- :mod:`mesh_utils`, :mod:`metrics_utils`.

What IS new here (the actual gap this module fills):

* a graded active region (artifact core -> fairing band -> transition band ->
  untouched exterior) instead of a single hard-edged vertex set;
* a per-vertex, provenance-weighted combination of THREE forces (local fairing,
  anatomy-target attraction, rest-shape restraint) instead of one indiscriminate
  Laplacian pass over the whole M5 region;
* a per-vertex trust-region step cap, so one tight vertex can never throttle the
  rest of the region the way a single shared line-search scalar did in
  :func:`smoothing_utils.constrained_smooth_mesh_region`;
* a "propose everywhere, repair only what actually broke, accept" outer loop
  instead of "reject the whole batch if anything is unsafe";
* multi-signal convergence with patience, so ``max_iterations`` is a safety cap,
  not the definition of "done" -- running longer than necessary should reach the
  same fixed point, not keep shrinking the face.

Historical milestones (M1-M5), the legacy V1/V2 repair modes, and the
strict / repair_after_batch / off smoothing safety modes in
:mod:`smoothing_utils` are all left exactly as they were, for reproducibility
and A/B comparison. This module adds a NEW, separate path; it changes nothing
in any other file.

Coordinate space is WORLD space throughout, consistent with the rest of the
project. This module imports cleanly outside Maya: all Maya access is isolated
in :mod:`mesh_utils` (read/write) and behind the ``anatomy_backend`` duck-typed
interface already defined by :class:`anatomy_constraint.MayaAnatomyBackend`, so
the pure region/force/convergence logic below can be (and is, see
``.cs_test/test_final_cleanup_solver.py``) unit-tested without a running Maya
session.
"""

from __future__ import print_function

import csv
import datetime
import json
import math
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import mesh_utils
import metrics_utils
import artifact_detection
import anatomy_constraint

try:
    import maya_io
except ImportError:  # pragma: no cover - maya_io present in this project
    maya_io = None


# =============================================================================
# DEFAULTS
# =============================================================================

DEFAULT_CLEANUP_GROWTH_RINGS = 6
DEFAULT_TRANSITION_RINGS = 4
DEFAULT_BOUNDARY_BUFFER_RINGS = 1

DEFAULT_W_FAIR = 0.5
DEFAULT_W_M4 = 0.25
DEFAULT_W_SHAPE = 0.15
DEFAULT_M4_ONLY_FAIR_SCALE = 0.3   # m4_only vertices still get a little fairing
DEFAULT_OVERLAP_M4_SCALE = 0.6     # overlap vertices get damped M4 pull (both forces already act)

DEFAULT_MAX_STEP_EDGE_RATIO = 0.5

DEFAULT_CLEARANCE_POLICY = "preserve_valid_baseline"
DEFAULT_CLEARANCE_TOLERANCE = 1e-4
DEFAULT_MAX_CONSTRAINT_ITERATIONS = 8
DEFAULT_INTERSECTION_TOLERANCE = 1e-6
DEFAULT_REPAIR_PROFILE = "broad_fair"

DEFAULT_DYNAMIC_GROWTH_ZSCORE = 2.0
DEFAULT_MAX_ACTIVE_GROWTH_FRACTION = 0.5

DEFAULT_MAX_ITERATIONS = 100
DEFAULT_CONVERGENCE_PATIENCE = 5
DEFAULT_MEAN_DISPLACEMENT_TOLERANCE = 1e-4
DEFAULT_MAX_DISPLACEMENT_TOLERANCE = 5e-4
DEFAULT_ROUGHNESS_REL_IMPROVEMENT_TOLERANCE = 0.01
DEFAULT_M4_REL_IMPROVEMENT_TOLERANCE = 0.01

STOP_MAX_ITERATIONS = "max_iterations"
STOP_CONVERGED = "convergence_tolerance"
STOP_NO_ARTIFACT_REGION = "no artifact region detected"
STOP_DRY_RUN = "dry run (no geometry modified)"
STOP_VALIDATION_FAILED = "post-hoc validation found residual intersections"

_BACKUP_SUFFIX = "_precleanupsolver"


# =============================================================================
# 1. PURE REGION / FORCE HELPERS
#    No Maya access. Operate on plain [[x, y, z], ...] lists and adjacency
#    lists, exactly like smoothing_utils.laplacian_smooth. Unit-tested in
#    .cs_test/test_final_cleanup_solver.py without a Maya session.
# =============================================================================

def _bfs_ring_distance(seeds: Sequence[int],
                       neighbors: List[List[int]],
                       blocked: Set[int],
                       max_rings: int,
                       ) -> Dict[int, int]:
    """Topological ring distance from ``seeds`` (distance 0), never crossing
    ``blocked`` vertices (protected boundaries). Stops at ``max_rings``."""
    dist: Dict[int, int] = {i: 0 for i in seeds if i not in blocked}
    frontier = set(dist.keys())
    for r in range(1, max(0, int(max_rings)) + 1):
        nxt: Set[int] = set()
        for u in frontier:
            if u < 0 or u >= len(neighbors):
                continue
            for w in neighbors[u]:
                if w in dist or w in blocked:
                    continue
                dist[w] = r
                nxt.add(w)
        if not nxt:
            break
        frontier = nxt
    return dist


def build_active_region(core: Sequence[int],
                        neighbors: List[List[int]],
                        boundary: Set[int],
                        cleanup_growth_rings: int,
                        transition_rings: int,
                        ) -> Dict[str, Any]:
    """Build the graded active region: fairing band -> transition band.

    ``core`` (the M5 / intersection seed set) is grown by ``cleanup_growth_rings``
    topological rings into the full-strength FAIRING region, then by a further
    ``transition_rings`` into a TRANSITION band whose ``anchor`` weight ramps
    smoothly from 0 (fairing edge) to 1 (outer edge, effectively frozen).
    Vertices in ``boundary`` (protected facial openings / true topological
    edges, already ring-buffered by the caller) are never entered.

    Returns
    -------
    dict
        ``{"fairing": [...], "transition": [...], "active": [...],
           "anchor": {i: float in [0, 1]}}``, all sorted / deterministic.
    """
    boundary = set(boundary or [])
    core = sorted(set(int(i) for i in core) - boundary)
    if not core:
        return {"fairing": [], "transition": [], "active": [], "anchor": {}}

    total_rings = max(0, int(cleanup_growth_rings)) + max(0, int(transition_rings))
    dist = _bfs_ring_distance(core, neighbors, boundary, total_rings)

    fairing = sorted(i for i, r in dist.items() if r <= cleanup_growth_rings)
    transition = sorted(i for i, r in dist.items() if cleanup_growth_rings < r)

    anchor: Dict[int, float] = {i: 0.0 for i in fairing}
    tr = max(1, int(transition_rings))
    for i in transition:
        t = (dist[i] - cleanup_growth_rings) / float(tr)
        t = min(1.0, max(0.0, t))
        anchor[i] = t * t * (3.0 - 2.0 * t)  # smoothstep, 0 -> 1

    active = sorted(set(fairing) | set(transition))
    return {"fairing": fairing, "transition": transition, "active": active, "anchor": anchor}


def classify_provenance(active: Sequence[int],
                        m3_only: Set[int],
                        m4_only: Set[int],
                        overlap: Set[int],
                        ) -> Dict[int, str]:
    """Tag every active vertex with WHY it is being corrected.

    ``"overlap"`` / ``"m4_only"`` / ``"m3_only"`` come straight from the M5
    report (see :func:`artifact_detection.detect_unified_artifacts`).
    ``"grown"`` is a vertex pulled in only by regional growth (fairing-band
    expansion, transition band, or dynamic active-set growth / intersection
    repair) with no M5 evidence of its own -- treated like m3_only (pure
    fairing, no anatomy-target pull), since it has no M4 target to attract to.
    """
    prov: Dict[int, str] = {}
    for i in active:
        if i in overlap:
            prov[i] = "overlap"
        elif i in m4_only:
            prov[i] = "m4_only"
        elif i in m3_only:
            prov[i] = "m3_only"
        else:
            prov[i] = "grown"
    return prov


def provenance_weights(active: Sequence[int],
                       provenance: Dict[int, str],
                       w_fair: float,
                       w_m4: float,
                       m4_only_fair_scale: float = DEFAULT_M4_ONLY_FAIR_SCALE,
                       overlap_m4_scale: float = DEFAULT_OVERLAP_M4_SCALE,
                       ) -> Dict[int, Tuple[float, float]]:
    """Return ``{i: (fair_weight, m4_weight)}`` from each vertex's provenance.

    m3_only / grown  -> fairing only (no M4 target pull; nothing to attract to
                        conceptually for m3_only, and "grown" halo vertices have
                        no M5 evidence at all).
    m4_only          -> mostly M4-target attraction, a SMALL fairing term so the
                        patch does not develop a crease at its own edge.
    overlap          -> both, with the M4 term damped so it does not simply
                        double the m4_only pull where geometry and anatomy
                        evidence already agree.
    """
    weights: Dict[int, Tuple[float, float]] = {}
    for i in active:
        p = provenance.get(i, "grown")
        if p == "m4_only":
            weights[i] = (w_fair * m4_only_fair_scale, w_m4)
        elif p == "overlap":
            weights[i] = (w_fair, w_m4 * overlap_m4_scale)
        else:  # "m3_only" or "grown"
            weights[i] = (w_fair, 0.0)
    return weights


def local_edge_lengths(positions: List[List[float]],
                       neighbors: List[List[int]],
                       indices: Sequence[int],
                       ) -> Dict[int, float]:
    """Mean 1-ring edge length per vertex (trust-region scale reference)."""
    out: Dict[int, float] = {}
    for i in indices:
        nbrs = neighbors[i] if 0 <= i < len(neighbors) else []
        if not nbrs:
            out[i] = 0.0
            continue
        out[i] = sum(mesh_utils.vec_length(mesh_utils.vec_sub(positions[i], positions[j]))
                     for j in nbrs) / len(nbrs)
    return out


def fairing_force(positions: List[List[float]],
                  neighbors: List[List[int]],
                  indices: Sequence[int],
                  fair_weight: Dict[int, float],
                  ) -> Dict[int, List[float]]:
    """Umbrella-Laplacian pull toward the local neighbour average.

    ``w * (mean_{j in N(i)} x_j - x_i)`` -- the same local-fairness force every
    other smoother in this project uses (see :func:`smoothing_utils.laplacian_smooth`),
    just computed as an explicit, weighted, per-vertex FORCE here instead of an
    unconditional replace-with-average step, so it can be combined with the
    other two forces before any trust-region clamp or anatomy projection.
    """
    out: Dict[int, List[float]] = {}
    for i in indices:
        w = fair_weight.get(i, 0.0)
        nbrs = neighbors[i] if 0 <= i < len(neighbors) else []
        if not nbrs or w == 0.0:
            out[i] = [0.0, 0.0, 0.0]
            continue
        avg = mesh_utils.vec_mean([positions[j] for j in nbrs])
        out[i] = mesh_utils.vec_scale(mesh_utils.vec_sub(avg, positions[i]), w)
    return out


def shape_force(positions: List[List[float]],
                rest_positions: List[List[float]],
                indices: Sequence[int],
                w_shape: float,
                ) -> Dict[int, List[float]]:
    """Restoring pull toward the registered ("rest") position.

    ``w_shape * (x_rest_i - x_i)`` -- this is what keeps repeated fairing from
    converging toward a fully flattened harmonic surface: the operator's fixed
    point is now a compromise between "locally smooth" and "close to the
    original registered shape", not smoothness alone.
    """
    out: Dict[int, List[float]] = {}
    for i in indices:
        if w_shape == 0.0:
            out[i] = [0.0, 0.0, 0.0]
            continue
        out[i] = mesh_utils.vec_scale(
            mesh_utils.vec_sub(rest_positions[i], positions[i]), w_shape)
    return out


def m4_force(positions: List[List[float]],
            targets: Dict[int, List[float]],
            indices: Sequence[int],
            m4_weight: Dict[int, float],
            ) -> Dict[int, List[float]]:
    """Damped attraction toward the anatomy-derived M4 target (where available).

    ``w * (t_i^M4 - x_i)``. Only applied where ``targets`` has a converged M4
    projection for ``i`` (see :func:`artifact_detection.compute_sdf_target_positions`);
    a directional/reference signal, never a hard snap -- ``m4_weight`` is a
    fraction well under 1 so this does not pull the skin onto the anatomy
    surface itself (see the module docstring / DEFAULT_W_M4).
    """
    out: Dict[int, List[float]] = {}
    for i in indices:
        w = m4_weight.get(i, 0.0)
        t = targets.get(i)
        if w == 0.0 or t is None:
            out[i] = [0.0, 0.0, 0.0]
            continue
        out[i] = mesh_utils.vec_scale(mesh_utils.vec_sub(t, positions[i]), w)
    return out


def combine_step(fair_f: Dict[int, List[float]],
                 shape_f: Dict[int, List[float]],
                 m4_f: Dict[int, List[float]],
                 indices: Sequence[int],
                 anchor: Dict[int, float],
                 max_step_edge_ratio: Optional[float],
                 local_edge: Dict[int, float],
                 ) -> Dict[int, List[float]]:
    """Sum the three forces, damp by ``(1 - anchor)``, clamp to a PER-VERTEX
    trust region.

    This is the direct replacement for the single shared line-search alpha in
    :func:`smoothing_utils.constrained_smooth_mesh_region`: every vertex here
    gets its OWN cap (``max_step_edge_ratio`` times ITS OWN local edge length,
    further damped by ITS OWN transition-band anchor weight), so one tight
    vertex can only ever limit itself, never the rest of the active region.
    """
    step: Dict[int, List[float]] = {}
    for i in indices:
        fx = fair_f.get(i, [0.0, 0.0, 0.0])
        sx = shape_f.get(i, [0.0, 0.0, 0.0])
        mx = m4_f.get(i, [0.0, 0.0, 0.0])
        raw = [fx[k] + sx[k] + mx[k] for k in range(3)]
        damp = 1.0 - float(anchor.get(i, 0.0))
        if damp <= 0.0:
            step[i] = [0.0, 0.0, 0.0]
            continue
        raw = [c * damp for c in raw]
        if max_step_edge_ratio is not None:
            cap = float(max_step_edge_ratio) * local_edge.get(i, 0.0)
            if cap > 0.0:
                mag = mesh_utils.vec_length(raw)
                if mag > cap and mag > 1e-12:
                    raw = mesh_utils.vec_scale(raw, cap / mag)
        step[i] = raw
    return step


def apply_step(positions: List[List[float]],
              step: Dict[int, List[float]],
              ) -> List[List[float]]:
    """Return a NEW position list with ``step`` added at its keys only."""
    out = [list(p) for p in positions]
    for i, d in step.items():
        out[i] = mesh_utils.vec_add(out[i], d)
    return out


def mean_laplacian_magnitude(positions: List[List[float]],
                             neighbors: List[List[int]],
                             indices: Sequence[int],
                             ) -> float:
    """Mean umbrella-Laplacian magnitude over ``indices`` -- the SAME formula
    as :func:`artifact_detection.compute_laplacian_scores` (M1), evaluated on
    an in-memory position array instead of a live mesh.

    This is a deliberate, minimal (4-line) re-expression, not a duplicated
    detector: the solver holds state in-memory across many iterations and must
    not round-trip every one of them through Maya just to measure roughness,
    so it cannot call the Maya-backed M1 function directly here.
    """
    if not indices:
        return 0.0
    vals = []
    for i in indices:
        nbrs = neighbors[i] if 0 <= i < len(neighbors) else []
        if not nbrs:
            continue
        avg = mesh_utils.vec_mean([positions[j] for j in nbrs])
        vals.append(mesh_utils.vec_length(mesh_utils.vec_sub(positions[i], avg)))
    return (sum(vals) / len(vals)) if vals else 0.0


def mean_target_disagreement(positions: List[List[float]],
                             targets: Dict[int, List[float]],
                             indices: Sequence[int],
                             ) -> float:
    """Mean distance from current position to the (static) M4 target, over
    whichever of ``indices`` have a converged target."""
    if not indices or not targets:
        return 0.0
    vals = [mesh_utils.vec_length(mesh_utils.vec_sub(positions[i], targets[i]))
            for i in indices if i in targets]
    return (sum(vals) / len(vals)) if vals else 0.0


def _stats(values: Sequence[float]) -> Dict[str, float]:
    vals = list(values)
    if not vals:
        return {"count": 0, "mean": 0.0, "max": 0.0}
    return {"count": len(vals), "mean": sum(vals) / len(vals), "max": max(vals)}


# =============================================================================
# 2. MAYA-TOUCHING ORCHESTRATION HELPERS
#    (thin; each delegates the actual work to artifact_detection /
#    anatomy_constraint / mesh_utils)
# =============================================================================

def _boundary_set(skin_mesh: str, neighbors: List[List[int]], rings: int) -> Set[int]:
    """Protected facial openings / true mesh edges, ring-buffered once."""
    boundary = mesh_utils.get_boundary_vertices(skin_mesh)
    if rings and rings > 0 and boundary:
        boundary = set(mesh_utils.grow_indices(neighbors, sorted(boundary), rings=int(rings)))
    return set(boundary or [])


def _m5_provenance_sets(report: Dict[str, Any]) -> Tuple[Set[int], Set[int], Set[int]]:
    return (set(report.get("m3_only_indices") or []),
            set(report.get("m4_only_indices") or []),
            set(report.get("overlap_indices") or []))


def _build_anatomy_backend(anatomical_meshes: Sequence[str]) -> Tuple[Any, List[str]]:
    """Build a :class:`anatomy_constraint.MayaAnatomyBackend` over the given
    mesh names, reusing :func:`mesh_utils.get_mesh_fn` for each (the SAME
    Maya API 2.0 accessor everything else in this project uses)."""
    mesh_fns = {}
    for name in anatomical_meshes:
        fn = mesh_utils.get_mesh_fn(name)
        if fn is not None:
            mesh_fns[name] = fn
    if not mesh_fns:
        raise ValueError(
            "final_cleanup_solver: could not load any anatomical mesh from "
            "anatomical_meshes ({0} requested). Is Maya running and are the "
            "meshes present?".format(len(list(anatomical_meshes))))
    backend = anatomy_constraint.MayaAnatomyBackend(mesh_fns)
    return backend, list(mesh_fns.keys())


def _scoped_intersection_core(skin_mesh: str,
                              positions: List[List[float]],
                              scope: Sequence[int],
                              backend: Any,
                              neighbors: List[List[int]],
                              skin_topology: Dict[str, Any],
                              boundary_buffer_rings: int,
                              intersection_tolerance: float,
                              ) -> Tuple[List[int], Dict[str, Any]]:
    """Real triangle/triangle intersection scan restricted to ``scope`` (plus
    its own incident faces), and the intersecting-face vertex core clipped
    back to ``scope``. Scoping to the active region is CORRECT and COMPLETE
    here: only active vertices ever move between calls, so any new
    intersecting face must be incident to one of them.
    """
    report = anatomy_constraint.analyze_skin_anatomy_intersections(
        skin_mesh, skin_indices=scope, backend=backend, positions=positions,
        neighbors=neighbors, skin_topology=skin_topology,
        boundary_buffer_rings=boundary_buffer_rings,
        intersection_tolerance=intersection_tolerance,
        detailed=False, verbose=False)
    core = anatomy_constraint.intersection_vertices_from_report(
        report, skin_topology, allowed=set(scope))
    return core, report


def _anatomy_project_clearance(positions: List[List[float]],
                               indices: Sequence[int],
                               backend: Any,
                               floors: Dict[int, float],
                               clearance_tolerance: float,
                               max_constraint_iterations: int,
                               normals: Optional[List[List[float]]],
                               ) -> Tuple[List[List[float]], Dict[int, float]]:
    """Per-vertex exact-clearance projection (cheap, local; NOT the triangle
    test). Reuses :func:`anatomy_constraint.enforce_anatomy_clearance` exactly
    as the existing constrained smoother does."""
    out = [list(p) for p in positions]
    moved: Dict[int, float] = {}
    for i in indices:
        floor_i = floors.get(i, 0.0)
        if floor_i <= 0.0:
            continue
        q = backend.exact_closest(out[i])
        if anatomy_constraint.is_clearance_satisfied(q["distance"], floor_i, clearance_tolerance):
            continue
        nrm = normals[i] if (normals and 0 <= i < len(normals)) else None
        sol = anatomy_constraint.enforce_anatomy_clearance(
            out[i], backend, floor_i,
            max_constraint_iterations=max_constraint_iterations,
            clearance_tolerance=clearance_tolerance, skin_normal=nrm)
        moved[i] = mesh_utils.vec_length(mesh_utils.vec_sub(sol["position"], out[i]))
        out[i] = sol["position"]
    return out, moved


def _repair_local(positions: List[List[float]],
                  core_now: Sequence[int],
                  backend: Any,
                  skin_mesh: str,
                  neighbors: List[List[int]],
                  normals: Optional[List[List[float]]],
                  skin_topology: Dict[str, Any],
                  min_clearance: float,
                  clearance_policy: str,
                  boundary_buffer_rings: int,
                  intersection_tolerance: float,
                  repair_kwargs: Dict[str, Any],
                  ) -> Tuple[List[List[float]], Optional[Dict[str, Any]]]:
    """Repair ONLY the vertices in ``core_now`` (the actually-offending
    component(s)), via the existing broad anatomy-supported patch repair.
    ``resolve_skin_anatomy_intersections`` partitions ``core_now`` into
    topology-connected components internally and solves each with its own
    local step -- exactly the "no single global alpha" property this solver
    needs, reused rather than rebuilt.
    """
    if not core_now:
        return positions, None
    result = anatomy_constraint.resolve_skin_anatomy_intersections(
        positions, list(core_now), backend, skin_mesh=skin_mesh,
        neighbors=neighbors, normals=normals, skin_topology=skin_topology,
        min_clearance=min_clearance, clearance_policy=clearance_policy,
        boundary_buffer_rings=boundary_buffer_rings,
        intersection_tolerance=intersection_tolerance, verbose=False,
        **repair_kwargs)
    return result["positions"], result


def _compute_m4_targets(skin_mesh: str,
                        anatomical_meshes: Sequence[str],
                        target_offset: float,
                        indices: Sequence[int],
                        sdf_query: Optional[Any] = None,
                        ) -> Dict[int, List[float]]:
    """Static anatomy-derived correction targets for ``indices`` (computed
    ONCE; anatomy does not move during cleanup). Reuses M4's own projector."""
    if not indices:
        return {}
    data = artifact_detection.compute_sdf_target_positions(
        skin_mesh, anatomical_meshes, target_offset, indices=sorted(indices),
        sdf_query=sdf_query)
    positions = data.get("target_positions") or {}
    converged = set(data.get("converged_indices") or [])
    return {i: positions[i] for i in converged if i in positions}


# =============================================================================
# 3. LOGGING (same conventions as cleanup_pipeline.py)
# =============================================================================

def _unique_path(path: str) -> str:
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
        print("[final_cleanup] wrote JSON log -> {0}".format(path))
        return path
    except (OSError, TypeError) as exc:
        print("[final_cleanup] WARNING: could not write JSON log ({0})".format(exc))
        return None


_CSV_COLUMNS = [
    "iteration", "active_count",
    "fair_disp_mean", "fair_disp_max",
    "m4_disp_mean", "m4_disp_max",
    "shape_disp_mean", "shape_disp_max",
    "clearance_disp_mean", "clearance_disp_max",
    "repair_disp_mean", "repair_disp_max",
    "net_disp_mean", "net_disp_max",
    "roughness_before", "roughness_after",
    "m4_disagreement_before", "m4_disagreement_after",
    "intersecting_face_count", "intersection_pair_count", "repaired_component_count",
    "min_exact_distance", "no_forbidden_intersections",
    "patience_motion", "patience_roughness", "patience_m4",
]


def _write_csv(rows: List[Dict[str, Any]], path: str) -> Optional[str]:
    path = _unique_path(path)
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k, "") for k in _CSV_COLUMNS})
        print("[final_cleanup] wrote CSV log  -> {0}".format(path))
        return path
    except OSError as exc:
        print("[final_cleanup] WARNING: could not write CSV log ({0})".format(exc))
        return None


def _iteration_csv_row(rec: Dict[str, Any]) -> Dict[str, Any]:
    row = {"iteration": rec["iteration"], "active_count": rec["active_count"]}
    for prefix, key in (("fair", "fairing_displacement"), ("m4", "m4_displacement"),
                        ("shape", "shape_displacement"), ("clearance", "clearance_displacement"),
                        ("repair", "repair_displacement"), ("net", "net_displacement")):
        row[prefix + "_disp_mean"] = round(rec[key]["mean"], 8)
        row[prefix + "_disp_max"] = round(rec[key]["max"], 8)
    row["roughness_before"] = round(rec["roughness"]["before"], 8)
    row["roughness_after"] = round(rec["roughness"]["after"], 8)
    row["m4_disagreement_before"] = round(rec["m4_disagreement"]["before"], 8)
    row["m4_disagreement_after"] = round(rec["m4_disagreement"]["after"], 8)
    row["intersecting_face_count"] = rec["intersections"]["face_count"]
    row["intersection_pair_count"] = rec["intersections"]["pair_count"]
    row["repaired_component_count"] = rec["intersections"]["repaired_component_count"]
    row["min_exact_distance"] = round(rec["min_exact_distance"], 6)
    row["no_forbidden_intersections"] = rec["no_forbidden_intersections"]
    row["patience_motion"] = rec["patience"]["motion"]
    row["patience_roughness"] = rec["patience"]["roughness"]
    row["patience_m4"] = rec["patience"]["m4"]
    return row


# =============================================================================
# 4. MAIN ENTRY POINT
# =============================================================================

def run_cleanup_solver(skin_mesh: str,
                       anatomical_meshes: Sequence[str],
                       target_offset: Optional[float] = None,
                       min_clearance: Optional[float] = None,
                       indices: Optional[Sequence[int]] = None,
                       m5_report: Optional[Dict[str, Any]] = None,
                       anatomy_backend: Optional[Any] = None,
                       sdf_query: Optional[Any] = None,
                       m5_detector_kwargs: Optional[Dict[str, Any]] = None,
                       # --- active region -------------------------------------
                       cleanup_growth_rings: int = DEFAULT_CLEANUP_GROWTH_RINGS,
                       transition_rings: int = DEFAULT_TRANSITION_RINGS,
                       boundary_buffer_rings: int = DEFAULT_BOUNDARY_BUFFER_RINGS,
                       # --- forces ----------------------------------------------
                       w_fair: float = DEFAULT_W_FAIR,
                       w_m4: float = DEFAULT_W_M4,
                       w_shape: float = DEFAULT_W_SHAPE,
                       m4_only_fair_scale: float = DEFAULT_M4_ONLY_FAIR_SCALE,
                       overlap_m4_scale: float = DEFAULT_OVERLAP_M4_SCALE,
                       max_step_edge_ratio: float = DEFAULT_MAX_STEP_EDGE_RATIO,
                       # --- anatomy feasibility -----------------------------
                       clearance_policy: str = DEFAULT_CLEARANCE_POLICY,
                       clearance_tolerance: float = DEFAULT_CLEARANCE_TOLERANCE,
                       max_constraint_iterations: int = DEFAULT_MAX_CONSTRAINT_ITERATIONS,
                       intersection_tolerance: float = DEFAULT_INTERSECTION_TOLERANCE,
                       repair_profile: str = DEFAULT_REPAIR_PROFILE,
                       repair_kwargs: Optional[Dict[str, Any]] = None,
                       # --- dynamic active set --------------------------------
                       dynamic_active_set: bool = True,
                       dynamic_growth_zscore: float = DEFAULT_DYNAMIC_GROWTH_ZSCORE,
                       max_active_growth_fraction: float = DEFAULT_MAX_ACTIVE_GROWTH_FRACTION,
                       redetect_every: Optional[int] = None,
                       # --- iteration / convergence --------------------------
                       max_iterations: int = DEFAULT_MAX_ITERATIONS,
                       convergence_patience: int = DEFAULT_CONVERGENCE_PATIENCE,
                       mean_displacement_tolerance: float = DEFAULT_MEAN_DISPLACEMENT_TOLERANCE,
                       max_displacement_tolerance: float = DEFAULT_MAX_DISPLACEMENT_TOLERANCE,
                       roughness_rel_improvement_tolerance: float = DEFAULT_ROUGHNESS_REL_IMPROVEMENT_TOLERANCE,
                       m4_rel_improvement_tolerance: float = DEFAULT_M4_REL_IMPROVEMENT_TOLERANCE,
                       # --- output / safety -----------------------------------
                       apply: bool = True,
                       verbose: bool = True,
                       report_interval: int = 5,
                       select_final: bool = True,
                       select_on_failure: bool = True,
                       create_backup: bool = True,
                       backup_suffix: str = _BACKUP_SUFFIX,
                       log_path: Optional[str] = None,
                       save_json: bool = True,
                       save_csv: bool = True,
                       ) -> Dict[str, Any]:
    """Run the unified fairing / anatomy-projection / repair solver to
    convergence (see the module docstring for the full design).

    Self-contained: needs only ``skin_mesh`` + ``anatomical_meshes`` (a fresh
    Maya session has no pre-existing Python state to rely on). ``target_offset``
    (M4's D0) and ``min_clearance`` (the hard anatomy floor) are both computed
    automatically from the scene's own distance distribution when omitted --
    see :func:`artifact_detection.summarize_skin_sdf_values`. Pass ``apply=False``
    for a dry run: every step still runs, in memory, but the scene is never
    written and no backup is made.

    Parameters
    ----------
    skin_mesh, anatomical_meshes:
        The registered skin and the internal anatomy mesh names.
    target_offset:
        M4's anatomy-offset iso-value ``d0``. Auto-picked as the scene's median
        skin->anatomy distance when ``None``.
    min_clearance:
        Hard anatomy floor (same role as the registration's
        ``collision_min_distance``). Defaults to ``target_offset`` when ``None``
        -- pass the project's actual collision floor explicitly for a faithful
        run (the d98 wrapper does this).
    indices, m5_report:
        Optional pre-computed M5 region / report, to avoid re-running M5 (e.g.
        for an A/B comparison against an identical baseline detection). If both
        are ``None``, M5 is run once here.
    anatomy_backend, sdf_query:
        Optional pre-built :class:`anatomy_constraint.MayaAnatomyBackend` /
        :class:`artifact_detection.AnatomySDFQuery` to reuse (e.g. the exact
        smooth-min field the registration itself uses). Built fresh over
        ``anatomical_meshes`` via :func:`mesh_utils.get_mesh_fn` when omitted,
        so this function works standalone.
    cleanup_growth_rings, transition_rings:
        Size of the full-strength fairing band and the damped transition band
        grown around the M5/intersection core (see :func:`build_active_region`).
    w_fair, w_m4, w_shape:
        Base weights for the three correction forces (see :func:`combine_step`
        and the module docstring's provenance rules).
    max_step_edge_ratio:
        Per-vertex trust-region cap, as a fraction of THAT vertex's own mean
        edge length. There is no shared/global step scalar anywhere in this
        solver.
    dynamic_active_set:
        If True, a vertex just outside the current fairing region is absorbed
        into it once its local roughness exceeds a scene-wide threshold
        (bounded by ``max_active_growth_fraction`` of the initial core size).
    redetect_every:
        If set (and ``apply=True``), re-run M5 every N iterations and union any
        newly detected vertices into the active region. Ignored when
        ``apply=False`` (M5 needs to read the live mesh).
    max_iterations, convergence_patience, *_tolerance:
        Hard cap and the multi-signal convergence gate (see the module
        docstring). ``max_iterations`` is a safety cap, not the target.

    Returns
    -------
    dict
        Full run report: ``converged``, ``iterations``, ``stop_reason``,
        before/after intersection / roughness / M4-disagreement / displacement
        metrics, the per-iteration log, and (on failure) the unresolved faces /
        anatomy meshes / components for manual inspection.
    """
    t_run0 = time.time()

    if not mesh_utils.mesh_exists(skin_mesh):
        raise ValueError("final_cleanup_solver: skin_mesh '{0}' does not exist".format(skin_mesh))
    if not anatomical_meshes:
        raise ValueError("final_cleanup_solver: anatomical_meshes is required "
                         "(list of internal mesh names)")
    if max_iterations < 1:
        raise ValueError("max_iterations must be >= 1, got {0}".format(max_iterations))
    if convergence_patience < 1:
        raise ValueError("convergence_patience must be >= 1, got {0}".format(convergence_patience))

    positions0 = mesh_utils.get_mesh_vertices(skin_mesh)
    if not positions0:
        raise ValueError("final_cleanup_solver: skin_mesh '{0}' has no readable "
                         "vertices".format(skin_mesh))
    neighbors = mesh_utils.get_vertex_neighbors(skin_mesh)
    normals = mesh_utils.get_vertex_normals(skin_mesh)
    mesh_fn = mesh_utils.get_mesh_fn(skin_mesh)
    skin_topology = mesh_utils.get_triangle_topology(mesh_fn) if mesh_fn is not None else None
    if not skin_topology or not skin_topology.get("triangles"):
        raise ValueError("final_cleanup_solver: could not read triangle topology "
                         "for '{0}'".format(skin_mesh))

    boundary = _boundary_set(skin_mesh, neighbors, boundary_buffer_rings)

    if anatomy_backend is None:
        anatomy_backend, valid_anatomical_meshes = _build_anatomy_backend(anatomical_meshes)
    else:
        valid_anatomical_meshes = list(anatomy_backend.mesh_names())

    if target_offset is None:
        summary = artifact_detection.summarize_skin_sdf_values(skin_mesh, valid_anatomical_meshes)
        if not summary.get("count"):
            raise ValueError("final_cleanup_solver: could not auto-compute "
                             "target_offset (no finite SDF samples); pass it explicitly")
        target_offset = summary["median"]
        if verbose:
            print("[final_cleanup] auto target_offset (D0) = {0:.4f} "
                  "(scene median skin->anatomy distance)".format(target_offset))

    if min_clearance is None:
        min_clearance = target_offset
        if verbose:
            print("[final_cleanup] min_clearance not given; using target_offset "
                  "({0:.4f}) as the anatomy floor".format(min_clearance))

    # --- initial M5 detection (unless the caller supplied one) ---------------
    if m5_report is None and indices is None:
        t0 = time.time()
        m5_indices, m5_report = artifact_detection.detect_unified_artifacts(
            skin_mesh, valid_anatomical_meshes, target_offset, select=False,
            **dict(m5_detector_kwargs or {}))
        m5_seconds = time.time() - t0
        if verbose:
            print("[final_cleanup] M5 baseline: {0} vertices ({1:.2f}s)".format(
                len(m5_indices), m5_seconds))
    elif m5_report is not None:
        m5_indices = sorted(set(m5_report.get("final_indices") or []))
    else:
        m5_indices = sorted(set(int(i) for i in indices))
        m5_report = {"final_indices": m5_indices, "m3_indices": [], "m4_indices": [],
                     "overlap_indices": [], "m3_only_indices": list(m5_indices),
                     "m4_only_indices": []}

    m3_only0, m4_only0, overlap0 = _m5_provenance_sets(m5_report)

    if create_backup and apply and m5_indices and maya_io is not None:
        backup_name = skin_mesh + backup_suffix
        if not mesh_utils.mesh_exists(backup_name):
            maya_io.duplicate_mesh(skin_mesh, suffix=backup_suffix)
        elif verbose:
            print("[final_cleanup] backup '{0}' already exists; keeping it".format(backup_name))

    # --- one whole-mesh intersection scan seeds the core with any PRE-EXISTING
    #     surface intersections M5's score-based detectors would not flag ------
    isect0_core, isect0_report = _scoped_intersection_core(
        skin_mesh, positions0, list(range(len(positions0))), anatomy_backend,
        neighbors, skin_topology, boundary_buffer_rings, intersection_tolerance)

    core0 = sorted((set(m5_indices) | set(isect0_core)) - boundary)
    if not core0:
        if verbose:
            print("[final_cleanup] no artifact / intersection region detected; nothing to do")
        return {
            "skin_mesh": skin_mesh, "converged": True, "iterations": 0,
            "stop_reason": STOP_NO_ARTIFACT_REGION, "dry_run": not apply,
            "target_offset": target_offset, "min_clearance": min_clearance,
            "m5_report": m5_report, "runtime_seconds": time.time() - t_run0,
        }

    region = build_active_region(core0, neighbors, boundary,
                                 cleanup_growth_rings, transition_rings)
    fairing_region: Set[int] = set(region["fairing"])
    active: Set[int] = set(region["active"])
    anchor: Dict[int, float] = region["anchor"]
    provenance = classify_provenance(sorted(active), m3_only0, m4_only0, overlap0)
    weights = provenance_weights(sorted(active), provenance, w_fair, w_m4,
                                 m4_only_fair_scale, overlap_m4_scale)

    # --- static M4 targets (anatomy does not move; computed once) ------------
    if sdf_query is None:
        resolved = artifact_detection.resolve_anatomical_meshes(valid_anatomical_meshes)
        sdf_query = (artifact_detection.build_anatomy_sdf_query(resolved["valid"])
                    if resolved["valid"] else None)
    m4_needed = sorted(i for i in active if weights[i][1] > 0.0)
    m4_targets: Dict[int, List[float]] = (
        _compute_m4_targets(skin_mesh, valid_anatomical_meshes, target_offset,
                           m4_needed, sdf_query=sdf_query) if m4_needed else {})

    # --- baseline-aware clearance floors --------------------------------------
    floors, _orig_dist, policy_used = anatomy_constraint.compute_clearance_floors(
        positions0, sorted(active), anatomy_backend, min_clearance,
        clearance_policy=clearance_policy, clearance_tolerance=clearance_tolerance)

    # --- dynamic-growth roughness threshold (one-time, whole-mesh) -----------
    roughness_threshold = float("inf")
    if dynamic_active_set:
        whole_scores = artifact_detection.compute_laplacian_scores(skin_mesh)
        whole_stats = artifact_detection.summarize_scores(whole_scores)
        roughness_threshold = whole_stats["mean"] + dynamic_growth_zscore * whole_stats["std"]

    x_rest = [list(p) for p in positions0]     # registered-shape reference (fixed for the run)
    x = [list(p) for p in positions0]          # current accepted state (in-memory working copy)

    roughness0 = mean_laplacian_magnitude(x, neighbors, sorted(fairing_region))
    m4_disagreement0 = mean_target_disagreement(x, m4_targets, sorted(m4_targets.keys()))
    min_dist0 = min((anatomy_backend.exact_closest(x[i])["distance"] for i in sorted(active)),
                    default=float("inf"))

    repair_kw = dict(repair_kwargs or {})
    repair_kw.setdefault("repair_profile", repair_profile)

    iterations_log: List[Dict[str, Any]] = []
    csv_rows: List[Dict[str, Any]] = []
    patience = {"motion": 0, "roughness": 0, "m4": 0}
    prev_roughness = roughness0
    prev_m4_disagreement = m4_disagreement0
    stop_reason: Optional[str] = None
    converged = False
    total_active_growth = 0
    max_growth_budget = int(math.ceil(max_active_growth_fraction * len(core0)))
    it = 0

    if verbose:
        print("[final_cleanup] core={0} fairing={1} transition={2} active={3} "
              "(provenance: m3_only={4} m4_only={5} overlap={6} grown={7})".format(
                  len(core0), len(fairing_region), len(active) - len(fairing_region),
                  len(active),
                  sum(1 for p in provenance.values() if p == "m3_only"),
                  sum(1 for p in provenance.values() if p == "m4_only"),
                  sum(1 for p in provenance.values() if p == "overlap"),
                  sum(1 for p in provenance.values() if p == "grown")))

    for it in range(1, int(max_iterations) + 1):
        x_before = [list(p) for p in x]
        active_list = sorted(active)

        local_edge = local_edge_lengths(x, neighbors, active_list)
        fair_w = {i: weights[i][0] for i in active_list}
        m4_w = {i: weights[i][1] for i in active_list}

        fair_f = fairing_force(x, neighbors, active_list, fair_w)
        shape_f = shape_force(x, x_rest, active_list, w_shape)
        m4_f = m4_force(x, m4_targets, active_list, m4_w)

        step = combine_step(fair_f, shape_f, m4_f, active_list, anchor,
                            max_step_edge_ratio, local_edge)
        x_proposed = apply_step(x, step)  # RAW proposal; may temporarily violate anatomy

        x_projected, clearance_moved = _anatomy_project_clearance(
            x_proposed, active_list, anatomy_backend, floors, clearance_tolerance,
            max_constraint_iterations, normals)

        core_now, isect_report = _scoped_intersection_core(
            skin_mesh, x_projected, active_list, anatomy_backend, neighbors,
            skin_topology, boundary_buffer_rings, intersection_tolerance)

        repair_result = None
        repair_disp: List[float] = []
        if core_now:
            x_accepted, repair_result = _repair_local(
                x_projected, core_now, anatomy_backend, skin_mesh, neighbors,
                normals, skin_topology, min_clearance, clearance_policy,
                boundary_buffer_rings, intersection_tolerance, repair_kw)
            if repair_result:
                patch = repair_result.get("patch_vertices") or []
                repair_disp = [mesh_utils.vec_length(mesh_utils.vec_sub(x_accepted[i], x_projected[i]))
                              for i in patch]
        else:
            x_accepted = x_projected

        # Final local verification: the ACCEPTED state must be intersection-free
        # within scope. If repair could not fully clear it (rare -- repair has
        # its own bounded pass count), roll back ONLY the still-offending
        # vertices, one component at a time -- never the whole iteration.
        residual_core, residual_report = _scoped_intersection_core(
            skin_mesh, x_accepted, active_list, anatomy_backend, neighbors,
            skin_topology, boundary_buffer_rings, intersection_tolerance)
        if residual_core:
            for i in residual_core:
                x_accepted[i] = list(x_before[i])
            residual_core, residual_report = _scoped_intersection_core(
                skin_mesh, x_accepted, active_list, anatomy_backend, neighbors,
                skin_topology, boundary_buffer_rings, intersection_tolerance)

        net_disp = [mesh_utils.vec_length(mesh_utils.vec_sub(x_accepted[i], x_before[i]))
                   for i in active_list]
        x = x_accepted

        roughness_after = mean_laplacian_magnitude(x, neighbors, sorted(fairing_region))
        m4_after = mean_target_disagreement(x, m4_targets, sorted(m4_targets.keys()))
        min_dist_now = min((anatomy_backend.exact_closest(x[i])["distance"] for i in active_list),
                           default=float("inf"))

        mean_net = sum(net_disp) / len(net_disp) if net_disp else 0.0
        max_net = max(net_disp) if net_disp else 0.0
        roughness_rel = abs(prev_roughness - roughness_after) / (prev_roughness + 1e-9)
        m4_rel = abs(prev_m4_disagreement - m4_after) / (prev_m4_disagreement + 1e-9)
        no_forbidden = not bool(residual_core)

        tiny_motion = (mean_net < mean_displacement_tolerance
                      and max_net < max_displacement_tolerance)
        patience["motion"] = patience["motion"] + 1 if tiny_motion else 0
        patience["roughness"] = (patience["roughness"] + 1
                                 if roughness_rel < roughness_rel_improvement_tolerance else 0)
        patience["m4"] = (patience["m4"] + 1
                          if (not m4_targets or m4_rel < m4_rel_improvement_tolerance) else 0)

        record = {
            "iteration": it,
            "active_count": len(active_list),
            "fairing_displacement": _stats([mesh_utils.vec_length(v) for v in fair_f.values()]),
            "m4_displacement": _stats([mesh_utils.vec_length(v) for v in m4_f.values()]),
            "shape_displacement": _stats([mesh_utils.vec_length(v) for v in shape_f.values()]),
            "clearance_displacement": _stats(list(clearance_moved.values())),
            "repair_displacement": _stats(repair_disp),
            "net_displacement": {"mean": mean_net, "max": max_net},
            "roughness": {"before": prev_roughness, "after": roughness_after},
            "m4_disagreement": {"before": prev_m4_disagreement, "after": m4_after},
            "intersections": {
                "face_count": isect_report.get("intersecting_skin_face_count", 0),
                "pair_count": isect_report.get("intersection_pair_count", 0),
                "repaired_component_count": (repair_result or {}).get("repair_component_count", 0),
            },
            "min_exact_distance": min_dist_now,
            "no_forbidden_intersections": no_forbidden,
            "patience": dict(patience),
        }
        iterations_log.append(record)
        csv_rows.append(_iteration_csv_row(record))

        if verbose and (it == 1 or it % max(1, report_interval) == 0):
            print("  [iter {0:4d}] active={1} net disp mean/max={2:.5f}/{3:.5f} "
                  "roughness {4:.5f}->{5:.5f} m4 {6:.5f}->{7:.5f} faces={8} "
                  "min_dist={9:.4f} patience(m/r/4)={10}/{11}/{12}".format(
                      it, len(active_list), mean_net, max_net, prev_roughness,
                      roughness_after, prev_m4_disagreement, m4_after,
                      record["intersections"]["face_count"], min_dist_now,
                      patience["motion"], patience["roughness"], patience["m4"]))

        prev_roughness = roughness_after
        prev_m4_disagreement = m4_after

        if (no_forbidden and patience["motion"] >= convergence_patience
                and patience["roughness"] >= convergence_patience
                and patience["m4"] >= convergence_patience):
            converged = True
            stop_reason = STOP_CONVERGED
            break

        # --- dynamic active-set growth (bounded, roughness-triggered) --------
        if dynamic_active_set and total_active_growth < max_growth_budget:
            frontier = sorted((set(mesh_utils.grow_indices(neighbors, sorted(active), rings=1))
                              - active) - boundary)
            newly: List[int] = []
            for j in frontier:
                if total_active_growth >= max_growth_budget:
                    break
                if mean_laplacian_magnitude(x, neighbors, [j]) > roughness_threshold:
                    newly.append(j)
                    total_active_growth += 1
            if newly:
                fairing_region |= set(newly)
                region = build_active_region(sorted(fairing_region), neighbors, boundary,
                                             0, transition_rings)
                active = set(region["active"])
                anchor = region["anchor"]
                for j in newly:
                    provenance[j] = "grown"
                    weights[j] = (w_fair, 0.0)
                new_active = sorted(active - set(floors.keys()))
                if new_active:
                    new_floors, _od, _pu = anatomy_constraint.compute_clearance_floors(
                        x, new_active, anatomy_backend, min_clearance,
                        clearance_policy=clearance_policy, clearance_tolerance=clearance_tolerance)
                    floors.update(new_floors)
                for j in active - set(weights.keys()):
                    provenance[j] = "grown"
                    weights[j] = (w_fair * 0.5, 0.0)
                if verbose:
                    print("  [iter {0:4d}] dynamic growth: +{1} vertices "
                          "(budget {2}/{3})".format(it, len(newly), total_active_growth,
                                                    max_growth_budget))

        # --- periodic M5 redetection (live mesh required) ---------------------
        if redetect_every and apply and (it % int(redetect_every) == 0):
            mesh_utils.set_mesh_vertices(skin_mesh, x)
            new_idx, new_report = artifact_detection.detect_unified_artifacts(
                skin_mesh, valid_anatomical_meshes, target_offset, select=False,
                **dict(m5_detector_kwargs or {}))
            extra = sorted((set(new_idx) - active) - boundary)
            if extra:
                fairing_region |= set(extra)
                region = build_active_region(sorted(fairing_region), neighbors, boundary,
                                             0, transition_rings)
                active = set(region["active"])
                anchor = region["anchor"]
                m3o, m4o, ov = _m5_provenance_sets(new_report)
                for j in extra:
                    if j in ov:
                        provenance[j] = "overlap"
                        weights[j] = (w_fair, w_m4 * overlap_m4_scale)
                    elif j in m4o:
                        provenance[j] = "m4_only"
                        weights[j] = (w_fair * m4_only_fair_scale, w_m4)
                    else:
                        provenance[j] = "m3_only"
                        weights[j] = (w_fair, 0.0)
                for j in active - set(weights.keys()):
                    provenance[j] = "grown"
                    weights[j] = (w_fair * 0.5, 0.0)
                more_m4 = sorted(j for j in extra if weights[j][1] > 0.0)
                if more_m4:
                    m4_targets.update(_compute_m4_targets(
                        skin_mesh, valid_anatomical_meshes, target_offset, more_m4,
                        sdf_query=sdf_query))
                new_active = sorted(active - set(floors.keys()))
                if new_active:
                    new_floors, _od, _pu = anatomy_constraint.compute_clearance_floors(
                        x, new_active, anatomy_backend, min_clearance,
                        clearance_policy=clearance_policy, clearance_tolerance=clearance_tolerance)
                    floors.update(new_floors)
                if verbose:
                    print("  [iter {0:4d}] M5 redetect: +{1} vertices".format(it, len(extra)))
    else:
        stop_reason = STOP_MAX_ITERATIONS

    if stop_reason is None:
        stop_reason = STOP_MAX_ITERATIONS

    # --- write once, at the end (matches constrained_smooth_mesh_region) -----
    if apply:
        mesh_utils.set_mesh_vertices(skin_mesh, x)

    final_active = sorted(active)
    final_core, final_scoped_report = _scoped_intersection_core(
        skin_mesh, x, final_active, anatomy_backend, neighbors, skin_topology,
        boundary_buffer_rings, intersection_tolerance)
    whole_report = anatomy_constraint.analyze_skin_anatomy_intersections(
        skin_mesh, skin_indices=list(range(len(x))), backend=anatomy_backend, positions=x,
        neighbors=neighbors, skin_topology=skin_topology,
        boundary_buffer_rings=boundary_buffer_rings,
        intersection_tolerance=intersection_tolerance, detailed=False, verbose=False)

    fully_clean = (not final_core) and whole_report.get("intersection_pair_count", 0) <= 0
    converged = bool(converged and fully_clean)
    if not converged and stop_reason == STOP_CONVERGED:
        stop_reason = STOP_VALIDATION_FAILED

    total_disp = metrics_utils.displacement_stats(positions0, x, indices=sorted(set(core0) | active))
    whole_disp = metrics_utils.displacement_stats(positions0, x)

    unresolved = None
    if not converged:
        unresolved = {
            "unresolved_faces": whole_report.get("intersecting_skin_faces") or [],
            "unresolved_face_count": whole_report.get("intersecting_skin_face_count", 0),
            "unresolved_vertices": final_core,
            "responsible_anatomy_meshes": whole_report.get("intersecting_anatomy_meshes") or [],
        }
        if verbose:
            print("[final_cleanup] NOT converged: {0} unresolved face(s), meshes={1}".format(
                unresolved["unresolved_face_count"], unresolved["responsible_anatomy_meshes"]))

    if apply and select_final and converged and final_active:
        mesh_utils.select_vertices(skin_mesh, final_active, replace=True)
    elif apply and not converged and select_on_failure:
        sel = final_core or final_active
        if sel:
            mesh_utils.select_vertices(skin_mesh, sel, replace=True)
            if verbose:
                print("[final_cleanup] selected {0} vertex(es) for manual inspection".format(len(sel)))

    result: Dict[str, Any] = {
        "skin_mesh": skin_mesh,
        "converged": converged,
        "iterations": it,
        "stop_reason": stop_reason,
        "dry_run": not apply,
        "target_offset": target_offset,
        "min_clearance": min_clearance,
        "clearance_policy": policy_used,
        "initial_core_count": len(core0),
        "fairing_count": len(fairing_region),
        "transition_count": len(final_active) - len(fairing_region),
        "active_count": len(final_active),
        "dynamic_growth_added": total_active_growth,
        "provenance_counts": {
            "m3_only": sum(1 for p in provenance.values() if p == "m3_only"),
            "m4_only": sum(1 for p in provenance.values() if p == "m4_only"),
            "overlap": sum(1 for p in provenance.values() if p == "overlap"),
            "grown": sum(1 for p in provenance.values() if p == "grown"),
        },
        "intersections": {
            "before": {"face_count": isect0_report.get("intersecting_skin_face_count", 0),
                      "pair_count": isect0_report.get("intersection_pair_count", 0)},
            "after": {"face_count": whole_report.get("intersecting_skin_face_count", 0),
                     "pair_count": whole_report.get("intersection_pair_count", 0)},
        },
        "roughness": {"before": roughness0, "after": prev_roughness},
        "m4_disagreement": {"before": m4_disagreement0, "after": prev_m4_disagreement},
        "min_exact_anatomy_distance": {"before": min_dist0,
                                      "after": min((anatomy_backend.exact_closest(x[i])["distance"]
                                                  for i in final_active), default=float("inf"))},
        "total_displacement": total_disp,
        "whole_mesh_displacement": whole_disp,
        "unresolved": unresolved,
        "m5_report": m5_report,
        "iterations_log": iterations_log,
        "runtime_seconds": time.time() - t_run0,
        "config": {
            "cleanup_growth_rings": cleanup_growth_rings, "transition_rings": transition_rings,
            "w_fair": w_fair, "w_m4": w_m4, "w_shape": w_shape,
            "max_step_edge_ratio": max_step_edge_ratio,
            "dynamic_active_set": dynamic_active_set, "redetect_every": redetect_every,
            "max_iterations": max_iterations, "convergence_patience": convergence_patience,
        },
    }

    if verbose:
        print_final_cleanup_report(result)

    if save_json or save_csv:
        log_dir = log_path if log_path else "cleanup_logs"
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base = "final_cleanup_{0}".format(timestamp)
        log_files = {}
        if save_json:
            p = _write_json(result, os.path.join(log_dir, base + ".json"))
            if p:
                log_files["json"] = p
        if save_csv:
            p = _write_csv(csv_rows, os.path.join(log_dir, base + ".csv"))
            if p:
                log_files["csv"] = p
        result["log_files"] = log_files

    return result


def print_final_cleanup_report(result: Dict[str, Any]) -> None:
    """Human-readable summary of a :func:`run_cleanup_solver` result."""
    print("\n" + "=" * 60)
    print("FINAL CLEANUP SOLVER {0}".format(
        "COMPLETE" if result.get("converged") else "STOPPED (not converged)"))
    print("=" * 60)
    print("converged:  {0}".format(result.get("converged")))
    print("iterations: {0}".format(result.get("iterations")))
    print("stop reason: {0}".format(result.get("stop_reason")))
    if result.get("dry_run"):
        print("(dry run -- scene NOT modified)")
    isect = result.get("intersections") or {}
    b, a = isect.get("before", {}), isect.get("after", {})
    print("intersections (faces): {0} -> {1}".format(
        b.get("face_count", 0), a.get("face_count", 0)))
    rough = result.get("roughness") or {}
    print("roughness (mean Laplacian magnitude): {0:.5f} -> {1:.5f}".format(
        rough.get("before", 0.0), rough.get("after", 0.0)))
    m4 = result.get("m4_disagreement") or {}
    print("M4 target disagreement: {0:.5f} -> {1:.5f}".format(
        m4.get("before", 0.0), m4.get("after", 0.0)))
    dist = result.get("min_exact_anatomy_distance") or {}
    print("min exact anatomy distance: {0:.4f} -> {1:.4f}".format(
        dist.get("before", float("inf")), dist.get("after", float("inf"))))
    disp = result.get("total_displacement") or {}
    print("mean final iteration displacement: see iterations_log[-1]['net_displacement']")
    print("displacement over touched region: mean={0:.5f} max={1:.5f}".format(
        disp.get("mean", 0.0), disp.get("max", 0.0)))
    whole = result.get("whole_mesh_displacement") or {}
    print("max shape deviation (whole mesh vs. registered): {0:.5f}".format(
        whole.get("max", 0.0)))
    prov = result.get("provenance_counts") or {}
    print("provenance: m3_only={0} m4_only={1} overlap={2} grown={3}".format(
        prov.get("m3_only", 0), prov.get("m4_only", 0), prov.get("overlap", 0),
        prov.get("grown", 0)))
    unresolved = result.get("unresolved")
    if unresolved:
        print("UNRESOLVED: {0} face(s); anatomy meshes: {1}".format(
            unresolved.get("unresolved_face_count", 0),
            unresolved.get("responsible_anatomy_meshes", [])))
    print("runtime: {0:.2f}s".format(result.get("runtime_seconds", 0.0)))
    print("=" * 60)


def compare_cleanup_result(before_positions: List[List[float]],
                          after_positions: List[List[float]],
                          indices: Optional[Sequence[int]] = None,
                          label: str = "final_cleanup",
                          ) -> Dict[str, Any]:
    """Thin convenience wrapper: displacement of the final mesh vs. the
    original (or vs. a saved snapshot), reusing :mod:`metrics_utils` (no new
    displacement math)."""
    return metrics_utils.print_displacement_report(
        before_positions, after_positions, indices=indices, label=label)
