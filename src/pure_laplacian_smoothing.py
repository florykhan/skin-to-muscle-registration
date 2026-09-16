"""
pure_laplacian_smoothing.py
============================
A DELIBERATELY UNSAFE, minimal, fully-transparent EXPERIMENTAL smoothing
pipeline. Completely separate from :mod:`final_cleanup_solver` -- does not
import it, does not import :mod:`anatomy_constraint`, and does not call
:mod:`smoothing_utils`'s smoothing operator (the Jacobi step below is
hand-written here for maximum transparency, per the exact formula requested).

Purpose
-------
Answer ONE diagnostic question before deciding how much safety machinery the
FINAL pipeline actually needs: if literally nothing resists Laplacian
smoothing, can the visible bumps/ridges on the registered skin actually be
flattened at all? If yes, the problem was how much fought the smoothing, not
the operator or the region. If no, the problem is the operator/region itself.

This module intentionally contains NONE of the following, by design:
    anatomy constraints, clearance projection, triangle intersection checks,
    repair/remediation, M4, M5 forces, trust regions, shape force,
    rest-reference smoothing, best-state rollback, oscillation damping,
    collision handling, target offsets, anatomy attraction.

It is EXPECTED and ACCEPTABLE for a strong/long run of this module to shrink
the skin, push it into anatomy, create intersections, or otherwise distort
the surface. Do not use this as a finishing stage. It is a diagnostic.

Reuses ONLY
-----------
* :mod:`mesh_utils` -- read/write Maya vertices, adjacency, vector math, and
  ``get_boundary_vertices`` (a pure TOPOLOGY fact -- which vertices sit on an
  open mesh edge such as the eye/mouth/neck rim -- not an anatomy or safety
  mechanism, so using it does not violate "zero anatomy/safety logic"; see
  ``protect_boundary`` below).
* :func:`artifact_detection.compute_laplacian_scores` -- the EXISTING,
  already visually-verified M1 roughness metric, reused for vertex SELECTION
  and for the before/after diagnostics ONLY. The smoothing operator itself
  does not call it and is not derived from it.
* :func:`maya_io.duplicate_mesh` -- the same optional pre-edit backup every
  other stage in this project uses.

Nothing here modifies M1-M5, artifact_detection.py, anatomy_constraint.py,
final_cleanup_solver.py, smoothing_utils.py, or the intersection detector.
"""

from __future__ import print_function

import time
from typing import Any, Dict, List, Optional, Sequence, Set

import mesh_utils
import artifact_detection

try:
    import maya_io
except ImportError:  # pragma: no cover - maya_io present in this project
    maya_io = None


DEFAULT_ROUGHNESS_PERCENTILE = 75.0
DEFAULT_GROWTH_RINGS = 3
DEFAULT_STRENGTH = 0.5
DEFAULT_ITERATIONS = 20
DEFAULT_REPORT_INTERVAL = 5
_BACKUP_SUFFIX = "_pre_pure_laplacian"


# =============================================================================
# 1. PURE HELPERS (no Maya access; unit-tested in isolation)
# =============================================================================

def jacobi_laplacian_step(positions: List[List[float]],
                          neighbors: List[List[int]],
                          weights: Dict[int, float],
                          ) -> List[List[float]]:
    """ONE simultaneous (Jacobi) umbrella-Laplacian step -- the exact formula::

        new_position[i] = position[i] + weight[i] * (neighbor_mean(i) - position[i])

    ALL proposed positions are computed from the SAME snapshot of
    ``positions`` (never from a partially-updated array within this call),
    then returned together as a new list -- the defining property of a Jacobi
    update, as opposed to Gauss-Seidel (updating vertices one at a time within
    the same pass, where later vertices would see already-moved neighbours).

    ``weights`` gives each vertex its own strength multiplier: 0 (or absent)
    means fixed, 1 means the formula above acts at its literal, unclamped
    value -- a hard selection and a graded falloff region are the SAME
    mechanism here, just different weight values. There is no hidden
    damping, clamp, adaptive reduction, or rejection anywhere in this
    function: if ``weights[i] == 1.0``, vertex ``i`` moves exactly to its
    neighbours' centroid on this call, full stop.
    """
    out = [list(p) for p in positions]
    for i, w in weights.items():
        if w == 0.0 or i < 0 or i >= len(positions):
            continue
        nbrs = neighbors[i] if 0 <= i < len(neighbors) else []
        if not nbrs:
            continue
        mean = mesh_utils.vec_mean([positions[j] for j in nbrs])
        p = positions[i]
        out[i] = [p[k] + w * (mean[k] - p[k]) for k in range(3)]
    return out


def ring_distances(seeds: Sequence[int],
                   neighbors: List[List[int]],
                   max_rings: int,
                   ) -> Dict[int, int]:
    """BFS topological ring distance (0 = a seed itself) from ``seeds``, out
    to ``max_rings``. Plain adjacency BFS; no mesh_utils/final_cleanup_solver
    dependency beyond the ``neighbors`` list already passed in."""
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


def linear_falloff_weight(ring: int, growth_rings: int) -> float:
    """Simple LINEAR falloff: ring 0 (the selection itself) -> 1.0, fading to
    0 just past ``growth_rings``. E.g. ``growth_rings=3`` gives
    ``[1.0, 0.75, 0.50, 0.25]`` for rings 0..3 and 0.0 beyond -- deliberately
    simple (no smoothstep, no second solver), matching the falloff table
    requested for this experiment.
    """
    if ring < 0:
        return 0.0
    denom = float(growth_rings) + 1.0
    return max(0.0, 1.0 - (float(ring) / denom))


def build_falloff_weights(core: Sequence[int],
                          neighbors: List[List[int]],
                          growth_rings: int,
                          exclude: Optional[Set[int]] = None,
                          ) -> Dict[int, float]:
    """``{vertex: weight}`` for ``core`` (weight 1.0, always -- every selected
    vertex is full strength regardless of ``growth_rings``) plus a
    linear-falloff halo grown ``growth_rings`` rings around it, so there is no
    hard seam at the edge of the selection. Vertices in ``exclude`` (e.g. true
    topological mesh-boundary vertices) never receive a weight, whether they
    are in ``core`` or only reached via the halo.
    """
    excl = set(exclude or [])
    core_set = set(int(i) for i in core) - excl
    dist = ring_distances(core_set, neighbors, growth_rings)
    weights: Dict[int, float] = {}
    for i, r in dist.items():
        if i in excl:
            continue
        weights[i] = 1.0 if i in core_set else linear_falloff_weight(r, growth_rings)
    return weights


def select_rough_vertices(scores: Dict[int, float], percentile: float) -> List[int]:
    """Vertex indices whose score is at or above the given percentile of
    ``scores`` (linear-interpolation percentile, same convention used
    throughout this project)."""
    if not scores:
        return []
    vals = sorted(scores.values())
    n = len(vals)
    k = (n - 1) * (float(percentile) / 100.0)
    lo, hi = int(k), min(int(k) + 1, n - 1)
    cutoff = vals[lo] + (vals[hi] - vals[lo]) * (k - lo)
    return sorted(i for i, s in scores.items() if s >= cutoff)


def roughness_percentile_summary(values: Sequence[float]) -> Dict[str, float]:
    """mean/p50/p90/p95/p99/max of a roughness sample. Standalone
    reimplementation (this module must not import final_cleanup_solver, which
    has its own equivalent)."""
    vals = sorted(float(v) for v in values)
    if not vals:
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}

    def _pct(p: float) -> float:
        k = (len(vals) - 1) * (p / 100.0)
        lo, hi = int(k), min(int(k) + 1, len(vals) - 1)
        return vals[lo] + (vals[hi] - vals[lo]) * (k - lo)

    return {
        "mean": sum(vals) / len(vals),
        "p50": _pct(50.0), "p90": _pct(90.0), "p95": _pct(95.0), "p99": _pct(99.0),
        "max": vals[-1],
    }


def _laplacian_magnitudes(positions: List[List[float]],
                          neighbors: List[List[int]],
                          indices: Sequence[int],
                          ) -> Dict[int, float]:
    """Per-vertex umbrella-Laplacian magnitude on an in-memory array -- the
    same formula as ``artifact_detection.compute_laplacian_scores``, used
    here only to report roughness mid-run without round-tripping every
    iteration through Maya."""
    out: Dict[int, float] = {}
    for i in indices:
        nbrs = neighbors[i] if 0 <= i < len(neighbors) else []
        if not nbrs:
            continue
        mean = mesh_utils.vec_mean([positions[j] for j in nbrs])
        out[i] = mesh_utils.vec_length(mesh_utils.vec_sub(positions[i], mean))
    return out


# =============================================================================
# 2. MAIN ENTRY POINT
# =============================================================================

def run_pure_laplacian_smoothing(skin_mesh: str,
                                 indices: Optional[Sequence[int]] = None,
                                 roughness_percentile: float = DEFAULT_ROUGHNESS_PERCENTILE,
                                 growth_rings: int = DEFAULT_GROWTH_RINGS,
                                 strength: float = DEFAULT_STRENGTH,
                                 iterations: int = DEFAULT_ITERATIONS,
                                 protect_boundary: bool = True,
                                 apply: bool = True,
                                 create_backup: bool = True,
                                 backup_suffix: str = _BACKUP_SUFFIX,
                                 report_interval: int = DEFAULT_REPORT_INTERVAL,
                                 verbose: bool = True,
                                 ) -> Dict[str, Any]:
    """EXPERIMENTAL, deliberately unsafe Jacobi Laplacian smoothing. See the
    module docstring: no anatomy constraint, no intersection check, no repair,
    no M4/M5, no trust region, no shape force, no rollback. Nothing stops this
    from shrinking the skin into anatomy given enough strength/iterations --
    that is the point of the experiment.

    Parameters
    ----------
    skin_mesh:
        Mesh to smooth (read fresh from Maya; no reliance on any pre-existing
        Python globals -- works in a brand new Maya session).
    indices:
        Vertices to treat as the full-strength "core". If omitted (the usual
        case), computed automatically: score every vertex with the EXISTING
        M1 metric (:func:`artifact_detection.compute_laplacian_scores`) and
        keep those at or above ``roughness_percentile``.
    growth_rings:
        Grow the core outward by this many topological rings with a simple
        LINEAR falloff (see :func:`linear_falloff_weight`) so there is no
        hard smoothing seam at the selection's edge. ``growth_rings=0``
        smooths exactly the selected/given vertices and nothing else.
    strength:
        Direct multiplier on the Jacobi step -- NOT damped, clamped, or
        adaptively reduced anywhere. ``strength=1.0`` moves every full-weight
        (core) vertex exactly to its neighbours' centroid on every iteration.
    protect_boundary:
        If True (default), true topological mesh-boundary vertices (open
        edges -- eye/mouth/nostril/neck rims, the outer mesh border) are
        never selected and never moved, using
        :func:`mesh_utils.get_boundary_vertices` (a pure topology fact, not
        an anatomy/safety mechanism). This is the one deliberate addition
        beyond your literal spec: without it, an open-boundary vertex's
        neighbour average is systematically one-sided (it has neighbours on
        only one side), which pulls those rims in a visually asymmetric way
        that would confuse the read on whether bumps are actually
        flattening. Pass ``protect_boundary=False`` for the literal
        zero-exceptions version.
    apply:
        If False, compute everything in memory and do not write or back up
        the scene.

    Returns
    -------
    dict
        ``selected_count``, ``active_count``, ``selection_source``,
        ``selected_indices``, ``active_indices``, roughness before/after
        (mean and the full percentile summary), ``mean_displacement``,
        ``max_displacement``, ``max_displacement_vertex`` -- all measured
        from the STARTING mesh at the top of this call.
    """
    t0 = time.time()

    if not mesh_utils.mesh_exists(skin_mesh):
        raise ValueError("pure_laplacian_smoothing: skin_mesh '{0}' does not exist".format(skin_mesh))
    if iterations < 1:
        raise ValueError("iterations must be >= 1, got {0}".format(iterations))
    if growth_rings < 0:
        raise ValueError("growth_rings must be >= 0, got {0}".format(growth_rings))

    positions0 = mesh_utils.get_mesh_vertices(skin_mesh)
    if not positions0:
        raise ValueError("pure_laplacian_smoothing: skin_mesh '{0}' has no readable "
                         "vertices".format(skin_mesh))
    neighbors = mesh_utils.get_vertex_neighbors(skin_mesh)
    n = len(positions0)

    boundary: Set[int] = set()
    if protect_boundary:
        try:
            boundary = set(mesh_utils.get_boundary_vertices(skin_mesh) or [])
        except Exception:
            boundary = set()

    scores: Optional[Dict[int, float]] = None
    if indices is not None:
        core = sorted((set(int(i) for i in indices if 0 <= int(i) < n)) - boundary)
        selection_source = "caller"
    else:
        scores = artifact_detection.compute_laplacian_scores(skin_mesh)
        core = sorted(set(select_rough_vertices(scores, roughness_percentile)) - boundary)
        selection_source = "auto_percentile"

    if not core:
        if verbose:
            print("[pure_laplacian] no vertices selected; nothing to do")
        return {"skin_mesh": skin_mesh, "selected_count": 0, "active_count": 0,
               "selection_source": selection_source, "selected_indices": [],
               "active_indices": [], "iterations": 0, "dry_run": not apply,
               "runtime_seconds": time.time() - t0}

    weights = build_falloff_weights(core, neighbors, growth_rings, exclude=boundary)
    active = sorted(weights.keys())

    if scores is None:
        scores = artifact_detection.compute_laplacian_scores(skin_mesh, indices=core)
    roughness_pct_before = roughness_percentile_summary(
        [scores[i] for i in core if i in scores])

    if create_backup and apply and maya_io is not None:
        backup_name = skin_mesh + backup_suffix
        if not mesh_utils.mesh_exists(backup_name):
            maya_io.duplicate_mesh(skin_mesh, suffix=backup_suffix)
        elif verbose:
            print("[pure_laplacian] backup '{0}' already exists; keeping it".format(backup_name))

    if verbose:
        print("[pure_laplacian] EXPERIMENTAL -- NO anatomy/safety logic of any kind.")
        print("[pure_laplacian] selected={0} ({1}) active incl. falloff={2} "
             "(boundary-protected={3}) strength={4} growth_rings={5} iterations={6}".format(
                 len(core), selection_source, len(active), protect_boundary,
                 strength, growth_rings, iterations))

    x = [list(p) for p in positions0]
    scaled_weights = {i: w * float(strength) for i, w in weights.items()}
    for it in range(1, int(iterations) + 1):
        x = jacobi_laplacian_step(x, neighbors, scaled_weights)
        if verbose and (it == 1 or it % max(1, report_interval) == 0 or it == iterations):
            live = roughness_percentile_summary(
                list(_laplacian_magnitudes(x, neighbors, core).values()))
            print("  [iter {0:3d}] roughness mean={1:.5f} p90={2:.5f} p95={3:.5f} "
                 "max={4:.5f}".format(it, live["mean"], live["p90"], live["p95"], live["max"]))

    if apply:
        mesh_utils.set_mesh_vertices(skin_mesh, x)

    roughness_pct_after = roughness_percentile_summary(
        list(_laplacian_magnitudes(x, neighbors, core).values()))

    disp = [mesh_utils.vec_length(mesh_utils.vec_sub(x[i], positions0[i])) for i in active]
    mean_disp = (sum(disp) / len(disp)) if disp else 0.0
    max_disp = max(disp) if disp else 0.0
    max_disp_idx = active[disp.index(max_disp)] if disp else -1

    result: Dict[str, Any] = {
        "skin_mesh": skin_mesh,
        "selected_count": len(core),
        "active_count": len(active),
        "selection_source": selection_source,
        "selected_indices": core,
        "active_indices": active,
        "iterations": int(iterations),
        "strength": float(strength),
        "growth_rings": int(growth_rings),
        "roughness_percentile": float(roughness_percentile),
        "protect_boundary": bool(protect_boundary),
        "dry_run": not apply,
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
        print_pure_laplacian_report(result)
    return result


def print_pure_laplacian_report(result: Dict[str, Any]) -> None:
    """Human-readable summary of a :func:`run_pure_laplacian_smoothing` result."""
    print("\n" + "=" * 60)
    print("PURE LAPLACIAN SMOOTHING -- EXPERIMENTAL, NO ANATOMY SAFETY")
    print("=" * 60)
    print("selected: {0} ({1})   active incl. falloff: {2}   boundary-protected: {3}".format(
        result.get("selected_count"), result.get("selection_source"),
        result.get("active_count"), result.get("protect_boundary")))
    print("strength={0}  growth_rings={1}  iterations={2}".format(
        result.get("strength"), result.get("growth_rings"), result.get("iterations")))
    if result.get("dry_run"):
        print("(dry run -- scene NOT modified)")
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
    print("NOTE: no intersection/clearance check was performed -- this run may have "
         "pushed skin into anatomy. That is expected for this experiment.")
    print("runtime: {0:.2f}s".format(result.get("runtime_seconds", 0.0)))
    print("=" * 60)
