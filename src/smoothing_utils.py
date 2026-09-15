"""
smoothing_utils.py
==================
Localized Laplacian smoothing for post-registration skin cleanup.

The core registration script (``d98``) already applies a global Laplacian
smoothing pass inside the shrink-wrap loop. This module generalizes that logic so
it can be applied to *only a selected region* of vertices after registration has
converged -- the practical task of cleaning up problem areas (lips, cheeks, nose,
etc.) without disturbing the rest of the mesh.

Two smoothing operators are provided:

* :func:`laplacian_smooth` -- plain Laplacian (simple, but shrinks volume when
  applied for many iterations).
* :func:`taubin_smooth` -- Laplacian + inflation (lambda/mu) that smooths noise
  while largely preserving volume. Prefer this for stronger cleanup.

All functions operate on plain ``[[x, y, z], ...]`` vertex lists so they are
easy to test and reuse. Convenience wrappers read from / write to a live Maya
mesh via :mod:`mesh_utils`.
"""

import math

from mesh_utils import (
    vec_mean,
    vec_lerp,
    vec_sub,
    vec_add,
    vec_scale,
    vec_length,
    get_mesh_vertices,
    set_mesh_vertices,
    get_vertex_neighbors,
    get_vertex_normals,
    get_mesh_fn,
    get_triangle_topology,
    grow_indices,
)

import anatomy_constraint as anatomy_constraint


def _region_mask(num_verts, indices):
    """Return a boolean list marking which vertices are allowed to move."""
    if indices is None:
        return [True] * num_verts
    mask = [False] * num_verts
    for i in indices:
        if 0 <= i < num_verts:
            mask[i] = True
    return mask


def laplacian_smooth(vertices, neighbors, strength=0.2, iterations=1,
                     indices=None):
    """Return a smoothed copy of ``vertices``.

    Parameters
    ----------
    vertices : list[[x, y, z]]
        Current vertex positions (index-aligned with ``neighbors``).
    neighbors : list[list[int]]
        1-ring adjacency from :func:`mesh_utils.get_vertex_neighbors`.
    strength : float
        Per-iteration blend toward the neighbor average (0..1). Higher is
        smoother/faster but can distort features.
    iterations : int
        How many smoothing passes to run.
    indices : iterable[int] or None
        If given, ONLY these vertices are moved. Their neighbors are still used
        as references, so the smoothed region blends into the frozen surroundings
        instead of tearing at the boundary.
    """
    n = len(vertices)
    mask = _region_mask(n, indices)
    verts = [list(v) for v in vertices]

    for _ in range(max(1, iterations)):
        new_verts = [list(v) for v in verts]
        for i in range(n):
            if not mask[i]:
                continue
            nbrs = neighbors[i]
            if not nbrs:
                continue
            avg = vec_mean([verts[j] for j in nbrs])
            new_verts[i] = vec_lerp(verts[i], avg, strength)
        verts = new_verts
    return verts


def taubin_smooth(vertices, neighbors, lamb=0.33, mu=-0.34, iterations=10,
                  indices=None):
    """Volume-preserving Taubin smoothing (lambda/mu passes).

    Each "pass" is a positive Laplacian step (``lamb``) followed by a negative
    inflation step (``mu``). With ``mu`` slightly larger in magnitude than
    ``lamb`` this counteracts the shrinkage of plain Laplacian smoothing, which
    matters for faces where you want to remove bumps without deflating lips or
    cheeks. ``iterations`` counts full lambda+mu pairs.
    """
    verts = [list(v) for v in vertices]
    for _ in range(max(1, iterations)):
        verts = laplacian_smooth(verts, neighbors, strength=lamb,
                                 iterations=1, indices=indices)
        verts = laplacian_smooth(verts, neighbors, strength=mu,
                                 iterations=1, indices=indices)
    return verts


# =============================================================================
# LIVE-MESH CONVENIENCE WRAPPERS
# =============================================================================

def smooth_mesh_region(mesh_name, indices=None, strength=0.2, iterations=5,
                       method="taubin", neighbors=None, apply=True):
    """Read a live Maya mesh, smooth (optionally a region), and write it back.

    Parameters
    ----------
    mesh_name : str
        Mesh to smooth (e.g. the skin mesh). Its NAME is not changed.
    indices : iterable[int] or None
        Region to smooth. None smooths the whole mesh.
    strength : float
        Blend strength; used as lambda for Taubin and as the direct strength for
        plain Laplacian.
    iterations : int
        Number of smoothing passes.
    method : {"taubin", "laplacian"}
        Smoothing operator to use.
    neighbors : list[list[int]] or None
        Precomputed adjacency; computed from the mesh if omitted.
    apply : bool
        If False, compute and return the new vertices WITHOUT modifying the
        scene (useful for previewing / metrics).

    Returns
    -------
    (before, after) : tuple[list, list]
        The original and smoothed vertex lists.
    """
    before = get_mesh_vertices(mesh_name)
    if not before:
        print("[smoothing_utils] mesh '{0}' has no vertices / not found".format(mesh_name))
        return [], []

    if neighbors is None:
        neighbors = get_vertex_neighbors(mesh_name)

    if method == "laplacian":
        after = laplacian_smooth(before, neighbors, strength=strength,
                                 iterations=iterations, indices=indices)
    else:
        after = taubin_smooth(before, neighbors, lamb=strength,
                              mu=-(strength + 0.01), iterations=iterations,
                              indices=indices)

    if apply:
        set_mesh_vertices(mesh_name, after)
        region_note = "whole mesh" if indices is None else "{0} verts".format(len(list(indices)))
        print("[smoothing_utils] smoothed {0} ({1}, method={2}, iters={3})".format(
            mesh_name, region_note, method, iterations))
    return before, after


# =============================================================================
# ANATOMY-CONSTRAINED LOCALIZED SMOOTHING
# =============================================================================
# Ordinary Laplacian/Taubin smoothing of a convex facial patch has an inward
# (shrinking) component: it flattens an outward registration bump but can pull the
# skin THROUGH the underlying fat/muscle/bone, exposing internal anatomy. This
# adds the anatomical FLOOR that unconstrained smoothing was missing:
#
#   for each iteration:
#       proposed = ONE localized smoothing step (only selected vertices move)
#       (optional per-vertex max-step cap relative to local edge length)
#       every `constraint_interval` iterations:
#           for each selected vertex whose proposed position is closer to anatomy
#           than `min_clearance`, PUSH it back out to the safe clearance using the
#           EXACT registration collision rule (closest_point + outward*clearance).
#       commit proposed -> current
#
# The smoothing motion stays geometrically meaningful (inward flattening is
# allowed); the anatomy constraint only supplies the lower bound. The distance
# field + push direction are INJECTED as ``sdf_query_fn`` so this module stays
# decoupled from the d98 registration script (which owns the smooth-min union SDF
# and collision field). ``smooth_mesh_region`` above is unchanged.
#
# The historical one-shot push (constraint_solver="legacy_single_push") reused
# the registration smooth-min field and accepted ``cp + outward * clearance``
# without re-querying. That is preserved for A/B comparison. The default
# ``constraint_solver="iterative_exact"`` instead:
#   1) optionally repairs high-confidence pre-existing penetrations
#   2) runs one Laplacian/Taubin step
#   3) optionally clamps segment crossings
#   4) iteratively projects onto an exact (unblended) anatomy floor
#   5) re-queries exact closest-point distance after every push
# Unsigned distance is still NOT treated as penetration; that classification
# lives in :mod:`anatomy_constraint`.


def _sc_stats(values):
    """min/mean/rms/max for a list of scalars (zeros for empty input)."""
    if not values:
        return {"count": 0, "min": 0.0, "mean": 0.0, "rms": 0.0, "max": 0.0}
    c = len(values)
    return {"count": c, "min": min(values), "mean": sum(values) / c,
            "rms": math.sqrt(sum(v * v for v in values) / c), "max": max(values)}


def _run_smoothing(vertices, neighbors, method, strength, indices, iterations):
    """N smoothing passes over ``indices`` using the EXISTING operators only."""
    if method == "taubin":
        return taubin_smooth(vertices, neighbors, lamb=strength,
                             mu=-(strength + 0.01), iterations=iterations,
                             indices=indices)
    return laplacian_smooth(vertices, neighbors, strength=strength,
                            iterations=iterations, indices=indices)


def _legacy_iterative_sdf_push(point, sdf_query_fn, min_clearance, max_iters, tol):
    """Iterate the historical smooth-min push, re-querying the blended field.

    Used only when no exact anatomy backend is available. Still not a strict
    geometric validator (smooth-min != min).
    """
    p = list(point)
    for k in range(max(1, int(max_iters))):
        dist, cp, outward = sdf_query_fn(p)
        if dist >= min_clearance - tol:
            return p, True, k + 1, max(0.0, min_clearance - dist)
        p = vec_add(list(cp), vec_scale(list(outward), min_clearance))
    dist, _, _ = sdf_query_fn(p)
    return p, False, max(1, int(max_iters)), max(0.0, min_clearance - dist)


DEFAULT_UNSAFE_STEP_POLICY = "largest_safe_fraction"
DEFAULT_SMOOTHING_LINE_SEARCH_STEPS = 8
DEFAULT_MIN_SMOOTHING_ALPHA = 1e-3
DEFAULT_SAFETY_MODE = "strict"
DEFAULT_SAFETY_BATCH_ITERATIONS = 10
DEFAULT_MAX_UNSAFE_BATCH_DISPLACEMENT_RATIO = 1.0


def _normalize_safety_mode(safety_mode, allow_safety_decline=None):
    """Map safety_mode / allow_safety_decline onto one of strict|repair_after_batch|off."""
    requested = None if safety_mode is None else str(safety_mode).strip().lower()
    if requested in ("repair-after-batch", "batch", "repair_after"):
        requested = "repair_after_batch"
    if requested is not None and requested not in ("strict", "repair_after_batch", "off"):
        raise ValueError(
            "safety_mode must be 'strict', 'repair_after_batch', or 'off', got {0!r}"
            .format(safety_mode))
    if allow_safety_decline is True:
        if requested not in (None, "repair_after_batch"):
            raise ValueError(
                "allow_safety_decline=True conflicts with safety_mode={0!r}"
                .format(safety_mode))
        return "repair_after_batch"
    if allow_safety_decline is False:
        if requested not in (None, "strict"):
            raise ValueError(
                "allow_safety_decline=False conflicts with safety_mode={0!r}"
                .format(safety_mode))
        return "strict"
    return requested or DEFAULT_SAFETY_MODE


def _region_disp_stats(before, after, indices):
    mags = [vec_length(vec_sub(after[i], before[i])) for i in indices]
    return {
        "mean": (sum(mags) / len(mags)) if mags else 0.0,
        "max": max(mags) if mags else 0.0,
    }


def _count_region_intersections(mesh_name, positions, region, backend, neighbors,
                                skin_topology, boundary_buffer_rings,
                                intersection_tolerance):
    if backend is None:
        return 0, 0
    rep = anatomy_constraint.analyze_skin_anatomy_intersections(
        mesh_name, skin_indices=region, backend=backend, positions=positions,
        neighbors=neighbors, skin_topology=skin_topology,
        boundary_buffer_rings=boundary_buffer_rings,
        intersection_tolerance=intersection_tolerance,
        detailed=False, verbose=False)
    return (int(rep.get("intersecting_skin_face_count", 0) or 0),
            int(rep.get("intersection_pair_count", 0) or 0))


def _unconstrained_smoothing_steps(work, neighbors, method, strength, active,
                                   n_iters, max_step_edge_ratio, local_edge):
    """Ordinary localized Laplacian/Taubin steps with no anatomy projection."""
    out = [list(v) for v in work]
    for _ in range(max(0, int(n_iters))):
        nxt = _run_smoothing(out, neighbors, method, strength, active, 1)
        if max_step_edge_ratio is not None:
            for i in active:
                cap = max_step_edge_ratio * local_edge.get(i, 0.0)
                if cap <= 0:
                    continue
                step = vec_sub(nxt[i], out[i])
                slen = vec_length(step)
                if slen > cap and slen > 1e-12:
                    nxt[i] = vec_add(out[i], vec_scale(step, cap / slen))
        out = nxt
    return out


def _scale_displacement_field(before, full, alpha, indices):
    """Return ``before + alpha * (full - before)`` on ``indices`` only."""
    out = [list(p) for p in before]
    a = float(alpha)
    if a <= 0.0:
        return out
    if a >= 1.0:
        for i in indices:
            out[i] = list(full[i])
        return out
    for i in indices:
        b = before[i]
        f = full[i]
        out[i] = [b[0] + a * (f[0] - b[0]),
                  b[1] + a * (f[1] - b[1]),
                  b[2] + a * (f[2] - b[2])]
    return out


def _smoothing_candidate_is_safe(report, baseline_faces, baseline_pair_count):
    """True iff the candidate adds no NEW intersecting faces and does not
    increase the local intersection-pair count vs the pre-iteration baseline.

    Pre-existing intersecting faces are allowed to remain. Total count == 0
    is NOT required.
    """
    faces = set(report.get("intersecting_skin_faces") or [])
    new_faces = faces - set(baseline_faces or [])
    pairs = int(report.get("intersection_pair_count", 0) or 0)
    base_pairs = int(baseline_pair_count or 0)
    if new_faces:
        return False, new_faces
    if pairs > base_pairs:
        return False, new_faces
    return True, set()


def _find_largest_safe_smoothing_step(
        positions_before, positions_full_proposal, moved_indices,
        is_safe_fn, steps=DEFAULT_SMOOTHING_LINE_SEARCH_STEPS,
        min_alpha=DEFAULT_MIN_SMOOTHING_ALPHA, project_fn=None):
    """Binary-search the largest patch-wide alpha in [0, 1] that is safe.

    ``positions_full_proposal`` is the FIXED Laplacian field for this
    iteration (after max-step, before projection). Every candidate is
    ``before + alpha * (full - before)``. Optional ``project_fn`` applies
    existing clearance / segment / tangent projection AFTER the lerp and
    BEFORE the safety test. The search origin never becomes a repaired mesh.

    ``is_safe_fn(projected_positions) -> (safe: bool, extra)``
    """
    moved = list(moved_indices or [])
    before = positions_before
    full = positions_full_proposal
    nsteps = max(1, int(steps))
    min_a = float(min_alpha) if min_alpha is not None else 0.0
    evaluations = 0

    def _eval(alpha):
        raw = _scale_displacement_field(before, full, alpha, moved)
        projected = project_fn(before, raw) if project_fn is not None else raw
        safe, extra = is_safe_fn(projected)
        return projected, raw, bool(safe), extra

    accepted, raw_full, ok, extra = _eval(1.0)
    evaluations += 1
    if ok:
        return {
            "positions": accepted,
            "raw_positions": raw_full,
            "alpha": 1.0,
            "safe": True,
            "status": "full",
            "evaluations": evaluations,
            "extra": extra,
        }

    lo, hi = 0.0, 1.0
    best_alpha = 0.0
    best_pos = [list(p) for p in before]
    best_raw = [list(p) for p in before]
    best_extra = extra
    for _ in range(nsteps):
        mid = 0.5 * (lo + hi)
        cand, raw, ok, extra = _eval(mid)
        evaluations += 1
        if ok:
            lo = mid
            best_alpha = mid
            best_pos = cand
            best_raw = raw
            best_extra = extra
        else:
            hi = mid

    if best_alpha >= min_a:
        status = "partial"
        safe = True
    else:
        status = "stalled_no_safe_step"
        safe = False
        best_pos = [list(p) for p in before]
        best_raw = [list(p) for p in before]
        best_alpha = 0.0

    return {
        "positions": best_pos,
        "raw_positions": best_raw,
        "alpha": best_alpha,
        "safe": safe,
        "status": status,
        "evaluations": evaluations,
        "extra": best_extra,
    }


def _apply_iteration_constraints(
        before, candidate, active, anatomy_backend, sdf_query_fn,
        floors, min_clearance, normals, solver, use_exact, do_segment,
        do_tangent, constrain_now, max_constraint_iterations,
        clearance_tolerance):
    """Existing per-vertex segment / tangent / clearance projection.

    Returns ``(projected_positions, stats)``. Does not write Maya.
    """
    proposed = [list(p) for p in candidate]
    stats = {
        "segment_detected": 0,
        "segment_prevented": 0,
        "constraint_events": 0,
        "constrained_vertices": set(),
        "corrections": [],
        "projection_steps": 0,
        "max_proj_used": 0,
        "unresolved": 0,
        "normal_removed": [],
        "tangent_kept": [],
    }
    for i in active:
        floor_i = floors.get(i, min_clearance)
        nrm = normals[i] if (normals and i < len(normals)) else None

        if do_segment:
            clamped, crossed, _hit = anatomy_constraint.clamp_segment_crossing(
                before[i], proposed[i], anatomy_backend, min_clearance=floor_i)
            if crossed:
                stats["segment_detected"] += 1
                stats["segment_prevented"] += 1
                proposed[i] = clamped

        if not constrain_now:
            continue

        if solver == "legacy_single_push":
            dist, cp, outward = sdf_query_fn(proposed[i])
            if dist < min_clearance:
                safe = vec_add(list(cp), vec_scale(list(outward), min_clearance))
                stats["corrections"].append(vec_length(vec_sub(proposed[i], safe)))
                proposed[i] = safe
                stats["constraint_events"] += 1
                stats["constrained_vertices"].add(i)
            continue

        if do_tangent:
            q_prop = anatomy_backend.exact_closest(proposed[i])
            if not anatomy_constraint.is_clearance_satisfied(
                    q_prop["distance"], floor_i, clearance_tolerance):
                cand, removed, tlen, used = anatomy_constraint.tangent_preserving_proposal(
                    before[i], proposed[i], anatomy_backend, floor_i,
                    clearance_tolerance=clearance_tolerance)
                if used:
                    proposed[i] = cand
                    stats["normal_removed"].append(removed)
                    stats["tangent_kept"].append(tlen)

        if use_exact:
            q = anatomy_backend.exact_closest(proposed[i])
            if anatomy_constraint.is_clearance_satisfied(
                    q["distance"], floor_i, clearance_tolerance):
                continue
            sol = anatomy_constraint.enforce_anatomy_clearance(
                proposed[i], anatomy_backend, floor_i,
                max_constraint_iterations=max_constraint_iterations,
                clearance_tolerance=clearance_tolerance, skin_normal=nrm)
            corr = vec_length(vec_sub(proposed[i], sol["position"]))
            proposed[i] = sol["position"]
            stats["constraint_events"] += 1
            stats["constrained_vertices"].add(i)
            stats["corrections"].append(corr)
            stats["projection_steps"] += sol["iterations"]
            if sol["iterations"] > stats["max_proj_used"]:
                stats["max_proj_used"] = sol["iterations"]
            if sol["unresolved"]:
                stats["unresolved"] += 1
        else:
            p2, ok, niter, _res = _legacy_iterative_sdf_push(
                proposed[i], sdf_query_fn, floor_i,
                max_constraint_iterations, clearance_tolerance)
            corr = vec_length(vec_sub(proposed[i], p2))
            if corr > 0.0:
                proposed[i] = p2
                stats["constraint_events"] += 1
                stats["constrained_vertices"].add(i)
                stats["corrections"].append(corr)
                stats["projection_steps"] += niter
                if niter > stats["max_proj_used"]:
                    stats["max_proj_used"] = niter
                if not ok:
                    stats["unresolved"] += 1
    return proposed, stats


def _local_intersection_report(
        mesh_name, positions, moved, anatomy_backend, neighbors, skin_topology,
        boundary_buffer_rings, intersection_tolerance):
    """Existing local (moved verts + 1 ring) surface-intersection query."""
    return anatomy_constraint.analyze_skin_anatomy_intersections(
        mesh_name, skin_indices=moved, backend=anatomy_backend,
        positions=positions, neighbors=neighbors,
        skin_topology=skin_topology,
        candidate_growth_rings=1,
        boundary_buffer_rings=boundary_buffer_rings,
        intersection_tolerance=intersection_tolerance,
        detailed=False, verbose=False)


def _legacy_repair_or_rollback_intersections(
        proposed, before_it, moved, active, anatomy_backend, mesh_name,
        neighbors, normals, skin_topology, min_clearance, clearance_tolerance,
        max_intersection_repair_iterations, intersection_repair_step_ratio,
        boundary_buffer_rings, intersection_tolerance, repair_mode,
        repair_blend_rings, repair_ring_weights, repair_binary_search,
        max_surface_repair_passes, max_repair_displacement_ratio,
        clearance_policy):
    """Historical per-iteration V2-repair-then-full-rollback path.

    Preserved for ``unsafe_step_policy="rollback"``. Any remaining local
    ``intersection_pair_count > 0`` after repair rolls the iteration back.
    Does not distinguish pre-existing vs new intersections.
    """
    irep = anatomy_constraint.resolve_skin_anatomy_intersections(
        proposed, moved, anatomy_backend, skin_mesh=mesh_name,
        neighbors=neighbors, normals=normals,
        skin_topology=skin_topology, min_clearance=min_clearance,
        clearance_tolerance=clearance_tolerance,
        max_intersection_repair_iterations=min(
            5, max_intersection_repair_iterations),
        intersection_repair_step_ratio=intersection_repair_step_ratio,
        intersection_repair_growth_rings=0,
        boundary_buffer_rings=boundary_buffer_rings,
        intersection_tolerance=intersection_tolerance,
        binary_search_min_step=False, verbose=False,
        repair_mode=repair_mode,
        repair_blend_rings=min(1, int(repair_blend_rings or 0)),
        repair_ring_weights=repair_ring_weights,
        repair_binary_search=repair_binary_search,
        max_surface_repair_passes=min(
            2, int(max_surface_repair_passes or 2)),
        max_repair_displacement_ratio=max_repair_displacement_ratio,
        post_repair_relax=False,
        clearance_policy=clearance_policy)
    proposed = irep["positions"]
    after_loc = irep.get("report_after") or {}
    if after_loc.get("intersection_pair_count", 0) <= 0:
        return proposed, "repaired"
    core = anatomy_constraint.intersection_vertices_from_report(
        after_loc, skin_topology, allowed=set(active))
    for i in core:
        proposed[i] = list(before_it[i])
    still = _local_intersection_report(
        mesh_name, proposed, moved, anatomy_backend, neighbors, skin_topology,
        boundary_buffer_rings, intersection_tolerance)
    if still.get("intersection_pair_count", 0) > 0:
        for i in moved:
            proposed[i] = list(before_it[i])
    return proposed, "rejected"


def constrained_smooth_mesh_region(mesh_name, indices, sdf_query_fn, min_clearance,
                                   method="laplacian", strength=0.2, iterations=20,
                                   constraint_interval=1, boundary_feather_rings=0,
                                   max_step_edge_ratio=None, neighbors=None,
                                   normals=None, apply=True, verbose=True,
                                   report_interval=10, verbose_iterations=False,
                                   anatomy_backend=None,
                                   constraint_solver="iterative_exact",
                                   resolve_initial_penetration=True,
                                   penetration_mode="auto",
                                   clearance_policy="preserve_valid_baseline",
                                   max_constraint_iterations=8,
                                   max_escape_iterations=10,
                                   clearance_tolerance=1e-4,
                                   prevent_segment_crossing=True,
                                   penetration_repair_feather_rings=0,
                                   preserve_tangential=True,
                                   resolve_initial_surface_intersections=True,
                                   prevent_surface_intersections=True,
                                   surface_intersection_check_interval=1,
                                   max_intersection_repair_iterations=20,
                                   intersection_repair_step_ratio=0.10,
                                   intersection_repair_growth_rings=0,
                                   skin_topology=None,
                                   intersection_tolerance=1e-6,
                                   boundary_buffer_rings=1,
                                   repair_mode="anatomy_supported_patch",
                                   repair_blend_rings=None,
                                   repair_ring_weights=(1.0, 0.6, 0.3),
                                   repair_binary_search=True,
                                   max_surface_repair_passes=5,
                                   max_repair_displacement_ratio=1.0,
                                   post_repair_relax=False,
                                   unsafe_step_policy=DEFAULT_UNSAFE_STEP_POLICY,
                                   smoothing_line_search_steps=DEFAULT_SMOOTHING_LINE_SEARCH_STEPS,
                                   min_smoothing_alpha=DEFAULT_MIN_SMOOTHING_ALPHA,
                                   line_search_surface_check=True,
                                   safety_mode=DEFAULT_SAFETY_MODE,
                                   allow_safety_decline=None,
                                   safety_batch_iterations=DEFAULT_SAFETY_BATCH_ITERATIONS,
                                   max_batches=None,
                                   max_unsafe_batch_displacement_ratio=DEFAULT_MAX_UNSAFE_BATCH_DISPLACEMENT_RATIO,
                                   redetect_m5_between_batches=False,
                                   repair_profile=None,
                                   repair_falloff=None,
                                   repair_field_iterations=None,
                                   repair_boundary_mode=None,
                                   repair_field_method=None):
    """Anatomy-constrained localized smoothing (see section header).

    Parameters
    ----------
    mesh_name : str
        Skin mesh to smooth (name preserved; not renamed).
    indices : iterable[int]
        The M5-selected region to smooth. Non-selected vertices stay fixed.
    sdf_query_fn : callable
        ``sdf_query_fn(point) -> (distance, closest_point, outward_dir)`` -- the
        registration smooth-min field. Exact safety uses ``anatomy_backend``
        when provided. Legacy one-shot mode still uses this field only.
    min_clearance : float
        Global requested anatomy floor. Required (no world-unit default).
        Under ``clearance_policy="preserve_valid_baseline"`` a non-penetrating
        vertex whose original exact distance is already below this value keeps
        that original distance as its personal floor.
    constraint_solver : {"iterative_exact", "legacy_single_push"}
        ``iterative_exact`` (default) re-queries exact closest-point distance
        after every push. ``legacy_single_push`` reproduces the historical
        one-shot ``cp + outward * clearance`` against the smooth-min field.
    resolve_initial_penetration : bool
        If True and an anatomy backend is available, high-confidence
        penetrating vertices are repaired BEFORE Laplacian smoothing starts.
    preserve_tangential : bool
        If True (iterative solver only), drop the inward normal component of a
        violating step and keep the tangential part when a closest-face normal
        is available.
    apply : bool
        If True, write the result and re-read to verify it persisted.
        If False, compute metrics only -- the scene is NOT modified.
    unsafe_step_policy : {"largest_safe_fraction", "rollback"}
        How to respond when the FULL Laplacian proposal creates a new
        skin/anatomy surface intersection. ``rollback`` reproduces the
        historical all-or-nothing reject (20/20 stalled). The default
        ``largest_safe_fraction`` binary-searches a single patch-wide alpha
        and accepts the largest safe fraction of that same Laplacian field.
    safety_mode : {"strict", "repair_after_batch", "off"}
        Smoothing-call safety policy only (standalone V2 repair is unchanged).
        ``strict`` never accepts a new forbidden intersection; ``repair_after_batch``
        allows temporary intersections for ``safety_batch_iterations`` then runs
        V2 repair; ``off`` is ordinary localized smoothing plus diagnostics.
    """
    if min_clearance is None:
        raise ValueError("min_clearance is required (no world-unit default is guessed)")

    solver = constraint_solver or "iterative_exact"
    if solver not in ("iterative_exact", "legacy_single_push"):
        raise ValueError("constraint_solver must be 'iterative_exact' or 'legacy_single_push'")
    step_policy = unsafe_step_policy or DEFAULT_UNSAFE_STEP_POLICY
    if step_policy not in ("largest_safe_fraction", "rollback"):
        raise ValueError("unsafe_step_policy must be 'largest_safe_fraction' or 'rollback'")
    mode_safety = _normalize_safety_mode(safety_mode, allow_safety_decline)

    current = get_mesh_vertices(mesh_name)
    if not current:
        print("[smoothing_utils] mesh '{0}' has no vertices / not found".format(mesh_name))
        return {"selected_vertex_count": 0}
    n = len(current)
    if neighbors is None:
        neighbors = get_vertex_neighbors(mesh_name)

    region = sorted(set(i for i in indices if 0 <= i < n))
    if not region:
        print("[smoothing_utils] no valid vertices to smooth")
        return {"selected_vertex_count": 0}

    if boundary_feather_rings and boundary_feather_rings > 0:
        active = sorted(set(grow_indices(neighbors, region, rings=boundary_feather_rings)))
    else:
        active = list(region)

    if normals is None:
        normals = get_vertex_normals(mesh_name)

    before = [list(v) for v in current]
    use_exact = anatomy_backend is not None and solver == "iterative_exact"
    do_prerepair = bool(resolve_initial_penetration) and anatomy_backend is not None
    do_segment = bool(prevent_segment_crossing) and anatomy_backend is not None
    do_tangent = bool(preserve_tangential) and use_exact
    do_isect_prerepair = (bool(resolve_initial_surface_intersections)
                          and anatomy_backend is not None)
    do_isect_prevent = (bool(prevent_surface_intersections)
                        and anatomy_backend is not None)
    if mode_safety == "off":
        do_prerepair = False
        do_isect_prerepair = False
        do_isect_prevent = False
        do_segment = False
        do_tangent = False
        use_exact = False
        if verbose:
            print("[constrained-smooth] WARNING: SAFETY DISABLED - "
                  "GEOMETRY MAY PENETRATE ANATOMY")
    elif mode_safety == "repair_after_batch":
        do_isect_prevent = False
        do_segment = False
        do_tangent = False
        use_exact = False
        if verbose:
            print("[constrained-smooth] safety_mode=repair_after_batch "
                  "batch_iters={0} (no per-iteration intersection reject)"
                  .format(int(safety_batch_iterations or DEFAULT_SAFETY_BATCH_ITERATIONS)))
        if redetect_m5_between_batches:
            print("[constrained-smooth] redetect_m5_between_batches=True is "
                  "ignored here (would couple smoothing_utils to artifact_detection)")

    if skin_topology is None and (do_isect_prerepair or do_isect_prevent
                                  or anatomy_backend is not None):
        try:
            fn = get_mesh_fn(mesh_name)
            if fn is not None:
                skin_topology = get_triangle_topology(fn)
        except Exception:
            skin_topology = None

    local_edge = {}
    for i in active:
        nbrs = neighbors[i] if i < len(neighbors) else []
        local_edge[i] = (sum(vec_length(vec_sub(before[i], before[j])) for j in nbrs)
                         / len(nbrs)) if nbrs else 0.0

    unconstrained = _run_smoothing(before, neighbors, method, strength, active, iterations)

    original_class = {}
    pen_before = {"penetrating_count": 0, "likely_penetrating_count": 0,
                  "unknown_count": 0, "below_clearance_count": 0,
                  "minimum_exact_distance": float("inf"),
                  "mean_exact_distance": 0.0}
    if anatomy_backend is not None:
        orig_report = anatomy_constraint.analyze_skin_anatomy_penetration(
            mesh_name, indices=region, min_clearance=min_clearance,
            backend=anatomy_backend, positions=before, normals=normals,
            clearance_tolerance=clearance_tolerance, detailed=True, verbose=False)
        pen_before = orig_report
        if orig_report.get("details_by_vertex"):
            original_class = orig_report["details_by_vertex"]

    floors = {i: float(min_clearance) for i in active}
    original_exact = {}
    policy_used = clearance_policy or "preserve_valid_baseline"
    if anatomy_backend is not None:
        floors, original_exact, policy_used = anatomy_constraint.compute_clearance_floors(
            before, active, anatomy_backend, min_clearance,
            clearance_policy=policy_used, classifications=original_class,
            clearance_tolerance=clearance_tolerance)
    elif solver == "legacy_single_push":
        policy_used = "global"

    work = [list(v) for v in before]
    repair_metrics = {
        "penetration_count_before": pen_before.get("penetrating_count", 0),
        "likely_penetration_count_before": pen_before.get("likely_penetrating_count", 0),
        "penetration_vertices_repaired": [],
        "penetration_vertices_unresolved": [],
        "mean_escape_displacement": 0.0,
        "max_escape_displacement": 0.0,
        "penetration_count_after_prerepair": pen_before.get("penetrating_count", 0),
        "likely_penetration_count_after_prerepair": pen_before.get("likely_penetrating_count", 0),
    }
    if do_prerepair and penetration_mode != "off":
        repair = anatomy_constraint.resolve_skin_anatomy_penetrations(
            work, region, anatomy_backend, min_clearance=min_clearance,
            normals=normals, neighbors=neighbors,
            max_escape_iterations=max_escape_iterations,
            clearance_tolerance=clearance_tolerance,
            penetration_repair_feather_rings=penetration_repair_feather_rings,
            classifications=original_class, verbose=verbose)
        work = repair["positions"]
        repair_metrics["penetration_vertices_repaired"] = repair["penetration_vertices_repaired"]
        repair_metrics["penetration_vertices_unresolved"] = repair["penetration_vertices_unresolved"]
        repair_metrics["mean_escape_displacement"] = repair["mean_escape_displacement"]
        repair_metrics["max_escape_displacement"] = repair["max_escape_displacement"]
        post = anatomy_constraint.analyze_skin_anatomy_penetration(
            mesh_name, indices=region, min_clearance=min_clearance,
            backend=anatomy_backend, positions=work, normals=normals,
            clearance_tolerance=clearance_tolerance, detailed=False, verbose=False)
        repair_metrics["penetration_count_after_prerepair"] = post.get("penetrating_count", 0)
        repair_metrics["likely_penetration_count_after_prerepair"] = post.get(
            "likely_penetrating_count", 0)
        for i in repair["penetration_vertices_repaired"]:
            floors[i] = float(min_clearance)

    after_prerepair = [list(v) for v in work]

    isect_metrics = {
        "intersecting_skin_face_count_before": 0,
        "intersecting_skin_vertex_count_before": 0,
        "intersection_pair_count_before": 0,
        "intersection_repair_iterations": 0,
        "intersection_vertices_moved": [],
        "intersection_faces_resolved": 0,
        "intersection_faces_unresolved": 0,
        "mean_intersection_repair_displacement": 0.0,
        "max_intersection_repair_displacement": 0.0,
        "intersecting_skin_face_count_after_prerepair": 0,
        "intersections_by_anatomy_mesh_before": {},
        "new_intersections_detected_during_smoothing": 0,
        "new_intersections_prevented": 0,
        "unsafe_iterations_repaired": 0,
        "unsafe_iterations_rejected": 0,
        "full_steps_safe": 0,
        "full_steps_unsafe": 0,
        "partial_steps_accepted": 0,
        "stalled_steps": 0,
        "line_search_evaluations": 0,
        "accepted_alpha_per_iteration": [],
        "unsafe_full_steps_recovered_by_line_search": 0,
        "full_step_proposed_displacements": [],
        "partial_step_displacements": [],
        "unsafe_step_policy": step_policy,
        "safety_mode": mode_safety,
        "safety_disabled": mode_safety == "off",
        "safety_batch_iterations": int(
            safety_batch_iterations or DEFAULT_SAFETY_BATCH_ITERATIONS),
        "batch_count": 0,
        "batch_reports": [],
        "batch_aborted": False,
        "batch_abort_reason": None,
    }
    if anatomy_backend is not None:
        isect0 = anatomy_constraint.analyze_skin_anatomy_intersections(
            mesh_name, skin_indices=region, backend=anatomy_backend,
            positions=before, neighbors=neighbors, skin_topology=skin_topology,
            boundary_buffer_rings=boundary_buffer_rings,
            intersection_tolerance=intersection_tolerance,
            detailed=False, verbose=False)
        isect_metrics["intersecting_skin_face_count_before"] = isect0.get(
            "intersecting_skin_face_count", 0)
        isect_metrics["intersecting_skin_vertex_count_before"] = isect0.get(
            "intersecting_skin_vertex_count", 0)
        isect_metrics["intersection_pair_count_before"] = isect0.get(
            "intersection_pair_count", 0)
        isect_metrics["intersections_by_anatomy_mesh_before"] = isect0.get(
            "intersections_by_anatomy_mesh") or {}
        isect_metrics["intersecting_skin_face_count_after_prerepair"] = isect0.get(
            "intersecting_skin_face_count", 0)

    if do_isect_prerepair:
        irep = anatomy_constraint.resolve_skin_anatomy_intersections(
            work, region, anatomy_backend, skin_mesh=mesh_name,
            neighbors=neighbors, normals=normals, skin_topology=skin_topology,
            min_clearance=min_clearance, clearance_tolerance=clearance_tolerance,
            max_intersection_repair_iterations=max_intersection_repair_iterations,
            intersection_repair_step_ratio=intersection_repair_step_ratio,
            intersection_repair_growth_rings=intersection_repair_growth_rings,
            boundary_buffer_rings=boundary_buffer_rings,
            intersection_tolerance=intersection_tolerance, verbose=verbose,
            repair_mode=repair_mode, repair_blend_rings=repair_blend_rings,
            repair_ring_weights=repair_ring_weights,
            repair_binary_search=repair_binary_search,
            max_surface_repair_passes=max_surface_repair_passes,
            max_repair_displacement_ratio=max_repair_displacement_ratio,
            post_repair_relax=post_repair_relax,
            clearance_policy=clearance_policy,
            repair_profile=repair_profile,
            repair_falloff=repair_falloff,
            repair_field_iterations=repair_field_iterations,
            repair_boundary_mode=repair_boundary_mode,
            repair_field_method=repair_field_method)
        work = irep["positions"]
        isect_metrics["intersection_repair_iterations"] = irep.get(
            "intersection_repair_iterations", 0)
        isect_metrics["intersection_vertices_moved"] = irep.get(
            "intersection_vertices_moved") or []
        isect_metrics["intersection_faces_resolved"] = irep.get(
            "intersection_faces_resolved", 0)
        isect_metrics["intersection_faces_unresolved"] = irep.get(
            "intersection_faces_unresolved", 0)
        isect_metrics["mean_intersection_repair_displacement"] = irep.get(
            "mean_intersection_repair_displacement", 0.0)
        isect_metrics["max_intersection_repair_displacement"] = irep.get(
            "max_intersection_repair_displacement", 0.0)
        isect_metrics["repair_mode"] = irep.get("repair_mode")
        isect_metrics["core_vertex_count"] = irep.get("core_vertex_count", 0)
        isect_metrics["patch_vertex_count"] = irep.get("patch_vertex_count", 0)
        isect_metrics["mean_displacement_gradient"] = irep.get(
            "mean_displacement_gradient", 0.0)
        isect_metrics["max_displacement_gradient"] = irep.get(
            "max_displacement_gradient", 0.0)
        after_i = irep.get("report_after") or {}
        isect_metrics["intersecting_skin_face_count_after_prerepair"] = after_i.get(
            "intersecting_skin_face_count", 0)
        after_prerepair = [list(v) for v in work]

    constrained_vertices = set()
    constraint_events = 0
    corrections = []
    projection_steps = 0
    max_proj_used = 0
    unresolved_constraints = 0
    segment_detected = 0
    segment_prevented = 0
    normal_removed = []
    tangent_kept = []
    run_strict = mode_safety == "strict"

    if mode_safety == "off":
        n_off = max(1, int(iterations))
        before_off = [list(v) for v in work]
        faces_b, _pairs_b = _count_region_intersections(
            mesh_name, before_off, region, anatomy_backend, neighbors,
            skin_topology, boundary_buffer_rings, intersection_tolerance)
        work = _unconstrained_smoothing_steps(
            work, neighbors, method, strength, active, n_off,
            max_step_edge_ratio, local_edge)
        faces_a, _pairs_a = _count_region_intersections(
            mesh_name, work, region, anatomy_backend, neighbors,
            skin_topology, boundary_buffer_rings, intersection_tolerance)
        off_disp = _region_disp_stats(before_off, work, active)
        isect_metrics["intersecting_skin_face_count_after_smoothing"] = faces_a
        isect_metrics["safety_decline"] = int(faces_a) - int(faces_b)
        isect_metrics["off_smoothing_displacement_mean"] = off_disp["mean"]
        isect_metrics["off_smoothing_displacement_max"] = off_disp["max"]
        for _it in range(n_off):
            isect_metrics["accepted_alpha_per_iteration"].append(1.0)
            isect_metrics["full_step_proposed_displacements"].append(off_disp["mean"])
        if verbose:
            print("[constrained-smooth] SAFETY DISABLED | faces {0}->{1} "
                  "(decline={2}) mean/max disp={3:.5f}/{4:.5f}".format(
                      faces_b, faces_a, isect_metrics["safety_decline"],
                      off_disp["mean"], off_disp["max"]))

    elif mode_safety == "repair_after_batch":
        remaining = max(0, int(iterations))
        batch_n = max(1, int(safety_batch_iterations or DEFAULT_SAFETY_BATCH_ITERATIONS))
        cap_ratio = max_unsafe_batch_displacement_ratio
        n_done = 0
        while remaining > 0:
            if max_batches is not None and n_done >= int(max_batches):
                break
            n_this = min(batch_n, remaining)
            before_batch = [list(v) for v in work]
            faces_b, _pb = _count_region_intersections(
                mesh_name, before_batch, region, anatomy_backend, neighbors,
                skin_topology, boundary_buffer_rings, intersection_tolerance)
            trial = _unconstrained_smoothing_steps(
                work, neighbors, method, strength, active, n_this,
                max_step_edge_ratio, local_edge)
            aborted = False
            if cap_ratio is not None:
                for i in active:
                    mag = vec_length(vec_sub(trial[i], before_batch[i]))
                    edge = local_edge.get(i, 0.0)
                    if edge > 1e-12 and mag > float(cap_ratio) * edge:
                        aborted = True
                        break
            if aborted:
                isect_metrics["batch_aborted"] = True
                isect_metrics["batch_abort_reason"] = (
                    "max_unsafe_batch_displacement_ratio exceeded")
                if verbose:
                    print("[constrained-smooth] WARNING: aborting remaining "
                          "batches; a vertex moved more than {0} x local edge"
                          .format(cap_ratio))
                break
            work = trial
            faces_s, _ps = _count_region_intersections(
                mesh_name, work, region, anatomy_backend, neighbors,
                skin_topology, boundary_buffer_rings, intersection_tolerance)
            smooth_disp = _region_disp_stats(before_batch, work, active)
            before_repair = [list(v) for v in work]
            if anatomy_backend is not None:
                irep = anatomy_constraint.resolve_skin_anatomy_intersections(
                    work, region, anatomy_backend, skin_mesh=mesh_name,
                    neighbors=neighbors, normals=normals,
                    skin_topology=skin_topology, min_clearance=min_clearance,
                    clearance_tolerance=clearance_tolerance,
                    max_intersection_repair_iterations=max_intersection_repair_iterations,
                    intersection_repair_step_ratio=intersection_repair_step_ratio,
                    intersection_repair_growth_rings=intersection_repair_growth_rings,
                    boundary_buffer_rings=boundary_buffer_rings,
                    intersection_tolerance=intersection_tolerance,
                    verbose=verbose, repair_mode=repair_mode,
                    repair_blend_rings=repair_blend_rings,
                    repair_ring_weights=repair_ring_weights,
                    repair_binary_search=repair_binary_search,
                    max_surface_repair_passes=max_surface_repair_passes,
                    max_repair_displacement_ratio=max_repair_displacement_ratio,
                    post_repair_relax=post_repair_relax,
                    clearance_policy=clearance_policy,
                    repair_profile=repair_profile,
                    repair_falloff=repair_falloff,
                    repair_field_iterations=repair_field_iterations,
                    repair_boundary_mode=repair_boundary_mode,
                    repair_field_method=repair_field_method)
                work = irep["positions"]
                faces_r = irep.get("intersecting_faces_after", 0)
                unresolved = irep.get("intersection_faces_unresolved", 0)
                isect_metrics["intersection_repair_iterations"] += irep.get(
                    "intersection_repair_iterations", 0)
                isect_metrics["mean_intersection_repair_displacement"] = irep.get(
                    "mean_intersection_repair_displacement", 0.0)
                isect_metrics["max_intersection_repair_displacement"] = irep.get(
                    "max_intersection_repair_displacement", 0.0)
            else:
                faces_r = faces_s
                unresolved = 0
            repair_disp = _region_disp_stats(before_repair, work, active)
            rec = {
                "smoothing_iterations": n_this,
                "smoothing_displacement": smooth_disp["mean"],
                "smoothing_displacement_max": smooth_disp["max"],
                "intersections_before": faces_b,
                "intersections_after_smoothing": faces_s,
                "intersections_after_repair": faces_r,
                "repair_displacement": repair_disp["mean"],
                "repair_displacement_max": repair_disp["max"],
                "unresolved_intersections": unresolved,
            }
            isect_metrics["batch_reports"].append(rec)
            n_done += 1
            remaining -= n_this
            for _it in range(n_this):
                isect_metrics["accepted_alpha_per_iteration"].append(1.0)
                isect_metrics["full_step_proposed_displacements"].append(
                    smooth_disp["mean"])
            if verbose:
                print("[constrained-smooth] batch {0}: smooth {1} iters "
                      "faces {2}->{3} then repair -> {4}  "
                      "smooth_disp={5:.5f} repair_disp={6:.5f}".format(
                          n_done, n_this, faces_b, faces_s, faces_r,
                          smooth_disp["mean"], repair_disp["mean"]))
        isect_metrics["batch_count"] = n_done

    it = 0
    n_strict = max(1, int(iterations))
    while run_strict and it < n_strict:
        before_it = [list(v) for v in work]
        laplacian_full = _run_smoothing(work, neighbors, method, strength, active, 1)

        if max_step_edge_ratio is not None:
            for i in active:
                cap = max_step_edge_ratio * local_edge.get(i, 0.0)
                if cap <= 0:
                    continue
                step = vec_sub(laplacian_full[i], work[i])
                slen = vec_length(step)
                if slen > cap and slen > 1e-12:
                    laplacian_full[i] = vec_add(work[i], vec_scale(step, cap / slen))

        step_mags = [vec_length(vec_sub(laplacian_full[i], before_it[i])) for i in active]
        isect_metrics["full_step_proposed_displacements"].append(
            (sum(step_mags) / len(step_mags)) if step_mags else 0.0)

        constrain_now = constraint_interval <= 1 or (it % constraint_interval == 0)
        moved = [i for i in active
                 if vec_length(vec_sub(laplacian_full[i], before_it[i])) > 1e-12]

        def _project(before_pos, cand_pos):
            return _apply_iteration_constraints(
                before_pos, cand_pos, active, anatomy_backend, sdf_query_fn,
                floors, min_clearance, normals, solver, use_exact, do_segment,
                do_tangent, constrain_now, max_constraint_iterations,
                clearance_tolerance)[0]

        isect_now = (do_isect_prevent and (
            surface_intersection_check_interval <= 1
            or (it % surface_intersection_check_interval == 0)))
        do_line_search = (
            step_policy == "largest_safe_fraction"
            and isect_now and skin_topology is not None
            and bool(line_search_surface_check)
            and moved)

        accepted_alpha = 1.0
        if do_line_search:
            baseline = _local_intersection_report(
                mesh_name, before_it, moved, anatomy_backend, neighbors,
                skin_topology, boundary_buffer_rings, intersection_tolerance)
            baseline_faces = set(baseline.get("intersecting_skin_faces") or [])
            baseline_pairs = int(baseline.get("intersection_pair_count", 0) or 0)

            def _is_safe(projected):
                rep = _local_intersection_report(
                    mesh_name, projected, moved, anatomy_backend, neighbors,
                    skin_topology, boundary_buffer_rings,
                    intersection_tolerance)
                ok, new_faces = _smoothing_candidate_is_safe(
                    rep, baseline_faces, baseline_pairs)
                return ok, {"report": rep, "new_faces": new_faces}

            ls = _find_largest_safe_smoothing_step(
                before_it, laplacian_full, moved, _is_safe,
                steps=smoothing_line_search_steps,
                min_alpha=min_smoothing_alpha,
                project_fn=_project)
            isect_metrics["line_search_evaluations"] += ls["evaluations"]
            accepted_alpha = ls["alpha"]
            proposed = ls["positions"]
            if ls["status"] == "full":
                isect_metrics["full_steps_safe"] += 1
            else:
                isect_metrics["full_steps_unsafe"] += 1
                isect_metrics["new_intersections_detected_during_smoothing"] += 1
                isect_metrics["new_intersections_prevented"] += 1
                if ls["status"] == "partial":
                    isect_metrics["partial_steps_accepted"] += 1
                    isect_metrics["unsafe_full_steps_recovered_by_line_search"] += 1
                    pd = [vec_length(vec_sub(proposed[i], before_it[i]))
                          for i in active]
                    isect_metrics["partial_step_displacements"].append(
                        (sum(pd) / len(pd)) if pd else 0.0)
                else:
                    isect_metrics["stalled_steps"] += 1
                    isect_metrics["unsafe_iterations_rejected"] += 1
                    proposed = [list(p) for p in before_it]
                    accepted_alpha = 0.0
            if ls["status"] != "stalled_no_safe_step":
                _accepted, acc_stats = _apply_iteration_constraints(
                    before_it, _scale_displacement_field(
                        before_it, laplacian_full, accepted_alpha, moved),
                    active, anatomy_backend, sdf_query_fn, floors, min_clearance,
                    normals, solver, use_exact, do_segment, do_tangent,
                    constrain_now, max_constraint_iterations, clearance_tolerance)
                proposed = _accepted
                segment_detected += acc_stats["segment_detected"]
                segment_prevented += acc_stats["segment_prevented"]
                constraint_events += acc_stats["constraint_events"]
                constrained_vertices |= acc_stats["constrained_vertices"]
                corrections.extend(acc_stats["corrections"])
                projection_steps += acc_stats["projection_steps"]
                if acc_stats["max_proj_used"] > max_proj_used:
                    max_proj_used = acc_stats["max_proj_used"]
                unresolved_constraints += acc_stats["unresolved"]
                normal_removed.extend(acc_stats["normal_removed"])
                tangent_kept.extend(acc_stats["tangent_kept"])
        else:
            proposed, acc_stats = _apply_iteration_constraints(
                before_it, laplacian_full, active, anatomy_backend,
                sdf_query_fn, floors, min_clearance, normals, solver,
                use_exact, do_segment, do_tangent, constrain_now,
                max_constraint_iterations, clearance_tolerance)
            segment_detected += acc_stats["segment_detected"]
            segment_prevented += acc_stats["segment_prevented"]
            constraint_events += acc_stats["constraint_events"]
            constrained_vertices |= acc_stats["constrained_vertices"]
            corrections.extend(acc_stats["corrections"])
            projection_steps += acc_stats["projection_steps"]
            if acc_stats["max_proj_used"] > max_proj_used:
                max_proj_used = acc_stats["max_proj_used"]
            unresolved_constraints += acc_stats["unresolved"]
            normal_removed.extend(acc_stats["normal_removed"])
            tangent_kept.extend(acc_stats["tangent_kept"])

            if (step_policy == "rollback" and isect_now
                    and skin_topology is not None):
                moved_r = [i for i in active
                           if vec_length(vec_sub(proposed[i], before_it[i])) > 1e-12]
                if moved_r:
                    loc = _local_intersection_report(
                        mesh_name, proposed, moved_r, anatomy_backend,
                        neighbors, skin_topology, boundary_buffer_rings,
                        intersection_tolerance)
                    if loc.get("intersection_pair_count", 0) > 0:
                        isect_metrics["full_steps_unsafe"] += 1
                        isect_metrics["new_intersections_detected_during_smoothing"] += 1
                        proposed, outcome = _legacy_repair_or_rollback_intersections(
                            proposed, before_it, moved_r, active,
                            anatomy_backend, mesh_name, neighbors, normals,
                            skin_topology, min_clearance, clearance_tolerance,
                            max_intersection_repair_iterations,
                            intersection_repair_step_ratio,
                            boundary_buffer_rings, intersection_tolerance,
                            repair_mode, repair_blend_rings,
                            repair_ring_weights, repair_binary_search,
                            max_surface_repair_passes,
                            max_repair_displacement_ratio, clearance_policy)
                        isect_metrics["new_intersections_prevented"] += 1
                        if outcome == "repaired":
                            isect_metrics["unsafe_iterations_repaired"] += 1
                            accepted_alpha = 1.0
                        else:
                            isect_metrics["unsafe_iterations_rejected"] += 1
                            isect_metrics["stalled_steps"] += 1
                            accepted_alpha = 0.0
                    else:
                        isect_metrics["full_steps_safe"] += 1
                else:
                    isect_metrics["full_steps_safe"] += 1
            else:
                isect_metrics["full_steps_safe"] += 1

        isect_metrics["accepted_alpha_per_iteration"].append(accepted_alpha)
        work = proposed

        if verbose_iterations and (it % max(1, report_interval) == 0):
            sd = [vec_length(vec_sub(work[i], before[i])) for i in active]
            if use_exact:
                minclr = min((anatomy_backend.exact_closest(work[i])["distance"]
                              for i in active), default=0.0)
            else:
                minclr = min((sdf_query_fn(work[i])[0] for i in active), default=0.0)
            print("  [iter {0:4d}] alpha={1:.4f} applied mean disp={2:.5f} "
                  "constrained so far={3} min clearance={4:.4f}".format(
                      it, accepted_alpha,
                      (sum(sd) / len(sd)) if sd else 0.0,
                      len(constrained_vertices), minclr))
        it += 1

    final = work

    ps = _sc_stats([vec_length(vec_sub(unconstrained[i], before[i])) for i in region])
    aps = _sc_stats([vec_length(vec_sub(final[i], before[i])) for i in region])
    sdf_before = [sdf_query_fn(before[i])[0] for i in region]
    sdf_after = [sdf_query_fn(final[i])[0] for i in region]
    db, da = _sc_stats(sdf_before), _sc_stats(sdf_after)
    below_sdf_before = sum(1 for d in sdf_before if d < min_clearance)
    below_sdf_after = sum(1 for d in sdf_after if d < min_clearance)

    exact_before_vals = []
    exact_after_vals = []
    below_exact_before = 0
    below_exact_after = 0
    if anatomy_backend is not None:
        for i in region:
            d0 = original_exact.get(i)
            if d0 is None:
                d0 = anatomy_backend.exact_closest(before[i])["distance"]
            d1 = anatomy_backend.exact_closest(final[i])["distance"]
            exact_before_vals.append(d0)
            exact_after_vals.append(d1)
            floor_i = floors.get(i, min_clearance)
            if not anatomy_constraint.is_clearance_satisfied(d0, floor_i, clearance_tolerance):
                below_exact_before += 1
            if not anatomy_constraint.is_clearance_satisfied(d1, floor_i, clearance_tolerance):
                below_exact_after += 1
        eb, ea = _sc_stats(exact_before_vals), _sc_stats(exact_after_vals)
    else:
        eb, ea = {"min": 0.0, "mean": 0.0}, {"min": 0.0, "mean": 0.0}

    inward = outward_c = tangential = 0
    normal_disps = []
    eps = 1e-6
    for i in region:
        delta = vec_sub(final[i], before[i])
        nrm = normals[i] if (normals and i < len(normals)) else [0.0, 0.0, 0.0]
        nd = delta[0] * nrm[0] + delta[1] * nrm[1] + delta[2] * nrm[2]
        normal_disps.append(nd)
        if nd < -eps:
            inward += 1
        elif nd > eps:
            outward_c += 1
        else:
            tangential += 1
    mean_nd = (sum(normal_disps) / len(normal_disps)) if normal_disps else 0.0
    corr = _sc_stats(corrections)

    final_pen = {"penetrating_count": 0, "likely_penetrating_count": 0, "unknown_count": 0}
    final_isect = {"intersecting_skin_face_count": 0, "intersecting_skin_vertex_count": 0,
                   "intersection_pair_count": 0, "intersections_by_anatomy_mesh": {}}
    if anatomy_backend is not None:
        final_pen = anatomy_constraint.analyze_skin_anatomy_penetration(
            mesh_name, indices=region, min_clearance=min_clearance,
            backend=anatomy_backend, positions=final, normals=normals,
            clearance_tolerance=clearance_tolerance, detailed=False, verbose=False)
        final_isect = anatomy_constraint.analyze_skin_anatomy_intersections(
            mesh_name, skin_indices=region, backend=anatomy_backend,
            positions=final, neighbors=neighbors, skin_topology=skin_topology,
            boundary_buffer_rings=boundary_buffer_rings,
            intersection_tolerance=intersection_tolerance,
            detailed=False, verbose=False)

    metrics = {
        "mesh": mesh_name,
        "selected_vertex_count": len(region),
        "active_vertex_count": len(active),
        "iterations": iterations,
        "method": method,
        "strength": strength,
        "min_clearance": min_clearance,
        "constraint_interval": constraint_interval,
        "boundary_feather_rings": boundary_feather_rings,
        "constraint_solver": solver,
        "clearance_policy": policy_used,
        "clearance_tolerance": clearance_tolerance,
        "resolve_initial_penetration": bool(do_prerepair),
        "prevent_segment_crossing": bool(do_segment),
        "preserve_tangential": bool(do_tangent),
        "max_constraint_iterations": max_constraint_iterations,
        "proposed_mean_displacement": ps["mean"],
        "proposed_rms_displacement": ps["rms"],
        "proposed_max_displacement": ps["max"],
        "applied_mean_displacement": aps["mean"],
        "applied_rms_displacement": aps["rms"],
        "applied_max_displacement": aps["max"],
        "number_of_constraint_events": constraint_events,
        "total_constraint_events": constraint_events,
        "unique_vertices_constrained": len(constrained_vertices),
        "number_safe_without_constraint": len(active) - len(constrained_vertices),
        "number_clamped_or_pushed": len(constrained_vertices),
        "max_constraint_correction": corr["max"],
        "mean_constraint_correction": corr["mean"],
        "iterative_projection_steps": projection_steps,
        "max_projection_iterations_used": max_proj_used,
        "unresolved_constraint_count": unresolved_constraints,
        "segment_crossings_detected": segment_detected,
        "segment_crossings_prevented": segment_prevented,
        "anatomy_distance_before": {"min": db["min"], "mean": db["mean"]},
        "anatomy_distance_after": {"min": da["min"], "mean": da["mean"]},
        "vertices_below_clearance_before": below_sdf_before,
        "vertices_below_clearance_after": below_sdf_after,
        "exact_min_distance_before": eb["min"],
        "exact_mean_distance_before": eb["mean"],
        "exact_min_distance_after": ea["min"],
        "exact_mean_distance_after": ea["mean"],
        "below_clearance_before": (below_exact_before if anatomy_backend is not None
                                   else below_sdf_before),
        "below_clearance_after": (below_exact_after if anatomy_backend is not None
                                  else below_sdf_after),
        "direction": {"count_inward": inward, "count_outward": outward_c,
                      "count_tangential": tangential,
                      "mean_normal_displacement": mean_nd},
        "normal_component_removed_mean": (
            (sum(normal_removed) / len(normal_removed)) if normal_removed else 0.0),
        "tangential_component_preserved_mean": (
            (sum(tangent_kept) / len(tangent_kept)) if tangent_kept else 0.0),
        "penetration_count_before": repair_metrics["penetration_count_before"],
        "likely_penetration_count_before": repair_metrics["likely_penetration_count_before"],
        "penetration_vertices_repaired": repair_metrics["penetration_vertices_repaired"],
        "penetration_vertices_unresolved": repair_metrics["penetration_vertices_unresolved"],
        "mean_escape_displacement": repair_metrics["mean_escape_displacement"],
        "max_escape_displacement": repair_metrics["max_escape_displacement"],
        "penetration_count_after_prerepair": repair_metrics["penetration_count_after_prerepair"],
        "likely_penetration_count_after_prerepair": repair_metrics[
            "likely_penetration_count_after_prerepair"],
        "penetrating_count_after": final_pen.get("penetrating_count", 0),
        "likely_penetrating_count_after": final_pen.get("likely_penetrating_count", 0),
        "unknown_count_after": final_pen.get("unknown_count", 0),
        "unknown_count_before": pen_before.get("unknown_count", 0),
        "intersecting_skin_face_count_before": isect_metrics[
            "intersecting_skin_face_count_before"],
        "intersecting_skin_vertex_count_before": isect_metrics[
            "intersecting_skin_vertex_count_before"],
        "intersection_pair_count_before": isect_metrics["intersection_pair_count_before"],
        "intersection_repair_iterations": isect_metrics["intersection_repair_iterations"],
        "intersection_vertices_moved": isect_metrics["intersection_vertices_moved"],
        "intersection_faces_resolved": isect_metrics["intersection_faces_resolved"],
        "intersection_faces_unresolved": isect_metrics["intersection_faces_unresolved"],
        "mean_intersection_repair_displacement": isect_metrics[
            "mean_intersection_repair_displacement"],
        "max_intersection_repair_displacement": isect_metrics[
            "max_intersection_repair_displacement"],
        "intersecting_skin_face_count_after_prerepair": isect_metrics[
            "intersecting_skin_face_count_after_prerepair"],
        "new_intersections_detected_during_smoothing": isect_metrics[
            "new_intersections_detected_during_smoothing"],
        "new_intersections_prevented": isect_metrics["new_intersections_prevented"],
        "unsafe_iterations_repaired": isect_metrics["unsafe_iterations_repaired"],
        "unsafe_iterations_rejected": isect_metrics["unsafe_iterations_rejected"],
        "unsafe_step_policy": isect_metrics.get("unsafe_step_policy"),
        "smoothing_line_search_steps": int(smoothing_line_search_steps),
        "min_smoothing_alpha": float(min_smoothing_alpha),
        "full_steps_safe": isect_metrics["full_steps_safe"],
        "full_steps_unsafe": isect_metrics["full_steps_unsafe"],
        "partial_steps_accepted": isect_metrics["partial_steps_accepted"],
        "stalled_steps": isect_metrics["stalled_steps"],
        "line_search_evaluations": isect_metrics["line_search_evaluations"],
        "accepted_alpha_per_iteration": list(
            isect_metrics["accepted_alpha_per_iteration"]),
        "unsafe_full_steps_recovered_by_line_search": isect_metrics[
            "unsafe_full_steps_recovered_by_line_search"],
        "full_step_proposed_mean_displacement": (
            (sum(isect_metrics["full_step_proposed_displacements"])
             / len(isect_metrics["full_step_proposed_displacements"]))
            if isect_metrics["full_step_proposed_displacements"] else 0.0),
        "final_applied_mean_displacement": aps["mean"],
        "partial_step_total_displacement": sum(
            isect_metrics["partial_step_displacements"]),
        "mean_accepted_alpha": (
            (sum(isect_metrics["accepted_alpha_per_iteration"])
             / len(isect_metrics["accepted_alpha_per_iteration"]))
            if isect_metrics["accepted_alpha_per_iteration"] else 0.0),
        "min_accepted_alpha": (
            min(isect_metrics["accepted_alpha_per_iteration"])
            if isect_metrics["accepted_alpha_per_iteration"] else 0.0),
        "max_accepted_alpha": (
            max(isect_metrics["accepted_alpha_per_iteration"])
            if isect_metrics["accepted_alpha_per_iteration"] else 0.0),
        "intersecting_skin_face_count_after": final_isect.get(
            "intersecting_skin_face_count", 0),
        "intersecting_skin_vertex_count_after": final_isect.get(
            "intersecting_skin_vertex_count", 0),
        "intersection_pair_count_after": final_isect.get("intersection_pair_count", 0),
        "intersections_by_anatomy_mesh_before": isect_metrics[
            "intersections_by_anatomy_mesh_before"],
        "intersections_by_anatomy_mesh_after": final_isect.get(
            "intersections_by_anatomy_mesh") or {},
        "resolve_initial_surface_intersections": bool(do_isect_prerepair),
        "prevent_surface_intersections": bool(do_isect_prevent),
        "apply": bool(apply),
        "safety_mode": mode_safety,
        "safety_disabled": bool(mode_safety == "off"),
        "allow_safety_decline": bool(mode_safety == "repair_after_batch"),
        "safety_batch_iterations": isect_metrics.get("safety_batch_iterations"),
        "batch_count": isect_metrics.get("batch_count", 0),
        "batch_reports": list(isect_metrics.get("batch_reports") or []),
        "batch_aborted": bool(isect_metrics.get("batch_aborted")),
        "batch_abort_reason": isect_metrics.get("batch_abort_reason"),
        "safety_decline": isect_metrics.get("safety_decline", (
            int(final_isect.get("intersecting_skin_face_count", 0) or 0)
            - int(isect_metrics.get("intersecting_skin_face_count_before", 0) or 0))),
        "after_prerepair_moved": any(
            vec_length(vec_sub(after_prerepair[i], before[i])) > 0.0 for i in region),
    }

    if apply:
        set_mesh_vertices(mesh_name, final)
        fresh = get_mesh_vertices(mesh_name)
        ss = _sc_stats([vec_length(vec_sub(fresh[i], before[i])) for i in region]
                       if fresh else [])
        metrics["scene_applied_mean_displacement"] = ss["mean"]
        metrics["scene_applied_max_displacement"] = ss["max"]
        if aps["max"] > 1e-5 and ss["max"] < 1e-5:
            metrics["fresh_read_warning"] = (
                "computed movement did NOT persist in the scene -- likely "
                "construction history / a deformer re-evaluating, or the visible "
                "shape is a different node. (History is NOT deleted automatically.)")
            print("[smoothing_utils] WARNING: {0}".format(metrics["fresh_read_warning"]))

    if verbose:
        _print_constrained_report(metrics)
    return metrics


def _print_constrained_report(m):
    d = m["direction"]
    print("[constrained-smooth] '{0}' {1} verts | method={2} strength={3} iters={4} "
          "clearance={5} solver={6} policy={7} safety_mode={8}".format(
              m["mesh"], m["selected_vertex_count"], m["method"],
              m["strength"], m["iterations"], m["min_clearance"],
              m.get("constraint_solver"), m.get("clearance_policy"),
              m.get("safety_mode")))
    if m.get("safety_disabled"):
        print("  *** WARNING: SAFETY DISABLED - GEOMETRY MAY PENETRATE ANATOMY ***")
        print("  safety_decline (faces after-before) = {0}".format(
            m.get("safety_decline")))
    print("  proposed disp (unconstrained): mean={0:.5f} rms={1:.5f} max={2:.5f}".format(
        m["proposed_mean_displacement"], m["proposed_rms_displacement"],
        m["proposed_max_displacement"]))
    print("  applied  disp (constrained):   mean={0:.5f} rms={1:.5f} max={2:.5f}".format(
        m["applied_mean_displacement"], m["applied_rms_displacement"],
        m["applied_max_displacement"]))
    print("  constraint: {0} events on {1} unique verts ({2} safe w/o constraint) | "
          "correction mean={3:.5f} max={4:.5f}".format(
              m["number_of_constraint_events"], m["unique_vertices_constrained"],
              m["number_safe_without_constraint"], m["mean_constraint_correction"],
              m["max_constraint_correction"]))
    print("  iterative projection steps={0}  max_iters_used={1}  unresolved={2}".format(
        m.get("iterative_projection_steps", 0),
        m.get("max_projection_iterations_used", 0),
        m.get("unresolved_constraint_count", 0)))
    print("  segment crossings: detected={0} prevented={1}".format(
        m.get("segment_crossings_detected", 0), m.get("segment_crossings_prevented", 0)))
    print("  SDF (smooth-min) clearance: before min={0:.4f} mean={1:.4f} -> after min={2:.4f} "
          "mean={3:.4f}".format(m["anatomy_distance_before"]["min"],
                                m["anatomy_distance_before"]["mean"],
                                m["anatomy_distance_after"]["min"],
                                m["anatomy_distance_after"]["mean"]))
    print("  EXACT clearance: before min={0:.4f} mean={1:.4f} -> after min={2:.4f} "
          "mean={3:.4f}".format(m.get("exact_min_distance_before", 0.0),
                                m.get("exact_mean_distance_before", 0.0),
                                m.get("exact_min_distance_after", 0.0),
                                m.get("exact_mean_distance_after", 0.0)))
    print("  below-clearance (exact floor): before={0} -> after={1}".format(
        m.get("below_clearance_before"), m.get("below_clearance_after")))
    print("  penetration: before={0} likely={1} -> after_prerepair={2}/{3} "
          "-> final={4}/{5} unknown_final={6}".format(
              m.get("penetration_count_before"), m.get("likely_penetration_count_before"),
              m.get("penetration_count_after_prerepair"),
              m.get("likely_penetration_count_after_prerepair"),
              m.get("penetrating_count_after"), m.get("likely_penetrating_count_after"),
              m.get("unknown_count_after")))
    print("  pre-repair: repaired={0} unresolved={1} mean_escape={2:.5f} max_escape={3:.5f}".format(
        len(m.get("penetration_vertices_repaired") or []),
        len(m.get("penetration_vertices_unresolved") or []),
        m.get("mean_escape_displacement", 0.0), m.get("max_escape_displacement", 0.0)))
    print("  direction: inward={0} outward={1} tangential={2} mean_normal_disp={3:.5f} "
          "(negative=inward)".format(d["count_inward"], d["count_outward"],
                                     d["count_tangential"], d["mean_normal_displacement"]))
    print("  tangent preserve: normal_removed_mean={0:.5f} tangential_kept_mean={1:.5f}".format(
        m.get("normal_component_removed_mean", 0.0),
        m.get("tangential_component_preserved_mean", 0.0)))
    print("  SURFACE INTERSECTION: faces {0}->{1}  pairs {2}->{3}  "
          "new_during_smooth={4} prevented={5} repaired_iters={6} rejected_iters={7}".format(
              m.get("intersecting_skin_face_count_before", 0),
              m.get("intersecting_skin_face_count_after", 0),
              m.get("intersection_pair_count_before", 0),
              m.get("intersection_pair_count_after", 0),
              m.get("new_intersections_detected_during_smoothing", 0),
              m.get("new_intersections_prevented", 0),
              m.get("unsafe_iterations_repaired", 0),
              m.get("unsafe_iterations_rejected", 0)))
    print("  SAFE STEP: policy={0}  full_safe={1} full_unsafe={2}  "
          "partial_accepted={3} stalled={4}  recovered_by_line_search={5}".format(
              m.get("unsafe_step_policy"),
              m.get("full_steps_safe", 0), m.get("full_steps_unsafe", 0),
              m.get("partial_steps_accepted", 0), m.get("stalled_steps", 0),
              m.get("unsafe_full_steps_recovered_by_line_search", 0)))
    print("  accepted alpha: mean={0:.4f} min={1:.4f} max={2:.4f}  "
          "line_search_evals={3}  per_iter={4}".format(
              m.get("mean_accepted_alpha", 0.0),
              m.get("min_accepted_alpha", 0.0),
              m.get("max_accepted_alpha", 0.0),
              m.get("line_search_evaluations", 0),
              m.get("accepted_alpha_per_iteration") or []))
    print("  per-iter full-step proposed mean={0:.5f}  "
          "partial-step total disp={1:.5f}".format(
              m.get("full_step_proposed_mean_displacement", 0.0),
              m.get("partial_step_total_displacement", 0.0)))
    if m.get("safety_mode") == "repair_after_batch":
        print("  BATCHES: count={0} size={1} aborted={2} {3}".format(
            m.get("batch_count", 0), m.get("safety_batch_iterations"),
            m.get("batch_aborted"), m.get("batch_abort_reason") or ""))
        for i, rec in enumerate(m.get("batch_reports") or []):
            print("    batch {0}: smooth_iters={1} faces {2}->{3} "
                  "repair->{4}  smooth_disp={5:.5f} repair_disp={6:.5f} "
                  "unresolved={7}".format(
                      i + 1, rec.get("smoothing_iterations"),
                      rec.get("intersections_before"),
                      rec.get("intersections_after_smoothing"),
                      rec.get("intersections_after_repair"),
                      float(rec.get("smoothing_displacement") or 0.0),
                      float(rec.get("repair_displacement") or 0.0),
                      rec.get("unresolved_intersections")))
    print("  intersection pre-repair: iters={0} moved={1} resolved_faces={2} "
          "unresolved={3} mean_disp={4:.5f}".format(
              m.get("intersection_repair_iterations", 0),
              len(m.get("intersection_vertices_moved") or []),
              m.get("intersection_faces_resolved", 0),
              m.get("intersection_faces_unresolved", 0),
              m.get("mean_intersection_repair_displacement", 0.0)))
    if "scene_applied_max_displacement" in m:
        print("  fresh-read scene disp: mean={0:.5f} max={1:.5f}".format(
            m["scene_applied_mean_displacement"], m["scene_applied_max_displacement"]))
