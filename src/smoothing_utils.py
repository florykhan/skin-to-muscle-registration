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
                                   preserve_tangential=True):
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
    """
    if min_clearance is None:
        raise ValueError("min_clearance is required (no world-unit default is guessed)")

    solver = constraint_solver or "iterative_exact"
    if solver not in ("iterative_exact", "legacy_single_push"):
        raise ValueError("constraint_solver must be 'iterative_exact' or 'legacy_single_push'")

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

    local_edge = {}
    if max_step_edge_ratio is not None:
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

    for it in range(max(1, iterations)):
        proposed = _run_smoothing(work, neighbors, method, strength, active, 1)

        if max_step_edge_ratio is not None:
            for i in active:
                cap = max_step_edge_ratio * local_edge.get(i, 0.0)
                if cap <= 0:
                    continue
                step = vec_sub(proposed[i], work[i])
                slen = vec_length(step)
                if slen > cap and slen > 1e-12:
                    proposed[i] = vec_add(work[i], vec_scale(step, cap / slen))

        constrain_now = constraint_interval <= 1 or (it % constraint_interval == 0)
        for i in active:
            floor_i = floors.get(i, min_clearance)
            nrm = normals[i] if (normals and i < len(normals)) else None

            if do_segment:
                clamped, crossed, _hit = anatomy_constraint.clamp_segment_crossing(
                    work[i], proposed[i], anatomy_backend, min_clearance=floor_i)
                if crossed:
                    segment_detected += 1
                    segment_prevented += 1
                    proposed[i] = clamped

            if not constrain_now:
                continue

            if solver == "legacy_single_push":
                dist, cp, outward = sdf_query_fn(proposed[i])
                if dist < min_clearance:
                    safe = vec_add(list(cp), vec_scale(list(outward), min_clearance))
                    corrections.append(vec_length(vec_sub(proposed[i], safe)))
                    proposed[i] = safe
                    constraint_events += 1
                    constrained_vertices.add(i)
                continue

            if do_tangent:
                q_prop = anatomy_backend.exact_closest(proposed[i])
                if not anatomy_constraint.is_clearance_satisfied(
                        q_prop["distance"], floor_i, clearance_tolerance):
                    cand, removed, tlen, used = anatomy_constraint.tangent_preserving_proposal(
                        work[i], proposed[i], anatomy_backend, floor_i,
                        clearance_tolerance=clearance_tolerance)
                    if used:
                        proposed[i] = cand
                        normal_removed.append(removed)
                        tangent_kept.append(tlen)

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
                constraint_events += 1
                constrained_vertices.add(i)
                corrections.append(corr)
                projection_steps += sol["iterations"]
                if sol["iterations"] > max_proj_used:
                    max_proj_used = sol["iterations"]
                if sol["unresolved"]:
                    unresolved_constraints += 1
            else:
                p2, ok, niter, _res = _legacy_iterative_sdf_push(
                    proposed[i], sdf_query_fn, floor_i,
                    max_constraint_iterations, clearance_tolerance)
                corr = vec_length(vec_sub(proposed[i], p2))
                if corr > 0.0:
                    proposed[i] = p2
                    constraint_events += 1
                    constrained_vertices.add(i)
                    corrections.append(corr)
                    projection_steps += niter
                    if niter > max_proj_used:
                        max_proj_used = niter
                    if not ok:
                        unresolved_constraints += 1

        work = proposed

        if verbose_iterations and (it % max(1, report_interval) == 0):
            sd = [vec_length(vec_sub(work[i], before[i])) for i in active]
            if use_exact:
                minclr = min((anatomy_backend.exact_closest(work[i])["distance"]
                              for i in active), default=0.0)
            else:
                minclr = min((sdf_query_fn(work[i])[0] for i in active), default=0.0)
            print("  [iter {0:4d}] applied mean disp={1:.5f} constrained so far={2} "
                  "min clearance={3:.4f}".format(
                      it, (sum(sd) / len(sd)) if sd else 0.0,
                      len(constrained_vertices), minclr))

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
    if anatomy_backend is not None:
        final_pen = anatomy_constraint.analyze_skin_anatomy_penetration(
            mesh_name, indices=region, min_clearance=min_clearance,
            backend=anatomy_backend, positions=final, normals=normals,
            clearance_tolerance=clearance_tolerance, detailed=False, verbose=False)

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
        "apply": bool(apply),
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
          "clearance={5} solver={6} policy={7}".format(
              m["mesh"], m["selected_vertex_count"], m["method"],
              m["strength"], m["iterations"], m["min_clearance"],
              m.get("constraint_solver"), m.get("clearance_policy")))
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
    if "scene_applied_max_displacement" in m:
        print("  fresh-read scene disp: mean={0:.5f} max={1:.5f}".format(
            m["scene_applied_mean_displacement"], m["scene_applied_max_displacement"]))
