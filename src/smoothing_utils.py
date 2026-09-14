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
# UNSIGNED-DISTANCE LIMITATION: the injected field uses UNSIGNED closest-surface
# distance plus an outward direction (from the closest anatomy point toward the
# query point). If a vertex ever ends up INSIDE anatomy that direction points
# deeper, so the push is only reliable while the skin stays OUTSIDE. Enforcing the
# constraint EVERY iteration (constraint_interval=1) keeps each step tiny and
# prevents the skin from crossing in the first place -- the same assumption and
# mechanism the registration's own collision pass uses. It is anatomy-aware
# clamping, NOT a mathematically guaranteed inside/outside / penetration solver.


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


def constrained_smooth_mesh_region(mesh_name, indices, sdf_query_fn, min_clearance,
                                   method="laplacian", strength=0.2, iterations=20,
                                   constraint_interval=1, boundary_feather_rings=0,
                                   max_step_edge_ratio=None, neighbors=None,
                                   normals=None, apply=True, verbose=True,
                                   report_interval=10, verbose_iterations=False):
    """Anatomy-constrained localized smoothing (see section header).

    Parameters
    ----------
    mesh_name : str
        Skin mesh to smooth (name preserved; not renamed).
    indices : iterable[int]
        The M5-selected region to smooth. Non-selected vertices stay fixed and act
        as frozen references so the region blends into its surroundings.
    sdf_query_fn : callable
        ``sdf_query_fn(point) -> (distance, closest_point, outward_dir)`` -- the
        anatomy distance field, injected by the d98 wrapper from the registration's
        ``compute_sdf_for_point`` (SAME field the registration collision uses).
        ``distance`` is the (unsigned) smooth-min distance to internal anatomy;
        ``outward_dir`` points from the closest anatomy point toward ``point``.
    min_clearance : float
        Minimum allowed skin-to-anatomy distance (the anatomical floor). Required
        (no world-unit default is guessed); the d98 wrapper defaults it to the
        registration's ``collision_min_distance``.
    method : {"laplacian", "taubin"}
        Per-iteration smoothing operator (reuses this module's existing functions).
    strength, iterations, constraint_interval, boundary_feather_rings,
    max_step_edge_ratio :
        ``constraint_interval=1`` enforces the floor after every smoothing
        iteration (recommended). ``boundary_feather_rings`` optionally extends the
        smoothed set outward for a softer transition (0 = exactly the region).
        ``max_step_edge_ratio`` optionally caps each per-iteration step to that
        fraction of the local mean edge length (secondary safety; ``None`` = off).
    neighbors, normals :
        Optional precomputed adjacency / vertex normals (read once if omitted).
    apply : bool
        If True, write the constrained result and RE-READ to verify it persisted.
        If False, compute metrics only -- the scene is NOT modified.

    Returns
    -------
    dict
        Metrics: proposed vs applied displacement, constraint events, anatomy
        clearance before/after, movement-direction diagnostic, and (when applied)
        a fresh-read verification.
    """
    if min_clearance is None:
        raise ValueError("min_clearance is required (no world-unit default is guessed)")

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

    # Active smoothed set = region (+ optional uniform feather rings).
    if boundary_feather_rings and boundary_feather_rings > 0:
        active = sorted(set(grow_indices(neighbors, region, rings=boundary_feather_rings)))
    else:
        active = list(region)

    if normals is None:
        normals = get_vertex_normals(mesh_name)   # for inward/outward diagnostic only

    before = [list(v) for v in current]

    # Local edge lengths (computed once) for the optional relative max-step cap.
    local_edge = {}
    if max_step_edge_ratio is not None:
        for i in active:
            nbrs = neighbors[i] if i < len(neighbors) else []
            local_edge[i] = (sum(vec_length(vec_sub(before[i], before[j])) for j in nbrs)
                             / len(nbrs)) if nbrs else 0.0

    # (A) What UNCONSTRAINED smoothing would do (the "proposed" baseline) -- pure
    # array math, no scene writes; this is the A/B reference for the constraint.
    unconstrained = _run_smoothing(before, neighbors, method, strength, active, iterations)

    # (B) Constrained loop: smooth one step, enforce the anatomy floor, repeat.
    work = [list(v) for v in before]
    constrained_vertices = set()
    constraint_events = 0
    corrections = []
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

        if constraint_interval <= 1 or (it % constraint_interval == 0):
            for i in active:
                dist, cp, outward = sdf_query_fn(proposed[i])
                if dist < min_clearance:
                    safe = vec_add(cp, vec_scale(outward, min_clearance))
                    corrections.append(vec_length(vec_sub(proposed[i], safe)))
                    proposed[i] = safe
                    constraint_events += 1
                    constrained_vertices.add(i)

        work = proposed

        if verbose_iterations and (it % max(1, report_interval) == 0):
            sd = [vec_length(vec_sub(work[i], before[i])) for i in active]
            minclr = min((sdf_query_fn(work[i])[0] for i in active), default=0.0)
            print("  [iter {0:4d}] applied mean disp={1:.5f} constrained so far={2} "
                  "min clearance={3:.4f}".format(
                      it, (sum(sd) / len(sd)) if sd else 0.0,
                      len(constrained_vertices), minclr))

    final = work

    # --- metrics -------------------------------------------------------------
    ps = _sc_stats([vec_length(vec_sub(unconstrained[i], before[i])) for i in region])
    aps = _sc_stats([vec_length(vec_sub(final[i], before[i])) for i in region])
    dist_before = [sdf_query_fn(before[i])[0] for i in region]
    dist_after = [sdf_query_fn(final[i])[0] for i in region]
    db, da = _sc_stats(dist_before), _sc_stats(dist_after)
    below_before = sum(1 for d in dist_before if d < min_clearance)
    below_after = sum(1 for d in dist_after if d < min_clearance)

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
        "proposed_mean_displacement": ps["mean"],
        "proposed_rms_displacement": ps["rms"],
        "proposed_max_displacement": ps["max"],
        "applied_mean_displacement": aps["mean"],
        "applied_rms_displacement": aps["rms"],
        "applied_max_displacement": aps["max"],
        "number_of_constraint_events": constraint_events,
        "unique_vertices_constrained": len(constrained_vertices),
        "number_safe_without_constraint": len(active) - len(constrained_vertices),
        "number_clamped_or_pushed": len(constrained_vertices),
        "max_constraint_correction": corr["max"],
        "mean_constraint_correction": corr["mean"],
        "anatomy_distance_before": {"min": db["min"], "mean": db["mean"]},
        "anatomy_distance_after": {"min": da["min"], "mean": da["mean"]},
        "vertices_below_clearance_before": below_before,
        "vertices_below_clearance_after": below_after,
        "direction": {"count_inward": inward, "count_outward": outward_c,
                      "count_tangential": tangential,
                      "mean_normal_displacement": mean_nd},
        "apply": bool(apply),
    }

    # --- apply + fresh-read verification -------------------------------------
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
          "clearance={5}".format(m["mesh"], m["selected_vertex_count"], m["method"],
                                 m["strength"], m["iterations"], m["min_clearance"]))
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
    print("  anatomy clearance: before min={0:.4f} mean={1:.4f} -> after min={2:.4f} "
          "mean={3:.4f}".format(m["anatomy_distance_before"]["min"],
                                m["anatomy_distance_before"]["mean"],
                                m["anatomy_distance_after"]["min"],
                                m["anatomy_distance_after"]["mean"]))
    print("  below-clearance verts: before={0} -> after={1}".format(
        m["vertices_below_clearance_before"], m["vertices_below_clearance_after"]))
    print("  direction: inward={0} outward={1} tangential={2} mean_normal_disp={3:.5f} "
          "(negative=inward)".format(d["count_inward"], d["count_outward"],
                                     d["count_tangential"], d["mean_normal_displacement"]))
    if "scene_applied_max_displacement" in m:
        print("  fresh-read scene disp: mean={0:.5f} max={1:.5f}".format(
            m["scene_applied_mean_displacement"], m["scene_applied_max_displacement"]))
