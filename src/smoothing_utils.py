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

from mesh_utils import (
    vec_mean,
    vec_lerp,
    get_mesh_vertices,
    set_mesh_vertices,
    get_vertex_neighbors,
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
