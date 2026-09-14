"""
mesh_utils.py
=============
Reusable Maya mesh-access helpers for the skin-to-muscle-registration project.

These are small, dependency-light wrappers around ``maya.cmds`` and
``maya.api.OpenMaya`` so that higher-level modules (smoothing, region selection,
metrics) do not each re-implement vertex queries and neighbor lookups.

Nothing here changes the scene by itself except :func:`set_mesh_vertices`, which
writes vertex positions back to a mesh. Reading functions are side-effect free.

This module is designed to run INSIDE Autodesk Maya (or ``mayapy``). It imports
the Maya API lazily-tolerantly so the file can still be inspected/imported for
linting outside Maya.
"""

import math

try:
    import maya.cmds as cmds
    import maya.api.OpenMaya as om
    MAYA_AVAILABLE = True
except ImportError:  # allows importing this file outside Maya (e.g. for linting)
    cmds = None
    om = None
    MAYA_AVAILABLE = False


# =============================================================================
# VECTOR MATH (kept here so smoothing/metrics can share one implementation)
# =============================================================================

def vec_length(v):
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def vec_sub(a, b):
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]


def vec_add(a, b):
    return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]


def vec_scale(v, s):
    return [v[0] * s, v[1] * s, v[2] * s]


def vec_mean(vectors):
    n = len(vectors)
    if n == 0:
        return [0.0, 0.0, 0.0]
    return [sum(v[0] for v in vectors) / n,
            sum(v[1] for v in vectors) / n,
            sum(v[2] for v in vectors) / n]


def vec_lerp(a, b, t):
    """Linear interpolation: a*(1-t) + b*t."""
    return [a[0] * (1 - t) + b[0] * t,
            a[1] * (1 - t) + b[1] * t,
            a[2] * (1 - t) + b[2] * t]


# =============================================================================
# MESH ACCESS
# =============================================================================

def mesh_exists(mesh_name):
    """Return True if a transform/shape with this name exists in the scene."""
    if not MAYA_AVAILABLE:
        return False
    return bool(cmds.objExists(mesh_name))


def get_mesh_fn(mesh_name):
    """Return an ``MFnMesh`` for ``mesh_name`` (transform or shape), or None."""
    if not MAYA_AVAILABLE:
        return None
    try:
        sel = om.MSelectionList()
        sel.add(mesh_name)
        dag = sel.getDagPath(0)
        if dag.apiType() == om.MFn.kTransform:
            dag.extendToShape()
        return om.MFnMesh(dag)
    except Exception:
        return None


def get_mesh_vertices(mesh_name):
    """Return world-space vertex positions as a list of [x, y, z]."""
    mesh_fn = get_mesh_fn(mesh_name)
    if mesh_fn is None:
        return []
    pts = mesh_fn.getPoints(om.MSpace.kWorld)
    return [[p.x, p.y, p.z] for p in pts]


def set_mesh_vertices(mesh_name, vertices):
    """Write world-space vertex positions back to a mesh.

    ``vertices`` must be ordered to match the mesh's vertex indices and have the
    same length as the mesh's vertex count.
    """
    mesh_fn = get_mesh_fn(mesh_name)
    if mesh_fn is None:
        return False
    pts = om.MPointArray()
    for v in vertices:
        pts.append(om.MPoint(v[0], v[1], v[2]))
    mesh_fn.setPoints(pts, om.MSpace.kWorld)
    mesh_fn.updateSurface()
    return True


def get_vertex_count(mesh_name):
    mesh_fn = get_mesh_fn(mesh_name)
    return mesh_fn.numVertices if mesh_fn is not None else 0


def get_vertex_neighbors(mesh_name):
    """Return per-vertex 1-ring neighbor lists: ``neighbors[i] -> [j, k, ...]``.

    This is the topological adjacency used by Laplacian smoothing and region
    growing. The result is stable for a fixed mesh topology, so callers should
    cache it rather than recomputing every operation.
    """
    mesh_fn = get_mesh_fn(mesh_name)
    if mesh_fn is None:
        return []
    n = mesh_fn.numVertices
    neighbors = [set() for _ in range(n)]
    edge_iter = om.MItMeshEdge(mesh_fn.object())
    while not edge_iter.isDone():
        v0, v1 = edge_iter.vertexId(0), edge_iter.vertexId(1)
        neighbors[v0].add(v1)
        neighbors[v1].add(v0)
        edge_iter.next()
    return [list(s) for s in neighbors]


def get_boundary_vertices(mesh_name):
    """Return the set of TRUE topological boundary vertex indices of a mesh.

    A boundary vertex is any vertex touching an edge that borders exactly one
    face (``MItMeshEdge.onBoundary()``) -- i.e. an open border of the surface.
    This is a purely topological test; it does NOT use vertex positions or any
    anatomical assumption. For a closed skin mesh this is empty; for a skin sheet
    with holes it returns the rims of every opening -- eye openings, lips/mouth
    opening, nostrils, the neck opening, and the outer mesh border.

    Consolidation note: M3 (multi-scale) and M4 (SDF-reference) each independently
    added an identical topological boundary helper here during development; they
    were merged into this single implementation. M3's ``find_boundary_vertices``
    and M4's ``find_skin_boundary_vertices`` both delegate to this function, so
    both milestones share one boundary definition.

    Parameters
    ----------
    mesh_name:
        Mesh to inspect (not modified).

    Returns
    -------
    set[int]
        Boundary vertex indices (empty if the mesh is missing, closed, or Maya
        is unavailable).
    """
    if not MAYA_AVAILABLE:
        return set()
    mesh_fn = get_mesh_fn(mesh_name)
    if mesh_fn is None:
        return set()
    boundary = set()
    edge_iter = om.MItMeshEdge(mesh_fn.object())
    while not edge_iter.isDone():
        if edge_iter.onBoundary():
            boundary.add(edge_iter.vertexId(0))
            boundary.add(edge_iter.vertexId(1))
        edge_iter.next()
    return boundary


def get_vertex_normals(mesh_name):
    """Return per-vertex unit normals in world space as a list of [x, y, z].

    Index-aligned with the mesh's vertices. Uses ``MFnMesh.getVertexNormals``
    (Maya API 2.0), which returns the averaged (not angle-weighted) normal per
    vertex. Empty list outside Maya or if the mesh is missing.
    """
    if not MAYA_AVAILABLE:
        return []
    mesh_fn = get_mesh_fn(mesh_name)
    if mesh_fn is None:
        return []
    normals = mesh_fn.getVertexNormals(False, om.MSpace.kWorld)
    return [[n.x, n.y, n.z] for n in normals]


def compute_closest_point_distances(points, meshes):
    """Closest-surface distance from each query point to a set of meshes.

    For every point in ``points`` (world-space ``[x, y, z]``), find the shortest
    Euclidean distance to the union of the given ``meshes`` and which mesh is
    nearest. One ``MMeshIntersector`` acceleration structure is built PER mesh
    ONCE (not per query), and every point is tested against each; no per-vertex
    ``cmds`` calls are made.

    Returns
    -------
    dict
        ``{"distances": [float, ...], "nearest": [mesh_name_or_None, ...],
           "valid_meshes": [...], "missing_meshes": [...]}``. Distances default
        to ``0.0`` (nearest ``None``) when no valid mesh exists.

    Notes
    -----
    ``MMeshIntersector.getClosestPoint`` takes the query point in world space
    (because the shape's inclusive world matrix is supplied to ``create``) and
    returns a point in object space, which is transformed back to world before
    measuring distance.
    """
    n_pts = len(points)
    if not MAYA_AVAILABLE:
        return {"distances": [0.0] * n_pts, "nearest": [None] * n_pts,
                "valid_meshes": [], "missing_meshes": list(meshes)}

    intersectors = []  # (name, MMeshIntersector, worldMatrix)
    valid, missing = [], []
    for name in meshes:
        if not cmds.objExists(name):
            missing.append(name)
            continue
        try:
            sel = om.MSelectionList()
            sel.add(name)
            dag = sel.getDagPath(0)
            if dag.apiType() == om.MFn.kTransform:
                dag.extendToShape()
            matrix = dag.inclusiveMatrix()
            intersector = om.MMeshIntersector()
            intersector.create(dag.node(), matrix)
            intersectors.append((name, intersector, matrix))
            valid.append(name)
        except Exception:
            missing.append(name)

    distances = [0.0] * n_pts
    nearest = [None] * n_pts
    if not intersectors:
        return {"distances": distances, "nearest": nearest,
                "valid_meshes": valid, "missing_meshes": missing}

    for k, p in enumerate(points):
        wp = om.MPoint(p[0], p[1], p[2])
        best_d = None
        best_name = None
        for name, intersector, matrix in intersectors:
            pom = intersector.getClosestPoint(wp)
            if pom is None:
                continue
            closest_world = om.MPoint(pom.point) * matrix
            d = wp.distanceTo(closest_world)
            if best_d is None or d < best_d:
                best_d = d
                best_name = name
        if best_d is not None:
            distances[k] = best_d
            nearest[k] = best_name

    return {"distances": distances, "nearest": nearest,
            "valid_meshes": valid, "missing_meshes": missing}


# =============================================================================
# SELECTION / COMPONENT HELPERS
# =============================================================================

def get_selected_vertex_indices(mesh_name=None):
    """Return vertex indices currently selected in Maya.

    If ``mesh_name`` is given, only indices belonging to that mesh are returned.
    Otherwise returns a dict of ``{mesh_transform: [indices]}``.

    Works with the usual component selection produced by right-click ->
    "Vertex" and dragging in the viewport (``mesh.vtx[12]``, ``mesh.vtx[3:9]``).
    """
    if not MAYA_AVAILABLE:
        return [] if mesh_name else {}

    sel = cmds.ls(selection=True, flatten=True) or []
    result = {}
    for item in sel:
        if ".vtx[" not in item:
            continue
        obj, comp = item.split(".vtx[")
        idx = int(comp.rstrip("]"))
        # Normalize to the transform name (strip shape/namespace path tail)
        transform = obj.split("|")[-1]
        result.setdefault(transform, []).append(idx)

    if mesh_name is not None:
        key = mesh_name.split("|")[-1]
        return sorted(result.get(key, []))
    return {k: sorted(v) for k, v in result.items()}


def select_vertices(mesh_name, indices, replace=True):
    """Select the given vertex indices on ``mesh_name`` in the Maya viewport."""
    if not MAYA_AVAILABLE or not indices:
        return
    comps = ["{0}.vtx[{1}]".format(mesh_name, i) for i in indices]
    cmds.select(comps, replace=replace)


def grow_indices(neighbors, indices, rings=1):
    """Grow a set of vertex indices outward by ``rings`` topological rings."""
    current = set(indices)
    for _ in range(max(0, rings)):
        added = set()
        for i in current:
            if 0 <= i < len(neighbors):
                added.update(neighbors[i])
        current |= added
    return sorted(current)


# =============================================================================
# CLOSEST-POINT / DISTANCE QUERIES (Maya API 2.0 acceleration)
# =============================================================================
# These are the shared, general-purpose distance primitives used by the
# artifact-detection SDF work (M4). They wrap ``MFnMesh.getClosestPoint``
# (Maya API 2.0), which maintains an internal spatial acceleration structure, so
# repeated queries against the SAME ``MFnMesh`` do not rebuild that structure and
# never fall back to slow per-call ``maya.cmds`` invocations. The d98
# registration script has its own private ``get_closest_point_on_mesh`` used by
# the (unchanged) registration algorithm; this is the reusable home for the same
# API-2.0 primitive so higher-level modules do not each re-implement it.
#
# Coordinate space: ALL functions here operate in WORLD space (``MSpace.kWorld``)
# and both accept and return world-space coordinates. Distances are UNSIGNED
# Euclidean distances to the nearest surface point (there is no inside/outside
# sign test).

def closest_point_on_mesh(mesh_fn, point):
    """Return ``(closest_xyz, distance)`` for the nearest point on a mesh.

    Parameters
    ----------
    mesh_fn:
        An ``MFnMesh`` (e.g. from :func:`get_mesh_fn`). Reused across many
        queries so Maya's internal acceleration structure is built once.
    point:
        World-space ``[x, y, z]`` query point.

    Returns
    -------
    (list[float], float)
        The world-space closest point ``[x, y, z]`` and the UNSIGNED Euclidean
        distance to it. Returns ``(point, inf)`` if the query fails or Maya is
        unavailable, so callers can treat ``inf`` as "no hit".
    """
    if not MAYA_AVAILABLE or mesh_fn is None:
        return list(point), float("inf")
    try:
        qp = om.MPoint(point[0], point[1], point[2])
        cp, _face_id = mesh_fn.getClosestPoint(qp, om.MSpace.kWorld)
        dist = math.sqrt((point[0] - cp.x) ** 2
                         + (point[1] - cp.y) ** 2
                         + (point[2] - cp.z) ** 2)
        return [cp.x, cp.y, cp.z], dist
    except Exception:
        return list(point), float("inf")
