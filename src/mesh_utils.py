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
