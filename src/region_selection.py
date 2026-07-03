"""
region_selection.py
===================
Define and select problematic vertex regions on the skin mesh so cleanup can be
applied locally (lips, cheeks, nose, eyes, chin, ...).

Three complementary ways to define a region are supported:

1. **Manual** -- pick vertices in the Maya viewport and read them back with
   :func:`from_current_selection`. This is the primary workflow for now: the
   researcher selects a bad patch by hand, then hands it to the smoothing tools.

2. **Saved presets** -- store/reload a region as JSON (:func:`save_region`,
   :func:`load_region`) so a hand-picked region is reproducible across sessions
   and scene versions (vertex indices are stable while topology is unchanged).

3. **Heuristic bounding-box** -- approximate a named facial region from vertex
   positions (:func:`region_by_bbox`, :func:`named_region`). These are rough
   helpers to bootstrap a selection; refine them by hand before smoothing.

The heuristics assume the model's conventional orientation:
    +X = character's left/right,  +Y = up,  +Z = forward (face front).
Adjust the fractions if your scene differs.
"""

import json
import os

from mesh_utils import (
    get_mesh_vertices,
    get_vertex_neighbors,
    get_selected_vertex_indices,
    select_vertices,
    grow_indices,
)


# =============================================================================
# MANUAL SELECTION (Maya viewport)
# =============================================================================

def from_current_selection(mesh_name):
    """Return vertex indices currently selected on ``mesh_name`` in Maya."""
    indices = get_selected_vertex_indices(mesh_name)
    print("[region_selection] {0} vertices selected on {1}".format(len(indices), mesh_name))
    return indices


def show_region(mesh_name, indices):
    """Select ``indices`` in the viewport so you can eyeball a region."""
    select_vertices(mesh_name, indices, replace=True)


def grow_region(mesh_name, indices, rings=1, neighbors=None):
    """Grow a region by ``rings`` topological rings (soft edges for blending)."""
    if neighbors is None:
        neighbors = get_vertex_neighbors(mesh_name)
    return grow_indices(neighbors, indices, rings=rings)


# =============================================================================
# SAVED PRESETS (JSON)
# =============================================================================

def save_region(indices, name, folder, mesh_name=None):
    """Save a region to ``<folder>/<name>.json``.

    The format matches the loose ``vertex_indices``/``indices`` convention used
    elsewhere in the project so presets are interchangeable.
    """
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, "{0}.json".format(name))
    data = {
        "name": name,
        "mesh": mesh_name,
        "indices": sorted(int(i) for i in indices),
        "vertex_count": len(list(indices)),
    }
    data["vertex_indices"] = data["indices"]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print("[region_selection] saved region '{0}' ({1} verts) -> {2}".format(
        name, data["vertex_count"], path))
    return path


def load_region(name, folder):
    """Load a region saved with :func:`save_region`. Returns a list of indices."""
    path = os.path.join(folder, "{0}.json".format(name))
    if not os.path.exists(path):
        print("[region_selection] region '{0}' not found at {1}".format(name, path))
        return []
    with open(path, "r") as f:
        data = json.load(f)
    indices = data.get("indices", data.get("vertex_indices", []))
    return sorted(int(i) for i in indices)


# =============================================================================
# HEURISTIC BOUNDING-BOX REGIONS
# =============================================================================

def _bounds(vertices):
    xs = [v[0] for v in vertices]
    ys = [v[1] for v in vertices]
    zs = [v[2] for v in vertices]
    return (min(xs), max(xs), min(ys), max(ys), min(zs), max(zs))


def region_by_bbox(vertices, x=(0.0, 1.0), y=(0.0, 1.0), z=(0.0, 1.0)):
    """Select vertices inside a normalized bounding-box slice of the mesh.

    Each of ``x``, ``y``, ``z`` is a ``(lo, hi)`` fraction in ``[0, 1]`` relative
    to the mesh's overall bounding box, so the same call works regardless of the
    scene's absolute scale/units.
    """
    xmin, xmax, ymin, ymax, zmin, zmax = _bounds(vertices)
    dx = (xmax - xmin) or 1.0
    dy = (ymax - ymin) or 1.0
    dz = (zmax - zmin) or 1.0

    xlo, xhi = xmin + x[0] * dx, xmin + x[1] * dx
    ylo, yhi = ymin + y[0] * dy, ymin + y[1] * dy
    zlo, zhi = zmin + z[0] * dz, zmin + z[1] * dz

    out = []
    for i, v in enumerate(vertices):
        if xlo <= v[0] <= xhi and ylo <= v[1] <= yhi and zlo <= v[2] <= zhi:
            out.append(i)
    return out


# Rough vertical/forward bands for common facial regions, as fractions of the
# mesh bounding box. These are STARTING points -- refine by hand in the viewport.
_NAMED_REGIONS = {
    "chin":   dict(y=(0.00, 0.22), z=(0.55, 1.00)),
    "lips":   dict(y=(0.22, 0.40), z=(0.60, 1.00)),
    "nose":   dict(x=(0.35, 0.65), y=(0.38, 0.62), z=(0.70, 1.00)),
    "cheeks": dict(y=(0.25, 0.55), z=(0.30, 0.80)),
    "eyes":   dict(y=(0.55, 0.72), z=(0.55, 1.00)),
}


def named_region(mesh_name, region, vertices=None):
    """Return an approximate vertex selection for a named facial ``region``.

    Supported names: ``chin``, ``lips``, ``nose``, ``cheeks``, ``eyes``.
    Falls back to an empty list for unknown names.
    """
    if region not in _NAMED_REGIONS:
        print("[region_selection] unknown region '{0}'. Known: {1}".format(
            region, ", ".join(sorted(_NAMED_REGIONS))))
        return []
    if vertices is None:
        vertices = get_mesh_vertices(mesh_name)
    if not vertices:
        return []
    box = _NAMED_REGIONS[region]
    indices = region_by_bbox(vertices, **box)
    print("[region_selection] '{0}' ~ {1} vertices (heuristic; refine by hand)".format(
        region, len(indices)))
    return indices
