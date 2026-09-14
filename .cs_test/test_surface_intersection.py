"""Offline tests for skin-anatomy SURFACE INTERSECTION (no Maya required)."""
from __future__ import print_function

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import anatomy_constraint as ac
import smoothing_utils


fails = []


def check(name, cond, detail=""):
    ok = bool(cond)
    print(("  ok   " if ok else "  FAIL ") + name + (
        ("  " + detail) if (detail and not ok) else ""))
    if not ok:
        fails.append(name)


def _aabb(pts):
    return ac._aabb_from_points(pts)


class SoupBackend(object):
    """Open anatomy: one triangle (or more) as a triangle soup. No inside test."""

    def __init__(self, name, triangles, points):
        # triangles: list of (face_id, i0, i1, i2)
        topo = {"triangles": triangles, "face_vertex_ids": [], "vertex_faces": []}
        self._cache = {
            name: ac.build_anatomy_surface_cache(
                None, name=name, points=points, topology=topo)
        }

    def anatomy_surface_cache(self):
        return self._cache

    def mesh_names(self):
        return list(self._cache.keys())

    def is_closed(self, name):
        return False

    def point_inside(self, p, name):
        return False

    def exact_closest(self, point, names=None):
        # unused for intersection tests
        return {"distance": 99.0, "closest": list(point), "normal": [0, 1, 0],
                "mesh": None, "face_id": -1, "category": "other", "hits": []}


def _skin_topo(triangles, nverts):
    face_verts = {}
    vertex_faces = [[] for _ in range(nverts)]
    for face_id, i0, i1, i2 in triangles:
        face_verts.setdefault(face_id, [])
        for v in (i0, i1, i2):
            if v not in face_verts[face_id]:
                face_verts[face_id].append(v)
            if v not in vertex_faces[v]:
                vertex_faces[v].append(face_id)
    nfaces = max(t[0] for t in triangles) + 1
    fvl = [face_verts.get(i, []) for i in range(nfaces)]
    return {"triangles": triangles, "face_vertex_ids": fvl, "vertex_faces": vertex_faces}


print("1. piercing triangles: vertices outside, faces intersect")
# Skin triangle in XY at z=0. Anatomy triangle in XZ piercing through z=0 at origin.
skin_pts = [
    [-1.0, -1.0, 0.0],
    [1.0, -1.0, 0.0],
    [0.0, 1.0, 0.0],
]
# Anatomy: vertical triangle that crosses the skin triangle interior
anat_pts = [
    [-0.2, 0.0, -1.0],
    [0.2, 0.0, -1.0],
    [0.0, 0.0, 1.0],
]
klass, pt = ac.triangle_triangle_intersection(
    (skin_pts[0], skin_pts[1], skin_pts[2]),
    (anat_pts[0], anat_pts[1], anat_pts[2]))
check("narrow-phase crossing", klass == "crossing", str(klass))

skin_topo = _skin_topo([(0, 0, 1, 2)], 3)
backend = SoupBackend("fat_2", [(0, 0, 1, 2)], anat_pts)
rep = ac.analyze_skin_anatomy_intersections(
    None, skin_indices=[0, 1, 2], backend=backend, positions=skin_pts,
    skin_topology=skin_topo, boundary_buffer_rings=0, exclude_boundary_faces=False,
    detailed=True, verbose=False)
check("intersecting face count 1", rep["intersecting_skin_face_count"] == 1,
      str(rep["intersecting_skin_face_count"]))
check("pair count >= 1", rep["intersection_pair_count"] >= 1)
check("anatomy mesh named", "fat_2" in rep["intersecting_anatomy_meshes"])
check("all 3 skin verts listed", set(rep["intersecting_skin_vertices"]) == {0, 1, 2})

print("2. close but NOT intersecting (distance is not intersection)")
# Skin at z=0, anatomy parallel at z=0.2 -- close, no intersection
anat_pts_far = [
    [-0.2, 0.0, 0.2],
    [0.2, 0.0, 0.2],
    [0.0, 1.0, 0.2],
]
klass2, _ = ac.triangle_triangle_intersection(
    (skin_pts[0], skin_pts[1], skin_pts[2]),
    (anat_pts_far[0], anat_pts_far[1], anat_pts_far[2]))
check("parallel offset is not an intersection", klass2 is None, str(klass2))
backend2 = SoupBackend("fat_2", [(0, 0, 1, 2)], anat_pts_far)
rep2 = ac.analyze_skin_anatomy_intersections(
    None, skin_indices=[0, 1, 2], backend=backend2, positions=skin_pts,
    skin_topology=skin_topo, boundary_buffer_rings=0, exclude_boundary_faces=False,
    verbose=False)
check("no intersecting faces when only close",
      rep2["intersecting_skin_face_count"] == 0)

print("3. edge-edge and vertex-face")
# Two triangles sharing a piercing edge-edge configuration
A = ([0, 0, 0], [2, 0, 0], [1, 2, 0])
B = ([1, -1, -1], [1, -1, 1], [1, 1, 0])  # vertical-ish through A's interior
k3, _ = ac.triangle_triangle_intersection(A, B)
check("edge/face piercing is crossing", k3 == "crossing", str(k3))

print("4. coplanar overlap vs touching")
C = ([0, 0, 0], [1, 0, 0], [0, 1, 0])
D = ([0.2, 0.2, 0], [1.2, 0.2, 0], [0.2, 1.2, 0])  # overlapping in plane
k4, _ = ac.triangle_triangle_intersection(C, D)
check("coplanar overlap", k4 == "coplanar_overlap", str(k4))
E = ([2, 0, 0], [3, 0, 0], [2, 1, 0])  # disjoint coplanar
k5, _ = ac.triangle_triangle_intersection(C, E)
check("disjoint coplanar is None", k5 is None, str(k5))
# touching at a vertex
F = ([1, 0, 0], [2, 0, 0], [1, 1, 0])  # shares vertex (1,0,0) / maybe edge
k6, _ = ac.triangle_triangle_intersection(C, F)
check("shared-edge coplanar is touching or overlap (not crossing)",
      k6 in ("touching", "coplanar_overlap"), str(k6))

print("5. BOTH directions (anatomy edge through skin AND skin edge through anatomy)")
# If we only tested skin edges vs anatomy face we could miss the converse.
# Construct: anatomy triangle large, a skin edge stabs it.
skin_stab = ([0, 0, -1], [0, 0, 1], [0.5, 0.5, 0])
anat_wall = ([-1, -1, 0], [1, -1, 0], [0, 1, 0])
k7, _ = ac.triangle_triangle_intersection(skin_stab, anat_wall)
check("skin-edge through anatomy is crossing", k7 == "crossing", str(k7))
# converse already covered by test 1 (anatomy edge through skin)

print("6. open anatomy soup does not need watertight inside/outside")
check("SoupBackend is_closed False", backend.is_closed("fat_2") is False)
check("still detected intersection on open mesh",
      rep["intersecting_skin_face_count"] == 1)

print("7. repair moves outward and can clear a crossing")
# Skin z=0, anatomy vertical through it. Push skin verts in +Z (outward normals).
normals = [[0, 0, 1], [0, 0, 1], [0, 0, 1]]
neighbors = [[1, 2], [0, 2], [0, 1]]
repair = ac.resolve_skin_anatomy_intersections(
    [list(p) for p in skin_pts], [0, 1, 2], backend,
    neighbors=neighbors, normals=normals, skin_topology=skin_topo,
    min_clearance=None, boundary_buffer_rings=0,
    max_intersection_repair_iterations=20,
    intersection_repair_step_ratio=0.5, verbose=False)
check("repair moved some verts",
      len(repair["intersection_vertices_moved"]) > 0)
check("repair reduced or cleared faces",
      repair["report_after"]["intersecting_skin_face_count"]
      <= repair["report_before"]["intersecting_skin_face_count"])
# After enough +Z motion the skin triangle leaves the anatomy triangle
# (anatomy only spans y=0). With step 0.5 * edge, should clear.
check("faces after repair ideally 0",
      repair["report_after"]["intersecting_skin_face_count"] == 0,
      str(repair["report_after"]["intersecting_skin_face_count"]))

print("8. combined prepare keeps penetration API working")
# Plane-like: no intersection soup, vertex likely-penetrating
class PlaneY(object):
    def mesh_names(self):
        return ["plane"]
    def is_closed(self, n):
        return False
    def point_inside(self, p, n):
        return False
    def exact_closest(self, point, names=None):
        cp = [point[0], 0.0, point[2]]
        return {"distance": abs(point[1]), "closest": cp, "normal": [0, 1, 0],
                "mesh": "plane", "face_id": 0, "category": "fat",
                "hits": [{"mesh": "plane", "distance": abs(point[1]),
                          "closest": cp, "normal": [0, 1, 0], "face_id": 0,
                          "category": "fat"}]}
    def anatomy_surface_cache(self):
        return {}
    def sdf_query(self, point):
        q = self.exact_closest(point)
        return q["distance"], q["closest"], [0, 1, 0] if point[1] >= 0 else [0, -1, 0]

plane = PlaneY()
c = ac.classify_skin_anatomy_vertex(
    [0, -0.2, 0], plane, skin_normal=[0, 1, 0], min_clearance=0.8)
check("vertex analyzer still flags likely_penetrating",
      c["label"] == "likely_penetrating")
prep = ac.prepare_skin_region_for_smoothing(
    [[0, -0.2, 0], [0, 2, 0]], [0, 1], plane,
    neighbors=[[1], [0]], normals=[[0, 1, 0], [0, 1, 0]],
    skin_topology=_skin_topo([(0, 0, 1, 1)], 2),  # degenerate dummy
    min_clearance=0.8, verbose=False)
check("combined prepare returns positions", "positions" in prep)
check("penetration after keys exist", "penetration_after" in prep)

print("9. triangle-triangle does not use distance threshold")
# Very close (1e-4) parallel triangles: NOT intersecting
G = ([0, 0, 0], [1, 0, 0], [0, 1, 0])
H = ([0, 0, 1e-4], [1, 0, 1e-4], [0, 1, 1e-4])
k9, _ = ac.triangle_triangle_intersection(G, H, eps=1e-6)
check("1e-4 gap is not intersection", k9 is None, str(k9))

print("10. constrained smoother still apply=False + legacy path")
verts = [
    [1.2, 0.2, 0.0],
    [1.2, 0.9, 0.0],
    [1.2, 0.2, 0.0],
    [5.0, 5.0, 0.0],
]
nbrs = [[1], [0, 2], [1], []]
nrms = [[0, 1, 0]] * 4
store = {"verts": [list(v) for v in verts]}
smoothing_utils.get_mesh_vertices = lambda name: [list(v) for v in store["verts"]]
smoothing_utils.set_mesh_vertices = lambda name, v: store.__setitem__("written", True)
smoothing_utils.get_vertex_neighbors = lambda name: nbrs
smoothing_utils.get_vertex_normals = lambda name: nrms
smoothing_utils.get_mesh_fn = lambda name: None
smoothing_utils.get_triangle_topology = lambda fn: None
m_dry = smoothing_utils.constrained_smooth_mesh_region(
    "skin", [1], plane.sdf_query, 0.8,
    method="laplacian", strength=0.5, iterations=2,
    anatomy_backend=plane, constraint_solver="iterative_exact",
    resolve_initial_penetration=False,
    resolve_initial_surface_intersections=True,
    prevent_surface_intersections=True,
    prevent_segment_crossing=False, preserve_tangential=False,
    clearance_policy="global", apply=False, verbose=False,
    skin_topology=_skin_topo([(0, 0, 1, 2)], 4))
check("apply=False did not write", store.get("written") is not True)
check("intersection metrics present",
      "intersecting_skin_face_count_after" in m_dry)
m_leg = smoothing_utils.constrained_smooth_mesh_region(
    "skin", [1], plane.sdf_query, 0.8,
    method="laplacian", strength=0.2, iterations=2,
    anatomy_backend=plane, constraint_solver="legacy_single_push",
    resolve_initial_penetration=False,
    resolve_initial_surface_intersections=False,
    prevent_surface_intersections=False,
    prevent_segment_crossing=False, preserve_tangential=False,
    clearance_policy="global", apply=False, verbose=False)
check("legacy solver still works", m_leg["constraint_solver"] == "legacy_single_push")

if fails:
    print("\nFAILED {0}: {1}".format(len(fails), fails))
    sys.exit(1)
print("\nAll surface-intersection checks passed.")
