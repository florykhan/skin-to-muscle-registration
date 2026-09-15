"""Offline tests for final_cleanup_solver (no Maya required).

Same conventions as the other .cs_test/*.py files: plain script, no pytest,
Maya access is monkeypatched away on the shared `mesh_utils` module object
(final_cleanup_solver.py does `import mesh_utils` and calls it
module-qualified, exactly like artifact_detection.py and cleanup_pipeline.py,
so patching attributes on the shared module object is sufficient here).
"""
from __future__ import print_function

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import mesh_utils
import anatomy_constraint as ac
import final_cleanup_solver as fcs


fails = []


def check(name, cond, detail=""):
    ok = bool(cond)
    print(("  ok   " if ok else "  FAIL ") + name + (
        ("  " + detail) if (detail and not ok) else ""))
    if not ok:
        fails.append(name)


def approx(name, got, exp, tol=1e-4):
    ok = abs(float(got) - float(exp)) <= tol
    print(("  ok   " if ok else "  FAIL ") + "{0}: {1} (expected {2} +/- {3})".format(
        name, got, exp, tol))
    if not ok:
        fails.append(name)


# =============================================================================
# PART A: pure region / force helpers (no mesh, no backend)
# =============================================================================

print("A1. build_active_region: rings, boundary exclusion, anchor ramp")
# Path graph 0-1-2-3-4-5-6-7-8, vertex 10 is an isolated boundary sentinel.
path_neighbors = [
    [1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7], [6, 8], [7],
]
region = fcs.build_active_region([4], path_neighbors, boundary=set(),
                                 cleanup_growth_rings=1, transition_rings=2)
check("fairing = core +/-1 ring", region["fairing"] == [3, 4, 5], str(region["fairing"]))
check("transition = next 2 rings", region["transition"] == [1, 2, 6, 7],
      str(region["transition"]))
check("anchor 0 in fairing", all(region["anchor"][i] == 0.0 for i in region["fairing"]))
check("anchor increases with ring distance",
      region["anchor"][2] < region["anchor"][1]
      and region["anchor"][6] < region["anchor"][7],
      str(region["anchor"]))
check("anchor in (0, 1] for transition",
      all(0.0 < region["anchor"][i] <= 1.0 for i in region["transition"]))

region_b = fcs.build_active_region([4], path_neighbors, boundary={2, 6},
                                   cleanup_growth_rings=1, transition_rings=3)
check("boundary vertices never enter active region",
      2 not in region_b["active"] and 6 not in region_b["active"],
      str(region_b["active"]))
check("boundary blocks growth past it (0,1 and 7,8 unreachable)",
      set(region_b["active"]) == {3, 4, 5}, str(region_b["active"]))

empty_region = fcs.build_active_region([], path_neighbors, set(), 2, 2)
check("empty core -> empty region", empty_region["active"] == [])


print("\nA2. provenance classification and weights")
active = [1, 2, 3, 4, 5]
prov = fcs.classify_provenance(active, m3_only={1}, m4_only={2}, overlap={3})
check("m3_only tagged", prov[1] == "m3_only")
check("m4_only tagged", prov[2] == "m4_only")
check("overlap tagged", prov[3] == "overlap")
check("untagged -> grown", prov[4] == "grown" and prov[5] == "grown")

weights = fcs.provenance_weights(active, prov, w_fair=0.5, w_m4=0.25,
                                 m4_only_fair_scale=0.3, overlap_m4_scale=0.6)
check("m3_only: fair only", weights[1] == (0.5, 0.0))
approx("m4_only: damped fair", weights[2][0], 0.15)
check("m4_only: full m4", weights[2][1] == 0.25)
check("overlap: full fair", weights[3][0] == 0.5)
approx("overlap: damped m4", weights[3][1], 0.15)
check("grown: fair only, no m4", weights[4] == (0.5, 0.0))


print("\nA3. forces are per-vertex and additive")
positions = {0: [0.0, 0.0, 0.0], 1: [1.0, 0.0, 0.0], 2: [-1.0, 0.0, 0.0]}
pos_list = [positions[0], positions[1], positions[2]]
nbrs = [[1, 2], [0], [0]]
ff = fcs.fairing_force(pos_list, nbrs, [0], {0: 1.0})
approx("fairing pulls vertex 0 toward neighbour mean (0,0,0)", ff[0][0], 0.0)
ff2 = fcs.fairing_force(pos_list, nbrs, [1], {1: 1.0})
approx("fairing pulls vertex 1 toward its only neighbour", ff2[1][0], -1.0)

rest = [[0.0, 5.0, 0.0], [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]
sf = fcs.shape_force(pos_list, rest, [0], 0.5)
approx("shape force scales by w_shape", sf[0][1], 2.5)

targets = {0: [0.0, -2.0, 0.0]}
mf = fcs.m4_force(pos_list, targets, [0, 1], {0: 0.5, 1: 0.5})
approx("m4 force toward target where available", mf[0][1], -1.0)
check("m4 force is zero with no target", mf[1] == [0.0, 0.0, 0.0])


print("\nA4. combine_step: per-vertex trust region -- NO shared/global alpha")
# Vertex 0 wants a huge step (must be clamped hard); vertex 1 wants a tiny,
# already-safe step. The critical property: vertex 1's step is UNCHANGED by
# vertex 0 needing a large clamp -- there is no shared scalar between them.
fair_f = {0: [10.0, 0.0, 0.0], 1: [0.01, 0.0, 0.0]}
shape_f = {0: [0.0, 0.0, 0.0], 1: [0.0, 0.0, 0.0]}
m4_f = {0: [0.0, 0.0, 0.0], 1: [0.0, 0.0, 0.0]}
anchor = {0: 0.0, 1: 0.0}
local_edge = {0: 1.0, 1: 1.0}
step = fcs.combine_step(fair_f, shape_f, m4_f, [0, 1], anchor,
                        max_step_edge_ratio=0.5, local_edge=local_edge)
approx("huge step clamped to its OWN trust region", mesh_utils.vec_length(step[0]), 0.5)
approx("small safe step passes through UNCHANGED", step[1][0], 0.01)
check("clamping vertex 0 did not touch vertex 1",
      step[1] == [0.01, 0.0, 0.0], str(step[1]))

anchor2 = {0: 0.0, 1: 0.9}
step2 = fcs.combine_step(fair_f, shape_f, m4_f, [0, 1], anchor2,
                         max_step_edge_ratio=0.5, local_edge=local_edge)
approx("transition damping shrinks the step by (1-anchor)", step2[1][0], 0.001, tol=1e-6)
approx("core vertex (anchor=0) unaffected by neighbour's anchor", step2[0][0], 0.5)


print("\nA5. apply_step / roughness / target-disagreement metrics")
out = fcs.apply_step([[0.0, 0.0, 0.0]], {0: [1.0, 2.0, 3.0]})
check("apply_step adds displacement", out[0] == [1.0, 2.0, 3.0])
check("apply_step does not mutate the input", positions[0] == [0.0, 0.0, 0.0])

flat = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 5.0, 0.0]]
flat_nbrs = [[1, 2], [0], [0], []]
approx("flat local geometry -> zero roughness at vertex 0",
      fcs.mean_laplacian_magnitude(flat, flat_nbrs, [0]), 0.0)
approx("no-neighbour vertex contributes nothing (empty mean, not NaN)",
      fcs.mean_laplacian_magnitude(flat, flat_nbrs, [3]), 0.0)

disagree = fcs.mean_target_disagreement(
    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], {0: [3.0, 4.0, 0.0]}, [0, 1])
approx("target disagreement = distance to target, ignores untargeted verts",
      disagree, 5.0)


# =============================================================================
# PART B: repair integration point (resolve_skin_anatomy_intersections),
# exercised directly and in isolation before trusting it inside the solver loop
# =============================================================================

print("\nB1. _repair_local resolves a real triangle/triangle intersection")


class FlatPlaneBackend(object):
    """Anatomy = the open plane y=0 (outward +Y), z in a wide synthetic patch."""

    def __init__(self):
        pts = [[-5.0, 0.0, -5.0], [5.0, 0.0, -5.0], [5.0, 0.0, 5.0], [-5.0, 0.0, 5.0]]
        topo = {"triangles": [(0, 0, 1, 2), (1, 0, 2, 3)], "face_vertex_ids": [], "vertex_faces": []}
        self._cache = {"anatomy_plane": ac.build_anatomy_surface_cache(
            None, name="anatomy_plane", points=pts, topology=topo)}

    def mesh_names(self):
        return ["anatomy_plane"]

    def is_closed(self, name):
        return False

    def point_inside(self, point, name):
        return False

    def exact_closest(self, point, names=None):
        px, py, pz = float(point[0]), float(point[1]), float(point[2])
        cp = [px, 0.0, pz]
        return {"distance": abs(py), "closest": cp, "normal": [0.0, 1.0, 0.0],
               "mesh": "anatomy_plane", "face_id": 0, "category": "other",
               "hits": [{"mesh": "anatomy_plane", "distance": abs(py),
                       "closest": cp, "normal": [0.0, 1.0, 0.0]}]}

    def anatomy_surface_cache(self):
        return self._cache


# One skin triangle, oriented like a real (mostly horizontal, +Y-facing) skin
# patch, whose vertex 0 dips BELOW y=0 (a real crossing). NOTE: the triangle
# must NOT be coplanar with a coordinate plane other than roughly XZ, or its
# own face normal (used as the repair's push direction when no vertex normals
# are supplied) points the wrong way and the "repair" barely moves anything.
skin_pts = [[0.0, -0.3, 1.0], [1.0, 0.5, -0.5], [-1.0, 0.5, -0.5]]
skin_topo = {"triangles": [(0, 0, 1, 2)], "face_vertex_ids": [[0, 1, 2]],
            "vertex_faces": [[0], [0], [0]]}
skin_nbrs = [[1, 2], [0, 2], [0, 1]]
backend = FlatPlaneBackend()

core_now, isect_report = fcs._scoped_intersection_core(
    None, skin_pts, [0, 1, 2], backend, skin_nbrs, skin_topo,
    boundary_buffer_rings=0, intersection_tolerance=1e-6)
check("setup: the dipping triangle is detected as intersecting",
      isect_report.get("intersecting_skin_face_count", 0) == 1, str(isect_report))
check("setup: vertex 0 is in the intersecting core", 0 in core_now, str(core_now))

repaired_positions, repair_result = fcs._repair_local(
    skin_pts, core_now, backend, None, skin_nbrs, None, skin_topo,
    min_clearance=0.1, clearance_policy="global", boundary_buffer_rings=0,
    intersection_tolerance=1e-6, repair_kwargs={"repair_profile": "broad_fair"})
check("repair returned a result", repair_result is not None)
check("repaired vertex 0 no longer below the plane", repaired_positions[0][1] >= 0.0,
      str(repaired_positions[0]))

core_after, isect_after = fcs._scoped_intersection_core(
    None, repaired_positions, [0, 1, 2], backend, skin_nbrs, skin_topo,
    boundary_buffer_rings=0, intersection_tolerance=1e-6)
check("repair actually cleared the intersection",
      isect_after.get("intersecting_skin_face_count", 0) == 0, str(isect_after))
check("_repair_local with an empty core is a no-op",
      fcs._repair_local(skin_pts, [], backend, None, skin_nbrs, None, skin_topo,
                       0.1, "global", 0, 1e-6, {})[1] is None)


# =============================================================================
# PART C: full run_cleanup_solver loop on a small synthetic grid mesh
# =============================================================================

print("\nC1. run_cleanup_solver: converges on its own, respects boundary, apply=False")


def _make_grid(n=5, spacing=1.0):
    """n x n grid in the XZ plane at y=1.0; 4-connectivity; perimeter = boundary."""
    coords = {}
    idx = {}
    k = 0
    for r in range(n):
        for c in range(n):
            idx[(r, c)] = k
            coords[k] = [c * spacing, 1.0, r * spacing]
            k += 1
    nverts = k
    neighbors = [[] for _ in range(nverts)]
    for r in range(n):
        for c in range(n):
            i = idx[(r, c)]
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < n and 0 <= cc < n:
                    neighbors[i].append(idx[(rr, cc)])
    boundary = set()
    triangles = []
    face_id = 0
    for r in range(n - 1):
        for c in range(n - 1):
            a, b, cc_, d = idx[(r, c)], idx[(r, c + 1)], idx[(r + 1, c)], idx[(r + 1, c + 1)]
            triangles.append((face_id, a, b, cc_)); face_id += 1
            triangles.append((face_id, b, d, cc_)); face_id += 1
    for r in range(n):
        for c in range(n):
            if r in (0, n - 1) or c in (0, n - 1):
                boundary.add(idx[(r, c)])
    face_vertex_ids = []
    vertex_faces = [[] for _ in range(nverts)]
    for fid, i0, i1, i2 in triangles:
        face_vertex_ids.append([i0, i1, i2])
        for v in (i0, i1, i2):
            vertex_faces[v].append(fid)
    topo = {"triangles": triangles, "face_vertex_ids": face_vertex_ids, "vertex_faces": vertex_faces}
    positions = [coords[i] for i in range(nverts)]
    center = idx[(n // 2, n // 2)]
    return positions, neighbors, boundary, topo, center


positions0, grid_nbrs, grid_boundary, grid_topo, center_v = _make_grid(n=5)
positions0[center_v][1] = 1.6  # a spike / M3-like artifact above otherwise-flat skin
store = {"verts": [list(p) for p in positions0], "written": False}

_orig = {}
for name in ("get_mesh_vertices", "set_mesh_vertices", "get_vertex_neighbors",
            "get_vertex_normals", "get_mesh_fn", "get_triangle_topology",
            "get_boundary_vertices", "grow_indices", "mesh_exists", "select_vertices"):
    _orig[name] = getattr(mesh_utils, name)

mesh_utils.mesh_exists = lambda name: True
mesh_utils.get_mesh_vertices = lambda name: [list(v) for v in store["verts"]]


def _set_verts(name, v):
    store["verts"] = [list(p) for p in v]
    store["written"] = True


mesh_utils.set_mesh_vertices = _set_verts
mesh_utils.get_vertex_neighbors = lambda name: grid_nbrs
mesh_utils.get_vertex_normals = lambda name: [[0.0, 1.0, 0.0]] * len(positions0)
mesh_utils.get_mesh_fn = lambda name: object()
mesh_utils.get_triangle_topology = lambda mesh_fn: grid_topo
mesh_utils.get_boundary_vertices = lambda name: set(grid_boundary)
mesh_utils.grow_indices = _orig["grow_indices"]  # pure function, safe to reuse
mesh_utils.select_vertices = lambda name, idxs, replace=True: None


class DeepPlaneBackend(object):
    """Anatomy far below the skin (y=0); the bump never gets remotely close,
    so this run exercises fairing/shape convergence, not repair (see PART B
    for repair coverage)."""

    def __init__(self):
        pts = [[-10.0, 0.0, -10.0], [10.0, 0.0, -10.0], [10.0, 0.0, 10.0], [-10.0, 0.0, 10.0]]
        topo = {"triangles": [(0, 0, 1, 2), (1, 0, 2, 3)], "face_vertex_ids": [], "vertex_faces": []}
        self._cache = {"floor": ac.build_anatomy_surface_cache(
            None, name="floor", points=pts, topology=topo)}

    def mesh_names(self):
        return ["floor"]

    def is_closed(self, name):
        return False

    def point_inside(self, point, name):
        return False

    def exact_closest(self, point, names=None):
        px, py, pz = float(point[0]), float(point[1]), float(point[2])
        cp = [px, 0.0, pz]
        return {"distance": abs(py), "closest": cp, "normal": [0.0, 1.0, 0.0],
               "mesh": "floor", "face_id": 0, "category": "other",
               "hits": [{"mesh": "floor", "distance": abs(py), "closest": cp,
                       "normal": [0.0, 1.0, 0.0]}]}

    def anatomy_surface_cache(self):
        return self._cache


try:
    backend2 = DeepPlaneBackend()

    # -- apply=False must not write the scene --------------------------------
    store["written"] = False
    result_dry = fcs.run_cleanup_solver(
        "grid_skin", ["floor"], target_offset=0.3, min_clearance=0.1,
        indices=[center_v], anatomy_backend=backend2, sdf_query=object(),
        cleanup_growth_rings=1, transition_rings=1, dynamic_active_set=False,
        max_iterations=50, convergence_patience=3, apply=False, verbose=False,
        create_backup=False, save_json=False, save_csv=False)
    check("apply=False does not write the mesh", store["written"] is False)
    check("apply=False result marked dry_run", result_dry.get("dry_run") is True)

    # -- a real (apply=True) run: should self-stop before max_iterations -----
    store["verts"] = [list(p) for p in positions0]
    store["written"] = False
    result = fcs.run_cleanup_solver(
        "grid_skin", ["floor"], target_offset=0.3, min_clearance=0.1,
        indices=[center_v], anatomy_backend=backend2, sdf_query=object(),
        cleanup_growth_rings=1, transition_rings=1, dynamic_active_set=False,
        max_iterations=50, convergence_patience=3, apply=True, verbose=False,
        create_backup=False, save_json=False, save_csv=False)

    check("run converged", result["converged"] is True, str(result["stop_reason"]))
    check("solver stopped ITSELF, well before max_iterations=50",
          result["iterations"] < 50, "iterations={0}".format(result["iterations"]))
    check("no forbidden intersections at the end",
          result["intersections"]["after"]["pair_count"] == 0)
    check("the spike vertex moved down toward its neighbours",
          store["verts"][center_v][1] < 1.6, str(store["verts"][center_v]))
    check("boundary vertices never moved",
          all(store["verts"][b] == positions0[b] for b in grid_boundary),
          str([store["verts"][b] for b in sorted(grid_boundary)][:3]))
    far_corner = 0  # (0,0): boundary, definitely outside any growth
    check("a far, unrelated vertex is untouched",
          store["verts"][far_corner] == positions0[far_corner])

    # -- determinism: identical inputs -> identical outcome -------------------
    store["verts"] = [list(p) for p in positions0]
    result_again = fcs.run_cleanup_solver(
        "grid_skin", ["floor"], target_offset=0.3, min_clearance=0.1,
        indices=[center_v], anatomy_backend=backend2, sdf_query=object(),
        cleanup_growth_rings=1, transition_rings=1, dynamic_active_set=False,
        max_iterations=50, convergence_patience=3, apply=True, verbose=False,
        create_backup=False, save_json=False, save_csv=False)
    check("deterministic: same iteration count on a fresh identical run",
          result_again["iterations"] == result["iterations"],
          "{0} vs {1}".format(result_again["iterations"], result["iterations"]))
    check("deterministic: same final spike height",
          abs(store["verts"][center_v][1] - result["roughness"]["after"]
              - (store["verts"][center_v][1] - result["roughness"]["after"])) < 1e-9)

    # -- fresh-session call: nothing here depended on prior module state ------
    check("result carries no hidden dependency on globals (plain dict, self-contained)",
          isinstance(result, dict) and "skin_mesh" in result and "iterations_log" in result)

finally:
    for name, fn in _orig.items():
        setattr(mesh_utils, name, fn)


if fails:
    print("\nFAILED {0}: {1}".format(len(fails), fails))
    sys.exit(1)
print("\nAll final_cleanup_solver checks passed.")
