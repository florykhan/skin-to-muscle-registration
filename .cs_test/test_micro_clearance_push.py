"""Offline tests for micro_clearance_push (no Maya required).

Same conventions as the other .cs_test/*.py files: plain script, no pytest,
Maya access monkeypatched on the shared `mesh_utils` module object. The real
triangle/triangle intersection detector AND the real
``anatomy_constraint.enforce_anatomy_clearance`` closest-point/outward
machinery are both exercised directly against a small, fully-synthetic
"fake but real" anatomy backend (a flat plane patch), the same established
pattern test_final_cleanup_solver.py uses (FlatPlaneBackend/DeepPlaneBackend).
"""
from __future__ import print_function

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import mesh_utils
import anatomy_constraint as ac
import micro_clearance_push as mcp


fails = []


def check(name, cond, detail=""):
    ok = bool(cond)
    print(("  ok   " if ok else "  FAIL ") + name + (
        ("  " + detail) if (detail and not ok) else ""))
    if not ok:
        fails.append(name)


def approx(name, got, exp, tol=1e-6):
    ok = abs(float(got) - float(exp)) <= tol
    print(("  ok   " if ok else "  FAIL ") + "{0}: {1} (expected {2} +/- {3})".format(
        name, got, exp, tol))
    if not ok:
        fails.append(name)


class PushPlaneBackend(object):
    """Fake anatomy backend: a flat horizontal CLOSED half-space plane at
    y=height (below = inside anatomy, above = outside/safe) -- realistic for
    this project's real anatomy meshes (closed organs), and the case
    anatomy_constraint.enforce_anatomy_clearance's inside-escape branch is
    designed for. Implements exactly the methods the real
    MayaAnatomyBackend/enforce_anatomy_clearance call: anatomy_surface_cache
    (intersection scan), exact_closest/is_closed/point_inside (clearance
    push)."""

    def __init__(self, height=1.0, half_extent=20.0, name="anatomy_plane"):
        h = float(half_extent)
        self._height = float(height)
        self._name = name
        pts = [[-h, height, -h], [h, height, -h], [h, height, h], [-h, height, h]]
        topo = {"triangles": [(0, 0, 1, 2), (1, 0, 2, 3)], "face_vertex_ids": [], "vertex_faces": []}
        self._cache = {name: ac.build_anatomy_surface_cache(None, name=name, points=pts, topology=topo)}

    def anatomy_surface_cache(self):
        return self._cache

    def is_closed(self, name):
        return True

    def point_inside(self, point, name):
        return point[1] < self._height

    def exact_closest(self, point, names=None):
        px, py, pz = float(point[0]), float(point[1]), float(point[2])
        cp = [px, self._height, pz]
        dist = abs(py - self._height)
        return {"distance": dist, "closest": cp, "normal": [0.0, 1.0, 0.0],
               "mesh": self._name, "face_id": 0, "category": "other",
               "hits": [{"mesh": self._name, "distance": dist, "closest": cp,
                       "normal": [0.0, 1.0, 0.0]}]}


# =============================================================================
# PART A: pure helpers (no mesh, no Maya, no real intersection scan)
# =============================================================================

print("A1. ring_distances: plain BFS")
path_neighbors = [[1], [0, 2], [1, 3], [2, 4], [3]]
d = mcp.ring_distances([2], path_neighbors, 2)
check("ring distances correct", d == {2: 0, 1: 1, 3: 1, 0: 2, 4: 2}, str(d))

print("\nA2. blend_corrections: core keeps its EXACT correction (weight 1.0)")
star_nbrs = [[1, 2, 3], [0], [0], [0]]  # 0=hub core, 1/2/3=rim, all adjacent to 0
core_corr = {0: [1.0, 2.0, 0.0]}
b0 = mcp.blend_corrections(core_corr, star_nbrs, blend_rings=0)
check("blend_rings=0 -> only the core, unmodified", b0 == {0: [1.0, 2.0, 0.0]}, str(b0))

print("\nA3. blend_corrections: ring1 = 0.5 * mean(core neighbor corrections)")
b1 = mcp.blend_corrections(core_corr, star_nbrs, blend_rings=1)
check("core unchanged", b1[0] == [1.0, 2.0, 0.0])
for j in (1, 2, 3):
    check("rim vertex {0} gets exactly half the core correction".format(j),
         b1[j] == [0.5, 1.0, 0.0], str(b1.get(j)))

print("\nA4. blend_corrections: TWO core neighbors touching the same halo vertex "
     "are averaged, then halved")
diamond_nbrs = [[2], [2], [0, 1, 3], [2]]  # 0,1 = two core hubs; 2 = shared halo; 3 = untouched
two_core = {0: [2.0, 0.0, 0.0], 1: [0.0, 4.0, 0.0]}
b_diamond = mcp.blend_corrections(two_core, diamond_nbrs, blend_rings=1)
approx("halo vertex 2's x: 0.5*mean(2.0,0.0)=0.5", b_diamond[2][0], 0.5)
approx("halo vertex 2's y: 0.5*mean(0.0,4.0)=1.0", b_diamond[2][1], 1.0)
check("vertex 3 (not adjacent to any core vertex) is untouched",
      3 not in b_diamond, str(b_diamond))

print("\nA5. blend_corrections: blend_rings=2 continues the halving pattern outward "
     "along a chain")
chain_nbrs = [[1], [0, 2], [1, 3], [2]]  # 0=core, 1=ring1, 2=ring2, 3=beyond
chain_core = {0: [4.0, 0.0, 0.0]}
b_chain = mcp.blend_corrections(chain_core, chain_nbrs, blend_rings=2)
approx("ring1 = 0.5*4.0 = 2.0", b_chain[1][0], 2.0)
approx("ring2 = 0.5*ring1 = 1.0 (continues the halving, not re-derived from the "
      "original core value)", b_chain[2][0], 1.0)
check("beyond blend_rings is never touched", 3 not in b_chain, str(b_chain))

print("\nA6. blend_corrections: excluded vertices never receive a blended value")
b_excl = mcp.blend_corrections(core_corr, star_nbrs, blend_rings=1, exclude={1})
check("excluded rim vertex 1 gets nothing even though adjacent to core",
      1 not in b_excl, str(b_excl))
check("other rim vertices unaffected by the exclusion", 2 in b_excl and 3 in b_excl)

print("\nA7. blend_corrections: empty core -> empty result")
check("no core corrections -> nothing blended",
      mcp.blend_corrections({}, star_nbrs, 1) == {})


# =============================================================================
# PART B: full run_micro_clearance_push, real intersection detector + real
# enforce_anatomy_clearance closest-point/outward machinery
# =============================================================================

_orig = {}
for name in ("get_mesh_vertices", "set_mesh_vertices", "get_vertex_neighbors",
            "get_mesh_fn", "get_triangle_topology", "get_boundary_vertices",
            "get_vertex_normals", "mesh_exists"):
    _orig[name] = getattr(mesh_utils, name)


def _install_mesh(positions, neighbors, topology, boundary=None):
    store = {"verts": [list(p) for p in positions], "written": False}
    mesh_utils.mesh_exists = lambda name: True
    mesh_utils.get_mesh_vertices = lambda name: [list(v) for v in store["verts"]]

    def _set_verts(name, v):
        store["verts"] = [list(p) for p in v]
        store["written"] = True

    mesh_utils.set_mesh_vertices = _set_verts
    mesh_utils.get_vertex_neighbors = lambda name: neighbors
    mesh_utils.get_mesh_fn = lambda name: object()
    mesh_utils.get_triangle_topology = lambda mesh_fn: topology
    mesh_utils.get_boundary_vertices = lambda name: set(boundary or [])
    mesh_utils.get_vertex_normals = lambda name: None
    return store


try:
    # -------------------------------------------------------------------
    # B1. Real crossing -> core vertex pushed to EXACTLY target_clearance;
    #     halo (its 1-ring) blended at half weight, no re-projection; final
    #     scan clean.
    # -------------------------------------------------------------------
    print("\nB1. one real crossing: core pushed to exact target_clearance, halo "
         "blended at 0.5x, final forbidden count is 0")
    combo_pos = [
        [0.0, 0.5, 0.0],    # 0: A (hub, genuinely crossing the plane at y=1.0)
        [1.0, 3.0, 0.0],    # 1: rim (fixed, safely above)
        [-1.0, 3.0, 0.5],   # 2: rim (fixed, safely above)
        [0.5, 2.0, 1.0],    # 3: Y (adjacent to A, no shared face -> halo only)
    ]
    combo_nbrs = [[1, 2, 3], [0], [0], [0]]
    combo_topo = {
        "triangles": [(0, 0, 1, 2)],       # ONLY A's fan; vertex 3 has no faces
        "face_vertex_ids": [[0, 1, 2], [], [], []],
        "vertex_faces": [[0], [0], [0], []],
    }
    plane = PushPlaneBackend(height=1.0, half_extent=20.0)

    _install_mesh(combo_pos, combo_nbrs, combo_topo)
    b1 = mcp.run_micro_clearance_push(
        "combo_skin", plane, target_clearance=0.15, blend_rings=1,
        apply=True, create_backup=False, verbose=False)

    check("initial forbidden face count is 1 (A's fan straddles the plane)",
          b1["initial_forbidden_face_count"] == 1, str(b1))
    check("core_indices is exactly [0] (A)", b1["core_indices"] == [0], str(b1["core_indices"]))
    check("corrected_core_vertex_count is 1", b1["corrected_core_vertex_count"] == 1)
    check("halo_indices includes every 1-ring neighbor of A (1, 2, and 3)",
          b1["halo_indices"] == [1, 2, 3], str(b1["halo_indices"]))
    check("blended_vertex_count (halo only) is 3", b1["blended_vertex_count"] == 3)
    check("final forbidden face count is 0 after the push", b1["final_forbidden_face_count"] == 0)

    committed = mesh_utils.get_mesh_vertices("combo_skin")
    approx("A (core) lands EXACTLY at height + target_clearance = 1.15",
          committed[0][1], 1.15, tol=1e-6)
    approx("A's correction is exactly 1.15 - 0.5 = 0.65",
          b1["max_correction_displacement"], 0.65, tol=1e-6)
    check("A is the max-displacement vertex", b1["max_correction_vertex"] == 0)
    for j in (1, 2, 3):
        approx("halo vertex {0} moved by exactly HALF of A's correction "
              "(0.5*0.65=0.325), no independent anatomy re-projection".format(j),
              committed[j][1] - combo_pos[j][1], 0.325, tol=1e-6)
    approx("mean correction displacement over core+halo: (0.65+0.325*3)/4",
          b1["mean_correction_displacement"], (0.65 + 0.325 * 3) / 4.0, tol=1e-6)

    # -------------------------------------------------------------------
    # B2. protect_boundary excludes a vertex from the halo blend even though
    #     it is topologically adjacent to the core.
    # -------------------------------------------------------------------
    print("\nB2. boundary-protected vertex never receives a blended correction")
    _install_mesh(combo_pos, combo_nbrs, combo_topo, boundary={1})
    b2 = mcp.run_micro_clearance_push(
        "combo_skin", plane, target_clearance=0.15, blend_rings=1,
        protect_boundary=True, apply=True, create_backup=False, verbose=False)
    check("boundary vertex 1 excluded from halo_indices",
          1 not in b2["halo_indices"], str(b2["halo_indices"]))
    check("vertices 2 and 3 still blended normally",
          2 in b2["halo_indices"] and 3 in b2["halo_indices"])
    committed2 = mesh_utils.get_mesh_vertices("combo_skin")
    check("boundary vertex 1's position is completely untouched",
          committed2[1] == combo_pos[1], str(committed2[1]))

    # -------------------------------------------------------------------
    # B3. No forbidden intersections at all -> clean no-op, nothing moved.
    # -------------------------------------------------------------------
    print("\nB3. nothing forbidden -> no-op, nothing moved")
    far_plane = PushPlaneBackend(height=-10000.0, half_extent=5.0)
    _install_mesh(combo_pos, combo_nbrs, combo_topo)
    b3 = mcp.run_micro_clearance_push(
        "combo_skin", far_plane, target_clearance=0.15, blend_rings=1,
        apply=True, create_backup=False, verbose=False)
    check("no core vertices found", b3["core_indices"] == [])
    check("no halo vertices either", b3["halo_indices"] == [])
    check("initial and final forbidden counts both 0",
          b3["initial_forbidden_face_count"] == 0 and b3["final_forbidden_face_count"] == 0)
    committed3 = mesh_utils.get_mesh_vertices("combo_skin")
    check("mesh completely unchanged", committed3 == combo_pos)

    # -------------------------------------------------------------------
    # B4. apply=False: computed in memory, scene not written.
    # -------------------------------------------------------------------
    print("\nB4. apply=False does not write the mesh")
    _install_mesh(combo_pos, combo_nbrs, combo_topo)
    store4 = _install_mesh(combo_pos, combo_nbrs, combo_topo)
    b4 = mcp.run_micro_clearance_push(
        "combo_skin", plane, target_clearance=0.15, blend_rings=1,
        apply=False, create_backup=False, verbose=False)
    check("apply=False -> nothing written", store4["written"] is False)
    check("dry_run flag set", b4["dry_run"] is True)
    check("core/halo are still computed and reported even in a dry run",
          b4["core_indices"] == [0] and b4["halo_indices"] == [1, 2, 3])

finally:
    for name, fn in _orig.items():
        setattr(mesh_utils, name, fn)


if fails:
    print("\nFAILED {0}: {1}".format(len(fails), fails))
    sys.exit(1)
print("\nAll micro_clearance_push checks passed.")
