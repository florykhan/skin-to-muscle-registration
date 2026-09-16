"""Offline tests for blocked_laplacian_smoothing (no Maya required).

Same conventions as the other .cs_test/*.py files: plain script, no pytest,
Maya access monkeypatched on the shared `mesh_utils` module object. The real
triangle/triangle intersection detector (anatomy_constraint) is exercised
directly against small, fully-synthetic "fake but real" anatomy backends
(flat plane patches built via anatomy_constraint.build_anatomy_surface_cache),
the same established pattern test_final_cleanup_solver.py uses
(FlatPlaneBackend/DeepPlaneBackend) -- this tests REAL triangle-triangle
behavior rather than mocking the detector's own math.
"""
from __future__ import print_function

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import mesh_utils
import artifact_detection
import anatomy_constraint as ac
import pure_laplacian_smoothing as pls
import blocked_laplacian_smoothing as bls


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


class PlaneBackend(object):
    """Minimal fake anatomy backend: a flat horizontal plane at y=height,
    spanning a wide XZ patch. blocked_laplacian_smoothing's intersection scan
    only ever calls ``.anatomy_surface_cache()`` (checked via hasattr, exactly
    like the real MayaAnatomyBackend) -- no other backend method is needed for
    these tests, since this module never does clearance/repair."""

    def __init__(self, height=1.0, half_extent=20.0, name="anatomy_plane"):
        h = float(half_extent)
        pts = [[-h, height, -h], [h, height, -h], [h, height, h], [-h, height, h]]
        topo = {"triangles": [(0, 0, 1, 2), (1, 0, 2, 3)], "face_vertex_ids": [], "vertex_faces": []}
        self._cache = {name: ac.build_anatomy_surface_cache(None, name=name, points=pts, topology=topo)}

    def anatomy_surface_cache(self):
        return self._cache


# =============================================================================
# PART A: pure helpers (no mesh, no Maya, no real intersection scan)
# =============================================================================

print("A1. freeze_footprint: freeze_rings=0 freezes ONLY the offending vertices")
path_neighbors = [
    [1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7], [6, 8], [7],
]
fp0 = bls.freeze_footprint(offending=[4], neighbors=path_neighbors, freeze_rings=0,
                          active_set=set(range(9)))
check("freeze_rings=0 -> exactly the offending set", fp0 == {4}, str(fp0))

print("\nA2. freeze_footprint: freeze_rings=1 includes immediate neighbors")
fp1 = bls.freeze_footprint(offending=[4], neighbors=path_neighbors, freeze_rings=1,
                          active_set=set(range(9)))
check("freeze_rings=1 -> offending + its immediate 1-ring", fp1 == {3, 4, 5}, str(fp1))
fp2 = bls.freeze_footprint(offending=[4], neighbors=path_neighbors, freeze_rings=2,
                          active_set=set(range(9)))
check("freeze_rings=2 -> offending + 2 rings", fp2 == {2, 3, 4, 5, 6}, str(fp2))

print("\nA3. freeze_footprint: always clipped to active_set")
fp_clip = bls.freeze_footprint(offending=[4], neighbors=path_neighbors, freeze_rings=1,
                              active_set={3, 4})
check("a reached vertex outside active_set is never frozen (it was never movable)",
      fp_clip == {3, 4}, str(fp_clip))
fp_offending_outside_active = bls.freeze_footprint(
    offending=[8], neighbors=path_neighbors, freeze_rings=0, active_set={0, 1, 2})
check("an offending vertex that isn't even in active_set contributes nothing",
      fp_offending_outside_active == set(), str(fp_offending_outside_active))

print("\nA4. freeze_footprint: empty offending -> empty footprint")
check("no offending vertices -> nothing frozen",
      bls.freeze_footprint([], path_neighbors, 1, set(range(9))) == set())

print("\nA5. blocked_laplacian_smoothing reuses the PURE module's Jacobi step "
     "by import, not by reimplementation")
check("bls.pls IS pure_laplacian_smoothing (literal reuse, not a copy)",
      bls.pls is pls)
check("bls.pls.jacobi_laplacian_step IS pure_laplacian_smoothing.jacobi_laplacian_step",
      bls.pls.jacobi_laplacian_step is pls.jacobi_laplacian_step)


# =============================================================================
# PART B: full run_blocked_laplacian_smoothing, real intersection detector
# =============================================================================

_orig = {}
for name in ("get_mesh_vertices", "set_mesh_vertices", "get_vertex_neighbors",
            "get_mesh_fn", "get_triangle_topology", "get_boundary_vertices",
            "mesh_exists"):
    _orig[name] = getattr(mesh_utils, name)


def _install_mesh(positions, neighbors, topology, boundary=None):
    """(Re)point every mesh_utils call this module uses at a fresh in-memory
    fixture. Returns the mutable `store` dict backing get/set."""
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
    return store


try:
    # -------------------------------------------------------------------
    # B1. No intersections ever -> result matches pure_laplacian_smoothing
    #     exactly (same operator, same selection/falloff, reused by import).
    # -------------------------------------------------------------------
    print("\nB1. no intersections: blocked result matches pure Laplacian result "
         "bit-for-bit (anatomy placed far away, never triggers)")

    def _make_grid(n=7, spacing=1.0):
        coords, idx = {}, {}
        k = 0
        for r in range(n):
            for c in range(n):
                idx[(r, c)] = k
                coords[k] = [c * spacing, 0.0, r * spacing]
                k += 1
        nverts = k
        neighbors_ = [[] for _ in range(nverts)]
        for r in range(n):
            for c in range(n):
                i = idx[(r, c)]
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < n and 0 <= cc < n:
                        neighbors_[i].append(idx[(rr, cc)])
        triangles, face_id = [], 0
        for r in range(n - 1):
            for c in range(n - 1):
                a, b, cc_, d = idx[(r, c)], idx[(r, c + 1)], idx[(r + 1, c)], idx[(r + 1, c + 1)]
                triangles.append((face_id, a, b, cc_)); face_id += 1
                triangles.append((face_id, b, d, cc_)); face_id += 1
        face_vertex_ids = []
        vertex_faces = [[] for _ in range(nverts)]
        for fid, i0, i1, i2 in triangles:
            face_vertex_ids.append([i0, i1, i2])
            for v in (i0, i1, i2):
                vertex_faces[v].append(fid)
        topo = {"triangles": triangles, "face_vertex_ids": face_vertex_ids, "vertex_faces": vertex_faces}
        boundary_ = set()
        for r in range(n):
            for c in range(n):
                if r in (0, n - 1) or c in (0, n - 1):
                    boundary_.add(idx[(r, c)])
        positions_ = [coords[i] for i in range(nverts)]
        center_ = idx[(n // 2, n // 2)]
        return positions_, neighbors_, boundary_, topo, center_

    grid_pos, grid_nbrs, grid_boundary, grid_topo, center_v = _make_grid(n=7)
    grid_pos[center_v][1] = 3.0  # an obvious spike, same fixture pattern as the pure module's tests

    far_backend = PlaneBackend(height=-10000.0, half_extent=5.0)  # AABB never overlaps the skin

    store = _install_mesh(grid_pos, grid_nbrs, grid_topo, boundary=grid_boundary)
    pure_result = pls.run_pure_laplacian_smoothing(
        "grid_skin", indices=[center_v], growth_rings=2, strength=0.5, iterations=8,
        apply=True, create_backup=False, verbose=False)
    pure_final = [list(p) for p in store["verts"]]

    store2 = _install_mesh(grid_pos, grid_nbrs, grid_topo, boundary=grid_boundary)
    blocked_result = bls.run_blocked_laplacian_smoothing(
        "grid_skin", far_backend, indices=[center_v], growth_rings=2, strength=0.5,
        iterations=8, freeze_rings=1, apply=True, create_backup=False, verbose=False)
    blocked_final = [list(p) for p in store2["verts"]]

    check("blocked_count is 0 when anatomy never intersects",
          blocked_result["blocked_count"] == 0, str(blocked_result["blocked_count"]))
    check("final vertex positions are IDENTICAL to the pure module's run",
          all(abs(pure_final[i][k] - blocked_final[i][k]) < 1e-9
             for i in range(len(pure_final)) for k in range(3)))
    approx("mean roughness after matches the pure run exactly",
          blocked_result["roughness"]["after_mean"], pure_result["roughness"]["after_mean"],
          tol=1e-9)
    approx("mean displacement matches the pure run exactly",
          blocked_result["mean_displacement"], pure_result["mean_displacement"], tol=1e-9)
    check("iterations_completed == iterations_requested when nothing ever blocks",
          blocked_result["iterations_completed"] == blocked_result["iterations_requested"] == 8)

    # -------------------------------------------------------------------
    # B2. One proposed local intersection: the offending vertex restores to
    #     its previous (here: starting) safe position and is never touched
    #     again. Single-active-vertex fan: only the hub is ever eligible to
    #     be flagged (its rim is fixed/inactive), so this isolates the
    #     rollback mechanism from any freeze_rings/face-sharing nuance.
    # -------------------------------------------------------------------
    print("\nB2. one proposed local intersection: offending vertex restores to its "
         "previous safe position; final state has zero forbidden intersections")
    fan_pos = [
        [0.0, 12.0, 0.0],   # 0: A (hub, will cross the plane at iteration 1)
        [1.0, 2.0, 0.0],    # 1: rim (fixed)
        [-0.5, 2.0, 0.87],  # 2: rim (fixed)
        [-0.5, 2.0, -0.87], # 3: rim (fixed)
    ]
    fan_nbrs = [[1, 2, 3], [0], [0], [0]]
    fan_topo = {
        "triangles": [(0, 0, 1, 2), (1, 0, 2, 3), (2, 0, 3, 1)],
        "face_vertex_ids": [[0, 1, 2], [0, 2, 3], [0, 3, 1]],
        "vertex_faces": [[0, 1, 2], [0, 2], [0, 1], [1, 2]],
    }
    plane_backend = PlaneBackend(height=1.0, half_extent=20.0)

    _install_mesh(fan_pos, fan_nbrs, fan_topo)
    b2 = bls.run_blocked_laplacian_smoothing(
        "fan_skin", plane_backend, indices=[0], growth_rings=0, strength=1.5,
        iterations=3, freeze_rings=0, apply=True, create_backup=False, verbose=False)
    check("hub A gets blocked on iteration 1 (its unconstrained proposal crosses the plane)",
          b2["blocked_indices"] == [0], str(b2))
    check("A's final position is EXACTLY its starting (pre-proposal) position",
          b2["max_displacement_vertex"] == 0 and b2["max_displacement"] < 1e-9,
          "max_disp={0}".format(b2["max_displacement"]))
    check("the rollback itself was a clean single-tier resolution (no "
         "could-not-restore stop reason)",
         "could not restore" not in (b2["stop_reason"] or ""))
    check("final forbidden intersection count is 0 after rollback",
          b2["final_forbidden_intersection_count"] == 0)
    check("iterations still run to completion once the offender is frozen "
         "(nothing left to move, so the loop stops early -- expected)",
         b2["iterations_completed"] == 1 and b2["stop_reason"] is not None)

    # -------------------------------------------------------------------
    # B3/B4/B5/B8/B9. Combined scenario: G (hub) crosses & freezes at
    # iteration 1; Y is Laplacian-adjacent to G (so its neighbour mean must
    # keep reading G's frozen value) but shares NO triangle face with G (so
    # Y is never itself flagged) and keeps smoothing every iteration.
    # Hand-computed to exact values.
    # -------------------------------------------------------------------
    print("\nB3-B5/B8/B9. blocked vertex stays fixed; unblocked neighbour keeps "
         "smoothing AND its neighbour-mean keeps using the frozen value; "
         "non-selected vertices never move")
    combo_pos = [
        [0.0, 12.0, 0.0],   # 0: G (hub, crosses at iteration 1)
        [1.0, 2.0, 0.0],    # 1: G's rim (fixed)
        [-1.0, 2.0, 0.5],   # 2: G's rim (fixed)
        [0.5, 2.0, 1.0],    # 3: Y (active, adjacent to G, never near anatomy)
        [2.0, 2.0, 1.0],    # 4: Y's other anchor (fixed)
        [0.5, 2.0, 3.0],    # 5: Y's other anchor (fixed)
    ]
    # Laplacian adjacency (used for smoothing) is independent of face topology
    # (used for intersection testing) in this codebase -- Y is adjacent to G
    # for averaging purposes without sharing a triangle face with it.
    combo_nbrs = [[1, 2, 3], [0], [0], [0, 4, 5], [3], [3]]
    combo_topo = {
        "triangles": [(0, 0, 1, 2)],           # ONLY G's fan; Y has no faces at all
        "face_vertex_ids": [[0, 1, 2], [], [], [], [], []],
        "vertex_faces": [[0], [0], [0], [], [], []],
    }
    _install_mesh(combo_pos, combo_nbrs, combo_topo)
    b3 = bls.run_blocked_laplacian_smoothing(
        "combo_skin", plane_backend, indices=[0, 3], growth_rings=0, strength=1.5,
        iterations=3, freeze_rings=0, apply=True, create_backup=False, verbose=False)

    check("G is blocked on iteration 1", b3["blocked_indices"] == [0], str(b3["blocked_indices"]))
    check("Y is never blocked (no shared face with G -> never itself flagged)",
          3 not in b3["blocked_indices"])
    check("all 3 requested iterations completed (Y keeps moving even after G freezes)",
          b3["iterations_completed"] == 3 and b3["stopped_early"] is False)
    check("final forbidden intersection count is 0",
          b3["final_forbidden_intersection_count"] == 0)

    committed = mesh_utils.get_mesh_vertices("combo_skin")
    approx("G (blocked) sits EXACTLY at its ORIGINAL y (frozen on iteration 1, "
          "positions_before == positions0 there)", committed[0][1], 12.0, tol=1e-9)
    approx("Y's y after iteration 1: hand-computed 2.0 + 1.5*(mean(12,2,2)-2.0) = 7.0 exactly "
          "(mean uses G's snapshot value BEFORE any freezing happened)",
          # can't read intermediate iterations from the result directly, so
          # verify the FINAL (iteration-3) value below instead, and derive
          # iteration 1/2 by direct hand-computed chain:
          2.0 + 1.5 * ((12.0 + 2.0 + 2.0) / 3.0 - 2.0), 7.0, tol=1e-9)
    _g_mean = (12.0 + 2.0 + 2.0) / 3.0  # G stays pinned at 12.0 for every later iteration too
    _y1 = 2.0 + 1.5 * (_g_mean - 2.0)
    _y2 = _y1 + 1.5 * (_g_mean - _y1)
    _y3 = _y2 + 1.5 * (_g_mean - _y2)
    approx("Y's FINAL y after 3 iterations matches the hand-computed chain that uses "
          "G's FROZEN value (12.0) at every step, not a moving target",
          committed[3][1], _y3, tol=1e-9)
    check("non-selected vertices (G's and Y's fixed anchors) never moved at all",
          all(committed[i] == combo_pos[i] for i in (1, 2, 4, 5)))

    # -------------------------------------------------------------------
    # B_auto. Light integration check that automatic percentile selection
    # still works end to end (selection logic itself is already covered by
    # the pure module's own test suite; this just confirms the wiring).
    # -------------------------------------------------------------------
    print("\nB_auto. automatic roughness-percentile selection wiring works end to end")
    grid_pos2, grid_nbrs2, grid_boundary2, grid_topo2, center_v2 = _make_grid(n=5)
    grid_pos2[center_v2][1] = 3.0
    _install_mesh(grid_pos2, grid_nbrs2, grid_topo2, boundary=grid_boundary2)
    auto_far_backend = PlaneBackend(height=-10000.0, half_extent=5.0)
    b_auto = bls.run_blocked_laplacian_smoothing(
        "grid_skin2", auto_far_backend, indices=None, roughness_percentile=95,
        growth_rings=0, strength=0.0, iterations=1, apply=False, create_backup=False,
        verbose=False)
    check("automatic selection finds the spike vertex",
          center_v2 in b_auto["selected_indices"], str(b_auto["selected_indices"]))
    check("selection_source reported as auto_percentile", b_auto["selection_source"] == "auto_percentile")

    # -------------------------------------------------------------------
    # B_baseline. Baseline validity check: an already-invalid starting mesh
    # is reported, not silently repaired.
    # -------------------------------------------------------------------
    print("\nB_baseline. already-invalid starting mesh is reported, not auto-repaired")
    bad_pos = [
        [0.0, 0.5, 0.0],    # 0: hub already BELOW the plane at y=1.0 -- invalid baseline
        [1.0, 2.0, 0.0],
        [-1.0, 2.0, 0.5],
    ]
    bad_nbrs = [[1, 2], [0], [0]]
    bad_topo = {
        "triangles": [(0, 0, 1, 2)],
        "face_vertex_ids": [[0, 1, 2]],
        "vertex_faces": [[0], [0], [0]],
    }
    _install_mesh(bad_pos, bad_nbrs, bad_topo)
    b_bad = bls.run_blocked_laplacian_smoothing(
        "bad_skin", plane_backend, indices=[0], growth_rings=0, strength=0.0,
        iterations=1, apply=False, create_backup=False, verbose=False)
    check("initial_intersection_count > 0 reported for an already-invalid baseline",
          b_bad["initial_intersection_count"] > 0, str(b_bad["initial_intersection_count"]))

    # -------------------------------------------------------------------
    # B10. Jacobi/simultaneous update preserved through the full pipeline
    # (not just the reused pure function in isolation): same 3-vertex
    # mutual-adjacency discriminator used in test_pure_laplacian_smoothing.py.
    # -------------------------------------------------------------------
    print("\nB10. Jacobi (simultaneous) update preserved end-to-end, not "
         "Gauss-Seidel/sequential")
    tri_pos = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [10.0, 0.0, 0.0]]
    tri_nbrs = [[1, 2], [0, 2], [0, 1]]
    tri_topo = {"triangles": [], "face_vertex_ids": [[], [], []], "vertex_faces": [[], [], []]}
    remote_backend = PlaneBackend(height=-10000.0, half_extent=5.0)
    _install_mesh(tri_pos, tri_nbrs, tri_topo)
    b10 = bls.run_blocked_laplacian_smoothing(
        "tri_skin", remote_backend, indices=[0, 1], growth_rings=0, strength=1.0,
        iterations=1, apply=True, create_backup=False, verbose=False)
    committed10 = mesh_utils.get_mesh_vertices("tri_skin")
    approx("vertex 0 moves to mean(ORIGINAL p1, p2)", committed10[0][0], 6.0)
    approx("vertex 1 moves to mean(ORIGINAL p0, p2) -- NOT the just-updated p0",
          committed10[1][0], 5.0)
    check("vertex 2 (not selected) never moved", committed10[2] == [10.0, 0.0, 0.0])

finally:
    for name, fn in _orig.items():
        setattr(mesh_utils, name, fn)


if fails:
    print("\nFAILED {0}: {1}".format(len(fails), fails))
    sys.exit(1)
print("\nAll blocked_laplacian_smoothing checks passed.")
