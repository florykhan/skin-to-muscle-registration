"""Offline tests for pure_laplacian_smoothing (no Maya required).

Same conventions as the other .cs_test/*.py files: plain script, no pytest,
Maya access monkeypatched on the shared `mesh_utils` module object (this
module and artifact_detection.py both do `import mesh_utils` and call it
module-qualified, so patching attributes on the shared module object reaches
both).
"""
from __future__ import print_function

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import mesh_utils
import artifact_detection
import pure_laplacian_smoothing as pls


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


# =============================================================================
# PART A: pure helpers (no mesh, no Maya)
# =============================================================================

print("A1. jacobi_laplacian_step: SIMULTANEOUS update, not sequential/Gauss-Seidel")
# Triangle 0-1-2, all mutually adjacent. 0 and 1 are selected (weight 1.0);
# 2 is fixed (no weight) and acts as an anchor distinct from 0/1.
tri_pos = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [10.0, 0.0, 0.0]]
tri_nbrs = [[1, 2], [0, 2], [0, 1]]
out = pls.jacobi_laplacian_step(tri_pos, tri_nbrs, {0: 1.0, 1: 1.0})
# Jacobi (correct): both use the ORIGINAL p0/p1 when computing each other's
# neighbour mean. new0 = mean(p1, p2) = (6,0,0). new1 = mean(p0, p2) = (5,0,0).
# A sequential/Gauss-Seidel bug (updating 0 first, then reading the NEW 0 when
# computing 1) would instead give new1 = mean(new0=(6,0,0), p2) = (8,0,0).
approx("vertex 0 moves to mean(ORIGINAL p1, p2)", out[0][0], 6.0)
approx("vertex 1 moves to mean(ORIGINAL p0, p2) -- NOT the just-updated p0 "
      "(this is the actual Jacobi-vs-sequential discriminator)", out[1][0], 5.0)
check("vertex 2 (no weight) is untouched", out[2] == [10.0, 0.0, 0.0])
check("input positions list is never mutated", tri_pos[0] == [0.0, 0.0, 0.0])

print("\nA2. non-selected vertices stay fixed; strength=0 changes nothing")
out2 = pls.jacobi_laplacian_step(tri_pos, tri_nbrs, {0: 0.0, 1: 1.0})
check("weight=0.0 vertex is untouched even though it's in the weights dict",
      out2[0] == [0.0, 0.0, 0.0])
check("vertex absent from weights dict entirely is untouched",
      pls.jacobi_laplacian_step(tri_pos, tri_nbrs, {1: 1.0})[2] == [10.0, 0.0, 0.0])
whole_zero = pls.jacobi_laplacian_step(tri_pos, tri_nbrs, {0: 0.0, 1: 0.0, 2: 0.0})
check("strength/weight 0 everywhere is a true no-op", whole_zero == tri_pos)

print("\nA3. strength=1.0 moves a full-weight vertex EXACTLY to its neighbour centroid")
out3 = pls.jacobi_laplacian_step(tri_pos, tri_nbrs, {0: 1.0})
approx("vertex 0 lands exactly on mean(p1, p2), no damping/clamp", out3[0][0], 6.0)
check("y/z components also land exactly on the mean", out3[0][1] == 0.0 and out3[0][2] == 0.0)
out_half = pls.jacobi_laplacian_step(tri_pos, tri_nbrs, {0: 0.5})
approx("strength=0.5 moves exactly halfway to the centroid, no hidden scaling",
      out_half[0][0], 3.0)  # halfway from 0.0 to 6.0


print("\nA4. linear_falloff_weight / build_falloff_weights: matches the requested table")
approx("ring 0 (core) = 1.0", pls.linear_falloff_weight(0, 3), 1.0)
approx("ring 1 = 0.75", pls.linear_falloff_weight(1, 3), 0.75)
approx("ring 2 = 0.50", pls.linear_falloff_weight(2, 3), 0.50)
approx("ring 3 = 0.25", pls.linear_falloff_weight(3, 3), 0.25)
check("ring beyond growth_rings would be <= 0 (clamped, never negative)",
      pls.linear_falloff_weight(4, 3) <= 1e-9)

path_neighbors = [
    [1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7], [6, 8], [7],
]
fw = pls.build_falloff_weights([4], path_neighbors, growth_rings=3)
approx("core vertex is full strength", fw[4], 1.0)
approx("1-ring neighbour is 0.75", fw[3], 0.75)
approx("2-ring neighbour is 0.50", fw[2], 0.50)
approx("3-ring neighbour is 0.25", fw[1], 0.25)
check("4 rings out (past growth_rings) never receives a weight at all",
      0 not in fw, str(fw))
check("growth_rings=0 smooths ONLY the exact selection, no halo",
      pls.build_falloff_weights([4], path_neighbors, growth_rings=0) == {4: 1.0})

fw_excl = pls.build_falloff_weights([4], path_neighbors, growth_rings=3, exclude={3})
check("excluded vertex (e.g. a topological boundary) never receives a weight, "
     "even though it would otherwise be in the falloff halo",
     3 not in fw_excl, str(fw_excl))
check("excluding a vertex does not remove ITS further neighbours from the halo",
      2 in fw_excl)


print("\nA5. select_rough_vertices: percentile selection")
scores = {i: float(i) for i in range(20)}  # 0..19, perfectly separated
sel75 = pls.select_rough_vertices(scores, 75.0)
check("p75 selection keeps roughly the top quartile", sel75 == sorted(range(15, 20)),
      str(sel75))
sel0 = pls.select_rough_vertices(scores, 0.0)
check("p0 selects everything", sel0 == sorted(scores.keys()))
sel100 = pls.select_rough_vertices(scores, 100.0)
check("p100 selects only the single maximum", sel100 == [19], str(sel100))
check("empty scores selects nothing", pls.select_rough_vertices({}, 75.0) == [])


print("\nA6. roughness_percentile_summary")
pct = pls.roughness_percentile_summary([1.0, 2.0, 3.0, 4.0, 100.0])
check("has all requested keys",
      all(k in pct for k in ("mean", "p50", "p90", "p95", "p99", "max")))
approx("max is the true maximum", pct["max"], 100.0)
check("empty input is safe (all zeros, not a crash)",
      pls.roughness_percentile_summary([])["max"] == 0.0)


# =============================================================================
# PART B: full run_pure_laplacian_smoothing on a small synthetic grid mesh
# =============================================================================

def _make_grid(n=7, spacing=1.0):
    """n x n grid in the XZ plane at y=0; 4-connectivity; perimeter = true
    topological boundary (open mesh edge, exactly like get_boundary_vertices
    would report on a real skin sheet's outer rim)."""
    coords = {}
    idx = {}
    k = 0
    for r in range(n):
        for c in range(n):
            idx[(r, c)] = k
            coords[k] = [c * spacing, 0.0, r * spacing]
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
    for r in range(n):
        for c in range(n):
            if r in (0, n - 1) or c in (0, n - 1):
                boundary.add(idx[(r, c)])
    positions = [coords[i] for i in range(nverts)]
    center = idx[(n // 2, n // 2)]
    return positions, neighbors, boundary, center


print("\nB1. run_pure_laplacian_smoothing: repeated iterations reduce roughness; "
     "boundary protection; apply=False; automatic percentile selection")
positions0, grid_nbrs, grid_boundary, center_v = _make_grid(n=7)
positions0[center_v][1] = 3.0  # an obvious spike -- should dominate the top percentile
store = {"verts": [list(p) for p in positions0], "written": False}

_orig = {}
for name in ("get_mesh_vertices", "set_mesh_vertices", "get_vertex_neighbors",
            "get_boundary_vertices", "mesh_exists"):
    _orig[name] = getattr(mesh_utils, name)

mesh_utils.mesh_exists = lambda name: True
mesh_utils.get_mesh_vertices = lambda name: [list(v) for v in store["verts"]]


def _set_verts(name, v):
    store["verts"] = [list(p) for p in v]
    store["written"] = True


mesh_utils.set_mesh_vertices = _set_verts
mesh_utils.get_vertex_neighbors = lambda name: grid_nbrs
mesh_utils.get_boundary_vertices = lambda name: set(grid_boundary)

try:
    # -- apply=False must not write the scene ---------------------------------
    store["written"] = False
    result_dry = pls.run_pure_laplacian_smoothing(
        "grid_skin", indices=[center_v], growth_rings=1, strength=0.5,
        iterations=5, apply=False, create_backup=False, verbose=False)
    check("apply=False does not write the mesh", store["written"] is False)
    check("apply=False result marked dry_run", result_dry.get("dry_run") is True)

    # -- automatic percentile selection finds the spike ------------------------
    store["verts"] = [list(p) for p in positions0]
    result_auto = pls.run_pure_laplacian_smoothing(
        "grid_skin", indices=None, roughness_percentile=95, growth_rings=0,
        strength=0.0, iterations=1, apply=False, create_backup=False, verbose=False)
    check("automatic selection (p95) finds the spike vertex",
          center_v in result_auto["selected_indices"], str(result_auto["selected_indices"]))
    check("automatic selection is reported as 'auto_percentile'",
          result_auto["selection_source"] == "auto_percentile")

    # -- explicit indices: 'caller' path ---------------------------------------
    result_caller = pls.run_pure_laplacian_smoothing(
        "grid_skin", indices=[center_v], growth_rings=0, strength=0.0,
        iterations=1, apply=False, create_backup=False, verbose=False)
    check("explicit indices reported as 'caller'",
          result_caller["selection_source"] == "caller")
    check("explicit indices used exactly as given (growth_rings=0)",
          result_caller["selected_indices"] == [center_v])

    # -- boundary protection: a boundary vertex is never in the active set ----
    boundary_v = sorted(grid_boundary)[0]
    result_boundary = pls.run_pure_laplacian_smoothing(
        "grid_skin", indices=[boundary_v], growth_rings=2, strength=1.0,
        iterations=1, protect_boundary=True, apply=False, create_backup=False,
        verbose=False)
    check("a boundary vertex passed explicitly as `indices` is dropped, not smoothed "
         "(protect_boundary=True)",
         result_boundary["selected_count"] == 0, str(result_boundary))
    result_no_protect = pls.run_pure_laplacian_smoothing(
        "grid_skin", indices=[boundary_v], growth_rings=0, strength=1.0,
        iterations=1, protect_boundary=False, apply=False, create_backup=False,
        verbose=False)
    check("protect_boundary=False allows a boundary vertex through, as requested",
          result_no_protect["selected_count"] == 1)

    # -- real run: strength=1.0 core vertex moves fully in iteration 1 --------
    store["verts"] = [list(p) for p in positions0]
    store["written"] = False
    result_full = pls.run_pure_laplacian_smoothing(
        "grid_skin", indices=[center_v], growth_rings=0, strength=1.0,
        iterations=1, apply=True, create_backup=False, verbose=False)
    check("apply=True writes the mesh", store["written"] is True)
    nbr_mean_y = sum(positions0[j][1] for j in grid_nbrs[center_v]) / len(grid_nbrs[center_v])
    check("strength=1.0, 1 iteration: core vertex lands exactly on its neighbours' "
         "mean height (no damping)",
         abs(store["verts"][center_v][1] - nbr_mean_y) < 1e-9,
         "{0} vs expected {1}".format(store["verts"][center_v][1], nbr_mean_y))

    # -- repeated iterations reduce roughness on the spike ---------------------
    store["verts"] = [list(p) for p in positions0]
    result_many = pls.run_pure_laplacian_smoothing(
        "grid_skin", indices=[center_v], growth_rings=3, strength=0.5,
        iterations=20, apply=True, create_backup=False, verbose=False)
    check("mean roughness improved after 20 iterations",
          result_many["roughness"]["after_mean"] < result_many["roughness"]["before_mean"],
          str(result_many["roughness"]))
    check("max roughness improved too (the spike itself flattened)",
          result_many["roughness"]["percentiles_after"]["max"]
          < result_many["roughness"]["percentiles_before"]["max"])
    check("displacement from the starting mesh is reported and nonzero",
          result_many["mean_displacement"] > 0.0 and result_many["max_displacement"] > 0.0)
    check("boundary vertices were never touched even after 20 aggressive iterations",
          all(store["verts"][b] == positions0[b] for b in grid_boundary))

    # -- no safety net: this experiment is ALLOWED to move a lot --------------
    store["verts"] = [list(p) for p in positions0]
    result_aggressive = pls.run_pure_laplacian_smoothing(
        "grid_skin", indices=[center_v], growth_rings=3, strength=1.0,
        iterations=30, apply=True, create_backup=False, verbose=False)
    check("strength=1.0 for many iterations is not silently reduced -- displacement "
         "keeps growing with no built-in ceiling",
         result_aggressive["max_displacement"] >= result_many["max_displacement"] - 1e-9,
         "{0} vs {1}".format(result_aggressive["max_displacement"], result_many["max_displacement"]))

finally:
    for name, fn in _orig.items():
        setattr(mesh_utils, name, fn)


if fails:
    print("\nFAILED {0}: {1}".format(len(fails), fails))
    sys.exit(1)
print("\nAll pure_laplacian_smoothing checks passed.")
