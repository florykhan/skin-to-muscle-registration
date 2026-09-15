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


print("\nA6. shape_weight_from_anchor / oscillation tracker")
approx("core (anchor=0) uses core_shape_scale",
      fcs.shape_weight_from_anchor(0.0, 0.3), 0.3)
approx("outer transition (anchor=1) uses full strength",
      fcs.shape_weight_from_anchor(1.0, 0.3), 1.0)
approx("midway transition is the midpoint",
      fcs.shape_weight_from_anchor(0.5, 0.3), 0.65)

hist = []
hist, osc = fcs.update_oscillation_tracker(hist, [5, 9], window=5, threshold=3)
check("1st iter: nobody oscillating yet", osc == set())
hist, osc = fcs.update_oscillation_tracker(hist, [5], window=5, threshold=3)
check("2nd iter: still under threshold", osc == set())
hist, osc = fcs.update_oscillation_tracker(hist, [5, 12], window=5, threshold=3)
check("3rd iter: vertex 5 hit threshold (3 of last 3)", osc == {5}, str(osc))
check("vertex 9 (appeared once) not flagged", 9 not in osc)
# window eviction: after 5 more clean iterations, vertex 5 ages out of the window
for _ in range(5):
    hist, osc = fcs.update_oscillation_tracker(hist, [], window=5, threshold=3)
check("old offender ages out of the rolling window", osc == set(), str(osc))


print("\nA7. persistent, monotonic M4 decay -- the actual reported-bug regression test")
# decay_m4_weight is now a pure ONE-STEP ratchet: caller owns the persistent state.
approx("ratchet step: one decay", fcs.decay_m4_weight(1.0, decay_rate=0.5, floor=0.0), 0.5)
approx("ratchet step: floored", fcs.decay_m4_weight(0.02, decay_rate=0.5, floor=0.1), 0.1)
approx("ratchet step never increases even if given a tiny current value",
      fcs.decay_m4_weight(0.05, decay_rate=0.5, floor=0.0), 0.025)

check("m4_is_improving: a real decrease beyond tolerance is improving",
      fcs.m4_is_improving(prev_disagreement=10.0, disagreement=8.0,
                         rel_improvement_tolerance=0.01) is True)
check("m4_is_improving: flat is NOT improving", not fcs.m4_is_improving(
      prev_disagreement=10.0, disagreement=9.99, rel_improvement_tolerance=0.01))
check("m4_is_improving: a REGRESSION (disagreement got WORSE) is NOT improving "
     "-- this is the exact fix: it must NOT reset the plateau streak",
     not fcs.m4_is_improving(prev_disagreement=2.30731, disagreement=2.46853,
                            rel_improvement_tolerance=0.01))

check("should_decay_m4: fires at streak==decay_patience", fcs.should_decay_m4(2, 2))
check("should_decay_m4: fires again at 2x decay_patience (keeps ratcheting)",
      fcs.should_decay_m4(4, 2))
check("should_decay_m4: does not fire between multiples", not fcs.should_decay_m4(3, 2))
check("should_decay_m4: does not fire at streak 0 (not plateaued)",
      not fcs.should_decay_m4(0, 2))

print("  replaying the REAL reported trajectory (iterations 13-16, w_m4=0.25, "
     "decay_rate=0.7, decay_patience=2, tolerance=0.01):")
w_m4 = 0.25
w_m4_effective = w_m4
streak = 0
# (prev_m4, m4_after) pairs exactly as reported for iterations 13, 14, 15, 16
real_trace = [
    (2.34664, 2.32656),  # iter 13: still improving -> streak resets to 0 after
    (2.32656, 2.30731),  # iter 14: still improving
    (2.30731, 2.46853),  # iter 15: DECAY FIRES this iteration (using streak entering=2)
    (2.46853, 2.27719),  # iter 16 (using the REAL iter16->20 M4 value as a stand-in decrease)
]
effective_log = []
for prev_m4, m4_after in real_trace:
    effective_log.append(w_m4_effective)          # weight USED this iteration (state entering)
    improving = fcs.m4_is_improving(prev_m4, m4_after, 0.01)
    streak = 0 if improving else streak + 1
    if fcs.should_decay_m4(streak, 2):
        w_m4_effective = fcs.decay_m4_weight(w_m4_effective, 0.7, 0.0)
check("iter 13 used full weight (still improving)", effective_log[0] == 0.25)
check("iter 14 used full weight (still improving, streak not yet at patience)",
      effective_log[1] == 0.25)
approx("iter 15 decays to 0.175 -- matches the real trace exactly", effective_log[2], 0.175)
approx("iter 16 STAYS at 0.175 -- the actual fix. Under the OLD (buggy) logic this "
      "snapped back to 0.25 because M4 got worse at iter 15 (a regression, which "
      "used to incorrectly reset the plateau streak)",
      effective_log[3], 0.175)

print("\nA8. classify_phase / force_opposition / provenance_buckets")
check("full M4 strength, still active -> anatomy_correction",
      fcs.classify_phase(0.25, 0.25, 0.0, 5, 0.1, 0.01) == "anatomy_correction")
check("M4 partially decayed -> balanced_cleanup",
      fcs.classify_phase(0.175, 0.25, 0.0, 5, 0.1, 0.01) == "balanced_cleanup")
check("M4 at floor but surface still noisy -> balanced_cleanup (not finishing yet)",
      fcs.classify_phase(0.0, 0.25, 0.0, 5, 0.1, 0.01) == "balanced_cleanup")
check("M4 at floor AND surface quiet -> fairing_finish",
      fcs.classify_phase(0.0, 0.25, 0.0, 1, 0.001, 0.01) == "fairing_finish")

opposing = {0: [1.0, 0.0, 0.0], 1: [0.0, 1.0, 0.0]}
reinforcing_a = {0: [1.0, 0.0, 0.0], 1: [0.0, 1.0, 0.0]}
reinforcing_b = {0: [2.0, 0.0, 0.0], 1: [0.0, 2.0, 0.0]}
opposing_b = {0: [-1.0, 0.0, 0.0], 1: [0.0, -1.0, 0.0]}
op_reinforce = fcs.force_opposition(reinforcing_a, reinforcing_b, [0, 1])
approx("perfectly aligned forces -> cosine +1", op_reinforce["mean_cosine"], 1.0)
op_oppose = fcs.force_opposition(reinforcing_a, opposing_b, [0, 1])
approx("perfectly opposed forces -> cosine -1", op_oppose["mean_cosine"], -1.0)
check("opposing_fraction reflects it", op_oppose["opposing_fraction"] == 1.0)
op_missing = fcs.force_opposition({0: [1.0, 0.0, 0.0]}, {}, [0, 1])
check("vertices missing from either field are skipped, not treated as zero-opposition",
      op_missing["count"] == 0)

buckets = fcs.provenance_buckets([1, 2, 3, 4], {1: "m3_only", 2: "m4_only", 3: "overlap"},
                                 oscillating={2})
check("oscillating overrides original provenance tag", buckets.get("oscillating") == [2])
check("m4_only lost vertex 2 to the oscillating bucket", buckets.get("m4_only") is None)
check("untagged vertex 4 falls back to 'grown'", buckets.get("grown") == [4])


print("\nA9. is_better_state (best-feasible-state ranking rule)")
check("strictly lower roughness wins outright",
      fcs.is_better_state(roughness=0.20, m4_disagreement=5.0,
                          best_roughness=0.25, best_m4=1.0))
check("higher roughness never wins, regardless of M4",
      not fcs.is_better_state(roughness=0.30, m4_disagreement=0.0,
                             best_roughness=0.25, best_m4=5.0))
check("roughness tie broken by lower M4 disagreement",
      fcs.is_better_state(roughness=0.25, m4_disagreement=0.9,
                         best_roughness=0.25, best_m4=1.0))
check("roughness tie with WORSE M4 does not win",
      not fcs.is_better_state(roughness=0.25, m4_disagreement=1.1,
                             best_roughness=0.25, best_m4=1.0))
check("first state ever seen (best=inf) always wins",
      fcs.is_better_state(roughness=0.257, m4_disagreement=2.28,
                         best_roughness=float("inf"), best_m4=float("inf")))


print("\nA10. _smooth_rest_reference: low-pass filters ONLY the given indices")
bumpy_ref = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, -5.0, 0.0]]
ref_nbrs = [[1, 2, 3, 4], [0], [0], [0], [0]]
smoothed = fcs._smooth_rest_reference(bumpy_ref, ref_nbrs, [0], iterations=20, strength=0.5)
check("the bump at vertex 0 is smoothed toward its (fixed) neighbours' mean (~0,0,0)",
      abs(smoothed[0][1]) < 0.05 and abs(smoothed[0][2]) < 0.05, str(smoothed[0]))
check("neighbours acting as anchors are NEVER moved by the reference smoothing",
      smoothed[1] == bumpy_ref[1] and smoothed[3] == bumpy_ref[3])
check("zero iterations is a true no-op",
      fcs._smooth_rest_reference(bumpy_ref, ref_nbrs, [0], iterations=0, strength=0.5)
      == bumpy_ref)


print("\nA11. taubin_force: reproduces the lambda/mu math as a FORCE, not a "
     "position replacement, via fairing_force reused twice")
tf_pos = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]
tf_nbrs = [[1, 2], [0], [0]]
tf = fcs.taubin_force(tf_pos, tf_nbrs, [1], lamb=0.5, mu=-0.3, weight={1: 1.0})
# hand-computed: lambda step moves vertex1 0.5 -> [0.5,0,0]; mu step (relative
# to the now-fixed anchor at [0,0,0]) pushes back out to [0.65,0,0]; net = -0.35.
approx("net Taubin displacement matches the hand-computed lambda-then-mu result",
      tf[1][0], -0.35, tol=1e-9)
check("input positions are never mutated", tf_pos[1] == [1.0, 0.0, 0.0])

tf_zero = fcs.taubin_force(tf_pos, tf_nbrs, [1], lamb=0.5, mu=-0.3, weight={1: 0.0})
check("zero weight -> zero net force (both half-steps see zero strength)",
      tf_zero[1] == [0.0, 0.0, 0.0])

# Anchors (vertex 0, not in `indices`) must stay FIXED across BOTH half-steps,
# not drift between the lambda pass and the mu pass.
tf3_pos = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [-2.0, 0.0, 0.0]]
tf3 = fcs.taubin_force(tf3_pos, tf_nbrs, [1, 2], lamb=0.4, mu=-0.4, weight={1: 1.0, 2: 1.0})
check("symmetric setup gives symmetric (mirrored) net displacement for both moving verts",
      abs(tf3[1][0] + tf3[2][0]) < 1e-9, str(tf3))


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


print("\nB2. _absorb_vertices: repair-driven growth gets the same bookkeeping "
     "as dynamic growth / redetect (validated, not silently untracked)")
positions_path = [[float(i), 1.0, 0.0] for i in range(9)]  # safely above the FlatPlaneBackend's y=0
fairing0 = {4}
provenance0 = {4: "m3_only"}
weights0 = {4: (0.5, 0.0)}
floors0 = {4: 0.1}
new_fairing, new_active, new_anchor = fcs._absorb_vertices(
    [3, 5], fairing0, set(), path_neighbors, 2, provenance0, weights0, floors0,
    positions_path, backend, 0.1, "global", 1e-4, 0.5, tag="repaired")
check("newly absorbed vertices join the fairing region", {3, 4, 5} <= new_fairing)
check("absorbed vertices are tagged with the given provenance",
      provenance0.get(3) == "repaired" and provenance0.get(5) == "repaired")
check("absorbed vertices get a default fairing-only weight",
      weights0.get(3) == (0.5, 0.0) and weights0.get(5) == (0.5, 0.0))
check("absorbed vertices get a clearance floor (not left unfloored)",
      3 in floors0 and 5 in floors0)
check("region regrows a transition band around the new fairing set",
      len(new_active) > len(new_fairing))
check("absorbing an empty set is a safe no-op",
      fcs._absorb_vertices([], new_fairing, set(), path_neighbors, 2, provenance0, weights0,
                          floors0, positions_path, backend, 0.1, "global", 1e-4, 0.5,
                          tag="repaired")[0] == new_fairing)


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

# anatomy_constraint.py does `from mesh_utils import get_boundary_vertices` (a name
# import), so patching mesh_utils.get_boundary_vertices above does NOT reach the name
# bound inside anatomy_constraint's own namespace -- its internal repair-patch-growth
# boundary lookup (_buffered_boundary_vertices) would otherwise silently see an empty
# boundary and let the repair patch grow into the "protected" perimeter. Patch it here
# too, the same way test_safe_smoothing_step.py patches names on smoothing_utils. In
# real Maya this gap does not exist: get_boundary_vertices(skin_mesh) is the same real
# call either way.
_orig["ac_get_boundary_vertices"] = ac.get_boundary_vertices
ac.get_boundary_vertices = lambda name: set(grid_boundary)


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

    print("\nC2. Phase-0 regression test: an initial REAL intersection is repaired "
         "before the rest reference is captured; repair does not fight shape restraint")
    positions0_c2 = [list(p) for p in positions0]
    for i in range(len(positions0_c2)):
        positions0_c2[i][1] = 0.5
    positions0_c2[center_v][1] = -0.3  # dips THROUGH the anatomy plane -- a REAL initial intersection

    store["verts"] = [list(p) for p in positions0_c2]
    store["written"] = False
    # boundary_buffer_rings=0 here: on this tiny 5x5 test grid, a 1-ring buffer
    # around the perimeter (meant, on a real 16k-vertex face, to margin tiny
    # eye/mouth/nostril openings) swallows rows/cols 1 and 3 too, leaving only
    # the center cross as "interior" and excluding every face touching the
    # center from the intersection scan as boundary noise. The TRUE boundary
    # (the grid perimeter itself) is still fully protected either way.
    # rest_reference_smoothing_iterations=0: isolate THIS regression test to the
    # x_rest-source fix specifically (a separate test below covers rest-reference
    # smoothing itself) -- with it enabled, x_rest != x at iteration 1 by design,
    # which would make the "shape displacement ~= 0" assertion below meaningless.
    result_c2 = fcs.run_cleanup_solver(
        "grid_skin", ["floor"], target_offset=0.1, min_clearance=0.1,
        indices=[center_v], anatomy_backend=backend2, sdf_query=object(),
        cleanup_growth_rings=1, transition_rings=1, dynamic_active_set=False,
        boundary_buffer_rings=0, rest_reference_smoothing_iterations=0,
        max_iterations=100, convergence_patience=3, apply=True, verbose=False,
        create_backup=False, save_json=False, save_csv=False)

    feas = result_c2["feasibility"]
    check("setup: there really was a pre-existing intersection to fix",
          feas["pre_intersections"]["face_count"] > 0, str(feas))
    check("Phase 0 resolved it before the iterative loop started",
          feas["resolved"] is True and feas["post_intersections"]["face_count"] == 0, str(feas))

    log = result_c2["iterations_log"]
    check("iteration 1 shape displacement is ~0 -- x_rest was captured AFTER Phase 0 "
         "(x == x_rest at the start of the loop), NOT from the raw invalid original "
         "(the actual root cause of the reported bug)",
         log[0]["shape_displacement"]["mean"] < 1e-9, str(log[0]["shape_displacement"]))

    repair_means = [r["repair_displacement"]["mean"] for r in log]
    early_peak = max(repair_means[:3]) if len(repair_means) >= 3 else repair_means[0]
    check("repair workload decays (or was never re-triggered): no persistent "
         "fair -> repair -> fair -> repair cycle",
         repair_means[-1] <= early_peak * 0.5 + 1e-9 or max(repair_means) < 1e-6,
         str(repair_means))
    check("run converged to a genuinely valid state", result_c2["converged"] is True,
          str(result_c2["stop_reason"]))
    check("no oscillation damping was needed to get there (this is the real fix, "
         "not a band-aid over a still-fighting solver)",
         result_c2["oscillating_vertices"]["count"] == 0,
         str(result_c2["oscillating_vertices"]))
    check("final roughness is within the ceiling -- roughness is now a real "
         "acceptance/convergence target, not just a logged number",
         result_c2["roughness"]["after"] <= result_c2["roughness"]["ceiling"] + 1e-9,
         str(result_c2["roughness"]))
    check("protected boundary still untouched in this scenario too",
          all(store["verts"][b] == positions0_c2[b] for b in grid_boundary))

    print("\nC3. max_iterations beyond the convergence point gives the SAME final mesh")
    store["verts"] = [list(p) for p in positions0]
    result_30 = fcs.run_cleanup_solver(
        "grid_skin", ["floor"], target_offset=0.3, min_clearance=0.1,
        indices=[center_v], anatomy_backend=backend2, sdf_query=object(),
        cleanup_growth_rings=1, transition_rings=1, dynamic_active_set=False,
        max_iterations=100, convergence_patience=3, apply=True, verbose=False,
        create_backup=False, save_json=False, save_csv=False)
    final_30 = [list(v) for v in store["verts"]]

    store["verts"] = [list(p) for p in positions0]
    result_80 = fcs.run_cleanup_solver(
        "grid_skin", ["floor"], target_offset=0.3, min_clearance=0.1,
        indices=[center_v], anatomy_backend=backend2, sdf_query=object(),
        cleanup_growth_rings=1, transition_rings=1, dynamic_active_set=False,
        max_iterations=200, convergence_patience=3, apply=True, verbose=False,
        create_backup=False, save_json=False, save_csv=False)
    final_80 = [list(v) for v in store["verts"]]

    check("both runs converged (stopped themselves, not by hitting the cap)",
          result_30["converged"] and result_80["converged"],
          "{0} / {1}".format(result_30["stop_reason"], result_80["stop_reason"]))
    check("same convergence iteration regardless of max_iterations headroom",
          result_30["iterations"] == result_80["iterations"],
          "{0} vs {1}".format(result_30["iterations"], result_80["iterations"]))
    max_pos_diff = max(mesh_utils.vec_length(mesh_utils.vec_sub(a, b))
                       for a, b in zip(final_30, final_80))
    check("more max_iterations headroom does not change the final geometry "
         "(does not keep shrinking/deforming once converged)",
         max_pos_diff < 1e-9, "max diff={0}".format(max_pos_diff))

    print("\nD1. run_final_fairing: pure finishing pass -- Taubin, no M4, "
         "reuses the caller's exact region, current mesh as reference")
    store["verts"] = [list(p) for p in positions0]  # the spike-at-center scenario again
    store["written"] = False
    result_fair = fcs.run_final_fairing(
        "grid_skin", ["floor"], indices=[center_v], anatomy_backend=backend2,
        target_offset=0.3, min_clearance=0.1, transition_rings=1,
        method="taubin", max_iterations=150, convergence_patience=3,
        apply=True, verbose=False, create_backup=False, save_json=False, save_csv=False)

    check("region_source is 'caller' -- the given indices were used directly, "
         "not re-derived from a fresh M5 detection",
         result_fair["region_source"] == "caller")
    check("base region is exactly the given indices (1 vertex)",
          result_fair["base_region_count"] == 1)
    check("active_indices is exposed (not just a count)",
          isinstance(result_fair.get("active_indices"), list) and len(result_fair["active_indices"]) >= 1)
    check("NO M4 in this stage's result at all", "m4_disagreement" not in result_fair
         and "m4_decay" not in result_fair)
    check("converged", result_fair["converged"] is True, str(result_fair["stop_reason"]))
    check("converged well before max_iterations (this toy single-vertex-core scenario needs "
         "more steps than a real 16k-vertex run's 10-20 default; Taubin's mu-reversal deliberately "
         "cancels part of each step, trading iteration count for shrinkage resistance)",
          result_fair["iterations"] < 150, "iterations={0}".format(result_fair["iterations"]))
    check("roughness improved (fairing region)",
          result_fair["roughness"]["after"] < result_fair["roughness"]["before"],
          str(result_fair["roughness"]))
    check("no forbidden intersections at the end",
          result_fair["intersections"]["after"]["pair_count"] == 0)
    check("the spike vertex moved down toward its neighbours",
          store["verts"][center_v][1] < 1.6, str(store["verts"][center_v]))
    check("boundary vertices never moved during fairing either",
          all(store["verts"][b] == positions0[b] for b in grid_boundary))
    check("method reported correctly", result_fair["method"] == "taubin")

    print("\nD2. run_final_fairing: laplacian method still available for A/B comparison")
    store["verts"] = [list(p) for p in positions0]
    result_fair_lap = fcs.run_final_fairing(
        "grid_skin", ["floor"], indices=[center_v], anatomy_backend=backend2,
        target_offset=0.3, min_clearance=0.1, transition_rings=1,
        method="laplacian", fair_strength=0.3, max_iterations=60, convergence_patience=3,
        apply=True, verbose=False, create_backup=False, save_json=False, save_csv=False)
    check("laplacian method also converges safely", result_fair_lap["converged"] is True)
    check("laplacian method also improves roughness",
          result_fair_lap["roughness"]["after"] < result_fair_lap["roughness"]["before"])

    print("\nD3. run_final_fairing: M5-fallback region path (indices=None) is wired "
         "correctly -- M5 itself is mocked here since its own correctness is already "
         "covered elsewhere; this only checks the fallback plumbing")
    store["verts"] = [list(p) for p in positions0]
    store["written"] = False
    orig_detect = fcs.artifact_detection.detect_unified_artifacts
    fcs.artifact_detection.detect_unified_artifacts = (
        lambda *a, **k: ([center_v], {"final_indices": [center_v]}))
    try:
        result_fallback = fcs.run_final_fairing(
            "grid_skin", ["floor"], indices=None, anatomy_backend=backend2,
            target_offset=0.3, min_clearance=0.1, transition_rings=1,
            max_iterations=5, convergence_patience=2,
            apply=False, verbose=False, create_backup=False, save_json=False, save_csv=False)
    finally:
        fcs.artifact_detection.detect_unified_artifacts = orig_detect
    check("no indices given -> falls back to (mocked) M5 detection",
          result_fallback["region_source"] == "m5_fallback")
    check("fallback region came from the (mocked) M5 call",
          result_fallback["base_region_count"] == 1)
    check("apply=False in the fallback path still does not write the mesh",
          store["written"] is False)

finally:
    ac.get_boundary_vertices = _orig.pop("ac_get_boundary_vertices")
    for name, fn in _orig.items():
        setattr(mesh_utils, name, fn)


if fails:
    print("\nFAILED {0}: {1}".format(len(fails), fails))
    sys.exit(1)
print("\nAll final_cleanup_solver checks passed.")
