"""Maya-independent tests for largest-safe-fraction smoothing steps."""
from __future__ import print_function

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import smoothing_utils as su


fails = []


def check(name, cond, detail=""):
    ok = bool(cond)
    print(("  ok   " if ok else "  FAIL ") + name + (
        ("  " + detail) if (detail and not ok) else ""))
    if not ok:
        fails.append(name)


def approx(name, got, exp, tol=1e-3):
    ok = abs(float(got) - float(exp)) <= tol
    print(("  ok   " if ok else "  FAIL ") + "{0}: {1} (expected {2} ± {3})".format(
        name, got, exp, tol))
    if not ok:
        fails.append(name)


print("1. candidate safety vs pre-existing intersections")
ok, newf = su._smoothing_candidate_is_safe(
    {"intersecting_skin_faces": [7], "intersection_pair_count": 2},
    baseline_faces={7}, baseline_pair_count=2)
check("CASE 4: same baseline intersection is safe", ok and not newf)
ok, newf = su._smoothing_candidate_is_safe(
    {"intersecting_skin_faces": [7, 9], "intersection_pair_count": 3},
    baseline_faces={7}, baseline_pair_count=2)
check("CASE 5: new face is unsafe", (not ok) and (9 in newf))
ok, _ = su._smoothing_candidate_is_safe(
    {"intersecting_skin_faces": [7], "intersection_pair_count": 5},
    baseline_faces={7}, baseline_pair_count=2)
check("same faces but more pairs is unsafe", not ok)
ok, _ = su._smoothing_candidate_is_safe(
    {"intersecting_skin_faces": [7], "intersection_pair_count": 1},
    baseline_faces={7}, baseline_pair_count=2)
check("fewer pairs on same faces is safe", ok)
ok, _ = su._smoothing_candidate_is_safe(
    {"intersecting_skin_faces": [], "intersection_pair_count": 0},
    baseline_faces={7}, baseline_pair_count=2)
check("clearing extra intersections is safe", ok)


print("2. line search alphas")
before = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
full = [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]
moved = [0]


def _y_threshold(limit):
    def _fn(pos):
        return pos[0][1] <= limit + 1e-12, None
    return _fn


ls1 = su._find_largest_safe_smoothing_step(
    before, full, moved, _y_threshold(2.0), steps=8, min_alpha=1e-3)
check("CASE 1: full step safe -> alpha=1", ls1["alpha"] == 1.0)
check("CASE 1 status full", ls1["status"] == "full")
check("CASE 1 single evaluation", ls1["evaluations"] == 1)

ls2 = su._find_largest_safe_smoothing_step(
    before, full, moved, _y_threshold(0.5), steps=8, min_alpha=1e-3)
check("CASE 2 status partial", ls2["status"] == "partial")
approx("CASE 2 alpha ~= 0.5", ls2["alpha"], 0.5, tol=0.02)
check("CASE 2 accepted y <= 0.5", ls2["positions"][0][1] <= 0.5 + 1e-12)
check("CASE 2 origin unchanged", before[0][1] == 0.0)

ls3 = su._find_largest_safe_smoothing_step(
    before, full, moved, _y_threshold(1e-6), steps=8, min_alpha=1e-3)
check("CASE 3 stalled when only tiny alpha is safe",
      ls3["status"] == "stalled_no_safe_step")
check("CASE 3 keeps before", ls3["positions"][0][1] == 0.0)
check("CASE 3 alpha 0", ls3["alpha"] == 0.0)


print("3. lerp is monotonic in the original field")
p25 = su._scale_displacement_field(before, full, 0.25, moved)
p50 = su._scale_displacement_field(before, full, 0.5, moved)
check("alpha 0.25 is 0.25", abs(p25[0][1] - 0.25) < 1e-12)
check("alpha 0.50 is 0.50", abs(p50[0][1] - 0.50) < 1e-12)
check("unmoved index stays", p50[1] == [0.0, 0.0, 0.0])


print("4. constrained smoother: apply=False + rollback vs line-search")
verts = [
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [5.0, 5.0, 0.0],
]
nbrs = [[1], [0, 2], [1], []]
nrms = [[0, 1, 0]] * 4
store = {"verts": [list(v) for v in verts]}


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
                          "closest": cp, "normal": [0, 1, 0]}]}
    def anatomy_surface_cache(self):
        return {}
    def sdf_query(self, point):
        q = self.exact_closest(point)
        return q["distance"], q["closest"], [0, 1, 0]


plane = PlaneY()
su.get_mesh_vertices = lambda name: [list(v) for v in store["verts"]]
su.set_mesh_vertices = lambda name, v: store.__setitem__("written", True)
su.get_vertex_neighbors = lambda name: nbrs
su.get_vertex_normals = lambda name: nrms
su.get_mesh_fn = lambda name: None

orig_local = su._local_intersection_report


def _fake_local(mesh_name, positions, moved, *args, **kwargs):
    faces, pairs = [], 0
    for i in moved or []:
        if positions[i][1] > 0.5 + 1e-12:
            faces, pairs = [99], 1
            break
    return {"intersecting_skin_faces": faces, "intersection_pair_count": pairs,
            "intersecting_skin_face_count": len(faces)}


try:
    su._local_intersection_report = _fake_local
    topo = {"triangles": [(0, 0, 1, 2)], "face_vertex_ids": [[0, 1, 2]],
            "vertex_faces": [[0], [0], [0], []]}
    m_ls = su.constrained_smooth_mesh_region(
        "skin", [1], plane.sdf_query, 0.01,
        method="laplacian", strength=1.0, iterations=1,
        anatomy_backend=plane, constraint_solver="iterative_exact",
        resolve_initial_penetration=False,
        resolve_initial_surface_intersections=False,
        prevent_surface_intersections=True,
        prevent_segment_crossing=False, preserve_tangential=False,
        clearance_policy="global", apply=False, verbose=False,
        skin_topology=topo,
        unsafe_step_policy="largest_safe_fraction",
        smoothing_line_search_steps=8,
        min_smoothing_alpha=1e-3)
    check("apply=False still does not write", store.get("written") is not True)
    check("line-search recovered or accepted a step",
          m_ls["partial_steps_accepted"] + m_ls["full_steps_safe"] >= 1,
          str(m_ls))
    check("not 1/1 rejected under line search",
          m_ls["unsafe_iterations_rejected"] == 0
          or m_ls["partial_steps_accepted"] > 0)
    check("accepted alpha in (0, 1]",
          0.0 < m_ls["accepted_alpha_per_iteration"][0] <= 1.0,
          str(m_ls.get("accepted_alpha_per_iteration")))
    store.pop("written", None)
    m_rb = su.constrained_smooth_mesh_region(
        "skin", [1], plane.sdf_query, 0.01,
        method="laplacian", strength=1.0, iterations=1,
        anatomy_backend=plane, constraint_solver="iterative_exact",
        resolve_initial_penetration=False,
        resolve_initial_surface_intersections=False,
        prevent_surface_intersections=True,
        prevent_segment_crossing=False, preserve_tangential=False,
        clearance_policy="global", apply=False, verbose=False,
        skin_topology=topo,
        unsafe_step_policy="rollback")
    check("rollback policy recorded", m_rb["unsafe_step_policy"] == "rollback")
    check("strict is the default safety_mode",
          m_ls.get("safety_mode") == "strict")
    store.pop("written", None)
    calls = {"repair": 0, "project": 0}
    orig_repair = su.anatomy_constraint.resolve_skin_anatomy_intersections
    orig_proj = su._apply_iteration_constraints

    def _repair_count(*a, **k):
        calls["repair"] += 1
        return orig_repair(*a, **k)

    def _proj_count(*a, **k):
        calls["project"] += 1
        return orig_proj(*a, **k)

    su.anatomy_constraint.resolve_skin_anatomy_intersections = _repair_count
    su._apply_iteration_constraints = _proj_count
    try:
        m_off = su.constrained_smooth_mesh_region(
            "skin", [1], plane.sdf_query, 0.01,
            method="laplacian", strength=0.2, iterations=4,
            anatomy_backend=plane, constraint_solver="iterative_exact",
            resolve_initial_penetration=False,
            resolve_initial_surface_intersections=False,
            prevent_surface_intersections=True,
            prevent_segment_crossing=True, preserve_tangential=True,
            clearance_policy="global", apply=False, verbose=False,
            skin_topology=topo, safety_mode="off")
        check("off apply=False does not write", store.get("written") is not True)
        check("off safety_disabled flag", m_off.get("safety_disabled") is True)
        check("off does not call intersection repair", calls["repair"] == 0,
              str(calls))
        check("off does not call safety projection", calls["project"] == 0,
              str(calls))
        check("off still reports intersection counts",
              "intersecting_skin_face_count_before" in m_off
              and "intersecting_skin_face_count_after" in m_off)
        check("off safety_decline present", "safety_decline" in m_off)
        store.pop("written", None)
        calls["repair"] = 0
        calls["project"] = 0
        m_batch = su.constrained_smooth_mesh_region(
            "skin", [1], plane.sdf_query, 0.01,
            method="laplacian", strength=0.2, iterations=4,
            anatomy_backend=plane, constraint_solver="iterative_exact",
            resolve_initial_penetration=False,
            resolve_initial_surface_intersections=False,
            prevent_surface_intersections=True,
            prevent_segment_crossing=True, preserve_tangential=True,
            clearance_policy="global", apply=False, verbose=False,
            skin_topology=topo, safety_mode="repair_after_batch",
            safety_batch_iterations=2)
        check("batch apply=False does not write", store.get("written") is not True)
        check("batch_count is 2 for 4 iters / 2",
              m_batch.get("batch_count") == 2, str(m_batch.get("batch_count")))
        check("batch calls V2 repair (not per-iter projection)",
              calls["repair"] >= 2 and calls["project"] == 0, str(calls))
        check("batch reports have repair keys",
              m_batch.get("batch_reports")
              and "intersections_after_smoothing" in m_batch["batch_reports"][0]
              and "intersections_after_repair" in m_batch["batch_reports"][0])
        check("alias allow_safety_decline maps to batch",
              su._normalize_safety_mode(None, True) == "repair_after_batch")
        raised = False
        try:
            su._normalize_safety_mode("off", True)
        except ValueError:
            raised = True
        check("conflicting alias+mode raises", raised)
    finally:
        su.anatomy_constraint.resolve_skin_anatomy_intersections = orig_repair
        su._apply_iteration_constraints = orig_proj
finally:
    su._local_intersection_report = orig_local

if fails:
    print("\nFAILED {0}: {1}".format(len(fails), fails))
    sys.exit(1)
print("\nAll safe-smoothing-step checks passed.")
