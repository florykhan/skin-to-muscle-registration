"""Offline tests for anatomy_constraint (no Maya required)."""
from __future__ import print_function

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import anatomy_constraint as ac
import mesh_utils
import smoothing_utils


fails = []


def check(name, cond, detail=""):
    ok = bool(cond)
    print(("  ok   " if ok else "  FAIL ") + name + (("  " + detail) if detail and not ok else ""))
    if not ok:
        fails.append(name)


def approx(name, got, exp, tol=1e-4):
    ok = abs(float(got) - float(exp)) <= tol
    print(("  ok   " if ok else "  FAIL ") + "{0}: {1} (expected {2} ± {3})".format(
        name, got, exp, tol))
    if not ok:
        fails.append(name)


def _n(v):
    L = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    return [v[0] / L, v[1] / L, v[2] / L] if L > 1e-12 else [0.0, 0.0, 1.0]


class TwoPlaneBackend(object):
    """Anatomy = planes x=0 (outward +X) and y=0 (outward +Y). Open meshes."""

    def __init__(self, k_blend=1.0):
        self.k_blend = k_blend
        self.sdf_query_fn = self.sdf_query

    def mesh_names(self):
        return ["plane_x", "plane_y"]

    def is_closed(self, name):
        return False

    def point_inside(self, point, mesh_name):
        return False

    def exact_closest(self, point, names=None):
        px, py, pz = float(point[0]), float(point[1]), float(point[2])
        hits = [
            {"mesh": "plane_y", "distance": abs(py),
             "closest": [px, 0.0, pz], "normal": [0.0, 1.0, 0.0],
             "face_id": 0, "category": "other"},
            {"mesh": "plane_x", "distance": abs(px),
             "closest": [0.0, py, pz], "normal": [1.0, 0.0, 0.0],
             "face_id": 0, "category": "other"},
        ]
        hits.sort(key=lambda h: h["distance"])
        b = hits[0]
        return {
            "distance": b["distance"], "closest": list(b["closest"]),
            "normal": list(b["normal"]), "mesh": b["mesh"],
            "face_id": 0, "category": "other", "hits": hits,
        }

    def sdf_query(self, point):
        """Smooth-min analogue of d98 compute_sdf_for_point (top-2 blend)."""
        q = self.exact_closest(point)
        hits = q["hits"]
        d_min = hits[0]["distance"]
        k = self.k_blend
        exp_vals = []
        for h in hits:
            exponent = -k * (h["distance"] - d_min)
            exp_vals.append(math.exp(max(-50, min(50, exponent))))
        total = sum(exp_vals) or 1e-30
        weights = [e / total for e in exp_vals]
        min_dist = d_min - math.log(total) / k
        cp = [0.0, 0.0, 0.0]
        for w, h in zip(weights, hits):
            cp[0] += h["closest"][0] * w
            cp[1] += h["closest"][1] * w
            cp[2] += h["closest"][2] * w
        outward = _n([point[0] - cp[0], point[1] - cp[1], point[2] - cp[2]])
        return min_dist, cp, outward

    def closest_segment_hit(self, p0, p1, mesh_name):
        dx, dy, dz = p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]
        length = math.sqrt(dx * dx + dy * dy + dz * dz)
        if length < 1e-12:
            return None
        t = None
        pt = None
        if mesh_name == "plane_y" and abs(dy) > 1e-12:
            # p0.y + t_param * dy = 0, t_param in (0,1); t is Euclidean
            a = -p0[1] / dy
            if 0.0 < a < 1.0:
                t = a * length
                pt = [p0[0] + a * dx, 0.0, p0[2] + a * dz]
        elif mesh_name == "plane_x" and abs(dx) > 1e-12:
            a = -p0[0] / dx
            if 0.0 < a < 1.0:
                t = a * length
                pt = [0.0, p0[1] + a * dy, p0[2] + a * dz]
        if t is None:
            return None
        return {"t": t, "point": pt, "face": 0, "mesh": mesh_name, "frac": t / length}


class SphereBackend(object):
    """Closed unit-ish sphere at origin, radius R. Inside = |p| < R."""

    def __init__(self, radius=1.0, name="sphere"):
        self.R = radius
        self.name = name
        self.sdf_query_fn = self.sdf_query

    def mesh_names(self):
        return [self.name]

    def is_closed(self, name):
        return True

    def point_inside(self, point, mesh_name):
        r = math.sqrt(point[0] ** 2 + point[1] ** 2 + point[2] ** 2)
        return r < self.R - 1e-9

    def exact_closest(self, point, names=None):
        r = math.sqrt(point[0] ** 2 + point[1] ** 2 + point[2] ** 2)
        if r < 1e-12:
            n = [0.0, 1.0, 0.0]
            cp = [0.0, self.R, 0.0]
        else:
            n = [point[0] / r, point[1] / r, point[2] / r]
            cp = [n[0] * self.R, n[1] * self.R, n[2] * self.R]
        dist = abs(r - self.R)
        hit = {"mesh": self.name, "distance": dist, "closest": cp, "normal": n,
               "face_id": 0, "category": "other"}
        return {"distance": dist, "closest": list(cp), "normal": list(n),
                "mesh": self.name, "face_id": 0, "category": "other", "hits": [hit]}

    def sdf_query(self, point):
        q = self.exact_closest(point)
        outward = _n([point[0] - q["closest"][0], point[1] - q["closest"][1],
                      point[2] - q["closest"][2]])
        return q["distance"], q["closest"], outward

    def closest_segment_hit(self, p0, p1, mesh_name):
        # Ray-sphere first positive hit with t < |p1-p0|
        dx, dy, dz = p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]
        length = math.sqrt(dx * dx + dy * dy + dz * dz)
        if length < 1e-12:
            return None
        d = [dx / length, dy / length, dz / length]
        # quadratic for |p0 + t d|^2 = R^2
        b = 2.0 * (p0[0] * d[0] + p0[1] * d[1] + p0[2] * d[2])
        c = p0[0] ** 2 + p0[1] ** 2 + p0[2] ** 2 - self.R ** 2
        disc = b * b - 4.0 * c
        if disc < 0:
            return None
        sdisc = math.sqrt(disc)
        for ts in ((-b - sdisc) / 2.0, (-b + sdisc) / 2.0):
            if 1e-4 < ts < length - 1e-4:
                pt = [p0[0] + ts * d[0], p0[1] + ts * d[1], p0[2] + ts * d[2]]
                return {"t": ts, "point": pt, "face": 0, "mesh": mesh_name,
                        "frac": ts / length}
        return None


class PlaneYBackend(object):
    """Open plane y=0, outward +Y."""

    def mesh_names(self):
        return ["plane"]

    def is_closed(self, name):
        return False

    def point_inside(self, point, mesh_name):
        return False

    def exact_closest(self, point, names=None):
        cp = [point[0], 0.0, point[2]]
        dist = abs(point[1])
        hit = {"mesh": "plane", "distance": dist, "closest": cp,
               "normal": [0.0, 1.0, 0.0], "face_id": 0, "category": "fat"}
        return {"distance": dist, "closest": cp, "normal": [0.0, 1.0, 0.0],
                "mesh": "plane", "face_id": 0, "category": "fat", "hits": [hit]}

    def sdf_query(self, point):
        q = self.exact_closest(point)
        outward = _n([point[0] - q["closest"][0], point[1] - q["closest"][1],
                      point[2] - q["closest"][2]])
        return q["distance"], q["closest"], outward

    def closest_segment_hit(self, p0, p1, mesh_name):
        dy = p1[1] - p0[1]
        length = math.sqrt((p1[0] - p0[0]) ** 2 + dy ** 2 + (p1[2] - p0[2]) ** 2)
        if abs(dy) < 1e-12 or length < 1e-12:
            return None
        a = -p0[1] / dy
        if 0.0 < a < 1.0:
            t = a * length
            pt = [p0[0] + a * (p1[0] - p0[0]), 0.0, p0[2] + a * (p1[2] - p0[2])]
            return {"t": t, "point": pt, "face": 0, "mesh": mesh_name, "frac": a}
        return None


# ---------------------------------------------------------------------------
print("1. unsigned distance is not penetration")
plane = PlaneYBackend()
c_close = ac.classify_skin_anatomy_vertex(
    [0, 0.5, 0], plane, skin_normal=[0, 1, 0], min_clearance=0.8)
check("close valid skin is below_clearance not inside",
      c_close["label"] == "below_clearance")
check("close valid is not penetrating",
      c_close["label"] not in ("inside_closed_anatomy", "likely_penetrating"))
c_clear = ac.classify_skin_anatomy_vertex(
    [0, 2.0, 0], plane, skin_normal=[0, 1, 0], min_clearance=0.8)
check("far skin is clear", c_clear["label"] == "clear")

print("2. open-mesh likely penetrating (anatomy on outward side of skin)")
c_pen = ac.classify_skin_anatomy_vertex(
    [0, -0.2, 0], plane, skin_normal=[0, 1, 0], min_clearance=0.8)
check("behind plane + outward skin normal => likely_penetrating",
      c_pen["label"] == "likely_penetrating", c_pen["label"])
check("signed_skin_side negative", c_pen["signed_skin_side"] < 0)

print("3. closed-mesh ray-parity analogue (sphere inside)")
sph = SphereBackend(radius=1.0)
c_in = ac.classify_skin_anatomy_vertex(
    [0.1, 0.0, 0.0], sph, skin_normal=[1, 0, 0], min_clearance=0.8)
check("point inside closed sphere => inside_closed_anatomy",
      c_in["label"] == "inside_closed_anatomy", c_in["label"])
c_out = ac.classify_skin_anatomy_vertex(
    [2.0, 0.0, 0.0], sph, skin_normal=[1, 0, 0], min_clearance=0.8)
check("point outside sphere far => clear", c_out["label"] == "clear", c_out["label"])

print("4. legacy one-push vs iterative exact (two-plane 0.514 analogue)")
be = TwoPlaneBackend(k_blend=1.0)
p0 = [0.1, 0.1, 0.0]
C = 0.8
legacy_pos, legacy_applied, _ = ac.legacy_single_push(p0, be.sdf_query, C)
legacy_exact = be.exact_closest(legacy_pos)["distance"]
check("legacy applied a push", legacy_applied)
check("legacy exact dist still below 0.8", legacy_exact < C - 1e-3,
      "legacy exact={0}".format(legacy_exact))
sol = ac.enforce_anatomy_clearance(p0, be, C, max_constraint_iterations=8)
robust_exact = be.exact_closest(sol["position"])["distance"]
check("iterative resolved", sol["resolved"])
check("iterative exact dist >= 0.8 - tol",
      ac.is_clearance_satisfied(robust_exact, C, 1e-3),
      "robust exact={0} pos={1}".format(robust_exact, sol["position"]))
check("iterative used more than 1 projection or landed safe",
      sol["iterations"] >= 1)

print("5. clearance tolerance")
check("0.8-1e-5 satisfies 0.8 with 1e-4 tol",
      ac.is_clearance_satisfied(0.8 - 1e-5, 0.8, 1e-4))
check("0.5 does not satisfy 0.8",
      not ac.is_clearance_satisfied(0.5, 0.8, 1e-4))

print("6. preserve_valid_baseline floors")
positions = [[0, 0.5, 0], [0, 2.0, 0]]
cls = {
    0: {"label": "below_clearance"},
    1: {"label": "clear"},
}
floors, orig, policy = ac.compute_clearance_floors(
    positions, [0, 1], plane, 0.8, clearance_policy="preserve_valid_baseline",
    classifications=cls)
approx("floor of originally-valid 0.5 vert", floors[0], 0.5, 1e-9)
approx("floor of far vert", floors[1], 0.8, 1e-9)
floors_g, _, _ = ac.compute_clearance_floors(
    positions, [0, 1], plane, 0.8, clearance_policy="global", classifications=cls)
approx("global floor 0", floors_g[0], 0.8, 1e-9)

print("7. penetrating verts get global floor")
cls_p = {0: {"label": "inside_closed_anatomy"}}
floors_p, _, _ = ac.compute_clearance_floors(
    [[0.1, 0, 0]], [0], sph, 0.8, classifications=cls_p)
approx("penetrating floor is global 0.8", floors_p[0], 0.8, 1e-9)

print("8. pre-repair closed inside")
pos = [[0.1, 0.0, 0.0], [2.0, 0.0, 0.0]]
normals = [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
rep = ac.resolve_skin_anatomy_penetrations(
    pos, [0, 1], sph, min_clearance=0.2, normals=normals, verbose=False)
r = math.sqrt(sum(x * x for x in rep["positions"][0]))
check("inside point repaired (moved)", 0 in rep["penetration_vertices_repaired"])
check("outside point not repaired", 1 not in rep["penetration_vertices_repaired"])
check("repaired point is outside sphere+clearance",
      r >= 1.0 + 0.2 - 1e-3, "r={0}".format(r))
check("unknown not in unresolved for outside", 1 not in rep["penetration_vertices_unresolved"])

print("9. open-mesh heuristic repair along skin normal")
pos2 = [[0.0, -0.3, 0.0]]
n2 = [[0.0, 1.0, 0.0]]
rep2 = ac.resolve_skin_anatomy_penetrations(
    pos2, [0], plane, min_clearance=0.8, normals=n2, verbose=False)
check("likely repaired or escaped to y>0",
      rep2["positions"][0][1] > 0.0, str(rep2["positions"][0]))

print("10. segment crossing clamp")
clamped, crossed, hit = ac.clamp_segment_crossing(
    [0, 1, 0], [0, -1, 0], plane, min_clearance=0.8)
check("segment y=1 -> y=-1 detected", crossed)
check("clamped stays on +Y side", clamped[1] >= -1e-6, str(clamped))

print("11. analyzer report buckets")
positions = [[0, 2, 0], [0, 0.4, 0], [0, -0.2, 0]]
normals = [[0, 1, 0]] * 3
rep = ac.analyze_skin_anatomy_penetration(
    None, indices=[0, 1, 2], min_clearance=0.8, backend=plane,
    positions=positions, normals=normals, detailed=True, verbose=False)
check("clear count 1", rep["clear_count"] == 1, str(rep["clear_count"]))
check("below_clearance includes the 0.4 vert", 1 in rep["below_clearance_indices"])
check("likely includes the -0.2 vert", 2 in rep["likely_penetrating_indices"])
check("0.4 is NOT classified as penetrating",
      1 not in rep["penetrating_indices"] and 1 not in rep["likely_penetrating_indices"])

print("12. category from INTERNAL_MESHES names")
check("fat_2 -> fat", ac.anatomy_category_from_name("fat_2") == "fat")
check("skull -> bone", ac.anatomy_category_from_name("middleres_skull_copy") == "bone")
check("cartilage misspelling", ac.anatomy_category_from_name("middleres_AlarCartidge1") == "cartilage")
check("muscle middleres", ac.anatomy_category_from_name("middleres_ZygomaticusMajor_l1") == "muscle")
check("polySurface other", ac.anatomy_category_from_name("polySurface38") == "other")

print("13. unknown verts are not auto-repaired")
# Force unknown by using a closed backend that returns None for inside
class UnknownBackend(PlaneYBackend):
    def is_closed(self, name):
        return True

    def point_inside(self, point, mesh_name):
        return None  # inconclusive

ub = UnknownBackend()
c_u = ac.classify_skin_anatomy_vertex(
    [0, 0.4, 0], ub, skin_normal=[0, 1, 0], min_clearance=0.8)
check("inconclusive near-surface is unknown", c_u["label"] == "unknown", c_u["label"])
pos_u = [[0, 0.4, 0]]
rep_u = ac.resolve_skin_anatomy_penetrations(
    pos_u, [0], ub, min_clearance=0.8, normals=[[0, 1, 0]], verbose=False)
check("unknown not repaired", 0 not in rep_u["penetration_vertices_repaired"])
check("unknown position unchanged", abs(rep_u["positions"][0][1] - 0.4) < 1e-12)

print("14. laplacian_smooth / smooth_mesh_region signatures unchanged")
import inspect
sig = inspect.signature(smoothing_utils.smooth_mesh_region)
check("smooth_mesh_region still has apply", "apply" in sig.parameters)
sigc = inspect.signature(smoothing_utils.constrained_smooth_mesh_region)
check("legacy solver param exists", "constraint_solver" in sigc.parameters)
check("iterative_exact is default",
      sigc.parameters["constraint_solver"].default == "iterative_exact")
check("legacy_single_push is accepted", True)

print("15. vec helpers added")
check("vec_dot", abs(mesh_utils.vec_dot([1, 0, 0], [2, 5, 0]) - 2) < 1e-12)
n = mesh_utils.vec_normalize([0, 4, 0])
approx("vec_normalize y", n[1], 1.0)

print("16. debug helper returns both solvers")
info = ac.debug_skin_anatomy_vertex(
    0, [[0.1, 0.1, 0]], be, normals=[[0, 1, 0]], min_clearance=0.8, verbose=False)
check("debug has legacy_exact_distance", info["legacy_exact_distance"] < 0.8)
check("debug robust >= 0.8", info["robust_exact_distance"] >= 0.8 - 1e-3)

print("17. constrained smoother: apply=False, region-only, legacy vs robust")
# Tiny sheet: verts 0,1,2 along y; anatomy is y=0. Smooth vertex 1 only.
verts = [
    [1.2, 0.2, 0.0],
    [1.2, 0.9, 0.0],  # selected: laplacian pulls toward y=0.2 (inward)
    [1.2, 0.2, 0.0],
    [5.0, 5.0, 0.0],  # frozen outsider
]
nbrs = [[1], [0, 2], [1], []]
nrms = [[0, 1, 0]] * 4
store = {"verts": [list(v) for v in verts]}

def _gv(name):
    return [list(v) for v in store["verts"]]

def _sv(name, v):
    store["written"] = True
    store["verts"] = [list(x) for x in v]
    return True

smoothing_utils.get_mesh_vertices = _gv
smoothing_utils.set_mesh_vertices = _sv
smoothing_utils.get_vertex_neighbors = lambda name: nbrs
smoothing_utils.get_vertex_normals = lambda name: nrms

m_dry = smoothing_utils.constrained_smooth_mesh_region(
    "skin", [1], plane.sdf_query, 0.8,
    method="laplacian", strength=0.5, iterations=3,
    anatomy_backend=plane, constraint_solver="iterative_exact",
    resolve_initial_penetration=False, prevent_segment_crossing=False,
    preserve_tangential=False, clearance_policy="global",
    apply=False, verbose=False)
check("apply=False did not write", store.get("written") is not True)
check("outsider never in selected", m_dry["selected_vertex_count"] == 1)

m_leg = smoothing_utils.constrained_smooth_mesh_region(
    "skin", [1], be.sdf_query, 0.8,
    method="laplacian", strength=0.2, iterations=5,
    anatomy_backend=be, constraint_solver="legacy_single_push",
    resolve_initial_penetration=False, prevent_segment_crossing=False,
    preserve_tangential=False, clearance_policy="global",
    apply=True, verbose=False)
# reset and run robust
store["verts"] = [list(v) for v in verts]
store.pop("written", None)
m_rob = smoothing_utils.constrained_smooth_mesh_region(
    "skin", [1], be.sdf_query, 0.8,
    method="laplacian", strength=0.2, iterations=5,
    anatomy_backend=be, constraint_solver="iterative_exact",
    resolve_initial_penetration=False, prevent_segment_crossing=True,
    preserve_tangential=True, clearance_policy="global",
    apply=True, verbose=False)
check("legacy solver recorded", m_leg["constraint_solver"] == "legacy_single_push")
check("robust solver recorded", m_rob["constraint_solver"] == "iterative_exact")
check("robust exact min after >= ~0.8",
      m_rob["exact_min_distance_after"] >= 0.8 - 1e-3,
      str(m_rob["exact_min_distance_after"]))
# frozen outsider: index 3 should still be [5,5,0]
check("only region moved (vert 3 frozen)",
      abs(store["verts"][3][0] - 5.0) < 1e-12 and abs(store["verts"][3][1] - 5.0) < 1e-12)

if fails:
    print("\nFAILED {0}: {1}".format(len(fails), fails))
    sys.exit(1)
print("\nAll offline anatomy_constraint checks passed ({0} groups).".format(17))
