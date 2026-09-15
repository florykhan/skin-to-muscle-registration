"""
anatomy_constraint.py
=====================
Exact anatomy clearance + geometric penetration analysis for post-registration
skin cleanup.

The registration collision field (``compute_sdf_for_point``) is a *smooth-min
blend* of unsigned closest-surface distances. That field is useful for motion,
but a single ``closest + outward * clearance`` push against it is **not** a
strict geometric validator: the blended closest point is not a real surface
point, so the pushed location can still be closer than the requested floor to
some unblended anatomical mesh.

This module adds:

* an **exact** (unblended) min-distance query over cached ``MFnMesh`` handles
* an **iterative** clearance projector that re-queries after every push
* a **penetration analyzer** that does NOT treat ``distance < threshold`` as
  inside (unsigned distance is not penetration)
* a **pre-repair** pass for high-confidence penetrating vertices
* optional **segment-crossing** clamps
* a **surface-intersection / protrusion analyzer** (triangle-triangle) that
  catches anatomy faces piercing skin faces even when every skin vertex is
  still classified outside
* **surface-intersection repair** before constrained smoothing (default:
  anatomy-supported patch; legacy per-vertex normal push kept for A/B)

Complexity (documented, not optimized away):
    Exact query: O(M) ``getClosestPoint`` per evaluation, M = # anatomy meshes
    (~43). Reuses caller-cached ``MFnMesh`` objects; does not rebuild
    acceleration structures and does not call ``maya.cmds`` per vertex.
    Ray-parity inside tests: O(R) ``allIntersections`` per closed mesh, R = 7
    directions. The full analyzer runs on the selected region (typically
    hundreds of M5 vertices), not on every smoothing iteration.
    Segment tests: broad-phase by endpoint distances, then
    ``closestIntersection`` on the shortlist.

Maya API 2.0 only. This module does not delete construction history, save the
scene, or delete objects.
"""

from __future__ import print_function

import math
import time

from mesh_utils import (
    vec_add,
    vec_cross,
    vec_dot,
    vec_length,
    vec_normalize,
    vec_scale,
    vec_sub,
    closest_point_and_normal,
    get_boundary_vertices,
    get_mesh_fn,
    get_mesh_vertices,
    get_triangle_topology,
    get_vertex_normals,
    get_vertex_neighbors,
    grow_indices,
    select_faces,
    select_vertices,
)

try:
    import maya.api.OpenMaya as om
    MAYA_AVAILABLE = True
except ImportError:
    om = None
    MAYA_AVAILABLE = False


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_MAX_CONSTRAINT_ITERATIONS = 8
DEFAULT_MAX_ESCAPE_ITERATIONS = 10
DEFAULT_CLEARANCE_TOLERANCE = 1e-4
DEFAULT_RAY_COUNT = 7
DEFAULT_RAY_MAJORITY = 4
INSIDE_CONFIDENCE_MIN_VALID_RAYS = 4
LIKELY_PROXIMITY_SCALE = 3.0
SEGMENT_HIT_EPS = 1e-4
HIGH_CONFIDENCE_LABELS = ("inside_closed_anatomy",)
HEURISTIC_LABELS = ("likely_penetrating", "surface_crossing")

# Tissue category is inferred from EXISTING INTERNAL_MESHES naming conventions
# (fat_*, skull/jaw, *cartil*, middleres_* muscles). Not a new tissue atlas.
_BONE_TOKENS = ("skull", "jaw")
_CARTILAGE_TOKENS = ("cartil", "cartige", "cartidge")


def anatomy_category_from_name(mesh_name):
    """Map an anatomy mesh name to fat / muscle / bone / cartilage / other.

    Uses the project's existing INTERNAL_MESHES naming (``fat_*``, skull/jaw,
    cartilage misspellings already in the scene, ``middleres_*`` muscles).
    Ambiguous names (e.g. ``polySurface38``) stay ``other``; reports always
    also keep the raw mesh name.
    """
    if not mesh_name:
        return "other"
    n = str(mesh_name).lower()
    if "fat" in n:
        return "fat"
    if any(tok in n for tok in _BONE_TOKENS):
        return "bone"
    if any(tok in n for tok in _CARTILAGE_TOKENS):
        return "cartilage"
    if n.startswith("middleres_"):
        return "muscle"
    return "other"


def _as_point(p):
    return [float(p[0]), float(p[1]), float(p[2])]


def _push_along(cp, direction, distance):
    n = vec_normalize(direction)
    if vec_length(n) < 1e-12:
        n = [0.0, 0.0, 1.0]
    return vec_add(cp, vec_scale(n, distance))


def _outward_from_query(point, closest, face_normal):
    """Direction from the surface toward the query point.

    Prefer the face normal if it agrees with ``point - closest``; otherwise
    flip it. If both are degenerate, fall back to +Z. This is the OUTSIDE
    direction for a point that is already outside. Do NOT use this as an
    inside-to-outside escape when the query is inside a closed mesh.
    """
    to_p = vec_sub(point, closest)
    n = list(face_normal) if face_normal is not None else [0.0, 0.0, 0.0]
    if vec_length(n) < 1e-12:
        n = vec_normalize(to_p)
    elif vec_dot(n, to_p) < 0.0:
        n = vec_scale(n, -1.0)
    if vec_length(n) < 1e-12:
        n = [0.0, 0.0, 1.0]
    return vec_normalize(n)


# ---------------------------------------------------------------------------
# Exact closest-point query (NO smooth-min blend)
# ---------------------------------------------------------------------------
def exact_closest_anatomy(point, mesh_fns, names=None):
    """True minimum unsigned closest-point distance over anatomy meshes.

    Does **not** blend distances. Returns a dict::

        distance, closest, normal, mesh, face_id, category, hits

    ``hits`` is every mesh result sorted by distance (for broad-phase).
    Complexity: O(M) closest-point queries on cached ``MFnMesh`` handles.
    """
    point = _as_point(point)
    hits = []
    items = mesh_fns.items() if names is None else (
        (n, mesh_fns[n]) for n in names if n in mesh_fns)
    for name, mesh_fn in items:
        if mesh_fn is None:
            continue
        cp, dist, nrm, face_id = closest_point_and_normal(mesh_fn, point)
        if dist < float("inf"):
            hits.append({
                "mesh": name,
                "distance": float(dist),
                "closest": list(cp),
                "normal": list(nrm),
                "face_id": int(face_id),
                "category": anatomy_category_from_name(name),
            })
    hits.sort(key=lambda h: h["distance"])
    if not hits:
        return {
            "distance": float("inf"),
            "closest": list(point),
            "normal": [0.0, 0.0, 1.0],
            "mesh": None,
            "face_id": -1,
            "category": "other",
            "hits": [],
        }
    best = hits[0]
    return {
        "distance": best["distance"],
        "closest": list(best["closest"]),
        "normal": list(best["normal"]),
        "mesh": best["mesh"],
        "face_id": best["face_id"],
        "category": best["category"],
        "hits": hits,
    }


def is_clearance_satisfied(distance, min_clearance, tolerance=DEFAULT_CLEARANCE_TOLERANCE):
    """True if ``distance >= min_clearance - tolerance`` (float-safe floor)."""
    if distance is None or not math.isfinite(distance):
        return True
    return float(distance) >= float(min_clearance) - float(tolerance)


# ---------------------------------------------------------------------------
# Maya ray intersection (API 2.0)
# ---------------------------------------------------------------------------
def _mfloat_point(p):
    return om.MFloatPoint(float(p[0]), float(p[1]), float(p[2]))


def _mfloat_vector(v):
    return om.MFloatVector(float(v[0]), float(v[1]), float(v[2]))


def mesh_all_intersections(mesh_fn, origin, direction, max_param=1.0e8):
    """List of hit dicts ``{t, point, face, bary}`` along a world-space ray.

    Uses ``MFnMesh.allIntersections`` (Maya API 2.0). Returns [] on failure.
    """
    if not MAYA_AVAILABLE or mesh_fn is None or om is None:
        return []
    n = vec_normalize(direction)
    if vec_length(n) < 1e-12:
        return []
    src = _mfloat_point(origin)
    dirv = _mfloat_vector(n)
    result = None
    try:
        result = mesh_fn.allIntersections(
            src, dirv, om.MSpace.kWorld, float(max_param), False)
    except TypeError:
        try:
            result = mesh_fn.allIntersections(
                src, dirv, space=om.MSpace.kWorld,
                maxParam=float(max_param), testBothDirections=False)
        except Exception:
            return []
    except Exception:
        return []
    if not result:
        return []
    hit_points = result[0] if len(result) > 0 else []
    hit_params = result[1] if len(result) > 1 else []
    hit_faces = result[2] if len(result) > 2 else []
    hit_bary1 = result[4] if len(result) > 4 else []
    hit_bary2 = result[5] if len(result) > 5 else []
    hits = []
    n_hits = len(hit_points) if hit_points is not None else 0
    for i in range(n_hits):
        pt = hit_points[i]
        t = float(hit_params[i]) if hit_params is not None and i < len(hit_params) else 0.0
        face = int(hit_faces[i]) if hit_faces is not None and i < len(hit_faces) else -1
        b1 = float(hit_bary1[i]) if hit_bary1 is not None and i < len(hit_bary1) else 0.0
        b2 = float(hit_bary2[i]) if hit_bary2 is not None and i < len(hit_bary2) else 0.0
        hits.append({
            "t": t,
            "point": [pt.x, pt.y, pt.z],
            "face": face,
            "bary": (b1, b2, 1.0 - b1 - b2),
        })
    hits.sort(key=lambda h: h["t"])
    return hits


def mesh_closest_intersection(mesh_fn, origin, direction, max_param):
    """First hit along a world-space ray, or None.

    Prefers ``closestIntersection``; falls back to the first
    ``allIntersections`` hit.
    """
    if not MAYA_AVAILABLE or mesh_fn is None or om is None:
        return None
    n = vec_normalize(direction)
    if vec_length(n) < 1e-12 or max_param <= 0.0:
        return None
    src = _mfloat_point(origin)
    dirv = _mfloat_vector(n)
    try:
        hit = mesh_fn.closestIntersection(
            src, dirv, om.MSpace.kWorld, float(max_param), False)
        if hit:
            pt = hit[0]
            if pt is None:
                return None
            t = float(hit[1]) if len(hit) > 1 else 0.0
            face = int(hit[2]) if len(hit) > 2 else -1
            if t <= 0.0 or (hasattr(pt, "x") is False and pt == 0):
                return None
            return {"t": t, "point": [pt.x, pt.y, pt.z], "face": face}
    except Exception:
        pass
    hits = mesh_all_intersections(mesh_fn, origin, n, max_param=max_param)
    return hits[0] if hits else None


def _ray_is_grazing(hit, edge_bary_eps=1e-3):
    """True if a hit is suspiciously close to a triangle edge/vertex."""
    bary = hit.get("bary")
    if not bary:
        return False
    return any(b < edge_bary_eps or b > 1.0 - edge_bary_eps for b in bary)


def _parity_directions(n_rays=DEFAULT_RAY_COUNT):
    dirs = [
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
        [1.0, 1.0, 1.0],
        [1.0, -1.0, 0.5],
        [-0.3, 0.8, 0.5],
    ]
    return dirs[:max(3, int(n_rays))]


def ray_parity_inside(point, mesh_fn, n_rays=DEFAULT_RAY_COUNT,
                      max_param=1.0e6):
    """Multi-ray intersection-parity inside test for one mesh.

    Returns ``(inside_or_none, info)`` where ``inside_or_none`` is True/False
    only when enough non-grazing rays agree; otherwise None (unknown).
    Does NOT rely on a single grazing ray.
    """
    votes_inside = 0
    votes_outside = 0
    valid = 0
    grazed = 0
    failed = 0
    for d in _parity_directions(n_rays):
        hits = mesh_all_intersections(mesh_fn, point, d, max_param=max_param)
        if not hits:
            # Zero hits: treat as outside for this ray (open space).
            votes_outside += 1
            valid += 1
            continue
        if any(_ray_is_grazing(h) for h in hits):
            grazed += 1
            continue
        if len(hits) % 2 == 1:
            votes_inside += 1
        else:
            votes_outside += 1
        valid += 1
    info = {
        "valid_rays": valid,
        "votes_inside": votes_inside,
        "votes_outside": votes_outside,
        "grazing_rays": grazed,
        "failed_rays": failed,
    }
    if valid < INSIDE_CONFIDENCE_MIN_VALID_RAYS:
        return None, info
    if votes_inside == votes_outside:
        return None, info
    inside = votes_inside > votes_outside
    # Require a majority, not a 1-vote squeaker on a tiny valid set.
    if max(votes_inside, votes_outside) < min(DEFAULT_RAY_MAJORITY, valid):
        if abs(votes_inside - votes_outside) < 2:
            return None, info
    return inside, info


# ---------------------------------------------------------------------------
# Anatomy backend
# ---------------------------------------------------------------------------
class MayaAnatomyBackend(object):
    """Cached anatomy queries over ``{name: MFnMesh}``.

    Optional ``sdf_query_fn(point) -> (dist, cp, outward)`` is the registration
    smooth-min field, used only for diagnostics / legacy comparison -- never as
    the exact safety validator.
    """

    def __init__(self, mesh_fns, sdf_query_fn=None, closed_cache=None):
        self.mesh_fns = dict(mesh_fns or {})
        self.sdf_query_fn = sdf_query_fn
        self._closed = dict(closed_cache or {})

    def mesh_names(self):
        return list(self.mesh_fns.keys())

    def sdf_query(self, point):
        if self.sdf_query_fn is not None:
            return self.sdf_query_fn(point)
        q = self.exact_closest(point)
        outward = _outward_from_query(point, q["closest"], q["normal"])
        return q["distance"], q["closest"], outward

    def exact_closest(self, point, names=None):
        return exact_closest_anatomy(point, self.mesh_fns, names=names)

    def is_closed(self, mesh_name):
        if mesh_name in self._closed:
            return self._closed[mesh_name]
        try:
            boundary = get_boundary_vertices(mesh_name)
            closed = (boundary is not None) and (len(boundary) == 0)
        except Exception:
            closed = False
        self._closed[mesh_name] = bool(closed)
        return bool(closed)

    def point_inside(self, point, mesh_name):
        """True/False/None (unknown) via multi-ray parity on a closed mesh."""
        if not self.is_closed(mesh_name):
            return False
        mesh_fn = self.mesh_fns.get(mesh_name)
        inside, _info = ray_parity_inside(point, mesh_fn)
        return inside

    def closest_segment_hit(self, p0, p1, mesh_name):
        mesh_fn = self.mesh_fns.get(mesh_name)
        if mesh_fn is None:
            return None
        delta = vec_sub(p1, p0)
        length = vec_length(delta)
        if length < 1e-12:
            return None
        hit = mesh_closest_intersection(mesh_fn, p0, delta, max_param=length)
        if not hit:
            return None
        t = hit["t"]
        # Param is Euclidean distance along the unit direction.
        if t <= SEGMENT_HIT_EPS or t >= length - SEGMENT_HIT_EPS:
            return None
        hit["mesh"] = mesh_name
        hit["frac"] = t / length
        return hit

    def anatomy_surface_cache(self):
        """Static triangulated anatomy with AABBs / uniform-grid broad phase.

        Built once per backend. Anatomy is not moved by this module.
        """
        if getattr(self, "_surface_cache", None) is not None:
            return self._surface_cache
        cache = {}
        for name, mesh_fn in self.mesh_fns.items():
            if mesh_fn is None:
                continue
            cache[name] = build_anatomy_surface_cache(mesh_fn, name=name)
        self._surface_cache = cache
        return cache


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
def signed_skin_side(point, closest, skin_normal):
    """``dot(point - closest, skin_outward_normal)``.

    > 0  anatomy closest point lies inward of the skin tangent plane
    < 0  anatomy lies on the *outward* side of the skin (likely behind/through)
    """
    if skin_normal is None or vec_length(skin_normal) < 1e-12:
        return 0.0
    return vec_dot(vec_sub(point, closest), vec_normalize(skin_normal))


def classify_skin_anatomy_vertex(point, backend, skin_normal=None,
                                 min_clearance=0.8,
                                 clearance_tolerance=DEFAULT_CLEARANCE_TOLERANCE,
                                 likely_proximity_scale=LIKELY_PROXIMITY_SCALE):
    """Geometric classification of ONE skin vertex vs anatomy.

    UNSIGNED distance below ``min_clearance`` is ``below_clearance``, never
    by itself ``inside``. Closed-mesh inside uses ray parity. Open meshes
    only produce ``likely_penetrating`` (heuristic) or ``unknown``.
    """
    q = backend.exact_closest(point)
    dist = q["distance"]
    mesh = q["mesh"]
    below = (math.isfinite(dist)
             and not is_clearance_satisfied(dist, min_clearance, clearance_tolerance))
    side = signed_skin_side(point, q["closest"], skin_normal)
    closed = bool(mesh) and backend.is_closed(mesh)
    method = "exact_distance"
    label = "clear"
    confidence = "high"
    inside = None
    inside_info = None

    # Closed-mesh inside/outside: test the nearest closed mesh AND any other
    # closed mesh we might be deep inside (unsigned distance can be large).
    closed_inside_mesh = None
    inside = False
    saw_unknown_inside = False
    if hasattr(backend, "mesh_names"):
        closed_candidates = [name for name in backend.mesh_names()
                             if backend.is_closed(name)]
        for name in closed_candidates:
            verdict = backend.point_inside(point, name)
            if verdict is True:
                inside = True
                closed_inside_mesh = name
                break
            if verdict is None:
                saw_unknown_inside = True
        if inside is not True and saw_unknown_inside:
            inside = None

    if inside is True:
        label = "inside_closed_anatomy"
        method = "ray_parity"
        confidence = "high"
        mesh = closed_inside_mesh or mesh
        q = dict(q)
        q["mesh"] = mesh
        q["category"] = anatomy_category_from_name(mesh)
    elif (closed and inside is None
          and (below or (math.isfinite(dist)
                         and dist <= float(min_clearance) * float(likely_proximity_scale)))):
        # Inconclusive ray parity near the surface: do not claim inside or clear.
        label = "unknown"
        method = "ray_parity_inconclusive"
        confidence = "low"
    else:
        # Open-mesh / outside heuristic: anatomy on the OUTWARD side of the
        # skin AND spatially close. Far vertices are never "likely penetrating"
        # from this signal alone.
        proximity = (math.isfinite(dist)
                     and dist <= max(float(min_clearance) * float(likely_proximity_scale),
                                     float(min_clearance)))
        if (skin_normal is not None and proximity
                and side < -clearance_tolerance):
            label = "likely_penetrating"
            method = "skin_side_heuristic"
            confidence = "medium" if closed else "medium"
        elif below:
            label = "below_clearance"
            method = "exact_distance"
            confidence = "high"
        else:
            label = "clear"
            method = "exact_distance"
            confidence = "high"

    return {
        "label": label,
        "confidence": confidence,
        "method": method,
        "distance": dist,
        "below_clearance": below,
        "signed_skin_side": side,
        "closed": closed,
        "inside_closed": inside,
        "mesh": mesh,
        "category": q.get("category") or anatomy_category_from_name(mesh),
        "closest": q["closest"],
        "normal": q["normal"],
        "inside_info": inside_info,
        "query": q,
    }


def _empty_penetration_report(vertex_count=0):
    return {
        "vertex_count": vertex_count,
        "penetrating_count": 0,
        "likely_penetrating_count": 0,
        "below_clearance_count": 0,
        "clear_count": 0,
        "unknown_count": 0,
        "surface_crossing_count": 0,
        "penetrating_indices": [],
        "likely_penetrating_indices": [],
        "below_clearance_indices": [],
        "clear_indices": [],
        "unknown_indices": [],
        "surface_crossing_indices": [],
        "minimum_exact_distance": float("inf"),
        "mean_exact_distance": 0.0,
        "per_anatomy_counts": {},
        "closed_mesh_inside_counts": {},
        "open_mesh_likely_counts": {},
        "classification_method_counts": {},
        "details_by_vertex": None,
    }


def analyze_skin_anatomy_penetration(skin_mesh, indices=None, anatomical_meshes=None,
                                     min_clearance=0.8, backend=None,
                                     positions=None, normals=None,
                                     clearance_tolerance=DEFAULT_CLEARANCE_TOLERANCE,
                                     likely_proximity_scale=LIKELY_PROXIMITY_SCALE,
                                     detailed=False, verbose=True):
    """Classify selected skin vertices vs internal anatomy.

    ``skin_mesh`` may be a mesh name (Maya) or ignored if ``positions`` and
    ``backend`` are supplied. UNSIGNED distance is recorded as
    ``below_clearance`` and is NEVER used as the sole penetration test.
    """
    if positions is None:
        positions = get_mesh_vertices(skin_mesh) if skin_mesh else []
    if not positions:
        report = _empty_penetration_report(0)
        if verbose:
            print("[anatomy] no skin vertices to analyze")
        return report
    if normals is None and skin_mesh:
        try:
            normals = get_vertex_normals(skin_mesh)
        except Exception:
            normals = None
    n = len(positions)
    if indices is None:
        region = list(range(n))
    else:
        region = sorted(set(int(i) for i in indices if 0 <= int(i) < n))
    if backend is None:
        if anatomical_meshes and isinstance(anatomical_meshes, dict):
            backend = MayaAnatomyBackend(anatomical_meshes)
        else:
            report = _empty_penetration_report(len(region))
            report["error"] = "no anatomy backend"
            return report

    penetrating = []
    likely = []
    below = []
    clear = []
    unknown = []
    crossing = []
    distances = []
    per_anatomy = {}
    closed_inside = {}
    open_likely = {}
    methods = {}
    details = {} if detailed else None

    for i in region:
        nrm = normals[i] if (normals and i < len(normals)) else None
        c = classify_skin_anatomy_vertex(
            positions[i], backend, skin_normal=nrm,
            min_clearance=min_clearance,
            clearance_tolerance=clearance_tolerance,
            likely_proximity_scale=likely_proximity_scale)
        label = c["label"]
        methods[c["method"]] = methods.get(c["method"], 0) + 1
        if math.isfinite(c["distance"]):
            distances.append(c["distance"])
        mesh = c["mesh"]
        if mesh:
            per_anatomy[mesh] = per_anatomy.get(mesh, 0) + 1
        if c["below_clearance"]:
            below.append(i)
        if label == "inside_closed_anatomy":
            penetrating.append(i)
            if mesh:
                closed_inside[mesh] = closed_inside.get(mesh, 0) + 1
        elif label == "likely_penetrating":
            likely.append(i)
            if mesh:
                open_likely[mesh] = open_likely.get(mesh, 0) + 1
        elif label == "surface_crossing":
            crossing.append(i)
        elif label == "unknown":
            unknown.append(i)
        elif label == "clear":
            clear.append(i)
        # below_clearance stays in `below` only -- not "clear" and not penetrating
        if details is not None:
            details[i] = c

    report = {
        "vertex_count": len(region),
        "penetrating_count": len(penetrating),
        "likely_penetrating_count": len(likely),
        "below_clearance_count": len(below),
        "clear_count": len(clear),
        "unknown_count": len(unknown),
        "surface_crossing_count": len(crossing),
        "penetrating_indices": penetrating,
        "likely_penetrating_indices": likely,
        "below_clearance_indices": below,
        "clear_indices": sorted(set(clear)),
        "unknown_indices": unknown,
        "surface_crossing_indices": crossing,
        "minimum_exact_distance": min(distances) if distances else float("inf"),
        "mean_exact_distance": (sum(distances) / len(distances)) if distances else 0.0,
        "per_anatomy_counts": per_anatomy,
        "closed_mesh_inside_counts": closed_inside,
        "open_mesh_likely_counts": open_likely,
        "classification_method_counts": methods,
        "min_clearance": min_clearance,
        "clearance_tolerance": clearance_tolerance,
        "details_by_vertex": details,
    }
    if verbose:
        print_penetration_report(report)
    return report


def print_penetration_report(report, label=""):
    prefix = "[penetration{0}]".format(" " + label if label else "")
    print("{0} verts={1}  penetrating={2}  likely={3}  below_clearance={4}  "
          "clear={5}  unknown={6}".format(
              prefix, report.get("vertex_count", 0),
              report.get("penetrating_count", 0),
              report.get("likely_penetrating_count", 0),
              report.get("below_clearance_count", 0),
              report.get("clear_count", 0),
              report.get("unknown_count", 0)))
    dmin = report.get("minimum_exact_distance", float("inf"))
    dmean = report.get("mean_exact_distance", 0.0)
    print("  exact distance: min={0}  mean={1:.5f}".format(
        "{0:.5f}".format(dmin) if math.isfinite(dmin) else "inf", dmean))
    ci = report.get("closed_mesh_inside_counts") or {}
    if ci:
        print("  closed-mesh inside: {0}".format(
            ", ".join("{0}:{1}".format(k, v) for k, v in sorted(ci.items()))))
    ol = report.get("open_mesh_likely_counts") or {}
    if ol:
        print("  open-mesh likely: {0}".format(
            ", ".join("{0}:{1}".format(k, v) for k, v in sorted(ol.items()))))


def select_penetrating_skin_vertices(report, mesh_name, include_likely=True,
                                     include_unknown=False):
    """Maya component-select the classified SKIN vertices (not fat/muscle)."""
    idx = list(report.get("penetrating_indices") or [])
    if include_likely:
        idx.extend(report.get("likely_penetrating_indices") or [])
    if include_unknown:
        idx.extend(report.get("unknown_indices") or [])
    idx = sorted(set(idx))
    if not idx:
        print("[penetration] no penetrating skin vertices to select")
        return []
    select_vertices(mesh_name, idx, replace=True)
    print("[penetration] selected {0} skin vertices "
          "(penetrating={1}, likely={2})".format(
              len(idx),
              len(report.get("penetrating_indices") or []),
              len(report.get("likely_penetrating_indices") or []) if include_likely else 0))
    return idx


# ---------------------------------------------------------------------------
# Iterative exact clearance projection
# ---------------------------------------------------------------------------
def legacy_single_push(point, sdf_query_fn, min_clearance):
    """Reproduce the historical one-shot collision push (no re-query)."""
    dist, cp, outward = sdf_query_fn(point)
    if dist < min_clearance:
        return vec_add(list(cp), vec_scale(list(outward), min_clearance)), True, dist
    return list(point), False, dist


def enforce_anatomy_clearance(point, backend, min_clearance,
                              max_constraint_iterations=DEFAULT_MAX_CONSTRAINT_ITERATIONS,
                              clearance_tolerance=DEFAULT_CLEARANCE_TOLERANCE,
                              skin_normal=None,
                              start_inside=False):
    """Iterative exact-distance projection onto the clearance floor.

    After every push the point is RE-QUERIED with the unblended exact closest
    anatomy distance. Safety is NEVER inferred from a single smooth-min push.

    If the point is inside a closed mesh, the push uses the anatomical *face
    normal* (not ``normalize(point - closest)``, which can point inward).

    Returns a dict: ``position, resolved, iterations, residual, corrections, ...``
    """
    p = _as_point(point)
    corrections = []
    resolved = False
    last_q = None
    used = 0
    inside_now = bool(start_inside)

    for k in range(max(1, int(max_constraint_iterations))):
        used = k + 1
        q = backend.exact_closest(p)
        last_q = q
        dist = q["distance"]
        mesh = q["mesh"]
        # Re-detect closed-mesh interior each iteration (normals may be flipped).
        if mesh and backend.is_closed(mesh):
            verdict = backend.point_inside(p, mesh)
            if verdict is True:
                inside_now = True
            elif verdict is False:
                inside_now = False
        safe_dist = is_clearance_satisfied(dist, min_clearance, clearance_tolerance)
        if safe_dist and not inside_now:
            resolved = True
            break

        cp = q["closest"]
        n_face = q["normal"]
        if inside_now:
            # Escape to the OUTWARD side of the stored face normal, then
            # verify; if still inside, try the opposite side.
            n = vec_normalize(n_face)
            if vec_length(n) < 1e-12:
                n = [0.0, 0.0, 1.0]
            cand_a = _push_along(cp, n, min_clearance)
            cand_b = _push_along(cp, vec_scale(n, -1.0), min_clearance)
            # Prefer the candidate that is not inside and has larger exact dist.
            best = cand_a
            best_score = -1.0
            for cand in (cand_a, cand_b):
                qq = backend.exact_closest(cand)
                inside_c = False
                if mesh and backend.is_closed(mesh):
                    v = backend.point_inside(cand, mesh)
                    inside_c = (v is True)
                score = (0.0 if inside_c else 10.0) + (
                    qq["distance"] if math.isfinite(qq["distance"]) else 0.0)
                if score > best_score:
                    best_score = score
                    best = cand
            new_p = best
        else:
            n = _outward_from_query(p, cp, n_face)
            new_p = _push_along(cp, n, min_clearance)
        corrections.append(vec_length(vec_sub(new_p, p)))
        p = new_p

    residual = 0.0
    if last_q is not None and math.isfinite(last_q["distance"]):
        residual = max(0.0, float(min_clearance) - float(last_q["distance"]))
        if residual <= float(clearance_tolerance):
            residual = 0.0
            if not inside_now:
                resolved = True

    return {
        "position": p,
        "resolved": bool(resolved),
        "iterations": used,
        "residual": residual,
        "corrections": corrections,
        "mean_correction": (sum(corrections) / len(corrections)) if corrections else 0.0,
        "max_correction": max(corrections) if corrections else 0.0,
        "final_distance": last_q["distance"] if last_q else float("inf"),
        "final_mesh": last_q["mesh"] if last_q else None,
        "unresolved": not bool(resolved),
    }


def first_segment_crossing(p0, p1, backend, min_clearance=0.8,
                           pad_scale=1.0):
    """First anatomy intersection along ``p0 -> p1``, or None.

    Broad-phase: meshes whose exact distance at either endpoint is less than
    ``|p1-p0| + min_clearance * pad_scale``. Then ``closestIntersection``.
    """
    delta = vec_sub(p1, p0)
    length = vec_length(delta)
    if length < 1e-12:
        return None
    q0 = backend.exact_closest(p0)
    q1 = backend.exact_closest(p1)
    pad = length + float(min_clearance) * float(pad_scale)
    names = set()
    for q in (q0, q1):
        if q.get("mesh"):
            names.add(q["mesh"])
        for h in q.get("hits") or []:
            if h["distance"] <= pad:
                names.add(h["mesh"])
    if not names and hasattr(backend, "mesh_names"):
        names = set(backend.mesh_names())
    best = None
    for name in names:
        hit = backend.closest_segment_hit(p0, p1, name)
        if not hit:
            continue
        if best is None or hit["t"] < best["t"]:
            best = hit
    return best


def clamp_segment_crossing(p0, p1, backend, min_clearance=0.8,
                           safety_offset=None):
    """If ``p0->p1`` hits anatomy, stop on the safe side of the first hit."""
    hit = first_segment_crossing(p0, p1, backend, min_clearance=min_clearance)
    if hit is None:
        return list(p1), False, None
    offset = float(safety_offset) if safety_offset is not None else max(
        float(min_clearance) * 0.05, 1e-3)
    frac = hit.get("frac", 0.0)
    # Back up slightly before the hit, then push out by clearance via solver.
    safe_frac = max(0.0, frac - (offset / max(vec_length(vec_sub(p1, p0)), 1e-12)))
    clamped = vec_add(p0, vec_scale(vec_sub(p1, p0), safe_frac))
    return clamped, True, hit


def tangent_preserving_proposal(current, proposed, backend, min_clearance,
                                clearance_tolerance=DEFAULT_CLEARANCE_TOLERANCE):
    """Remove the inward normal component toward anatomy; keep tangential.

    Uses the exact closest-face normal at ``current``. If that normal is
    degenerate, returns ``proposed`` unchanged (no invented tangent basis).
    """
    q = backend.exact_closest(current)
    n = _outward_from_query(current, q["closest"], q["normal"])
    if vec_length(n) < 1e-12:
        return list(proposed), 0.0, vec_length(vec_sub(proposed, current)), False
    delta = vec_sub(proposed, current)
    n_comp = vec_dot(delta, n)
    tangential = vec_sub(delta, vec_scale(n, n_comp))
    t_len = vec_length(tangential)
    removed = 0.0
    used = False
    if n_comp < 0.0:
        # Moving toward anatomy: drop the inward component.
        removed = -n_comp
        candidate = vec_add(current, tangential)
        used = True
    else:
        candidate = list(proposed)
    return candidate, removed, t_len, used


# ---------------------------------------------------------------------------
# Baseline-aware per-vertex floors
# ---------------------------------------------------------------------------
def compute_clearance_floors(positions, indices, backend, global_min_clearance,
                             clearance_policy="preserve_valid_baseline",
                             classifications=None,
                             clearance_tolerance=DEFAULT_CLEARANCE_TOLERANCE):
    """Per-vertex allowed clearance floor.

    ``preserve_valid_baseline`` (research default): a NON-penetrating vertex
    whose original exact distance is already below ``global_min_clearance``
    keeps ``floor_i = original_distance_i`` so smoothing cannot make it worse
    but the solver does not inflate originally-valid close skin to 0.8.

    Penetrating vertices use ``global_min_clearance`` (after repair).

    ``global``: every vertex uses ``global_min_clearance``.
    """
    floors = {}
    original_distances = {}
    policy = clearance_policy or "preserve_valid_baseline"
    for i in indices:
        q = backend.exact_closest(positions[i])
        d = q["distance"]
        original_distances[i] = d
        label = None
        if classifications and i in classifications:
            label = classifications[i].get("label")
        penetrating = label in HIGH_CONFIDENCE_LABELS or label in HEURISTIC_LABELS
        if policy == "global" or penetrating:
            floors[i] = float(global_min_clearance)
        else:
            if math.isfinite(d):
                floors[i] = min(float(global_min_clearance), max(0.0, float(d)))
            else:
                floors[i] = float(global_min_clearance)
        # Never allow a negative/NaN floor.
        if not math.isfinite(floors[i]):
            floors[i] = float(global_min_clearance)
    return floors, original_distances, policy


# ---------------------------------------------------------------------------
# Penetration pre-repair
# ---------------------------------------------------------------------------
def resolve_skin_anatomy_penetrations(positions, indices, backend,
                                      min_clearance=0.8,
                                      normals=None, neighbors=None,
                                      max_escape_iterations=DEFAULT_MAX_ESCAPE_ITERATIONS,
                                      clearance_tolerance=DEFAULT_CLEARANCE_TOLERANCE,
                                      penetration_repair_feather_rings=0,
                                      classifications=None,
                                      verbose=True):
    """Push high-confidence penetrating vertices outside anatomy.

    Only ``inside_closed_anatomy`` is repaired automatically as guaranteed
    interior. ``likely_penetrating`` (open-mesh heuristic) is also repaired,
    explicitly as a heuristic escape along the skin outward normal.
    ``unknown`` vertices are left untouched.
    """
    work = [list(v) for v in positions]
    repaired = []
    unresolved = []
    displacements = []
    per_vertex = {}

    def _cls(i, pos):
        if classifications and i in classifications and pos is positions[i]:
            return classifications[i]
        nrm = normals[i] if (normals and i < len(normals)) else None
        return classify_skin_anatomy_vertex(
            pos, backend, skin_normal=nrm, min_clearance=min_clearance,
            clearance_tolerance=clearance_tolerance)

    for i in indices:
        c0 = _cls(i, work[i])
        label = c0["label"]
        if label not in HIGH_CONFIDENCE_LABELS and label not in ("likely_penetrating",):
            continue
        p = list(work[i])
        ok = False
        last_disp = 0.0
        for step in range(max(1, int(max_escape_iterations))):
            c = _cls(i, p)
            if c["label"] not in HIGH_CONFIDENCE_LABELS and c["label"] != "likely_penetrating":
                # Still enforce clearance on the escaped point.
                floor = min_clearance
                sol = enforce_anatomy_clearance(
                    p, backend, floor,
                    max_constraint_iterations=DEFAULT_MAX_CONSTRAINT_ITERATIONS,
                    clearance_tolerance=clearance_tolerance,
                    skin_normal=(normals[i] if normals and i < len(normals) else None),
                    start_inside=False)
                p = sol["position"]
                ok = True
                break
            q = c["query"]
            cp = q["closest"]
            if c["label"] == "inside_closed_anatomy":
                sol = enforce_anatomy_clearance(
                    p, backend, min_clearance,
                    max_constraint_iterations=DEFAULT_MAX_CONSTRAINT_ITERATIONS,
                    clearance_tolerance=clearance_tolerance,
                    skin_normal=(normals[i] if normals and i < len(normals) else None),
                    start_inside=True)
                new_p = sol["position"]
            else:
                # Open-mesh heuristic: escape along skin outward normal.
                n_skin = (normals[i] if (normals and i < len(normals)) else None)
                n_skin = vec_normalize(n_skin) if n_skin is not None else [0.0, 0.0, 0.0]
                if vec_length(n_skin) < 1e-12:
                    n_skin = _outward_from_query(p, cp, q["normal"])
                escape = max(float(min_clearance), abs(c["signed_skin_side"]) + min_clearance)
                new_p = vec_add(p, vec_scale(n_skin, escape))
                sol = enforce_anatomy_clearance(
                    new_p, backend, min_clearance,
                    max_constraint_iterations=DEFAULT_MAX_CONSTRAINT_ITERATIONS,
                    clearance_tolerance=clearance_tolerance,
                    skin_normal=n_skin, start_inside=False)
                new_p = sol["position"]
            last_disp += vec_length(vec_sub(new_p, p))
            p = new_p

        work[i] = p
        displacements.append(vec_length(vec_sub(p, positions[i])))
        final_c = _cls(i, p)
        still_bad = final_c["label"] in HIGH_CONFIDENCE_LABELS or final_c["label"] == "likely_penetrating"
        rec = {
            "index": i,
            "label_before": label,
            "label_after": final_c["label"],
            "displacement": vec_length(vec_sub(p, positions[i])),
            "resolved": not still_bad,
        }
        per_vertex[i] = rec
        if still_bad:
            unresolved.append(i)
        else:
            repaired.append(i)

    # Optional 1-ring feather AFTER core repair; every feathered point must
    # still pass the anatomy constraint. Default rings=0 (off until correctness
    # of the core repair is established).
    feathered = []
    rings = int(penetration_repair_feather_rings or 0)
    if rings > 0 and neighbors is not None and repaired:
        core = set(repaired)
        grown = set(grow_indices(neighbors, repaired, rings=rings))
        ring = [j for j in grown if j not in core and j not in set(indices) or (
            j not in core)]
        # Only feather vertices that are in the selected region (or its ring)
        # and were NOT high-confidence penetrating (those already moved).
        ring = [j for j in grown if j not in core]
        weights = {0: 1.0, 1: 0.5}
        # Ring 1 = neighbors of core.
        ring1 = set()
        for i in core:
            if 0 <= i < len(neighbors):
                ring1.update(neighbors[i])
        ring1 -= core
        for j in ring1:
            if j < 0 or j >= len(work):
                continue
            # Mean core displacement of adjacent repaired verts.
            disp_list = []
            for i in (neighbors[j] if j < len(neighbors) else []):
                if i in core:
                    disp_list.append(vec_sub(work[i], positions[i]))
            if not disp_list:
                continue
            mean_d = [
                sum(d[0] for d in disp_list) / len(disp_list),
                sum(d[1] for d in disp_list) / len(disp_list),
                sum(d[2] for d in disp_list) / len(disp_list),
            ]
            cand = vec_add(positions[j], vec_scale(mean_d, weights.get(1, 0.5)))
            nrm = normals[j] if (normals and j < len(normals)) else None
            # Do not push a valid neighbour into anatomy.
            sol = enforce_anatomy_clearance(
                cand, backend, min_clearance,
                clearance_tolerance=clearance_tolerance, skin_normal=nrm)
            c_after = classify_skin_anatomy_vertex(
                sol["position"], backend, skin_normal=nrm,
                min_clearance=min_clearance,
                clearance_tolerance=clearance_tolerance)
            if c_after["label"] in HIGH_CONFIDENCE_LABELS:
                continue
            work[j] = sol["position"]
            feathered.append(j)

    mean_esc = (sum(displacements) / len(displacements)) if displacements else 0.0
    max_esc = max(displacements) if displacements else 0.0
    result = {
        "positions": work,
        "penetration_vertices_repaired": repaired,
        "penetration_vertices_unresolved": unresolved,
        "repaired_count": len(repaired),
        "unresolved_count": len(unresolved),
        "mean_escape_displacement": mean_esc,
        "max_escape_displacement": max_esc,
        "feathered_indices": feathered,
        "per_vertex": per_vertex,
    }
    if verbose:
        print("[penetration-repair] repaired={0} unresolved={1} "
              "mean_escape={2:.5f} max_escape={3:.5f} feathered={4}".format(
                  len(repaired), len(unresolved), mean_esc, max_esc, len(feathered)))
    return result


# ---------------------------------------------------------------------------
# Debug one vertex
# ---------------------------------------------------------------------------
def debug_skin_anatomy_vertex(vertex_index, positions, backend,
                              normals=None, neighbors=None,
                              min_clearance=0.8,
                              clearance_tolerance=DEFAULT_CLEARANCE_TOLERANCE,
                              laplacian_strength=0.2,
                              verbose=True):
    """Print a full anatomy diagnosis for one skin vertex."""
    if vertex_index < 0 or vertex_index >= len(positions):
        print("[debug] vertex {0} out of range".format(vertex_index))
        return None
    p = list(positions[vertex_index])
    nrm = normals[vertex_index] if (normals and vertex_index < len(normals)) else None
    exact = backend.exact_closest(p)
    sdf_dist = sdf_cp = sdf_out = None
    if getattr(backend, "sdf_query_fn", None) is not None or hasattr(backend, "sdf_query"):
        try:
            sdf_dist, sdf_cp, sdf_out = backend.sdf_query(p)
        except Exception:
            pass
    c = classify_skin_anatomy_vertex(
        p, backend, skin_normal=nrm, min_clearance=min_clearance,
        clearance_tolerance=clearance_tolerance)
    side = c["signed_skin_side"]

    proposed = list(p)
    if neighbors is not None and vertex_index < len(neighbors) and neighbors[vertex_index]:
        nbrs = neighbors[vertex_index]
        avg = [
            sum(positions[j][0] for j in nbrs) / len(nbrs),
            sum(positions[j][1] for j in nbrs) / len(nbrs),
            sum(positions[j][2] for j in nbrs) / len(nbrs),
        ]
        proposed = [
            p[0] + (avg[0] - p[0]) * float(laplacian_strength),
            p[1] + (avg[1] - p[1]) * float(laplacian_strength),
            p[2] + (avg[2] - p[2]) * float(laplacian_strength),
        ]
    hit = first_segment_crossing(p, proposed, backend, min_clearance=min_clearance)
    legacy_pos = list(p)
    legacy_applied = False
    if hasattr(backend, "sdf_query"):
        legacy_pos, legacy_applied, _ = legacy_single_push(
            proposed, backend.sdf_query, min_clearance)
    robust = enforce_anatomy_clearance(
        proposed, backend, min_clearance,
        clearance_tolerance=clearance_tolerance, skin_normal=nrm)
    robust_exact = backend.exact_closest(robust["position"])
    legacy_exact = backend.exact_closest(legacy_pos)

    info = {
        "vertex": vertex_index,
        "current": p,
        "skin_normal": nrm,
        "exact_mesh": exact["mesh"],
        "exact_category": exact["category"],
        "exact_closest": exact["closest"],
        "exact_normal": exact["normal"],
        "exact_distance": exact["distance"],
        "sdf_distance": sdf_dist,
        "sdf_closest": sdf_cp,
        "sdf_outward": sdf_out,
        "signed_skin_side": side,
        "closed": c["closed"],
        "classification": c["label"],
        "method": c["method"],
        "proposed_laplacian": proposed,
        "segment_crosses": hit is not None,
        "segment_hit": hit,
        "legacy_applied": legacy_applied,
        "legacy_position": legacy_pos,
        "legacy_exact_distance": legacy_exact["distance"],
        "robust_position": robust["position"],
        "robust_resolved": robust["resolved"],
        "robust_iterations": robust["iterations"],
        "robust_exact_distance": robust_exact["distance"],
        "min_clearance": min_clearance,
    }
    if verbose:
        def _r(v):
            if v is None:
                return None
            if isinstance(v, (list, tuple)) and v and isinstance(v[0], (int, float)):
                return [round(float(x), 5) for x in v]
            return v
        print("[debug vtx {0}]".format(vertex_index))
        print("  current           = {0}".format(_r(p)))
        print("  skin normal       = {0}".format(_r(nrm)))
        print("  exact mesh        = {0}  category={1}  closed={2}".format(
            exact["mesh"], exact["category"], c["closed"]))
        print("  exact closest     = {0}".format(_r(exact["closest"])))
        print("  exact normal      = {0}".format(_r(exact["normal"])))
        print("  exact distance    = {0:.5f}".format(exact["distance"] if math.isfinite(exact["distance"]) else float("nan")))
        print("  sdf (smooth-min)  = {0}".format(
            "{0:.5f}".format(sdf_dist) if sdf_dist is not None else "n/a"))
        print("  signed skin-side  = {0:.5f}  (>0 anatomy inward of skin)".format(side))
        print("  classification    = {0}  method={1}".format(c["label"], c["method"]))
        print("  proposed Laplace  = {0}".format(_r(proposed)))
        print("  segment crosses?  = {0}".format(hit is not None))
        print("  legacy one-push   = applied={0} pos={1} exact_dist={2:.5f}".format(
            legacy_applied, _r(legacy_pos),
            legacy_exact["distance"] if math.isfinite(legacy_exact["distance"]) else float("nan")))
        print("  robust iterative  = resolved={0} iters={1} pos={2} exact_dist={3:.5f}".format(
            robust["resolved"], robust["iterations"], _r(robust["position"]),
            robust_exact["distance"] if math.isfinite(robust_exact["distance"]) else float("nan")))
        print("  requested floor   = {0}".format(min_clearance))
    return info


# Suggested aliases from the research brief.
project_to_safe_anatomy_position = enforce_anatomy_clearance
repair_skin_penetrations = resolve_skin_anatomy_penetrations

# ---------------------------------------------------------------------------
# SKIN–ANATOMY SURFACE INTERSECTION (triangle-triangle)
# ---------------------------------------------------------------------------
# Vertex penetration (a skin *point* inside / behind anatomy) is NOT the same
# as an anatomy *face* crossing a skin *face*. All three vertices of a skin
# triangle can be classified "clear" while an anatomy triangle still pierces
# the triangle interior. This section detects that protrusion geometrically.
#
# Approach:
#   Broad phase: mesh AABB, then uniform-grid of anatomy triangles, then
#                per-triangle AABB overlap.
#   Narrow phase: non-coplanar -- finite-segment vs triangle tests on ALL 6
#                edges (3 skin + 3 anatomy) so either piercing direction is
#                caught. Coplanar -- 2D SAT on the shared plane.
# Distance < threshold is NEVER treated as an intersection.
#
# Open anatomy meshes are first-class: triangle-triangle does not need
# inside/outside. Intentional openings (eyes/mouth/nostrils/neck) have no
# skin face spanning the hole, so visibility-through-a-hole is not flagged.

DEFAULT_INTERSECTION_TOLERANCE = 1e-6
DEFAULT_MAX_INTERSECTION_REPAIR_ITERATIONS = 20
DEFAULT_INTERSECTION_REPAIR_STEP_RATIO = 0.10
AUTO_REPAIR_INTERSECTION_CLASSES = ("crossing", "coplanar_overlap")
DEBUG_ISECT_PREFIX = "smr_isect_dbg_"
DEFAULT_REPAIR_MODE = "anatomy_supported_patch"
DEFAULT_REPAIR_PROFILE = "broad_fair"
DEFAULT_REPAIR_BLEND_RINGS = 6
DEFAULT_REPAIR_RING_WEIGHTS = (1.0, 0.6, 0.3)
DEFAULT_MAX_SURFACE_REPAIR_PASSES = 5
DEFAULT_MAX_REPAIR_DISPLACEMENT_RATIO = 1.0
DEFAULT_REPAIR_BINARY_SEARCH_STEPS = 8
DEFAULT_SUPPORT_OFFSET_RATIO = 0.02
DEFAULT_HARMONIC_DISPLACEMENT_ITERS = 50
DEFAULT_REPAIR_FALLOFF = "harmonic"
DEFAULT_REPAIR_BOUNDARY_MODE = "soft"
DEFAULT_REPAIR_FIELD_METHOD = "harmonic"
DEFAULT_REPAIR_FIELD_STOP_EPS = 1e-8


def _aabb_from_points(pts, pad=0.0):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    zs = [p[2] for p in pts]
    return (min(xs) - pad, min(ys) - pad, min(zs) - pad,
            max(xs) + pad, max(ys) + pad, max(zs) + pad)


def _aabb_union(aabbs):
    if not aabbs:
        return None
    xmin = min(a[0] for a in aabbs)
    ymin = min(a[1] for a in aabbs)
    zmin = min(a[2] for a in aabbs)
    xmax = max(a[3] for a in aabbs)
    ymax = max(a[4] for a in aabbs)
    zmax = max(a[5] for a in aabbs)
    return (xmin, ymin, zmin, xmax, ymax, zmax)


def _aabb_overlap(a, b, pad=0.0):
    if a is None or b is None:
        return False
    return not (a[3] + pad < b[0] or b[3] + pad < a[0]
                or a[4] + pad < b[1] or b[4] + pad < a[1]
                or a[5] + pad < b[2] or b[5] + pad < a[2])


class _UniformGrid(object):
    """Integer-hash grid over triangle AABBs (broad phase)."""

    def __init__(self, tri_aabbs, cell):
        self.cell = max(float(cell), 1e-6)
        self.buckets = {}
        for idx, aabb in enumerate(tri_aabbs):
            for key in self._keys(aabb):
                self.buckets.setdefault(key, []).append(idx)

    def _keys(self, aabb):
        c = self.cell
        i0 = int(math.floor(aabb[0] / c))
        j0 = int(math.floor(aabb[1] / c))
        k0 = int(math.floor(aabb[2] / c))
        i1 = int(math.floor(aabb[3] / c))
        j1 = int(math.floor(aabb[4] / c))
        k1 = int(math.floor(aabb[5] / c))
        keys = []
        for i in range(i0, i1 + 1):
            for j in range(j0, j1 + 1):
                for k in range(k0, k1 + 1):
                    keys.append((i, j, k))
        return keys

    def query(self, aabb):
        seen = set()
        out = []
        for key in self._keys(aabb):
            for idx in self.buckets.get(key, ()):
                if idx not in seen:
                    seen.add(idx)
                    out.append(idx)
        return out


def build_anatomy_surface_cache(mesh_fn, name=None, points=None, topology=None):
    """Build a static triangle cache for one anatomy mesh (Maya or synthetic)."""
    if topology is None and mesh_fn is not None:
        topology = get_triangle_topology(mesh_fn)
    if topology is None:
        topology = {"triangles": [], "face_vertex_ids": []}
    if points is None and mesh_fn is not None and MAYA_AVAILABLE and om is not None:
        try:
            pts = mesh_fn.getPoints(om.MSpace.kWorld)
            points = [[p.x, p.y, p.z] for p in pts]
        except Exception:
            points = []
    points = points or []
    triangles = topology.get("triangles") or []
    tri_points = []
    tri_aabb = []
    for face_id, i0, i1, i2 in triangles:
        if max(i0, i1, i2) >= len(points):
            continue
        tp = (list(points[i0]), list(points[i1]), list(points[i2]))
        tri_points.append(tp)
        tri_aabb.append(_aabb_from_points(tp))
    mesh_aabb = _aabb_union(tri_aabb)
    grid = None
    if tri_aabb:
        extents = [max(a[3] - a[0], a[4] - a[1], a[5] - a[2]) for a in tri_aabb]
        mean_e = sum(extents) / len(extents)
        if mesh_aabb is not None:
            diag = math.sqrt((mesh_aabb[3] - mesh_aabb[0]) ** 2
                             + (mesh_aabb[4] - mesh_aabb[1]) ** 2
                             + (mesh_aabb[5] - mesh_aabb[2]) ** 2)
            cell = max(mean_e * 1.5, diag / 32.0, 1e-4)
        else:
            cell = max(mean_e * 1.5, 1e-4)
        if len(tri_aabb) >= 8:
            grid = _UniformGrid(tri_aabb, cell)
    return {
        "name": name,
        "triangles": triangles,
        "points": points,
        "tri_points": tri_points,
        "tri_aabb": tri_aabb,
        "mesh_aabb": mesh_aabb,
        "grid": grid,
        "topology": topology,
        "category": anatomy_category_from_name(name),
    }


def _segment_triangle_hit(p0, p1, tri, eps):
    """Möller–Trumbore on the finite segment p0->p1 vs triangle tri.

    Returns dict {t, u, v, point, interior} or None.
    ``interior`` is True when the hit is clearly inside both the segment and
    the triangle (not on a vertex/edge within ``eps``).
    """
    a, b, c = tri
    e1 = vec_sub(b, a)
    e2 = vec_sub(c, a)
    direction = vec_sub(p1, p0)
    h = vec_cross(direction, e2)
    det = vec_dot(e1, h)
    if abs(det) < eps:
        return None  # parallel; coplanar handled separately
    inv = 1.0 / det
    s = vec_sub(p0, a)
    u = inv * vec_dot(s, h)
    if u < -eps or u > 1.0 + eps:
        return None
    q = vec_cross(s, e1)
    v = inv * vec_dot(direction, q)
    if v < -eps or (u + v) > 1.0 + eps:
        return None
    t = inv * vec_dot(e2, q)
    if t < -eps or t > 1.0 + eps:
        return None
    point = vec_add(p0, vec_scale(direction, t))
    interior = (eps < t < 1.0 - eps and u > eps and v > eps and (u + v) < 1.0 - eps)
    return {"t": t, "u": u, "v": v, "point": point, "interior": interior}


def _project_tri_2d(tri, normal):
    ax = abs(normal[0])
    ay = abs(normal[1])
    az = abs(normal[2])
    if ax >= ay and ax >= az:
        return [[p[1], p[2]] for p in tri]
    if ay >= ax and ay >= az:
        return [[p[0], p[2]] for p in tri]
    return [[p[0], p[1]] for p in tri]


def _sat_2d(A, B, eps):
    """Return ('overlap'|'touching'|None) for two 2D triangles."""
    def axes(T):
        out = []
        for i in range(3):
            x0, y0 = T[i]
            x1, y1 = T[(i + 1) % 3]
            n = (y0 - y1, x1 - x0)
            ln = math.hypot(n[0], n[1])
            if ln > eps:
                out.append((n[0] / ln, n[1] / ln))
        return out
    min_overlap = float("inf")
    for n in axes(A) + axes(B):
        pa = [p[0] * n[0] + p[1] * n[1] for p in A]
        pb = [p[0] * n[0] + p[1] * n[1] for p in B]
        overlap = min(max(pa), max(pb)) - max(min(pa), min(pb))
        if overlap < -eps:
            return None
        if overlap < min_overlap:
            min_overlap = overlap
    if min_overlap > eps:
        return "overlap"
    return "touching"


def triangle_triangle_intersection(tri_a, tri_b, eps=DEFAULT_INTERSECTION_TOLERANCE):
    """Classify the intersection of two triangles in R^3.

    Returns ``(classification, point_or_none)`` where classification is one of
    ``crossing``, ``touching``, ``coplanar_overlap``, ``uncertain``, or None
    (no intersection). Tests edges of BOTH triangles (skin vs anatomy and
    anatomy vs skin). Close-but-not-intersecting pairs are not flagged.
    """
    a0, a1, a2 = tri_a
    b0, b1, b2 = tri_b
    n_a = vec_cross(vec_sub(a1, a0), vec_sub(a2, a0))
    n_b = vec_cross(vec_sub(b1, b0), vec_sub(b2, b0))
    area_a = vec_length(n_a)
    area_b = vec_length(n_b)
    if area_a < eps or area_b < eps:
        return "uncertain", None
    n_a = vec_scale(n_a, 1.0 / area_a)
    n_b = vec_scale(n_b, 1.0 / area_b)
    da = -vec_dot(n_a, a0)
    db = -vec_dot(n_b, b0)
    dist_b = [vec_dot(n_a, p) + da for p in tri_b]
    dist_a = [vec_dot(n_b, p) + db for p in tri_a]
    if (all(d > eps for d in dist_b) or all(d < -eps for d in dist_b)
            or all(d > eps for d in dist_a) or all(d < -eps for d in dist_a)):
        return None, None
    coplanar = (all(abs(d) <= eps for d in dist_b)
                or all(abs(d) <= eps for d in dist_a))
    if coplanar:
        A2 = _project_tri_2d(tri_a, n_a)
        B2 = _project_tri_2d(tri_b, n_a)
        sat = _sat_2d(A2, B2, eps)
        if sat == "overlap":
            mid = [(a0[i] + a1[i] + a2[i]) / 3.0 for i in range(3)]
            return "coplanar_overlap", mid
        if sat == "touching":
            return "touching", list(a0)
        return None, None
    hits = []
    interior = False
    for (p, q) in ((a0, a1), (a1, a2), (a2, a0)):
        h = _segment_triangle_hit(p, q, tri_b, eps)
        if h:
            hits.append(h)
            interior = interior or h["interior"]
    for (p, q) in ((b0, b1), (b1, b2), (b2, b0)):
        h = _segment_triangle_hit(p, q, tri_a, eps)
        if h:
            hits.append(h)
            interior = interior or h["interior"]
    if not hits:
        return None, None
    pt = hits[0]["point"]
    if interior:
        return "crossing", pt
    return "touching", pt


def _buffered_boundary_vertices(skin_mesh, neighbors, rings, boundary_set=None):
    if boundary_set is None:
        try:
            boundary_set = set(get_boundary_vertices(skin_mesh) or [])
        except Exception:
            boundary_set = set()
    if rings and rings > 0 and neighbors is not None and boundary_set:
        return set(grow_indices(neighbors, list(boundary_set), rings=int(rings)))
    return set(boundary_set or [])


def _faces_incident_to_vertices(vertex_faces, indices, extra_face_ids=None):
    faces = set(extra_face_ids or [])
    for i in indices:
        if 0 <= i < len(vertex_faces):
            faces.update(vertex_faces[i])
    return faces


def _empty_intersection_report():
    return {
        "candidate_skin_vertex_count": 0,
        "candidate_skin_face_count": 0,
        "intersecting_skin_face_count": 0,
        "intersecting_skin_vertex_count": 0,
        "intersecting_skin_faces": [],
        "intersecting_skin_vertices": [],
        "intersection_core_vertices": [],
        "intersection_pair_count": 0,
        "touching_pair_count": 0,
        "uncertain_pair_count": 0,
        "coplanar_overlap_pair_count": 0,
        "crossing_pair_count": 0,
        "intersections_by_anatomy_mesh": {},
        "intersecting_anatomy_meshes": [],
        "boundary_excluded_face_count": 0,
        "runtime_seconds": 0.0,
        "details": None,
    }


def analyze_skin_anatomy_intersections(skin_mesh, skin_indices=None,
                                       anatomical_meshes=None, backend=None,
                                       positions=None, neighbors=None,
                                       skin_topology=None,
                                       boundary_buffer_rings=1,
                                       candidate_growth_rings=0,
                                       intersection_tolerance=DEFAULT_INTERSECTION_TOLERANCE,
                                       detailed=False, verbose=True,
                                       exclude_boundary_faces=True):
    """Detect skin-face / anatomy-face intersections (protrusions).

    ``skin_indices=None`` scans the whole skin. Otherwise every skin face
    incident to the given vertices is tested (a face is in scope if *any* of
    its vertices is in the region). Not a vertex-path test: this is actual
    triangle-triangle intersection. Open anatomy meshes are supported.
    """
    t0 = time.time()
    report = _empty_intersection_report()
    if positions is None and skin_mesh:
        positions = get_mesh_vertices(skin_mesh)
    if not positions:
        if verbose:
            print("[intersection] no skin vertices")
        return report
    if neighbors is None and skin_mesh:
        try:
            neighbors = get_vertex_neighbors(skin_mesh)
        except Exception:
            neighbors = None
    if skin_topology is None:
        mesh_fn = get_mesh_fn(skin_mesh) if skin_mesh else None
        if mesh_fn is not None:
            skin_topology = get_triangle_topology(mesh_fn)
        else:
            skin_topology = {"triangles": [], "face_vertex_ids": [],
                             "vertex_faces": [[] for _ in positions]}
    triangles = skin_topology.get("triangles") or []
    face_vertex_ids = skin_topology.get("face_vertex_ids") or []
    vertex_faces = skin_topology.get("vertex_faces") or []
    n = len(positions)
    if skin_indices is None:
        region = list(range(n))
    else:
        region = sorted(set(int(i) for i in skin_indices if 0 <= int(i) < n))
        if candidate_growth_rings and neighbors is not None:
            region = sorted(set(grow_indices(neighbors, region,
                                             rings=int(candidate_growth_rings))))
    candidate_faces = _faces_incident_to_vertices(vertex_faces, region)
    if not candidate_faces and triangles:
        # topology without vertex_faces (synthetic): include faces whose
        # triangle verts intersect the region.
        region_set = set(region)
        for face_id, i0, i1, i2 in triangles:
            if i0 in region_set or i1 in region_set or i2 in region_set:
                candidate_faces.add(face_id)
    buffered = _buffered_boundary_vertices(
        skin_mesh, neighbors, boundary_buffer_rings) if exclude_boundary_faces else set()
    excluded_faces = set()
    if buffered:
        for fi in list(candidate_faces):
            verts = face_vertex_ids[fi] if fi < len(face_vertex_ids) else []
            if verts and all(v in buffered for v in verts):
                excluded_faces.add(fi)
            elif verts and any(v in buffered for v in verts) and len(verts) >= 3:
                # Face that *touches* a buffered opening: exclude only if it
                # has a true boundary vertex (opening edge), already in buffer.
                if any(v in buffered for v in verts):
                    # Conservative: skip faces with ANY buffered-boundary vert
                    # so eye/mouth/nostril/neck rims are not treated as errors
                    # merely because anatomy is visible through the hole.
                    excluded_faces.add(fi)
    tested_faces = candidate_faces - excluded_faces
    report["candidate_skin_vertex_count"] = len(region)
    report["candidate_skin_face_count"] = len(candidate_faces)
    report["boundary_excluded_face_count"] = len(excluded_faces)

    cache = {}
    if backend is not None and hasattr(backend, "anatomy_surface_cache"):
        cache = backend.anatomy_surface_cache() or {}
    elif anatomical_meshes and isinstance(anatomical_meshes, dict):
        # dict of name -> prebuilt cache entries or {points, triangles}
        cache = anatomical_meshes

    # Skin triangle list restricted to tested faces.
    skin_tris = []  # (face_id, (p0,p1,p2), aabb, (i0,i1,i2))
    for face_id, i0, i1, i2 in triangles:
        if face_id not in tested_faces:
            continue
        if max(i0, i1, i2) >= n:
            continue
        tp = (positions[i0], positions[i1], positions[i2])
        skin_tris.append((face_id, tp, _aabb_from_points(tp), (i0, i1, i2)))
    skin_union_aabb = _aabb_union([t[2] for t in skin_tris])

    crossing_faces = set()
    touching_faces = set()
    overlap_faces = set()
    uncertain_faces = set()
    pairs_by_mesh = {}
    pair_count = {"crossing": 0, "touching": 0, "coplanar_overlap": 0, "uncertain": 0}
    details = [] if detailed else None
    eps = float(intersection_tolerance)

    for mesh_name, entry in cache.items():
        if not entry:
            continue
        mesh_aabb = entry.get("mesh_aabb")
        if skin_union_aabb is not None and mesh_aabb is not None:
            if not _aabb_overlap(skin_union_aabb, mesh_aabb):
                continue
        tri_points = entry.get("tri_points") or []
        tri_aabb = entry.get("tri_aabb") or []
        tri_ids = entry.get("triangles") or []
        grid = entry.get("grid")
        mesh_pairs = 0
        for face_id, stp, saabb, sidx in skin_tris:
            if grid is not None:
                cand = grid.query(saabb)
            else:
                cand = range(len(tri_points))
            for ti in cand:
                if ti >= len(tri_points):
                    continue
                aaabb = tri_aabb[ti] if ti < len(tri_aabb) else None
                if aaabb is not None and not _aabb_overlap(saabb, aaabb):
                    continue
                atp = tri_points[ti]
                klass, pt = triangle_triangle_intersection(stp, atp, eps=eps)
                if klass is None:
                    continue
                pair_count[klass] = pair_count.get(klass, 0) + 1
                mesh_pairs += 1
                aface = tri_ids[ti][0] if ti < len(tri_ids) else -1
                if klass == "crossing":
                    crossing_faces.add(face_id)
                elif klass == "coplanar_overlap":
                    overlap_faces.add(face_id)
                elif klass == "touching":
                    touching_faces.add(face_id)
                elif klass == "uncertain":
                    uncertain_faces.add(face_id)
                if details is not None:
                    details.append({
                        "skin_face_id": face_id,
                        "skin_vertex_indices": list(sidx),
                        "anatomy_mesh": mesh_name,
                        "anatomy_category": anatomy_category_from_name(mesh_name),
                        "anatomy_face_id": aface,
                        "classification": klass,
                        "point": pt,
                    })
        if mesh_pairs:
            pairs_by_mesh[mesh_name] = pairs_by_mesh.get(mesh_name, 0) + mesh_pairs

    intersecting_faces = sorted(crossing_faces | overlap_faces)
    # Report vertices of any flagged intersecting (repair-class) face.
    isect_verts = set()
    for fi in intersecting_faces:
        if fi < len(face_vertex_ids):
            isect_verts.update(face_vertex_ids[fi])
        else:
            for face_id, i0, i1, i2 in triangles:
                if face_id == fi:
                    isect_verts.update((i0, i1, i2))
    report.update({
        "intersecting_skin_face_count": len(intersecting_faces),
        "intersecting_skin_vertex_count": len(isect_verts),
        "intersecting_skin_faces": intersecting_faces,
        "intersecting_skin_vertices": sorted(isect_verts),
        "intersection_core_vertices": sorted(isect_verts),
        "intersection_pair_count": pair_count.get("crossing", 0) + pair_count.get("coplanar_overlap", 0),
        "crossing_pair_count": pair_count.get("crossing", 0),
        "touching_pair_count": pair_count.get("touching", 0),
        "uncertain_pair_count": pair_count.get("uncertain", 0),
        "coplanar_overlap_pair_count": pair_count.get("coplanar_overlap", 0),
        "intersections_by_anatomy_mesh": pairs_by_mesh,
        "intersecting_anatomy_meshes": sorted(pairs_by_mesh.keys()),
        "touching_skin_faces": sorted(touching_faces),
        "uncertain_skin_faces": sorted(uncertain_faces),
        "runtime_seconds": time.time() - t0,
        "details": details,
        "intersection_tolerance": eps,
        "skin_topology": None,  # not dumped; caller already has it
    })
    if verbose:
        print_intersection_report(report)
    return report


def print_intersection_report(report, label=""):
    prefix = "[intersection{0}]".format(" " + label if label else "")
    print("{0} faces={1}/{2} verts={3} pairs={4} (crossing={5} overlap={6} "
          "touching={7} uncertain={8})  {9:.3f}s".format(
              prefix,
              report.get("intersecting_skin_face_count", 0),
              report.get("candidate_skin_face_count", 0),
              report.get("intersecting_skin_vertex_count", 0),
              report.get("intersection_pair_count", 0),
              report.get("crossing_pair_count", 0),
              report.get("coplanar_overlap_pair_count", 0),
              report.get("touching_pair_count", 0),
              report.get("uncertain_pair_count", 0),
              report.get("runtime_seconds", 0.0)))
    bym = report.get("intersections_by_anatomy_mesh") or {}
    if bym:
        parts = ["{0}:{1}".format(k, v) for k, v in sorted(bym.items(),
                                                           key=lambda kv: -kv[1])]
        print("  by anatomy mesh: {0}".format(", ".join(parts[:12])
                                              + (" ..." if len(parts) > 12 else "")))


def select_intersecting_skin_faces(report, mesh_name, replace=True):
    faces = report.get("intersecting_skin_faces") or []
    if not faces:
        print("[intersection] no intersecting skin faces to select")
        return []
    select_faces(mesh_name, faces, replace=replace)
    print("[intersection] selected {0} intersecting SKIN faces".format(len(faces)))
    return list(faces)


def select_intersecting_skin_vertices(report, mesh_name, replace=True):
    idx = report.get("intersecting_skin_vertices") or []
    if not idx:
        print("[intersection] no intersecting skin vertices to select")
        return []
    select_vertices(mesh_name, idx, replace=replace)
    print("[intersection] selected {0} intersecting SKIN vertices".format(len(idx)))
    return list(idx)


def _mean_local_edge(positions, neighbors, index):
    if neighbors is None or index >= len(neighbors) or not neighbors[index]:
        return 0.0
    nbrs = neighbors[index]
    return (sum(vec_length(vec_sub(positions[index], positions[j])) for j in nbrs)
            / float(len(nbrs)))


def _outward_skin_normal(index, normals, positions, skin_topology):
    n = None
    if normals is not None and index < len(normals):
        n = vec_normalize(normals[index])
    if n is not None and vec_length(n) > 1e-12:
        return n
    # Area-weighted average of incident triangle normals.
    vertex_faces = (skin_topology or {}).get("vertex_faces") or []
    triangles = (skin_topology or {}).get("triangles") or []
    acc = [0.0, 0.0, 0.0]
    faces = vertex_faces[index] if index < len(vertex_faces) else []
    face_set = set(faces)
    for face_id, i0, i1, i2 in triangles:
        if face_set and face_id not in face_set:
            continue
        if not face_set and index not in (i0, i1, i2):
            continue
        if max(i0, i1, i2) >= len(positions):
            continue
        a, b, c = positions[i0], positions[i1], positions[i2]
        cr = vec_cross(vec_sub(b, a), vec_sub(c, a))
        acc = vec_add(acc, cr)
    n = vec_normalize(acc)
    if vec_length(n) < 1e-12:
        n = [0.0, 0.0, 1.0]
    return n


def intersection_vertices_from_report(report, skin_topology, allowed=None,
                                      boundary=None):
    """Map intersecting faces -> vertex ids, then clip by allowed/boundary."""
    core = set()
    faces = report.get("intersecting_skin_faces") or []
    face_vertex_ids = (skin_topology or {}).get("face_vertex_ids") or []
    triangles = (skin_topology or {}).get("triangles") or []
    for fi in faces:
        if fi < len(face_vertex_ids) and face_vertex_ids[fi]:
            core.update(face_vertex_ids[fi])
        else:
            for face_id, i0, i1, i2 in triangles:
                if face_id == fi:
                    core.update((i0, i1, i2))
    if allowed is not None:
        core &= set(allowed)
    if boundary:
        core -= set(boundary)
    return sorted(core)


def _legacy_normal_push_intersection_repair(
        positions, skin_indices, backend,
        skin_mesh=None, neighbors=None, normals=None, skin_topology=None,
        min_clearance=0.8, clearance_tolerance=DEFAULT_CLEARANCE_TOLERANCE,
        max_intersection_repair_iterations=DEFAULT_MAX_INTERSECTION_REPAIR_ITERATIONS,
        intersection_repair_step_ratio=DEFAULT_INTERSECTION_REPAIR_STEP_RATIO,
        intersection_repair_growth_rings=0,
        boundary_buffer_rings=1,
        intersection_tolerance=DEFAULT_INTERSECTION_TOLERANCE,
        repair_classes=AUTO_REPAIR_INTERSECTION_CLASSES,
        binary_search_min_step=True,
        verbose=True):
    """V1 intersection repair: repeated per-vertex skin-normal escape.

    Preserved for A/B comparison (``repair_mode="legacy_normal_push"``).
    This is the spike-prone method: isolated core vertices step along their
    own normals by a fraction of local edge length. Do NOT use as the default.
    """
    work = [list(v) for v in positions]
    n = len(work)
    allowed = set(i for i in (skin_indices or range(n)) if 0 <= i < n)
    if intersection_repair_growth_rings and neighbors is not None:
        allowed = set(grow_indices(neighbors, list(allowed),
                                   rings=int(intersection_repair_growth_rings)))
    boundary = _buffered_boundary_vertices(
        skin_mesh, neighbors, boundary_buffer_rings)
    allowed -= boundary

    def _analyze(pos, indices=None):
        return analyze_skin_anatomy_intersections(
            skin_mesh, skin_indices=indices if indices is not None else sorted(allowed),
            backend=backend, positions=pos, neighbors=neighbors,
            skin_topology=skin_topology,
            boundary_buffer_rings=boundary_buffer_rings,
            intersection_tolerance=intersection_tolerance,
            detailed=False, verbose=False)

    before = _analyze(work)
    moved = set()
    displacements = []
    last_report = before
    resolved_faces = set()
    iters_used = 0
    auto = set(repair_classes or AUTO_REPAIR_INTERSECTION_CLASSES)

    for it in range(max(1, int(max_intersection_repair_iterations))):
        iters_used = it + 1
        last_report = _analyze(work)
        # Only auto-repair crossing / optional coplanar_overlap faces.
        repair_faces = set(last_report.get("intersecting_skin_faces") or [])
        if "coplanar_overlap" not in auto:
            # re-filter is approximate; intersecting_skin_faces already is
            # crossing ∪ overlap. If overlap excluded, drop overlap-only via details
            # not available; treat all intersecting_skin_faces as repair targets
            # when crossing is in auto (default includes both).
            pass
        if last_report.get("intersection_pair_count", 0) <= 0:
            break
        core = intersection_vertices_from_report(
            last_report, skin_topology, allowed=allowed, boundary=boundary)
        if not core:
            break
        unsafe = {i: list(work[i]) for i in core}
        step_ratio = float(intersection_repair_step_ratio)
        # Mild adaptive increase if the same faces persist.
        if it >= 4:
            step_ratio *= 1.5
        if it >= 10:
            step_ratio *= 1.5
        for i in core:
            nrm = _outward_skin_normal(i, normals, work, skin_topology)
            edge = _mean_local_edge(work, neighbors, i)
            if edge <= 1e-12:
                edge = vec_length(nrm) or 1.0
            step = step_ratio * edge
            work[i] = vec_add(work[i], vec_scale(nrm, step))
            if min_clearance is not None and backend is not None:
                sol = enforce_anatomy_clearance(
                    work[i], backend, min_clearance,
                    clearance_tolerance=clearance_tolerance, skin_normal=nrm)
                work[i] = sol["position"]
            moved.add(i)
        trial = _analyze(work, indices=core)
        if (trial.get("intersection_pair_count", 0) <= 0
                and binary_search_min_step):
            lo, hi = 0.0, 1.0
            best = {i: list(work[i]) for i in core}
            for _ in range(6):
                mid = 0.5 * (lo + hi)
                for i in core:
                    work[i] = [
                        unsafe[i][k] + mid * (best[i][k] - unsafe[i][k])
                        for k in range(3)]
                mid_rep = _analyze(work, indices=core)
                if mid_rep.get("intersection_pair_count", 0) <= 0:
                    hi = mid
                    best = {i: list(work[i]) for i in core}
                else:
                    lo = mid
            for i in core:
                work[i] = best[i]
        for i in core:
            displacements.append(vec_length(vec_sub(work[i], positions[i])))
        now_faces = set((_analyze(work).get("intersecting_skin_faces") or []))
        resolved_faces |= (repair_faces - now_faces)

    after = _analyze(work)
    unresolved_faces = after.get("intersecting_skin_faces") or []
    result = {
        "positions": work,
        "intersection_repair_iterations": iters_used,
        "intersection_vertices_moved": sorted(moved),
        "intersection_faces_resolved": len(resolved_faces),
        "intersection_faces_unresolved": len(unresolved_faces),
        "mean_intersection_repair_displacement": (
            (sum(displacements) / len(displacements)) if displacements else 0.0),
        "max_intersection_repair_displacement": (
            max(displacements) if displacements else 0.0),
        "report_before": before,
        "report_after": after,
        "unresolved_intersecting_faces": unresolved_faces,
    }
    if verbose:
        print("[intersection-repair] iters={0} moved={1} faces {2}->{3} "
              "resolved={4} unresolved={5} mean_disp={6:.5f} max_disp={7:.5f}".format(
                  iters_used, len(moved),
                  before.get("intersecting_skin_face_count", 0),
                  after.get("intersecting_skin_face_count", 0),
                  len(resolved_faces), len(unresolved_faces),
                  result["mean_intersection_repair_displacement"],
                  result["max_intersection_repair_displacement"]))
    result["repair_mode"] = "legacy_normal_push"
    result["repair_passes"] = iters_used
    result["intersecting_faces_before"] = before.get("intersecting_skin_face_count", 0)
    result["intersecting_faces_after"] = after.get("intersecting_skin_face_count", 0)
    result["intersection_pairs_before"] = before.get("intersection_pair_count", 0)
    result["intersection_pairs_after"] = after.get("intersection_pair_count", 0)
    result["core_vertex_count"] = len(moved)
    result["patch_vertex_count"] = len(moved)
    result["core_vertices"] = sorted(moved)
    result["patch_vertices"] = sorted(moved)
    result["unresolved_intersection_faces"] = unresolved_faces
    result["unresolved_core_vertices"] = intersection_vertices_from_report(
        after, skin_topology, allowed=allowed, boundary=boundary)
    result["offending_anatomy_meshes"] = list(
        after.get("intersecting_anatomy_meshes") or before.get("intersecting_anatomy_meshes") or [])
    result["mean_core_target_distance"] = result["mean_intersection_repair_displacement"]
    result["max_core_target_distance"] = result["max_intersection_repair_displacement"]
    result["mean_applied_displacement"] = result["mean_intersection_repair_displacement"]
    result["max_applied_displacement"] = result["max_intersection_repair_displacement"]
    result["binary_search_count"] = 0
    result["post_relax_displacement"] = 0.0
    v1_field = {}
    for i in moved:
        v1_field[i] = vec_sub(work[i], positions[i])
    qv1 = _measure_patch_repair_quality(v1_field, neighbors, work)
    result["mean_displacement_gradient"] = qv1["mean_displacement_gradient"]
    result["max_displacement_gradient"] = qv1["max_displacement_gradient"]
    return result


def _exact_closest_named(backend, point, mesh_name):
    if backend is None:
        return {
            "distance": float("inf"), "closest": list(point),
            "normal": [0.0, 0.0, 1.0], "mesh": mesh_name, "face_id": -1,
            "category": "other", "hits": [],
        }
    if mesh_name:
        try:
            return backend.exact_closest(point, names=[mesh_name])
        except TypeError:
            pass
    return backend.exact_closest(point)


def _closest_point_on_triangle(point, tri):
    """Return (closest, unnormalized_normal) of ``point`` on triangle ``tri``."""
    a, b, c = tri
    ab = vec_sub(b, a)
    ac = vec_sub(c, a)
    n_raw = vec_cross(ab, ac)
    ap = vec_sub(point, a)
    d1 = vec_dot(ab, ap)
    d2 = vec_dot(ac, ap)
    if d1 <= 0.0 and d2 <= 0.0:
        return list(a), n_raw
    bp = vec_sub(point, b)
    d3 = vec_dot(ab, bp)
    d4 = vec_dot(ac, bp)
    if d3 >= 0.0 and d4 <= d3:
        return list(b), n_raw
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3) if abs(d1 - d3) > 1e-18 else 0.0
        return vec_add(a, vec_scale(ab, v)), n_raw
    cp = vec_sub(point, c)
    d5 = vec_dot(ab, cp)
    d6 = vec_dot(ac, cp)
    if d6 >= 0.0 and d5 <= d6:
        return list(c), n_raw
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6) if abs(d2 - d6) > 1e-18 else 0.0
        return vec_add(a, vec_scale(ac, w)), n_raw
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        denom = (d4 - d3) + (d5 - d6)
        w = (d4 - d3) / denom if abs(denom) > 1e-18 else 0.0
        return vec_add(b, vec_scale(vec_sub(c, b), w)), n_raw
    denom = va + vb + vc
    if abs(denom) < 1e-18:
        return list(a), n_raw
    v = vb / denom
    w = vc / denom
    return vec_add(a, vec_add(vec_scale(ab, v), vec_scale(ac, w))), n_raw


def _orient_support_normal(n_anat, n_skin):
    """Agree anatomy normal with skin outward; fall back to skin if degenerate."""
    n_skin_u = vec_normalize(n_skin) if n_skin is not None else [0.0, 0.0, 1.0]
    if vec_length(n_skin_u) < 1e-12:
        n_skin_u = [0.0, 0.0, 1.0]
    if n_anat is None or vec_length(n_anat) < 1e-12:
        return n_skin_u, "skin_fallback"
    n = vec_normalize(n_anat)
    if vec_dot(n, n_skin_u) < 0.0:
        n = vec_scale(n, -1.0)
    return n, "oriented_anatomy"


def _support_from_record(backend, point, rec):
    """Closest point + raw anatomy normal for one intersection record.

    Prefers the recorded anatomy face in the EXISTING surface cache (no
    accelerator rebuild), then ``exact_closest`` on that mesh, then the
    recorded intersection point.
    """
    mesh = (rec or {}).get("anatomy_mesh")
    face_id = (rec or {}).get("anatomy_face_id")
    rec_pt = (rec or {}).get("point")
    closest = None
    n_raw = None
    dist = float("inf")
    source = "none"
    cache = {}
    if backend is not None and hasattr(backend, "anatomy_surface_cache"):
        try:
            cache = backend.anatomy_surface_cache() or {}
        except Exception:
            cache = {}
    entry = cache.get(mesh) if mesh else None
    if entry:
        tris = entry.get("triangles") or []
        tpts = entry.get("tri_points") or []
        if face_id is not None:
            for ti, tdef in enumerate(tris):
                if tdef[0] != face_id or ti >= len(tpts):
                    continue
                cp, n_raw = _closest_point_on_triangle(point, tpts[ti])
                closest = cp
                dist = vec_length(vec_sub(point, cp))
                source = "cache_face"
                break
        if closest is None:
            for tp in tpts:
                cp, nr = _closest_point_on_triangle(point, tp)
                d = vec_length(vec_sub(point, cp))
                if d < dist:
                    dist, closest, n_raw = d, cp, nr
                    source = "cache_mesh"
    if closest is None:
        q = _exact_closest_named(backend, point, mesh)
        qdist = q.get("distance", float("inf"))
        if q.get("mesh") and math.isfinite(qdist):
            closest = list(q["closest"])
            n_raw = list(q.get("normal") or [0.0, 0.0, 0.0])
            dist = qdist
            source = "exact_closest"
    if closest is None and rec_pt is not None:
        closest = list(rec_pt)
        source = "intersection_point"
        dist = vec_length(vec_sub(point, closest))
    if closest is None:
        closest = list(point)
        dist = 0.0
        source = "identity"
    return {
        "closest": closest,
        "normal": n_raw,
        "distance": dist,
        "mesh": mesh,
        "source": source,
    }


def _collect_offending_anatomy(report, skin_topology, core):
    """Map each core vertex to intersection records of its incident faces.

    Reuses the EXISTING analyzer's ``details`` when present; otherwise falls
    back to ``intersecting_anatomy_meshes`` for every core vertex.
    """
    by_vert = {i: [] for i in core}
    details = report.get("details") or []
    face_verts = (skin_topology or {}).get("face_vertex_ids") or []
    meshes = list(report.get("intersecting_anatomy_meshes") or [])
    stub = [{"anatomy_mesh": m, "anatomy_face_id": None, "point": None,
             "classification": "crossing"} for m in meshes]
    if details:
        for rec in details:
            if rec.get("classification") not in AUTO_REPAIR_INTERSECTION_CLASSES:
                continue
            fi = rec.get("skin_face_id")
            verts = list(rec.get("skin_vertex_indices") or [])
            if not verts and fi is not None and fi < len(face_verts):
                verts = list(face_verts[fi])
            for v in verts:
                if v in by_vert:
                    by_vert[v].append(rec)
        for i in core:
            if not by_vert[i]:
                by_vert[i] = list(stub)
        return by_vert
    for i in core:
        by_vert[i] = list(stub)
    return by_vert


def _compute_anatomy_supported_target(point, skin_normal, offending_records,
                                      backend, offset, max_disp):
    """Place ``point`` just outside ALL offending anatomy surfaces.

    Motion is along the oriented support normal only (no tangential snap to
    the closest point). Multiple meshes are applied sequentially -- never
    averaged, because an average can recross one of the surfaces.
    """
    cand = list(point)
    method = "none"
    last_n = list(skin_normal) if skin_normal is not None else [0.0, 0.0, 1.0]
    last_p = list(point)
    candidates_info = []
    recs = list(offending_records or [])
    if not recs:
        recs = [{}]
    for _pass in range(3):
        moved = False
        for rec in recs:
            sup = _support_from_record(backend, cand, rec)
            n, method_i = _orient_support_normal(sup.get("normal"), skin_normal)
            p_i = sup["closest"]
            side = vec_dot(vec_sub(cand, p_i), n)
            if side < offset - 1e-12:
                cand = vec_add(cand, vec_scale(n, offset - side))
                last_n, last_p = n, list(p_i)
                method = method_i + "+" + sup.get("source", "")
                moved = True
            candidates_info.append({
                "mesh": rec.get("anatomy_mesh"),
                "source": sup.get("source"),
                "method": method_i,
                "closest": list(p_i),
                "normal": list(n),
            })
        if not moved:
            break
    disp = vec_sub(cand, point)
    mag = vec_length(disp)
    if max_disp is not None and mag > max_disp > 0.0:
        cand = vec_add(point, vec_scale(disp, max_disp / mag))
        mag = max_disp
        method = method + "+clamped"
    meshes = []
    for rec in recs:
        m = rec.get("anatomy_mesh")
        if m and m not in meshes:
            meshes.append(m)
    return cand, {
        "support_method": method,
        "support_normal": last_n,
        "closest": last_p,
        "target": list(cand),
        "target_distance": mag,
        "offending_meshes": meshes,
        "candidate_supports": candidates_info,
    }


def _find_minimal_safe_target(base, field, analyze_fn, steps=8,
                              position_tolerance=None):
    """Binary-search the scale of a patch displacement field.

    Face intersections cannot be decided per vertex, so the search is on the
    whole field: ``x' = x + alpha * d``, smallest ``alpha`` in [0, 1] that
    locally clears. If alpha=1 is still intersecting, returns that candidate
    for the next repair pass (does not invent extra extrusion).
    """
    full = [list(v) for v in base]
    for i, di in field.items():
        full[i] = vec_add(base[i], di)
    full_rep = analyze_fn(full)
    if full_rep.get("intersection_pair_count", 0) > 0:
        return full, 1.0, 0, False
    lo, hi = 0.0, 1.0
    best = [list(v) for v in full]
    nsteps = max(1, int(steps))
    max_mag = max((vec_length(d) for d in field.values()), default=0.0)
    searches = 0
    for _ in range(nsteps):
        if (position_tolerance is not None and max_mag > 0.0
                and (hi - lo) * max_mag <= float(position_tolerance)):
            break
        searches += 1
        mid = 0.5 * (lo + hi)
        trial = [list(v) for v in base]
        for i, di in field.items():
            trial[i] = vec_add(base[i], vec_scale(di, mid))
        tr = analyze_fn(trial)
        if tr.get("intersection_pair_count", 0) <= 0:
            hi = mid
            best = trial
        else:
            lo = mid
    return best, hi, searches, True


def _apply_repair_profile(profile, repair_blend_rings=None, repair_falloff=None,
                          repair_field_iterations=None, repair_boundary_mode=None,
                          repair_field_method=None):
    """Fill unspecified V2 repair knobs from ``legacy_v2_local`` or ``broad_fair``."""
    name = profile or DEFAULT_REPAIR_PROFILE
    if name in ("legacy_v2_local", "legacy_v2", "local"):
        name = "legacy_v2_local"
        spec = {
            "repair_blend_rings": 2,
            "repair_falloff": "harmonic",
            "repair_field_iterations": 40,
            "repair_boundary_mode": "fixed",
            "repair_field_method": "harmonic",
        }
    else:
        name = "broad_fair"
        spec = {
            "repair_blend_rings": DEFAULT_REPAIR_BLEND_RINGS,
            "repair_falloff": DEFAULT_REPAIR_FALLOFF,
            "repair_field_iterations": DEFAULT_HARMONIC_DISPLACEMENT_ITERS,
            "repair_boundary_mode": DEFAULT_REPAIR_BOUNDARY_MODE,
            "repair_field_method": DEFAULT_REPAIR_FIELD_METHOD,
        }
    if repair_blend_rings is not None:
        spec["repair_blend_rings"] = int(repair_blend_rings)
    if repair_falloff is not None:
        spec["repair_falloff"] = str(repair_falloff)
    if repair_field_iterations is not None:
        spec["repair_field_iterations"] = int(repair_field_iterations)
    if repair_boundary_mode is not None:
        spec["repair_boundary_mode"] = str(repair_boundary_mode)
    if repair_field_method is not None:
        spec["repair_field_method"] = str(repair_field_method)
    spec["repair_profile"] = name
    return spec


def _connected_index_components(indices, neighbors):
    """Connected components of ``indices`` under 1-ring adjacency."""
    remaining = set(indices or [])
    comps = []
    while remaining:
        start = remaining.pop()
        stack = [start]
        comp = [start]
        while stack:
            i = stack.pop()
            nbrs = neighbors[i] if (neighbors is not None and i < len(neighbors)) else []
            for j in nbrs:
                if j in remaining:
                    remaining.remove(j)
                    stack.append(j)
                    comp.append(j)
        comps.append(sorted(comp))
    comps.sort(key=len, reverse=True)
    return comps


def _topology_falloff_weight(dist, n_rings, mode="smoothstep"):
    """Weight at topological distance ``dist`` (0=core, n_rings=outer)."""
    if dist <= 0:
        return 1.0
    n = max(1, int(n_rings or 1))
    if dist >= n:
        return 0.0
    t = float(dist) / float(n)
    t = max(0.0, min(1.0, t))
    if mode == "linear":
        return 1.0 - t
    # smoothstep (also used as a gentle envelope; harmonic solve is separate)
    s = t * t * (3.0 - 2.0 * t)
    return 1.0 - s


def _build_intersection_repair_patch(core, neighbors, allowed, boundary,
                                     blend_rings, ring_weights=None):
    """Core + topological rings. Boundary verts never enter the patch.

    Stores ``distance`` (ring index from core) for every patch vertex.
    """
    core_set = set(core)
    boundary = set(boundary or [])
    allowed = set(allowed or core_set)
    distance = {i: 0 for i in core_set}
    weights = {i: 1.0 for i in core_set}
    rings = [set(core_set)]
    current = set(core_set)
    n_rings = max(0, int(blend_rings or 0))
    for r in range(1, n_rings + 1):
        nxt = set()
        for i in current:
            if neighbors is None or i >= len(neighbors):
                continue
            for j in neighbors[i]:
                if j in current or j in boundary or j not in allowed:
                    continue
                nxt.add(j)
                if j not in distance:
                    distance[j] = r
                    w = _topology_falloff_weight(r, n_rings, "smoothstep")
                    # Only honor an explicit ring-weight table that covers the
                    # full radius. The legacy 3-tuple (1.0, 0.6, 0.3) must not
                    # clip a 6-ring patch back into a local mound.
                    if ring_weights and len(ring_weights) >= (n_rings + 1) and r < len(ring_weights):
                        w = float(ring_weights[r])
                    weights[j] = w
        rings.append(nxt)
        current |= nxt
    outer = []
    last_ring = rings[-1] if rings else set()
    for i in current:
        if i in core_set:
            continue
        nbrs = neighbors[i] if (neighbors is not None and i < len(neighbors)) else []
        if (i in last_ring) or (not nbrs) or any(j not in current for j in nbrs):
            if i not in outer:
                outer.append(i)
    return {
        "core": sorted(core_set),
        "patch": sorted(current),
        "rings": [sorted(s) for s in rings],
        "weights": weights,
        "distance": distance,
        "outer": sorted(set(outer)),
        "n_rings": n_rings,
    }


def _merge_overlapping_repair_patches(patch_list):
    """Union-find merge of grown patches that share any vertex.

    Policy: overlapping components become ONE displacement field so a vertex
    is never given two sequential lifts. Disjoint patches stay independent.
    """
    n = len(patch_list)
    if n <= 1:
        return list(patch_list)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    sets = [set(p["patch"]) for p in patch_list]
    for i in range(n):
        for j in range(i + 1, n):
            if sets[i] & sets[j]:
                parent[find(j)] = find(i)
    groups = {}
    for i, p in enumerate(patch_list):
        groups.setdefault(find(i), []).append(p)
    merged = []
    for parts in groups.values():
        if len(parts) == 1:
            merged.append(parts[0])
            continue
        core = sorted(set(v for p in parts for v in p["core"]))
        # Rebuild from combined core using the first part's ring count / maps
        # is done by the caller via _build; here concatenate and mark dirty.
        merged.append({
            "core": core,
            "_rebuild": True,
            "n_rings": max(p.get("n_rings", 0) for p in parts),
        })
    return merged


def _blend_patch_displacements(core_disp, patch_info, neighbors,
                               method="harmonic", harmonic_iters=None,
                               falloff="harmonic", boundary_mode="fixed",
                               stop_eps=DEFAULT_REPAIR_FIELD_STOP_EPS):
    """Fair a displacement field: core pinned, outer ring ~0.

    ``method``:
        harmonic -- Laplacian of *displacement vectors* (default)
        feather  -- topology-weight * mean core displacement
        biharmonic -- extra harmonic passes (approximate fairing; no new deps)
    ``boundary_mode``:
        fixed -- every true outer vertex pinned to 0
        soft  -- only the last grown ring is pinned; near-outer rings stay free
    """
    patch = patch_info["patch"]
    core_set = set(patch_info["core"])
    patch_set = set(patch)
    dist = patch_info.get("distance") or {}
    n_rings = int(patch_info.get("n_rings") or 0)
    last_ring = set((patch_info.get("rings") or [[]])[-1] if n_rings else [])
    if boundary_mode == "soft" and last_ring:
        outer_set = set(i for i in last_ring if i not in core_set)
    else:
        outer_set = set(patch_info.get("outer") or [])
    outer_set &= patch_set
    d = {i: [0.0, 0.0, 0.0] for i in patch}
    for i, vec in (core_disp or {}).items():
        if i in d:
            d[i] = list(vec)
    field_method = method or "harmonic"
    if neighbors is None:
        field_method = "feather"
    if field_method == "feather" or (falloff in ("linear", "smoothstep") and field_method != "harmonic"):
        if core_disp:
            mean_c = [
                sum(v[k] for v in core_disp.values()) / float(len(core_disp))
                for k in range(3)]
        else:
            mean_c = [0.0, 0.0, 0.0]
        mode = falloff if falloff in ("linear", "smoothstep") else "smoothstep"
        for i in patch:
            if i in core_set:
                continue
            w = _topology_falloff_weight(dist.get(i, n_rings), n_rings, mode)
            if boundary_mode == "soft":
                di = dist.get(i, n_rings)
                if di >= n_rings:
                    w = 0.0
                elif n_rings >= 1 and di == n_rings - 1:
                    w = min(w, 0.12)
                elif n_rings >= 2 and di == n_rings - 2:
                    w = min(w, 0.25)
            d[i] = vec_scale(mean_c, w)
        for i in outer_set:
            d[i] = [0.0, 0.0, 0.0]
        return d
    interior = [i for i in patch if i not in core_set and i not in outer_set]
    n_iter = int(harmonic_iters if harmonic_iters is not None
                 else DEFAULT_HARMONIC_DISPLACEMENT_ITERS)
    if field_method == "biharmonic":
        n_iter = max(n_iter, 2 * n_iter)
    eps = float(stop_eps) if stop_eps is not None else DEFAULT_REPAIR_FIELD_STOP_EPS
    for _ in range(max(1, n_iter)):
        new_d = {}
        max_ch = 0.0
        for i in interior:
            nbrs = [j for j in neighbors[i] if j in patch_set] if (
                neighbors is not None and i < len(neighbors)) else []
            if not nbrs:
                continue
            avg = [sum(d[j][k] for j in nbrs) / float(len(nbrs)) for k in range(3)]
            ch = vec_length(vec_sub(avg, d[i]))
            if ch > max_ch:
                max_ch = ch
            new_d[i] = avg
        d.update(new_d)
        if max_ch < eps:
            break
    for i in outer_set:
        d[i] = [0.0, 0.0, 0.0]
    if boundary_mode == "soft" and core_disp:
        mean_core = (sum(vec_length(v) for v in core_disp.values())
                     / float(len(core_disp)))
        for i in patch:
            if i in core_set or i in outer_set:
                continue
            di = int(dist.get(i, n_rings))
            mag = vec_length(d[i])
            cap = None
            if di >= n_rings:
                cap = 0.0
            elif n_rings >= 1 and di == n_rings - 1:
                cap = 0.12 * mean_core
            elif n_rings >= 2 and di == n_rings - 2:
                cap = 0.25 * mean_core
            if cap is not None and mag > cap and mag > 1e-12:
                d[i] = vec_scale(d[i], cap / mag) if cap > 0.0 else [0.0, 0.0, 0.0]
    return d


def _measure_patch_repair_quality(displacements, neighbors, positions=None):
    grads, mags = [], []
    for i, di in (displacements or {}).items():
        mags.append(vec_length(di))
        if neighbors is None or i >= len(neighbors):
            continue
        g = 0.0
        for j in neighbors[i]:
            dj = displacements.get(j, [0.0, 0.0, 0.0])
            g = max(g, vec_length(vec_sub(di, dj)))
        grads.append(g)
    lap = []
    if positions is not None and neighbors is not None:
        for i in displacements or {}:
            nbrs = neighbors[i] if i < len(neighbors) else []
            if not nbrs:
                continue
            avg = [
                sum(positions[j][k] for j in nbrs) / float(len(nbrs))
                for k in range(3)]
            lap.append(vec_length(vec_sub(positions[i], avg)))
    return {
        "mean_displacement_gradient": (sum(grads) / len(grads)) if grads else 0.0,
        "max_displacement_gradient": max(grads) if grads else 0.0,
        "mean_applied_displacement": (sum(mags) / len(mags)) if mags else 0.0,
        "max_applied_displacement": max(mags) if mags else 0.0,
        "mean_laplacian_magnitude": (sum(lap) / len(lap)) if lap else 0.0,
        "max_laplacian_magnitude": max(lap) if lap else 0.0,
    }


def _local_taubin_step(positions, neighbors, indices, strength):
    """One lambda+mu pair on ``indices`` only (no module-level Laplacian import)."""
    idx = [i for i in indices]
    lamb = float(strength)
    mu = -(lamb + 0.01)

    def _step(pos, s):
        out = [list(p) for p in pos]
        for i in idx:
            nbrs = neighbors[i] if (neighbors is not None and i < len(neighbors)) else []
            if not nbrs:
                continue
            avg = [
                sum(pos[j][k] for j in nbrs) / float(len(nbrs)) for k in range(3)]
            out[i] = [pos[i][k] + s * (avg[k] - pos[i][k]) for k in range(3)]
        return out

    mid = _step(positions, lamb)
    return _step(mid, mu)


def _anatomy_supported_patch_repair(
        positions, skin_indices, backend,
        skin_mesh=None, neighbors=None, normals=None, skin_topology=None,
        min_clearance=0.8, clearance_tolerance=DEFAULT_CLEARANCE_TOLERANCE,
        clearance_policy="preserve_valid_baseline",
        boundary_buffer_rings=1,
        intersection_tolerance=DEFAULT_INTERSECTION_TOLERANCE,
        repair_blend_rings=None,
        repair_ring_weights=DEFAULT_REPAIR_RING_WEIGHTS,
        repair_binary_search=True,
        repair_binary_search_steps=DEFAULT_REPAIR_BINARY_SEARCH_STEPS,
        max_surface_repair_passes=DEFAULT_MAX_SURFACE_REPAIR_PASSES,
        max_repair_displacement_ratio=DEFAULT_MAX_REPAIR_DISPLACEMENT_RATIO,
        post_repair_relax=True,
        post_repair_relax_iterations=3,
        post_repair_relax_strength=0.1,
        repair_position_tolerance=None,
        verbose=True,
        repair_profile=DEFAULT_REPAIR_PROFILE,
        repair_falloff=None,
        repair_field_iterations=None,
        repair_boundary_mode=None,
        repair_field_method=None):
    """V2: place the intersecting core just outside offending anatomy, then
    distribute that displacement as a smooth patch field (not per-vertex
    normal extrusion). Re-tests with the UNCHANGED intersection analyzer.
    """
    t0 = time.time()
    spec = _apply_repair_profile(
        repair_profile, repair_blend_rings=repair_blend_rings,
        repair_falloff=repair_falloff,
        repair_field_iterations=repair_field_iterations,
        repair_boundary_mode=repair_boundary_mode,
        repair_field_method=repair_field_method)
    repair_blend_rings = spec["repair_blend_rings"]
    repair_falloff = spec["repair_falloff"]
    repair_field_iterations = spec["repair_field_iterations"]
    repair_boundary_mode = spec["repair_boundary_mode"]
    repair_field_method = spec["repair_field_method"]
    profile_name = spec["repair_profile"]
    work = [list(v) for v in positions]
    n = len(work)
    region = [i for i in (skin_indices or range(n)) if 0 <= i < n]
    boundary = _buffered_boundary_vertices(
        skin_mesh, neighbors, boundary_buffer_rings)
    allowed = set(region)
    if repair_blend_rings and neighbors is not None:
        allowed = set(grow_indices(neighbors, list(region),
                                   rings=int(repair_blend_rings)))
    allowed -= set(boundary)
    floors, orig_dist, policy = {}, {}, clearance_policy or "preserve_valid_baseline"
    if backend is not None and min_clearance is not None:
        floors, orig_dist, policy = compute_clearance_floors(
            work, sorted(allowed) or region, backend, min_clearance,
            clearance_policy=policy)

    def _analyze(pos, indices=None, detailed=False):
        return analyze_skin_anatomy_intersections(
            skin_mesh,
            skin_indices=indices if indices is not None else region,
            backend=backend, positions=pos, neighbors=neighbors,
            skin_topology=skin_topology,
            boundary_buffer_rings=boundary_buffer_rings,
            intersection_tolerance=intersection_tolerance,
            detailed=detailed, verbose=False)

    before = _analyze(work, detailed=True)
    last = before
    patch_info = {"core": [], "patch": [], "weights": {}, "outer": [], "rings": []}
    quality = _measure_patch_repair_quality({}, neighbors)
    binary_count = 0
    unresolved_core = []
    per_vertex = {}
    target_dists = []
    offending_meshes = set(before.get("intersecting_anatomy_meshes") or [])
    applied_field = {}
    passes = 0
    roughness_before = 0.0
    roughness_recorded = False
    weights_t = tuple(repair_ring_weights or DEFAULT_REPAIR_RING_WEIGHTS)

    for p_i in range(max(1, int(max_surface_repair_passes))):
        passes = p_i + 1
        last = _analyze(work, detailed=True)
        if last.get("intersection_pair_count", 0) <= 0:
            break
        core = intersection_vertices_from_report(
            last, skin_topology, allowed=allowed, boundary=boundary)
        if not core:
            break
        components = _connected_index_components(core, neighbors)
        raw_patches = []
        for comp in components:
            raw_patches.append(_build_intersection_repair_patch(
                comp, neighbors, allowed, boundary,
                repair_blend_rings, weights_t))
        merged = _merge_overlapping_repair_patches(raw_patches)
        patches = []
        for p in merged:
            if p.get("_rebuild"):
                p = _build_intersection_repair_patch(
                    p["core"], neighbors, allowed, boundary,
                    p.get("n_rings", repair_blend_rings), weights_t)
            bset = set(boundary)
            p["core"] = [i for i in p["core"] if i not in bset]
            p["patch"] = [i for i in p["patch"] if i not in bset]
            p["outer"] = [i for i in p.get("outer") or [] if i not in bset]
            p["weights"] = {i: w for i, w in (p.get("weights") or {}).items()
                            if i not in bset}
            p["distance"] = {i: d for i, d in (p.get("distance") or {}).items()
                             if i not in bset}
            if p["core"]:
                patches.append(p)
        if not patches:
            break
        # Combined field so overlapping clusters cannot double-move a vertex.
        combined = {}
        combined_core = []
        combined_patch = []
        combined_outer = []
        combined_weights = {}
        combined_dist = {}
        per_vertex = {}
        target_dists = []
        component_summaries = []
        all_core_disp = {}
        patch_fields = []
        offending = _collect_offending_anatomy(
            last, skin_topology, [i for p in patches for i in p["core"]])
        for p in patches:
            core_i = list(p["core"])
            core_disp = {}
            for i in core_i:
                nrm = _outward_skin_normal(i, normals, work, skin_topology)
                edge = _mean_local_edge(work, neighbors, i)
                standoff = max(1e-4, DEFAULT_SUPPORT_OFFSET_RATIO * (
                    edge if edge > 1e-12 else 1.0))
                if policy == "global" and min_clearance is not None:
                    offset = max(standoff, float(min_clearance))
                else:
                    offset = standoff
                max_disp = None
                if max_repair_displacement_ratio is not None and edge > 1e-12:
                    max_disp = float(max_repair_displacement_ratio) * edge
                target, info = _compute_anatomy_supported_target(
                    work[i], nrm, offending.get(i), backend, offset, max_disp)
                core_disp[i] = vec_sub(target, work[i])
                all_core_disp[i] = core_disp[i]
                info["weight"] = 1.0
                info["local_edge"] = edge
                info["offset"] = offset
                info["skin_position"] = list(work[i])
                info["skin_normal"] = list(nrm) if nrm is not None else None
                info["topo_distance"] = 0
                per_vertex[i] = info
                target_dists.append(info["target_distance"])
                offending_meshes.update(info.get("offending_meshes") or [])
            field = _blend_patch_displacements(
                core_disp, p, neighbors,
                method=repair_field_method,
                harmonic_iters=repair_field_iterations,
                falloff=repair_falloff,
                boundary_mode=repair_boundary_mode)
            patch_fields.append(field)
            combined.update(field)
            combined_core.extend(p["core"])
            combined_patch.extend(p["patch"])
            combined_outer.extend(p.get("outer") or [])
            combined_weights.update(p.get("weights") or {})
            combined_dist.update(p.get("distance") or {})
            cdisps = [vec_length(core_disp[i]) for i in core_i]
            pdisps = [vec_length(field[i]) for i in p["patch"]]
            meshes_c = sorted(set(
                m for i in core_i
                for m in ((per_vertex.get(i) or {}).get("offending_meshes") or [])))
            component_summaries.append({
                "core_count": len(core_i),
                "patch_count": len(p["patch"]),
                "ring_radius": p.get("n_rings", repair_blend_rings),
                "anatomy_meshes": meshes_c,
                "max_displacement": max(pdisps) if pdisps else 0.0,
                "core_vertices": list(core_i),
                "patch_vertices": list(p["patch"]),
                "outer_vertices": list(p.get("outer") or []),
            })
        patch_info = {
            "core": sorted(set(combined_core)),
            "patch": sorted(set(combined_patch)),
            "outer": sorted(set(combined_outer)),
            "weights": combined_weights,
            "distance": combined_dist,
            "rings": [],
            "n_rings": repair_blend_rings,
            "components": component_summaries,
        }
        core = list(patch_info["core"])
        if not roughness_recorded and patch_info.get("patch"):
            dummy = {i: [0.0, 0.0, 0.0] for i in patch_info["patch"]}
            qb = _measure_patch_repair_quality(dummy, neighbors, work)
            roughness_before = qb.get("mean_laplacian_magnitude", 0.0)
            roughness_recorded = True
        chosen = [list(v) for v in work]
        applied_field = {}
        alpha_used = 1.0
        for field in patch_fields:
            if max_repair_displacement_ratio is not None:
                for i, di in list(field.items()):
                    mag = vec_length(di)
                    edge_i = _mean_local_edge(work, neighbors, i)
                    cap = float(max_repair_displacement_ratio) * (
                        edge_i if edge_i > 1e-12 else 1.0)
                    if mag > cap > 0.0:
                        field[i] = vec_scale(di, cap / mag)
            if repair_binary_search:
                chosen, alpha_used, nsearch, _cleared = _find_minimal_safe_target(
                    chosen, field, lambda pos: _analyze(pos),
                    steps=repair_binary_search_steps,
                    position_tolerance=repair_position_tolerance)
                binary_count += nsearch
            else:
                for i, di in field.items():
                    chosen[i] = vec_add(chosen[i], di)
                alpha_used = 1.0
            for i, di in field.items():
                applied_field[i] = vec_scale(di, alpha_used)
                if i in per_vertex:
                    per_vertex[i]["applied_displacement"] = list(applied_field[i])
                    per_vertex[i]["binary_search_alpha"] = alpha_used
                    per_vertex[i]["final_position"] = list(chosen[i])
        work = chosen
        if min_clearance is not None and backend is not None:
            for i in patch_info["patch"]:
                if i in boundary:
                    continue
                edge_i = _mean_local_edge(work, neighbors, i)
                standoff_i = max(
                    1e-4,
                    DEFAULT_SUPPORT_OFFSET_RATIO * (
                        edge_i if edge_i > 1e-12 else 1.0))
                if policy == "global":
                    floor_i = floors.get(i, float(min_clearance))
                else:
                    floor_i = floors.get(i, standoff_i)
                nrm = _outward_skin_normal(i, normals, work, skin_topology)
                sol = enforce_anatomy_clearance(
                    work[i], backend, floor_i,
                    clearance_tolerance=clearance_tolerance, skin_normal=nrm)
                work[i] = sol["position"]
        last = _analyze(work, detailed=True)
        if last.get("intersection_pair_count", 0) <= 0:
            break
        unresolved_core = intersection_vertices_from_report(
            last, skin_topology, allowed=allowed, boundary=boundary)

    after_geom = last
    post_relax_disp = 0.0
    if (post_repair_relax and patch_info.get("patch")
            and after_geom.get("intersection_pair_count", 0) <= 0):
        before_relax = [list(v) for v in work]
        relaxed = [list(v) for v in work]
        ok = True
        interior = [i for i in patch_info["patch"]
                    if i not in set(patch_info.get("outer") or [])
                    and i not in set(boundary)]
        for _ in range(max(0, int(post_repair_relax_iterations))):
            trial = _local_taubin_step(
                relaxed, neighbors, interior, post_repair_relax_strength)
            tr = _analyze(trial, indices=patch_info["patch"])
            if tr.get("intersection_pair_count", 0) > 0:
                ok = False
                break
            relaxed = trial
        if ok:
            work = relaxed
            after_geom = _analyze(work, detailed=True)
            post_relax_disp = (
                sum(vec_length(vec_sub(work[i], before_relax[i]))
                    for i in patch_info["patch"])
                / float(len(patch_info["patch"]))) if patch_info["patch"] else 0.0

    after = after_geom
    quality = _measure_patch_repair_quality(applied_field, neighbors, work)
    moved = sorted(i for i, di in applied_field.items() if vec_length(di) > 1e-12)
    unresolved_faces = after.get("intersecting_skin_faces") or []
    result = {
        "positions": work,
        "repair_mode": "anatomy_supported_patch",
        "report_before": before,
        "report_after": after,
        "intersecting_faces_before": before.get("intersecting_skin_face_count", 0),
        "intersecting_faces_after": after.get("intersecting_skin_face_count", 0),
        "intersection_pairs_before": before.get("intersection_pair_count", 0),
        "intersection_pairs_after": after.get("intersection_pair_count", 0),
        "intersection_repair_iterations": passes,
        "repair_passes": passes,
        "intersection_vertices_moved": moved,
        "intersection_faces_resolved": max(
            0, before.get("intersecting_skin_face_count", 0)
            - after.get("intersecting_skin_face_count", 0)),
        "intersection_faces_unresolved": len(unresolved_faces),
        "unresolved_intersecting_faces": unresolved_faces,
        "unresolved_intersection_faces": unresolved_faces,
        "unresolved_core_vertices": unresolved_core,
        "core_vertex_count": len(patch_info.get("core") or []),
        "patch_vertex_count": len(patch_info.get("patch") or []),
        "core_vertices": list(patch_info.get("core") or []),
        "patch_vertices": list(patch_info.get("patch") or []),
        "patch_weights": dict(patch_info.get("weights") or {}),
        "offending_anatomy_meshes": sorted(offending_meshes),
        "mean_core_target_distance": (
            (sum(target_dists) / len(target_dists)) if target_dists else 0.0),
        "max_core_target_distance": max(target_dists) if target_dists else 0.0,
        "mean_applied_displacement": quality["mean_applied_displacement"],
        "max_applied_displacement": quality["max_applied_displacement"],
        "mean_intersection_repair_displacement": quality["mean_applied_displacement"],
        "max_intersection_repair_displacement": quality["max_applied_displacement"],
        "mean_displacement_gradient": quality["mean_displacement_gradient"],
        "max_displacement_gradient": quality["max_displacement_gradient"],
        "binary_search_count": binary_count,
        "post_relax_displacement": post_relax_disp,
        "clearance_policy": policy,
        "per_vertex": per_vertex,
        "runtime_seconds": time.time() - t0,
        "mean_laplacian_magnitude": quality.get("mean_laplacian_magnitude", 0.0),
        "max_laplacian_magnitude": quality.get("max_laplacian_magnitude", 0.0),
        "repair_profile": profile_name,
        "repair_falloff": repair_falloff,
        "repair_field_method": repair_field_method,
        "repair_field_iterations": repair_field_iterations,
        "repair_boundary_mode": repair_boundary_mode,
        "repair_blend_rings": repair_blend_rings,
        "repair_patch_count": len(patch_info.get("components") or []),
        "repair_component_count": len(patch_info.get("components") or []),
        "repair_components": list(patch_info.get("components") or []),
        "mean_patch_radius_rings": (
            (sum(c.get("ring_radius", 0) for c in (patch_info.get("components") or []))
             / float(len(patch_info.get("components") or [1])))
            if patch_info.get("components") else float(repair_blend_rings or 0)),
        "max_patch_radius_rings": max(
            [c.get("ring_radius", 0) for c in (patch_info.get("components") or [])]
            or [repair_blend_rings or 0]),
        "core_mean_displacement": (
            (sum(vec_length(applied_field[i]) for i in (patch_info.get("core") or [])
                 if i in applied_field)
             / float(len(patch_info.get("core") or [1])))
            if patch_info.get("core") else 0.0),
        "core_max_displacement": max(
            [vec_length(applied_field[i]) for i in (patch_info.get("core") or [])
             if i in applied_field] or [0.0]),
        "patch_mean_displacement": quality["mean_applied_displacement"],
        "patch_max_displacement": quality["max_applied_displacement"],
        "displacement_gradient_mean": quality["mean_displacement_gradient"],
        "displacement_gradient_max": quality["max_displacement_gradient"],
        "laplacian_roughness_before": roughness_before,
        "laplacian_roughness_after": quality.get("mean_laplacian_magnitude", 0.0),
        "patch_distance": dict(patch_info.get("distance") or {}),
        "outer_vertices": list(patch_info.get("outer") or []),
        "intersections_before": before.get("intersecting_skin_face_count", 0),
        "intersections_after": after.get("intersecting_skin_face_count", 0),
        "overlap_merge_policy": (
            "union-find merge of grown patches that share a vertex; one "
            "harmonic field per merged cluster. Disjoint clusters keep "
            "independent fields and independent binary-search alphas "
            "(never sequential additive lifts on the same vertex)"),
    }
    if verbose:
        print("[intersection-repair V2] profile={0} rings={1} field={2}/{3} "
              "bound={4} passes={5} faces {6}->{7} comps={8} "
              "core={9} patch={10} mean/max disp={11:.5f}/{12:.5f} "
              "max_grad={13:.5f} unresolved={14}".format(
                  profile_name, repair_blend_rings, repair_field_method,
                  repair_field_iterations, repair_boundary_mode, passes,
                  result["intersecting_faces_before"],
                  result["intersecting_faces_after"],
                  result["repair_component_count"],
                  result["core_vertex_count"], result["patch_vertex_count"],
                  result["mean_applied_displacement"],
                  result["max_applied_displacement"],
                  result["max_displacement_gradient"],
                  result["intersection_faces_unresolved"]))
    return result


def resolve_skin_anatomy_intersections(
        positions, skin_indices, backend,
        skin_mesh=None, neighbors=None, normals=None, skin_topology=None,
        min_clearance=0.8, clearance_tolerance=DEFAULT_CLEARANCE_TOLERANCE,
        max_intersection_repair_iterations=DEFAULT_MAX_INTERSECTION_REPAIR_ITERATIONS,
        intersection_repair_step_ratio=DEFAULT_INTERSECTION_REPAIR_STEP_RATIO,
        intersection_repair_growth_rings=0,
        boundary_buffer_rings=1,
        intersection_tolerance=DEFAULT_INTERSECTION_TOLERANCE,
        repair_classes=AUTO_REPAIR_INTERSECTION_CLASSES,
        binary_search_min_step=True,
        verbose=True,
        repair_mode=DEFAULT_REPAIR_MODE,
        repair_blend_rings=None,
        repair_ring_weights=DEFAULT_REPAIR_RING_WEIGHTS,
        repair_binary_search=True,
        repair_binary_search_steps=DEFAULT_REPAIR_BINARY_SEARCH_STEPS,
        max_surface_repair_passes=None,
        max_repair_displacement_ratio=DEFAULT_MAX_REPAIR_DISPLACEMENT_RATIO,
        post_repair_relax=True,
        post_repair_relax_iterations=3,
        post_repair_relax_strength=0.1,
        clearance_policy="preserve_valid_baseline",
        repair_position_tolerance=None,
        select_repair_patch=False,
        repair_profile=DEFAULT_REPAIR_PROFILE,
        repair_falloff=None,
        repair_field_iterations=None,
        repair_boundary_mode=None,
        repair_field_method=None):
    """Repair skin/anatomy *surface intersections* (not vertex penetration).

    Uses the EXISTING intersection analyzer as a black box. Default
    ``repair_mode="anatomy_supported_patch"`` (V2). Pass
    ``repair_mode="legacy_normal_push"`` to reproduce the V1 per-vertex
    normal-escape behaviour (spike-prone, kept for A/B).

    ``repair_profile="broad_fair"`` (default) uses a wide topology patch and
    a harmonic displacement field. ``repair_profile="legacy_v2_local"``
    reproduces the original 2-ring fixed-boundary V2. Explicit knobs override
    the profile. Detection / identification is unchanged.
    """
    mode = repair_mode or DEFAULT_REPAIR_MODE
    if mode in ("legacy_normal_push", "legacy", "v1"):
        result = _legacy_normal_push_intersection_repair(
            positions, skin_indices, backend, skin_mesh=skin_mesh,
            neighbors=neighbors, normals=normals, skin_topology=skin_topology,
            min_clearance=min_clearance, clearance_tolerance=clearance_tolerance,
            max_intersection_repair_iterations=max_intersection_repair_iterations,
            intersection_repair_step_ratio=intersection_repair_step_ratio,
            intersection_repair_growth_rings=intersection_repair_growth_rings,
            boundary_buffer_rings=boundary_buffer_rings,
            intersection_tolerance=intersection_tolerance,
            repair_classes=repair_classes,
            binary_search_min_step=binary_search_min_step, verbose=verbose)
        if select_repair_patch and skin_mesh:
            select_surface_repair_patch(result, skin_mesh, which="core")
        return result
    if mode not in ("anatomy_supported_patch", "v2", "patch"):
        raise ValueError("repair_mode must be 'anatomy_supported_patch' or "
                         "'legacy_normal_push', got {0!r}".format(mode))
    passes = (max_surface_repair_passes if max_surface_repair_passes is not None
              else DEFAULT_MAX_SURFACE_REPAIR_PASSES)
    result = _anatomy_supported_patch_repair(
        positions, skin_indices, backend, skin_mesh=skin_mesh,
        neighbors=neighbors, normals=normals, skin_topology=skin_topology,
        min_clearance=min_clearance, clearance_tolerance=clearance_tolerance,
        clearance_policy=clearance_policy,
        boundary_buffer_rings=boundary_buffer_rings,
        intersection_tolerance=intersection_tolerance,
        repair_blend_rings=repair_blend_rings,
        repair_ring_weights=repair_ring_weights,
        repair_binary_search=repair_binary_search,
        repair_binary_search_steps=repair_binary_search_steps,
        max_surface_repair_passes=passes,
        max_repair_displacement_ratio=max_repair_displacement_ratio,
        post_repair_relax=post_repair_relax,
        post_repair_relax_iterations=post_repair_relax_iterations,
        post_repair_relax_strength=post_repair_relax_strength,
        repair_position_tolerance=repair_position_tolerance,
        verbose=verbose,
        repair_profile=repair_profile,
        repair_falloff=repair_falloff,
        repair_field_iterations=repair_field_iterations,
        repair_boundary_mode=repair_boundary_mode,
        repair_field_method=repair_field_method)
    if select_repair_patch and skin_mesh:
        select_surface_repair_patch(result, skin_mesh, which="patch")
    return result


repair_skin_anatomy_intersections = resolve_skin_anatomy_intersections


def select_surface_repair_patch(report, mesh_name, which="patch", replace=True,
                                component=None):
    """Select V2 repair core, full patch, outer boundary, or component N.

    ``which``:
        ``core`` / ``patch`` / ``full`` / ``outer`` / ``component`` / ``componentN``
    ``component``:
        0-based index into ``report['repair_components']`` (optional).
    """
    key = str(which or "patch").lower().strip().replace("-", "_")
    comps = list((report or {}).get("repair_components") or [])
    idx = []
    if key in ("core",):
        idx = report.get("core_vertices") or []
    elif key in ("outer", "boundary", "outer_boundary"):
        idx = report.get("outer_vertices") or []
    elif key in ("both", "full", "full_patch", "patch"):
        idx = report.get("patch_vertices") or report.get("intersection_vertices_moved") or []
        if key == "both":
            idx = sorted(set(report.get("core_vertices") or [])
                         | set(idx))
    elif key.startswith("component"):
        rest = key.replace("component", "").replace("_", "").strip()
        try:
            cidx = int(rest) if rest else int(component if component is not None else 0)
        except (TypeError, ValueError):
            cidx = 0
        if 0 <= cidx < len(comps):
            idx = comps[cidx].get("patch_vertices") or []
            which = "component{0}".format(cidx)
    elif component is not None:
        cidx = int(component)
        if 0 <= cidx < len(comps):
            idx = comps[cidx].get("patch_vertices") or []
            which = "component{0}".format(cidx)
    else:
        idx = report.get("patch_vertices") or report.get("intersection_vertices_moved") or []
    if not idx:
        print("[intersection-repair] no patch vertices to select ({0})".format(which))
        return []
    select_vertices(mesh_name, idx, replace=replace)
    print("[intersection-repair] selected {0} SKIN verts ({1})".format(len(idx), which))
    return list(idx)


def print_repair_patch_summary(report):
    """Print per-component V2 patch size / radius / anatomy / displacement."""
    report = report or {}
    comps = list(report.get("repair_components") or [])
    print("[repair-patch] profile={0} rings={1} field={2}/{3} bound={4} "
          "comps={5} core={6} patch={7}".format(
              report.get("repair_profile"),
              report.get("repair_blend_rings"),
              report.get("repair_field_method"),
              report.get("repair_field_iterations"),
              report.get("repair_boundary_mode"),
              report.get("repair_component_count", len(comps)),
              report.get("core_vertex_count", 0),
              report.get("patch_vertex_count", 0)))
    print("  faces {0}->{1}  mean/max disp={2:.5f}/{3:.5f}  "
          "max_grad={4:.5f}  roughness {5:.5f}->{6:.5f}".format(
              float(report.get("intersections_before")
                    or report.get("intersecting_faces_before") or 0),
              float(report.get("intersections_after")
                    or report.get("intersecting_faces_after") or 0),
              float(report.get("patch_mean_displacement")
                    or report.get("mean_applied_displacement") or 0.0),
              float(report.get("patch_max_displacement")
                    or report.get("max_applied_displacement") or 0.0),
              float(report.get("displacement_gradient_max")
                    or report.get("max_displacement_gradient") or 0.0),
              float(report.get("laplacian_roughness_before") or 0.0),
              float(report.get("laplacian_roughness_after") or 0.0)))
    if not comps:
        print("  (no repair components)")
        return comps
    for i, c in enumerate(comps):
        print("  component {0}: core={1} patch={2} radius={3} "
              "anatomy={4} max_disp={5:.5f}".format(
                  i, c.get("core_count", 0), c.get("patch_count", 0),
                  c.get("ring_radius"), c.get("anatomy_meshes"),
                  float(c.get("max_displacement") or 0.0)))
    return comps


def debug_surface_repair_vertex(vertex_index, repair_report, verbose=True):
    """Print V2 per-vertex support / target / displacement from a repair report."""
    info = (repair_report or {}).get("per_vertex") or {}
    rec = info.get(vertex_index)
    weights = (repair_report or {}).get("patch_weights") or {}
    after = (repair_report or {}).get("report_after") or {}
    remaining_verts = set(after.get("intersecting_skin_vertices") or [])
    remaining_faces = after.get("intersecting_skin_faces") or []
    if verbose:
        print("[debug repair vtx {0}]".format(vertex_index))
        print("  in core?  {0}  patch weight={1}".format(
            vertex_index in ((repair_report or {}).get("core_vertices") or []),
            weights.get(vertex_index)))
        print("  remaining intersecting vertex? {0}".format(
            vertex_index in remaining_verts))
        print("  remaining intersecting faces  = {0}".format(remaining_faces))
        if not rec:
            print("  (no per-vertex V2 record; vertex was not a core this pass)")
            return rec
        print("  skin position    = {0}".format(rec.get("skin_position")))
        print("  skin normal      = {0}".format(rec.get("skin_normal")))
        print("  offending meshes = {0}".format(rec.get("offending_meshes")))
        print("  candidate supports:")
        for c in rec.get("candidate_supports") or []:
            print("    mesh={0} source={1} method={2} n={3}".format(
                c.get("mesh"), c.get("source"), c.get("method"), c.get("normal")))
        print("  support method   = {0}".format(rec.get("support_method")))
        print("  chosen support n = {0}".format(rec.get("support_normal")))
        print("  closest          = {0}".format(rec.get("closest")))
        print("  target           = {0}".format(rec.get("target")))
        print("  target distance  = {0:.5f}".format(rec.get("target_distance") or 0.0))
        print("  binary-search a  = {0}  final={1}".format(
            rec.get("binary_search_alpha"), rec.get("final_position")))
        print("  local edge       = {0:.5f}  offset={1:.5f}".format(
            rec.get("local_edge") or 0.0, rec.get("offset") or 0.0))
        print("  applied disp     = {0}".format(rec.get("applied_displacement")))
    return rec


def prepare_skin_region_for_smoothing(
        positions, skin_indices, backend,
        skin_mesh=None, neighbors=None, normals=None, skin_topology=None,
        min_clearance=0.8, clearance_tolerance=DEFAULT_CLEARANCE_TOLERANCE,
        max_escape_iterations=DEFAULT_MAX_ESCAPE_ITERATIONS,
        max_intersection_repair_iterations=DEFAULT_MAX_INTERSECTION_REPAIR_ITERATIONS,
        intersection_repair_step_ratio=DEFAULT_INTERSECTION_REPAIR_STEP_RATIO,
        intersection_repair_growth_rings=0,
        penetration_repair_feather_rings=0,
        boundary_buffer_rings=1,
        verbose=True,
        repair_mode=DEFAULT_REPAIR_MODE,
        repair_blend_rings=None,
        repair_ring_weights=DEFAULT_REPAIR_RING_WEIGHTS,
        repair_binary_search=True,
        max_surface_repair_passes=None,
        max_repair_displacement_ratio=DEFAULT_MAX_REPAIR_DISPLACEMENT_RATIO,
        post_repair_relax=True,
        clearance_policy="preserve_valid_baseline",
        repair_profile=DEFAULT_REPAIR_PROFILE,
        repair_falloff=None,
        repair_field_iterations=None,
        repair_boundary_mode=None,
        repair_field_method=None):
    """Combined pre-repair: vertex penetration then remaining surface crossings.

    Recommended order (this function)::

        analyze intersections + vertex penetration
        repair high-confidence vertex penetration
        repair remaining surface intersections
        re-run BOTH analyzers
    """
    work = [list(v) for v in positions]
    isect0 = analyze_skin_anatomy_intersections(
        skin_mesh, skin_indices=skin_indices, backend=backend, positions=work,
        neighbors=neighbors, skin_topology=skin_topology,
        boundary_buffer_rings=boundary_buffer_rings, verbose=verbose)
    pen0 = analyze_skin_anatomy_penetration(
        skin_mesh, indices=skin_indices, backend=backend, positions=work,
        normals=normals, min_clearance=min_clearance,
        clearance_tolerance=clearance_tolerance, detailed=True, verbose=verbose)
    pen_repair = resolve_skin_anatomy_penetrations(
        work, skin_indices, backend, min_clearance=min_clearance,
        normals=normals, neighbors=neighbors,
        max_escape_iterations=max_escape_iterations,
        clearance_tolerance=clearance_tolerance,
        penetration_repair_feather_rings=penetration_repair_feather_rings,
        classifications=pen0.get("details_by_vertex"), verbose=verbose)
    work = pen_repair["positions"]
    isect_repair = resolve_skin_anatomy_intersections(
        work, skin_indices, backend, skin_mesh=skin_mesh, neighbors=neighbors,
        normals=normals, skin_topology=skin_topology, min_clearance=min_clearance,
        clearance_tolerance=clearance_tolerance,
        max_intersection_repair_iterations=max_intersection_repair_iterations,
        intersection_repair_step_ratio=intersection_repair_step_ratio,
        intersection_repair_growth_rings=intersection_repair_growth_rings,
        boundary_buffer_rings=boundary_buffer_rings, verbose=verbose,
        repair_mode=repair_mode, repair_blend_rings=repair_blend_rings,
        repair_ring_weights=repair_ring_weights,
        repair_binary_search=repair_binary_search,
        max_surface_repair_passes=max_surface_repair_passes,
        max_repair_displacement_ratio=max_repair_displacement_ratio,
        post_repair_relax=post_repair_relax,
        clearance_policy=clearance_policy,
        repair_profile=repair_profile,
        repair_falloff=repair_falloff,
        repair_field_iterations=repair_field_iterations,
        repair_boundary_mode=repair_boundary_mode,
        repair_field_method=repair_field_method)
    work = isect_repair["positions"]
    isect1 = isect_repair.get("report_after") or analyze_skin_anatomy_intersections(
        skin_mesh, skin_indices=skin_indices, backend=backend, positions=work,
        neighbors=neighbors, skin_topology=skin_topology,
        boundary_buffer_rings=boundary_buffer_rings, verbose=False)
    pen1 = analyze_skin_anatomy_penetration(
        skin_mesh, indices=skin_indices, backend=backend, positions=work,
        normals=normals, min_clearance=min_clearance,
        clearance_tolerance=clearance_tolerance, detailed=False, verbose=verbose)
    return {
        "positions": work,
        "penetration_before": pen0,
        "penetration_after": pen1,
        "intersection_before": isect0,
        "intersection_after": isect1,
        "penetration_repair": pen_repair,
        "intersection_repair": isect_repair,
    }


repair_m5_region_anatomy_conflicts = prepare_skin_region_for_smoothing


def debug_skin_anatomy_intersection(skin_face_id, positions, backend,
                                    skin_mesh=None, neighbors=None, normals=None,
                                    skin_topology=None,
                                    intersection_record=None,
                                    min_clearance=0.8,
                                    intersection_repair_step_ratio=DEFAULT_INTERSECTION_REPAIR_STEP_RATIO,
                                    intersection_tolerance=DEFAULT_INTERSECTION_TOLERANCE,
                                    verbose=True):
    """Diagnose one intersecting skin face and a trial outward step."""
    if skin_topology is None and skin_mesh:
        fn = get_mesh_fn(skin_mesh)
        skin_topology = get_triangle_topology(fn) if fn is not None else {}
    face_vertex_ids = (skin_topology or {}).get("face_vertex_ids") or []
    triangles = (skin_topology or {}).get("triangles") or []
    verts = list(face_vertex_ids[skin_face_id]) if skin_face_id < len(face_vertex_ids) else []
    if not verts:
        verts = []
        for face_id, i0, i1, i2 in triangles:
            if face_id == skin_face_id:
                verts.extend([i0, i1, i2])
        verts = sorted(set(verts))
    rec = intersection_record
    if rec is None:
        rep = analyze_skin_anatomy_intersections(
            skin_mesh, skin_indices=verts, backend=backend, positions=positions,
            neighbors=neighbors, skin_topology=skin_topology,
            detailed=True, verbose=False,
            intersection_tolerance=intersection_tolerance)
        details = [d for d in (rep.get("details") or [])
                   if d.get("skin_face_id") == skin_face_id]
        rec = details[0] if details else None
        report_local = rep
    else:
        report_local = None
    nrm_face = [0.0, 0.0, 0.0]
    tris = [t for t in triangles if t[0] == skin_face_id]
    for _fid, i0, i1, i2 in tris:
        if max(i0, i1, i2) < len(positions):
            cr = vec_cross(vec_sub(positions[i1], positions[i0]),
                           vec_sub(positions[i2], positions[i0]))
            nrm_face = vec_add(nrm_face, cr)
    nrm_face = vec_normalize(nrm_face)
    v_normals = []
    steps = []
    for i in verts:
        vn = _outward_skin_normal(i, normals, positions, skin_topology)
        v_normals.append(vn)
        edge = _mean_local_edge(positions, neighbors, i)
        steps.append(float(intersection_repair_step_ratio) * (edge or 1.0))
    trial = [list(p) for p in positions]
    for i, vn, st in zip(verts, v_normals, steps):
        trial[i] = vec_add(trial[i], vec_scale(vn, st))
    after = analyze_skin_anatomy_intersections(
        skin_mesh, skin_indices=verts, backend=backend, positions=trial,
        neighbors=neighbors, skin_topology=skin_topology,
        detailed=False, verbose=False,
        intersection_tolerance=intersection_tolerance)
    still = skin_face_id in (after.get("intersecting_skin_faces") or [])
    info = {
        "skin_face_id": skin_face_id,
        "skin_face_vertices": verts,
        "anatomy_mesh": rec.get("anatomy_mesh") if rec else None,
        "anatomy_face_id": rec.get("anatomy_face_id") if rec else None,
        "classification": rec.get("classification") if rec else None,
        "intersection_point": rec.get("point") if rec else None,
        "skin_face_normal": nrm_face,
        "skin_vertex_normals": v_normals,
        "local_edge_steps": steps,
        "trial_still_intersecting": still,
        "pairs_after_trial": after.get("intersection_pair_count", 0),
        "record": rec,
        "local_report": report_local,
    }
    if verbose:
        print("[debug isect face {0}] verts={1}".format(skin_face_id, verts))
        print("  anatomy mesh/face = {0} / {1}".format(
            info["anatomy_mesh"], info["anatomy_face_id"]))
        print("  classification    = {0}".format(info["classification"]))
        print("  point             = {0}".format(info["intersection_point"]))
        print("  skin face normal  = {0}".format(
            [round(x, 5) for x in nrm_face] if nrm_face else None))
        print("  trial step ratio  = {0}  still_intersecting={1}".format(
            intersection_repair_step_ratio, still))
    return info


def create_intersection_debug_markers(records, prefix=DEBUG_ISECT_PREFIX):
    """Optional locators at intersection points. Off by default in analyzers."""
    if not records:
        return []
    try:
        import maya.cmds as cmds
    except ImportError:
        return []
    created = []
    for k, rec in enumerate(records):
        pt = rec.get("point") if isinstance(rec, dict) else None
        if not pt:
            continue
        name = "{0}{1}".format(prefix, k)
        if cmds.objExists(name):
            cmds.delete(name)
        loc = cmds.spaceLocator(name=name)[0]
        cmds.xform(loc, ws=True, t=(pt[0], pt[1], pt[2]))
        created.append(loc)
    print("[intersection] created {0} debug locators ({1}*)".format(
        len(created), prefix))
    return created


def cleanup_intersection_debug_markers(prefix=DEBUG_ISECT_PREFIX):
    try:
        import maya.cmds as cmds
    except ImportError:
        return
    hits = cmds.ls(prefix + "*", type="transform") or []
    if hits:
        cmds.delete(hits)
        print("[intersection] deleted {0} debug locators".format(len(hits)))
