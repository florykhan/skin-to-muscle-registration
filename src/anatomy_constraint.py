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

from mesh_utils import (
    vec_add,
    vec_dot,
    vec_length,
    vec_normalize,
    vec_scale,
    vec_sub,
    closest_point_and_normal,
    get_boundary_vertices,
    get_mesh_vertices,
    get_vertex_normals,
    get_vertex_neighbors,
    grow_indices,
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
