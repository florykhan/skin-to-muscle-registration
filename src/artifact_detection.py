"""
artifact_detection.py
=====================
Automatic detection of locally irregular / artifact-prone skin vertices.

This is the first *algorithmic* contribution toward reducing manual cleanup after
registration. Instead of the researcher hand-selecting every problem patch, this
module flags candidate vertices using a local irregularity score, so cleanup can
be targeted automatically.

Irregularity score (umbrella / uniform Laplacian magnitude)
-----------------------------------------------------------
For every vertex ``i`` with 1-ring topological neighbourhood ``N(i)``::

    score_i = || x_i - mean(x_j for j in N(i)) ||

where ``x_i`` is the vertex position and the mean is the centroid of its
neighbours. A low score means the vertex sits on the local surface; a high score
suggests a spike, dent, fold, or local registration artifact.

Scope / safety
--------------
This version is **detection only**: it never moves vertices, never renames
objects, and never touches the d98 registration algorithm. It reuses the mesh
access in :mod:`mesh_utils` and the region growing in :mod:`region_selection`
rather than duplicating that code.

The module is importable outside Maya (Maya access is isolated in
:mod:`mesh_utils`, which degrades gracefully), so scoring/summary/detection logic
can be linted and reasoned about without a running Maya session.
"""

from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import math

import mesh_utils
import region_selection


# =============================================================================
# 1. SCORING
# =============================================================================

def compute_laplacian_scores(mesh_name: str,
                             indices: Optional[List[int]] = None,
                             ) -> Dict[int, float]:
    """Compute the umbrella-Laplacian irregularity score for each vertex.

    Parameters
    ----------
    mesh_name:
        Mesh to analyze (e.g. the registered skin mesh). Not modified.
    indices:
        If given, scores are computed ONLY for these vertex indices (their
        neighbours are still read from the full mesh, so scores are exact). If
        omitted, every vertex is scored.

    Returns
    -------
    dict
        ``{vertex_index: score}``. Empty if the mesh is missing/empty. Vertices
        with no neighbours get a score of ``0.0``.

    Notes
    -----
    The mesh is queried exactly once (positions + adjacency) regardless of how
    many vertices are scored, to avoid expensive repeated Maya calls.
    """
    vertices = mesh_utils.get_mesh_vertices(mesh_name)
    if not vertices:
        print("[artifact_detection] mesh '{0}' not found or has no vertices".format(mesh_name))
        return {}

    neighbors = mesh_utils.get_vertex_neighbors(mesh_name)
    if not neighbors:
        print("[artifact_detection] could not read topology for '{0}'".format(mesh_name))
        return {}

    n = len(vertices)
    if indices is None:
        target_indices: List[int] = list(range(n))
    else:
        target_indices = [i for i in indices if 0 <= i < n]

    scores: Dict[int, float] = {}
    for i in target_indices:
        nbrs = neighbors[i]
        if not nbrs:
            scores[i] = 0.0
            continue
        centroid = mesh_utils.vec_mean([vertices[j] for j in nbrs])
        scores[i] = mesh_utils.vec_length(mesh_utils.vec_sub(vertices[i], centroid))
    return scores


# =============================================================================
# 2. SUMMARY STATISTICS
# =============================================================================

def _median(values: List[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    m = len(s)
    mid = m // 2
    if m % 2 == 1:
        return s[mid]
    return 0.5 * (s[mid - 1] + s[mid])


def _std(values: List[float], mean: float) -> float:
    """Population standard deviation (0.0 for <2 samples)."""
    m = len(values)
    if m < 2:
        return 0.0
    var = sum((v - mean) ** 2 for v in values) / m
    return math.sqrt(var)


def summarize_scores(scores: Dict[int, float]) -> Dict[str, float]:
    """Return summary statistics for a score dict.

    Keys: ``count``, ``mean``, ``median``, ``std``, ``max``, ``max_index``.
    All numeric; ``max_index`` is ``-1`` when there are no scores.
    """
    if not scores:
        return {"count": 0, "mean": 0.0, "median": 0.0, "std": 0.0,
                "max": 0.0, "max_index": -1}

    items = list(scores.items())
    values = [v for _, v in items]
    count = len(values)
    mean = sum(values) / count
    max_index, max_val = max(items, key=lambda kv: kv[1])
    return {
        "count": count,
        "mean": mean,
        "median": _median(values),
        "std": _std(values, mean),
        "max": max_val,
        "max_index": max_index,
    }


# =============================================================================
# 3. OUTLIER DETECTION
# =============================================================================

def _percentile(sorted_vals: List[float], pct: float) -> float:
    """Linear-interpolation percentile of an already-sorted ascending list."""
    if not sorted_vals:
        return 0.0
    if pct <= 0:
        return sorted_vals[0]
    if pct >= 100:
        return sorted_vals[-1]
    k = (len(sorted_vals) - 1) * (pct / 100.0)
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return sorted_vals[int(k)]
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def detect_outliers(scores: Dict[int, float],
                    method: str = "zscore",
                    threshold: float = 2.5,
                    percentile: float = 95.0,
                    min_score: float = 0.0,
                    ) -> List[int]:
    """Return sorted vertex indices whose score is considered suspicious.

    Parameters
    ----------
    scores:
        ``{vertex_index: score}`` from :func:`compute_laplacian_scores`.
    method:
        ``"zscore"``  -> flag vertices with ``(score - mean) / std >= threshold``.
        ``"percentile"`` -> flag vertices with ``score >= percentile(scores, percentile)``.
    threshold:
        Z-score cutoff (used when ``method == "zscore"``).
    percentile:
        Percentile cutoff in ``[0, 100]`` (used when ``method == "percentile"``).
    min_score:
        Absolute floor: vertices below this raw score are never flagged. Useful
        to suppress trivially small irregularities regardless of the statistics.

    Returns
    -------
    list[int]
        Sorted suspicious vertex indices (possibly empty).

    Safety
    ------
    Zero standard deviation (all scores equal) yields no z-score outliers rather
    than a divide-by-zero. Unknown methods raise ``ValueError``.
    """
    if not scores:
        return []

    if method == "zscore":
        values = list(scores.values())
        mean = sum(values) / len(values)
        std = _std(values, mean)
        if std <= 1e-12:
            return []  # no variation -> nothing stands out
        flagged = [idx for idx, s in scores.items()
                   if s >= min_score and (s - mean) / std >= threshold]

    elif method == "percentile":
        sorted_vals = sorted(scores.values())
        cutoff = _percentile(sorted_vals, percentile)
        cutoff = max(cutoff, min_score)
        flagged = [idx for idx, s in scores.items() if s >= cutoff]

    else:
        raise ValueError(
            "unknown method '{0}'; expected 'zscore' or 'percentile'".format(method))

    return sorted(flagged)


# =============================================================================
# 4 & 5. MAYA SELECTION / REGION GROWTH (reuse existing helpers)
# =============================================================================

def select_vertices(mesh_name: str, indices: List[int]) -> None:
    """Select detected vertices in Maya for visual inspection.

    Thin pass-through to :func:`mesh_utils.select_vertices` so component
    selection logic lives in one place.
    """
    if not indices:
        print("[artifact_detection] nothing to select (0 vertices)")
        return
    mesh_utils.select_vertices(mesh_name, indices, replace=True)


def grow_detected_region(mesh_name: str,
                         indices: List[int],
                         rings: int = 1,
                         ) -> List[int]:
    """Grow the detected set outward by ``rings`` topological rings.

    Reuses :func:`region_selection.grow_region`. Returns the (sorted) expanded
    index list; returns the input unchanged when ``rings <= 0`` or empty.
    """
    if not indices or rings <= 0:
        return sorted(indices)
    return region_selection.grow_region(mesh_name, indices, rings=rings)


# =============================================================================
# 6. ORCHESTRATION (detection only -- never modifies the mesh)
# =============================================================================

def detect_irregular_region(mesh_name: str,
                            method: str = "percentile",
                            threshold: float = 2.5,
                            percentile: float = 97.5,
                            rings: int = 1,
                            select: bool = True,
                            min_score: float = 0.0,
                            ) -> Tuple[List[int], Dict[str, float]]:
    """Detect an irregular region end-to-end and (optionally) select it in Maya.

    Steps: score all vertices -> summarize -> flag outliers -> grow by ``rings``
    -> optionally select in the viewport. **No vertex positions are changed.**

    Parameters
    ----------
    mesh_name:
        Mesh to analyze (not modified, not renamed).
    method, threshold, percentile, min_score:
        Passed to :func:`detect_outliers`.
    rings:
        Grow the detected set by this many neighbour rings (softer patches).
    select:
        If True, select the final indices in Maya for inspection.

    Returns
    -------
    (indices, stats):
        ``indices`` are the final (possibly grown) suspicious vertex indices;
        ``stats`` is the score summary from :func:`summarize_scores`.
    """
    if not mesh_utils.mesh_exists(mesh_name):
        print("[artifact_detection] object '{0}' does not exist".format(mesh_name))
        return [], summarize_scores({})

    scores = compute_laplacian_scores(mesh_name)
    stats = summarize_scores(scores)
    if stats["count"] == 0:
        print("[artifact_detection] no scores computed for '{0}'".format(mesh_name))
        return [], stats

    detected = detect_outliers(scores, method=method, threshold=threshold,
                               percentile=percentile, min_score=min_score)
    grown = grow_detected_region(mesh_name, detected, rings=rings)

    if method == "zscore":
        crit = "z-score >= {0}".format(threshold)
    else:
        crit = "top {0:.1f}%% (percentile >= {1})".format(100.0 - percentile, percentile)

    print("[artifact_detection] '{0}': scored {1} verts | mean={2:.4f} "
          "median={3:.4f} std={4:.4f} max={5:.4f} (vtx {6})".format(
              mesh_name, stats["count"], stats["mean"], stats["median"],
              stats["std"], stats["max"], stats["max_index"]))
    print("[artifact_detection] flagged {0} by {1}; grown to {2} verts "
          "(+{3} ring(s))".format(len(detected), crit, len(grown), rings))

    if select and grown:
        select_vertices(mesh_name, grown)
        print("[artifact_detection] selected {0} vertices for inspection "
              "(mesh unchanged)".format(len(grown)))

    return grown, stats


# =============================================================================
# M2. REFERENCE-BASED LAPLACIAN COMPARISON
# =============================================================================
# WHY M2 REDUCES FALSE POSITIVES (vs. the M1 magnitude score)
# -----------------------------------------------------------
# M1 flags a vertex when its own umbrella Laplacian is large:
#       score_i = ||x_i - mean(x_j for j in N(i))||
# That quantity is essentially discrete local curvature, so it is intrinsically
# large wherever the FACE is genuinely curved -- nostrils, lip border, eyelid
# creases, brow, jawline -- even when those areas are registered perfectly. M1
# therefore cannot tell "sharp because it's an artifact" from "sharp because the
# anatomy is sharp," and drowns real artifacts in valid high-curvature features.
#
# M2 instead compares the CURRENT skin's local shape against a REFERENCE (the
# target mesh) that shares topology and vertex correspondence:
#       L_current(i) = x_i - mean(x_j for j in N(i))     # local shape, current
#       L_target(i)  = t_i - mean(t_j for j in N(i))     # local shape, target
#       score_i      = ||L_current(i) - L_target(i)||    # DISAGREEMENT in shape
# Comparing the full Laplacian VECTORS (not just magnitudes) means a valid
# feature that is present in BOTH meshes largely cancels: L_current ~= L_target,
# so its score is small. What survives is where the current mesh's local shape
# DEVIATES from the reference -- i.e. spikes/dents/folds introduced by
# registration -- which is exactly what we want to flag. High but faithful
# curvature is suppressed; genuine local disagreement is preserved.
#
# The optional normalization divides by the current mesh's average local edge
# length so the score becomes scale-relative (a small absolute wobble in a dense
# region counts comparably to the same relative wobble in a coarse region).

def validate_corresponding_topology(current_mesh: str,
                                    target_mesh: str,
                                    check_adjacency: bool = True,
                                    ) -> Dict[str, object]:
    """Verify two meshes share topology / vertex correspondence for M2.

    Checks (in order): both objects exist, both non-empty, equal vertex counts,
    and -- when ``check_adjacency`` is True -- identical 1-ring adjacency for
    every vertex (so index ``i`` refers to the same point on both meshes).

    Parameters
    ----------
    current_mesh, target_mesh:
        Meshes to compare. Neither is modified.
    check_adjacency:
        If True (default) compare per-vertex neighbour sets. This is O(V*d) and
        the strongest available correspondence guarantee short of UVs/IDs.

    Returns
    -------
    dict
        ``{"ok": True, "vertex_count": int, "adjacency_checked": bool,
           "adjacency_match": bool}`` on success.

    Raises
    ------
    ValueError
        With a descriptive message if any check fails.
    """
    for m in (current_mesh, target_mesh):
        if not mesh_utils.mesh_exists(m):
            raise ValueError("mesh '{0}' does not exist".format(m))

    nc = mesh_utils.get_vertex_count(current_mesh)
    nt = mesh_utils.get_vertex_count(target_mesh)
    if nc == 0 or nt == 0:
        raise ValueError("empty mesh: '{0}' has {1} verts, '{2}' has {3} verts".format(
            current_mesh, nc, target_mesh, nt))
    if nc != nt:
        raise ValueError(
            "vertex count mismatch: '{0}'={1} vs '{2}'={3}; meshes are not "
            "corresponding".format(current_mesh, nc, target_mesh, nt))

    adjacency_match = True
    if check_adjacency:
        na = mesh_utils.get_vertex_neighbors(current_mesh)
        nb = mesh_utils.get_vertex_neighbors(target_mesh)
        if len(na) != len(nb):
            raise ValueError("adjacency length mismatch ({0} vs {1})".format(
                len(na), len(nb)))
        for i in range(len(na)):
            if set(na[i]) != set(nb[i]):
                adjacency_match = False
                raise ValueError(
                    "adjacency mismatch at vertex {0}: meshes do not share "
                    "topology / correspondence".format(i))

    return {"ok": True, "vertex_count": nc,
            "adjacency_checked": bool(check_adjacency),
            "adjacency_match": adjacency_match}


def compute_reference_laplacian_scores(current_mesh: str,
                                       target_mesh: str,
                                       indices: Optional[List[int]] = None,
                                       normalize: bool = False,
                                       epsilon: float = 1e-8,
                                       ) -> Dict[int, float]:
    """Score vertices by disagreement between current and target local shape.

    For each vertex ``i`` (using the shared 1-ring ``N(i)``)::

        L_current(i) = x_i - mean(x_j for j in N(i))
        L_target(i)  = t_i - mean(t_j for j in N(i))
        score_i      = || L_current(i) - L_target(i) ||

    If ``normalize`` is True the score is divided by the current mesh's local
    scale ``mean(||x_i - x_j|| for j in N(i)) + epsilon`` to make it
    scale-relative. See the module section header for why this reduces the
    false positives seen with the M1 magnitude score.

    Parameters
    ----------
    current_mesh, target_mesh:
        Corresponding meshes (identical topology / vertex order). Not modified.
    indices:
        If given, score only these vertex indices (neighbours are still read
        from the full meshes, so scores are exact).
    normalize:
        Divide by current-mesh average local edge length when True.
    epsilon:
        Small constant guarding the normalization denominator (default 1e-8).

    Returns
    -------
    dict
        ``{vertex_index: score}``. Empty if either mesh is missing/empty or the
        vertex counts differ. No-neighbour (boundary) vertices score ``0.0``.

    Notes
    -----
    Each mesh's positions are read once and the (shared) topology is read once;
    the target's Laplacian is evaluated over the same neighbour indices, which is
    valid precisely because the meshes correspond (validate this up front with
    :func:`validate_corresponding_topology`).
    """
    current = mesh_utils.get_mesh_vertices(current_mesh)
    target = mesh_utils.get_mesh_vertices(target_mesh)
    if not current or not target:
        print("[artifact_detection][M2] missing/empty mesh "
              "('{0}': {1}, '{2}': {3})".format(
                  current_mesh, len(current), target_mesh, len(target)))
        return {}
    if len(current) != len(target):
        print("[artifact_detection][M2] vertex count mismatch "
              "({0} vs {1}); meshes are not corresponding".format(
                  len(current), len(target)))
        return {}

    neighbors = mesh_utils.get_vertex_neighbors(current_mesh)
    if not neighbors:
        print("[artifact_detection][M2] could not read topology for "
              "'{0}'".format(current_mesh))
        return {}

    n = len(current)
    if indices is None:
        target_indices: List[int] = list(range(n))
    else:
        target_indices = [i for i in indices if 0 <= i < n]

    scores: Dict[int, float] = {}
    for i in target_indices:
        nbrs = neighbors[i]
        if not nbrs:
            scores[i] = 0.0
            continue
        cen_cur = mesh_utils.vec_mean([current[j] for j in nbrs])
        cen_tgt = mesh_utils.vec_mean([target[j] for j in nbrs])
        lap_cur = mesh_utils.vec_sub(current[i], cen_cur)
        lap_tgt = mesh_utils.vec_sub(target[i], cen_tgt)
        diff = mesh_utils.vec_sub(lap_cur, lap_tgt)
        score = mesh_utils.vec_length(diff)

        if normalize:
            local_scale = sum(
                mesh_utils.vec_length(mesh_utils.vec_sub(current[i], current[j]))
                for j in nbrs) / len(nbrs)
            score = score / (local_scale + epsilon)

        scores[i] = score
    return scores


def detect_reference_irregular_region(current_mesh: str,
                                      target_mesh: str,
                                      method: str = "percentile",
                                      threshold: float = 2.5,
                                      percentile: float = 97.5,
                                      min_score: float = 0.0,
                                      normalize: bool = True,
                                      rings: int = 1,
                                      select: bool = True,
                                      epsilon: float = 1e-8,
                                      ) -> Tuple[List[int], Dict[str, float]]:
    """M2 end-to-end: reference-Laplacian scoring -> detection -> selection.

    Validates correspondence, scores by current-vs-target local shape
    disagreement, flags outliers, grows by ``rings``, and (optionally) selects
    the result in Maya. Reuses :func:`summarize_scores`, :func:`detect_outliers`,
    :func:`grow_detected_region`, and :func:`select_vertices`.

    **Detection only -- no vertex positions are ever changed.**

    Returns
    -------
    (indices, stats):
        Final (possibly grown) suspicious vertex indices and the score summary.
        Returns ``([], empty_stats)`` if the topology check fails or no scores
        can be computed (fails gracefully instead of raising inside Maya).
    """
    try:
        info = validate_corresponding_topology(current_mesh, target_mesh)
    except ValueError as exc:
        print("[artifact_detection][M2] topology check FAILED: {0}".format(exc))
        return [], summarize_scores({})
    print("[artifact_detection][M2] topology OK ({0} verts, adjacency {1})".format(
        info["vertex_count"],
        "checked" if info["adjacency_checked"] else "skipped"))

    scores = compute_reference_laplacian_scores(
        current_mesh, target_mesh, normalize=normalize, epsilon=epsilon)
    stats = summarize_scores(scores)
    if stats["count"] == 0:
        print("[artifact_detection][M2] no scores computed")
        return [], stats

    detected = detect_outliers(scores, method=method, threshold=threshold,
                               percentile=percentile, min_score=min_score)
    grown = grow_detected_region(current_mesh, detected, rings=rings)

    if method == "zscore":
        crit = "z-score >= {0}".format(threshold)
    else:
        crit = "top {0:.1f}%% (percentile >= {1})".format(100.0 - percentile, percentile)
    norm_note = "normalized" if normalize else "absolute"

    print("[artifact_detection][M2] '{0}' vs '{1}' ({2}): scored {3} verts | "
          "mean={4:.4f} median={5:.4f} std={6:.4f} max={7:.4f} (vtx {8})".format(
              current_mesh, target_mesh, norm_note, stats["count"],
              stats["mean"], stats["median"], stats["std"],
              stats["max"], stats["max_index"]))
    print("[artifact_detection][M2] flagged {0} by {1}; grown to {2} verts "
          "(+{3} ring(s))".format(len(detected), crit, len(grown), rings))

    if select and grown:
        select_vertices(current_mesh, grown)
        print("[artifact_detection][M2] selected {0} vertices for inspection "
              "(mesh unchanged)".format(len(grown)))

    return grown, stats


# =============================================================================
# M3 - Multi-Scale Boundary-Aware Laplacian Detection
# =============================================================================
# EXPERIMENTAL. This section ENHANCES M1 (it does not replace it, and it does not
# use the target mesh at all). All M1 functions above are untouched so M1 stays
# reproducible.
#
# Research hypothesis
# -------------------
# M1's single 1-ring umbrella-Laplacian magnitude cannot distinguish a valid
# sharp anatomical feature (eyelid, lip, nostril, nose, jaw, chin, neck border)
# from a genuine local artifact (spike/dent/fold): both produce a large 1-ring
# score. M3 tests three ideas to separate them:
#
#   1. MULTI-SCALE. Compute the Laplacian at several neighbourhood radii
#      r in `scales` (N_r(i) = all vertices within r edge rings). A valid smooth
#      feature tends to "wash out" as the neighbourhood grows (its curvature is
#      consistent with the surrounding surface), whereas a true local defect
#      often stays unusual across scales. Combining scales (mean / max /
#      persistence) is more selective than any single scale.
#
#   2. SCALE NORMALIZATION. Divide each scale's score by the local geometric
#      scale (mean edge length within N_r(i)) so results are less dependent on
#      triangle density, making a global percentile/z-score threshold fairer
#      across dense and coarse mesh regions.
#
#   3. BOUNDARY AWARENESS + COMPONENT FILTERING. Open mesh borders (neck cut,
#      eye/mouth openings, nostrils) are topological boundaries whose Laplacian
#      is intrinsically large but not an artifact; they are detected by edge
#      topology (never by position) and optionally excluded with a buffer. Tiny
#      isolated detections are unlikely to be meaningful cleanup regions, so
#      connected components below a size threshold are dropped.
#
# This is a HYPOTHESIS to be evaluated visually; no accuracy claim is made here.
# M3 is detection-only: it never moves vertices or edits topology.


def _ring_layers(adjacency: List[List[int]],
                 vertex_index: int,
                 max_ring: int) -> Dict[int, Set[int]]:
    """Single BFS returning cumulative n-ring sets for r in 1..max_ring.

    ``layers[r]`` is the set of all vertices reachable within ``r`` edge steps of
    ``vertex_index`` (excluding itself). One traversal serves every scale, so
    neighbourhoods are computed once per vertex rather than once per (vertex,
    scale). If the local component is exhausted early, larger rings reuse the
    fully-accumulated set.
    """
    visited = {vertex_index}
    frontier = {vertex_index}
    accumulated: Set[int] = set()
    layers: Dict[int, Set[int]] = {}
    for r in range(1, max_ring + 1):
        nxt: Set[int] = set()
        for u in frontier:
            for w in adjacency[u]:
                if w not in visited:
                    visited.add(w)
                    nxt.add(w)
        accumulated |= nxt
        layers[r] = set(accumulated)
        frontier = nxt
        if not frontier:
            for rr in range(r + 1, max_ring + 1):
                layers[rr] = set(accumulated)
            break
    return layers


def get_n_ring_neighbors(mesh_name: str,
                         vertex_index: int,
                         rings: int,
                         adjacency: Optional[List[List[int]]] = None,
                         ) -> Set[int]:
    """Return unique vertex indices reachable within 1..``rings`` edge steps.

    Excludes ``vertex_index`` itself. Pass a precomputed ``adjacency`` (from
    :func:`mesh_utils.get_vertex_neighbors`) to avoid querying Maya repeatedly
    inside per-vertex loops; it is fetched once only if omitted.

    Raises
    ------
    ValueError
        If ``rings < 1``.
    """
    if rings < 1:
        raise ValueError("rings must be >= 1, got {0}".format(rings))
    if adjacency is None:
        adjacency = mesh_utils.get_vertex_neighbors(mesh_name)
    if not adjacency or vertex_index < 0 or vertex_index >= len(adjacency):
        return set()
    return _ring_layers(adjacency, vertex_index, rings).get(rings, set())


def find_boundary_vertices(mesh_name: str,
                           adjacency: Optional[List[List[int]]] = None,
                           ) -> Set[int]:
    """Return the set of true topological boundary vertices.

    Boundaries are identified purely from EDGE topology: an edge that borders
    exactly one face is a boundary edge, and its endpoints are boundary vertices
    (delegated to :func:`mesh_utils.get_boundary_vertices`, Maya API 2.0). This
    never uses vertex positions or anatomical assumptions.

    The ``adjacency`` argument is accepted for call-site symmetry with the other
    M3 helpers but is not needed for the test: vertex-vertex adjacency alone does
    not carry the edge/face incidence required to identify open borders.
    """
    return mesh_utils.get_boundary_vertices(mesh_name)


def grow_vertex_set(indices: Sequence[int],
                    adjacency: List[List[int]],
                    rings: int = 1,
                    ) -> List[int]:
    """Generic topology-based growth on precomputed adjacency.

    Reuses :func:`mesh_utils.grow_indices` (unlike
    :func:`grow_detected_region`, this takes adjacency directly and issues no
    Maya query, so it is safe to call repeatedly inside the detector). Returns a
    sorted index list; a no-op for ``rings <= 0``.
    """
    if not indices:
        return []
    if rings <= 0:
        return sorted(set(indices))
    return mesh_utils.grow_indices(adjacency, list(indices), rings=rings)


def compute_multiscale_laplacian_scores(mesh_name: str,
                                        scales: Sequence[int] = (1, 2, 3),
                                        indices: Optional[List[int]] = None,
                                        normalize: bool = True,
                                        epsilon: float = 1e-8,
                                        aggregation: str = "mean",
                                        ) -> Dict[str, object]:
    """Compute normalized umbrella-Laplacian scores at several neighbourhood scales.

    For each vertex ``i`` and each ``r`` in ``scales`` (with ``N_r(i)`` = all
    vertices within ``r`` edge rings)::

        L_i^(r)          = x_i - mean(x_j for j in N_r(i))
        score_i^(r)      = || L_i^(r) ||
        local_scale_i^(r)= mean(||x_i - x_j|| for j in N_r(i))
        normalized_i^(r) = score_i^(r) / (local_scale_i^(r) + epsilon)   # if normalize

    Research hypothesis: a valid sharp feature loses prominence as ``r`` grows,
    while a genuine local defect stays unusual across scales, so a multi-scale
    combination is more selective than M1's single 1-ring score.

    Parameters
    ----------
    scales:
        Neighbourhood radii (edge rings), each ``>= 1``.
    indices:
        Restrict scoring to these vertices (neighbours still read from the full
        mesh). ``None`` scores every vertex.
    normalize:
        Divide each scale's score by that scale's local mean edge length.
    aggregation:
        ``"mean"`` -> ``combined = mean(normalized per scale)``;
        ``"max"``  -> ``combined = max(normalized per scale)``.
        (Persistence is intentionally NOT handled here; the detector uses the
        per-scale ``scale_scores`` for that, keeping this return contract clear.)

    Returns
    -------
    dict
        ``{"combined_scores": {i: s}, "scale_scores": {r: {i: s}},
           "scales": tuple(scales), "aggregation": aggregation}``.
        Empty score maps if the mesh is missing/empty. No-neighbour vertices
        score ``0.0`` at every scale.

    Notes
    -----
    Positions and adjacency are read exactly once; each vertex's ring layers are
    computed with a single BFS shared across all scales. Cost is
    ``O(V * |N_max|)`` (no ``O(V^2)`` operations).
    """
    scales = tuple(int(r) for r in scales)
    if not scales:
        raise ValueError("scales must be a non-empty sequence")
    if any(r < 1 for r in scales):
        raise ValueError("every scale must be >= 1, got {0}".format(scales))
    if aggregation not in ("mean", "max"):
        raise ValueError(
            "aggregation must be 'mean' or 'max' here, got '{0}'".format(aggregation))
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0, got {0}".format(epsilon))

    empty = {"combined_scores": {}, "scale_scores": {r: {} for r in scales},
             "scales": scales, "aggregation": aggregation}

    vertices = mesh_utils.get_mesh_vertices(mesh_name)
    if not vertices:
        print("[artifact_detection][M3] mesh '{0}' not found or empty".format(mesh_name))
        return empty
    adjacency = mesh_utils.get_vertex_neighbors(mesh_name)
    if not adjacency:
        print("[artifact_detection][M3] could not read topology for '{0}'".format(mesh_name))
        return empty

    n = len(vertices)
    if indices is None:
        target_indices: List[int] = list(range(n))
    else:
        target_indices = [i for i in indices if 0 <= i < n]

    max_ring = max(scales)
    scale_scores: Dict[int, Dict[int, float]] = {r: {} for r in scales}

    for i in target_indices:
        if not adjacency[i]:
            for r in scales:
                scale_scores[r][i] = 0.0
            continue
        layers = _ring_layers(adjacency, i, max_ring)
        xi = vertices[i]
        for r in scales:
            members = layers.get(r, set())
            if not members:
                scale_scores[r][i] = 0.0
                continue
            centroid = mesh_utils.vec_mean([vertices[j] for j in members])
            score = mesh_utils.vec_length(mesh_utils.vec_sub(xi, centroid))
            if normalize:
                local_scale = sum(
                    mesh_utils.vec_length(mesh_utils.vec_sub(xi, vertices[j]))
                    for j in members) / len(members)
                score = score / (local_scale + epsilon)
            scale_scores[r][i] = score

    combined_scores: Dict[int, float] = {}
    for i in target_indices:
        per_scale = [scale_scores[r][i] for r in scales]
        if aggregation == "mean":
            combined_scores[i] = sum(per_scale) / len(per_scale)
        else:  # "max"
            combined_scores[i] = max(per_scale)

    return {"combined_scores": combined_scores, "scale_scores": scale_scores,
            "scales": scales, "aggregation": aggregation}


def connected_vertex_components(indices: Sequence[int],
                                adjacency: List[List[int]],
                                ) -> List[List[int]]:
    """Split ``indices`` into topology-connected components (within the set only).

    Traversal is restricted to edges between detected vertices, so two flagged
    patches that are not adjacent become separate components. Each component is
    returned as a sorted index list; components are ordered largest-first, with
    ties broken by smallest starting index for deterministic output.
    """
    idx_set = set(indices)
    visited: Set[int] = set()
    components: List[List[int]] = []
    for start in sorted(idx_set):
        if start in visited:
            continue
        stack = [start]
        visited.add(start)
        comp: List[int] = []
        while stack:
            u = stack.pop()
            comp.append(u)
            if 0 <= u < len(adjacency):
                for w in adjacency[u]:
                    if w in idx_set and w not in visited:
                        visited.add(w)
                        stack.append(w)
        components.append(sorted(comp))
    components.sort(key=lambda c: (-len(c), c[0] if c else -1))
    return components


def filter_small_components(indices: Sequence[int],
                            adjacency: List[List[int]],
                            min_component_size: int,
                            ) -> Tuple[List[int], Dict[str, object]]:
    """Drop connected components smaller than ``min_component_size``.

    Returns ``(filtered_indices, stats)`` where ``stats`` reports component and
    vertex counts before/after filtering for the detector report.
    """
    if min_component_size < 1:
        raise ValueError(
            "min_component_size must be >= 1, got {0}".format(min_component_size))

    components = connected_vertex_components(indices, adjacency)
    kept = [c for c in components if len(c) >= min_component_size]
    filtered = sorted(v for c in kept for v in c)

    stats: Dict[str, object] = {
        "components_before": len(components),
        "components_after": len(kept),
        "vertices_before": len(set(indices)),
        "vertices_after": len(filtered),
        "removed_vertices": len(set(indices)) - len(filtered),
        "removed_components": len(components) - len(kept),
        "kept_component_sizes": [len(c) for c in kept],
    }
    return filtered, stats


def _validate_multiscale_params(scales, method, percentile, threshold,
                                aggregation, min_persistent_scales,
                                boundary_buffer_rings, min_component_size,
                                final_growth_rings, epsilon):
    """Raise descriptive ValueError for invalid detector parameters."""
    if not scales:
        raise ValueError("scales must be a non-empty sequence")
    if any(int(r) < 1 for r in scales):
        raise ValueError("every scale must be >= 1, got {0}".format(tuple(scales)))
    if method not in ("percentile", "zscore"):
        raise ValueError(
            "method must be 'percentile' or 'zscore', got '{0}'".format(method))
    if not (0.0 <= percentile <= 100.0):
        raise ValueError("percentile must be in [0, 100], got {0}".format(percentile))
    if threshold < 0:
        raise ValueError("threshold must be >= 0, got {0}".format(threshold))
    if aggregation not in ("persistence", "mean", "max"):
        raise ValueError("aggregation must be 'persistence', 'mean' or 'max', "
                         "got '{0}'".format(aggregation))
    if aggregation == "persistence":
        if min_persistent_scales < 1:
            raise ValueError("min_persistent_scales must be >= 1, got {0}".format(
                min_persistent_scales))
        if min_persistent_scales > len(scales):
            raise ValueError(
                "min_persistent_scales ({0}) cannot exceed number of scales "
                "({1})".format(min_persistent_scales, len(scales)))
    if boundary_buffer_rings < 0:
        raise ValueError("boundary_buffer_rings must be >= 0, got {0}".format(
            boundary_buffer_rings))
    if min_component_size < 1:
        raise ValueError("min_component_size must be >= 1, got {0}".format(
            min_component_size))
    if final_growth_rings < 0:
        raise ValueError("final_growth_rings must be >= 0, got {0}".format(
            final_growth_rings))
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0, got {0}".format(epsilon))


def detect_multiscale_irregular_region(mesh_name: str,
                                       scales: Sequence[int] = (1, 2, 3),
                                       method: str = "percentile",
                                       percentile: float = 97.5,
                                       threshold: float = 2.5,
                                       normalize: bool = True,
                                       aggregation: str = "persistence",
                                       min_persistent_scales: int = 2,
                                       exclude_boundaries: bool = True,
                                       boundary_buffer_rings: int = 1,
                                       min_component_size: int = 5,
                                       final_growth_rings: int = 1,
                                       select: bool = True,
                                       epsilon: float = 1e-8,
                                       ) -> Tuple[List[int], Dict[str, object]]:
    """M3 detector: multi-scale, boundary-aware, component-filtered detection.

    Pipeline (detection only -- never edits vertices or topology):
    validate params -> read positions/topology once -> per-scale normalized
    Laplacian scores -> per-scale outlier detection -> aggregate
    (persistence / mean / max) -> optional boundary exclusion (+ buffer) ->
    connected components -> drop small components -> grow survivors ->
    optional Maya selection.

    Parameters mirror the research knobs described in the M3 section header. See
    :func:`compute_multiscale_laplacian_scores` for the scoring definition.

    Returns
    -------
    (final_indices, report):
        ``final_indices`` is the deterministic sorted vertex list; ``report`` is
        a dict of per-stage counts/statistics (see keys built below).
    """
    _validate_multiscale_params(scales, method, percentile, threshold,
                                aggregation, min_persistent_scales,
                                boundary_buffer_rings, min_component_size,
                                final_growth_rings, epsilon)
    scales = tuple(int(r) for r in scales)

    def _empty_report():
        return {
            "mesh": mesh_name, "scales": scales, "normalization": normalize,
            "aggregation": aggregation, "per_scale_stats": {},
            "per_scale_detected_counts": {}, "persistent_count": 0,
            "boundary_vertex_count": 0, "boundary_excluded_count": 0,
            "components_before_filter": 0, "components_after_filter": 0,
            "removed_small_component_vertices": 0,
            "final_count_before_growth": 0, "final_count": 0,
        }

    if not mesh_utils.mesh_exists(mesh_name):
        print("[artifact_detection][M3] object '{0}' does not exist".format(mesh_name))
        return [], _empty_report()

    # Topology read once for boundary/components/growth (scoring reads its own
    # single copy internally).
    adjacency = mesh_utils.get_vertex_neighbors(mesh_name)
    if not adjacency:
        print("[artifact_detection][M3] could not read topology for '{0}'".format(mesh_name))
        return [], _empty_report()

    # --- C. per-scale normalized scores -------------------------------------
    score_agg = aggregation if aggregation in ("mean", "max") else "mean"
    ms = compute_multiscale_laplacian_scores(
        mesh_name, scales=scales, normalize=normalize, epsilon=epsilon,
        aggregation=score_agg)
    scale_scores: Dict[int, Dict[int, float]] = ms["scale_scores"]  # type: ignore
    combined_scores: Dict[int, float] = ms["combined_scores"]        # type: ignore

    if not any(scale_scores[r] for r in scales):
        print("[artifact_detection][M3] no scores computed")
        return [], _empty_report()

    per_scale_stats: Dict[int, Dict[str, float]] = {
        r: summarize_scores(scale_scores[r]) for r in scales}

    # --- D. per-scale outlier detection -------------------------------------
    per_scale_detected: Dict[int, Set[int]] = {}
    for r in scales:
        flagged = detect_outliers(scale_scores[r], method=method,
                                  threshold=threshold, percentile=percentile,
                                  min_score=0.0)
        per_scale_detected[r] = set(flagged)
    per_scale_detected_counts = {r: len(per_scale_detected[r]) for r in scales}

    # --- E/F. aggregate ------------------------------------------------------
    if aggregation == "persistence":
        tally: Dict[int, int] = {}
        for r in scales:
            for v in per_scale_detected[r]:
                tally[v] = tally.get(v, 0) + 1
        detected_set = {v for v, c in tally.items() if c >= min_persistent_scales}
        persistent_count = len(detected_set)
    else:
        detected_set = set(detect_outliers(
            combined_scores, method=method, threshold=threshold,
            percentile=percentile, min_score=0.0))
        persistent_count = len(detected_set)

    # --- G. boundary exclusion ----------------------------------------------
    boundary_vertex_count = 0
    boundary_excluded_count = 0
    if exclude_boundaries:
        boundary = find_boundary_vertices(mesh_name)
        boundary_vertex_count = len(boundary)
        if boundary:
            if boundary_buffer_rings > 0:
                exclusion = set(grow_vertex_set(sorted(boundary), adjacency,
                                                rings=boundary_buffer_rings))
            else:
                exclusion = set(boundary)
            before = len(detected_set)
            detected_set = detected_set - exclusion
            boundary_excluded_count = before - len(detected_set)

    # --- H. connected components --------------------------------------------
    components_before = connected_vertex_components(detected_set, adjacency)

    # --- I. drop small components -------------------------------------------
    filtered, comp_stats = filter_small_components(
        detected_set, adjacency, min_component_size)
    final_count_before_growth = len(filtered)

    # --- J. grow survivors (after filtering only) ---------------------------
    final_indices = grow_vertex_set(filtered, adjacency, rings=final_growth_rings)

    # --- report --------------------------------------------------------------
    report: Dict[str, object] = {
        "mesh": mesh_name,
        "scales": scales,
        "normalization": normalize,
        "aggregation": aggregation,
        "per_scale_stats": per_scale_stats,
        "per_scale_detected_counts": per_scale_detected_counts,
        "persistent_count": persistent_count,
        "boundary_vertex_count": boundary_vertex_count,
        "boundary_excluded_count": boundary_excluded_count,
        "components_before_filter": len(components_before),
        "components_after_filter": comp_stats["components_after"],
        "removed_small_component_vertices": comp_stats["removed_vertices"],
        "final_count_before_growth": final_count_before_growth,
        "final_count": len(final_indices),
    }

    # --- N. concise report ---------------------------------------------------
    if method == "zscore":
        crit = "z-score >= {0}".format(threshold)
    else:
        crit = "top {0:.1f}% (pct >= {1})".format(100.0 - percentile, percentile)
    print("[artifact_detection][M3] '{0}' | scales={1} norm={2} agg={3} ({4})".format(
        mesh_name, scales, normalize, aggregation, crit))
    for r in scales:
        st = per_scale_stats[r]
        print("  scale {0}: detected {1:5d} | mean={2:.4f} median={3:.4f} "
              "std={4:.4f} max={5:.4f}".format(
                  r, per_scale_detected_counts[r], st["mean"], st["median"],
                  st["std"], st["max"]))
    if aggregation == "persistence":
        print("  persistence: kept {0} verts flagged at >= {1}/{2} scales".format(
            persistent_count, min_persistent_scales, len(scales)))
    else:
        print("  {0}-combined: {1} verts flagged".format(aggregation, persistent_count))
    print("  boundary: {0} boundary verts, excluded {1} detections "
          "(+{2} buffer ring(s))".format(
              boundary_vertex_count, boundary_excluded_count, boundary_buffer_rings))
    print("  components: {0} -> {1} after dropping < {2} verts "
          "(removed {3} verts)".format(
              report["components_before_filter"], report["components_after_filter"],
              min_component_size, report["removed_small_component_vertices"]))
    print("  final: {0} verts -> {1} after +{2} growth ring(s)".format(
        final_count_before_growth, report["final_count"], final_growth_rings))

    # --- K. selection --------------------------------------------------------
    if select and final_indices:
        select_vertices(mesh_name, final_indices)
        print("[artifact_detection][M3] selected {0} vertices for inspection "
              "(mesh unchanged)".format(len(final_indices)))

    return final_indices, report


# =============================================================================
# M3 V2 - Hybrid Geometry and Anatomy-Aware Artifact Detection
# =============================================================================
# EXPERIMENTAL. Enhances (does not replace) M1 and M3 V1, which remain untouched
# and reproducible. Does NOT use the target mesh.
#
# Motivation
# ----------
# M3 V1 relies on the umbrella Laplacian, which measures LOCAL geometric
# irregularity. Visual evaluation showed it catches sharp local defects (e.g. a
# chin spike) but misses BROAD, smooth registration errors (e.g. cheeks that
# drifted off the anatomy but stayed locally smooth), and still lights up
# edge-like regions. Parameter tuning alone cannot fix this because the missed
# errors genuinely have low Laplacian scores.
#
# M3 V2 combines three complementary signals so different error types can be
# caught by whichever signal is sensitive to them:
#   1. Multi-scale Laplacian irregularity  -> local spikes/dents/folds (reuses V1).
#   2. Surface-normal inconsistency        -> creased / flipped / noisy shading.
#   3. Local deviation in distance-to-anatomy -> broad drift off the underlying
#      internal meshes even where the surface is locally smooth (the class of
#      error the Laplacian misses).
# Each signal is robustly normalized (median / MAD) so a global threshold is
# comparable across signals, then combined with configurable weights. Instead of
# hard boundary deletion (V1), a SOFT boundary weight down-scales scores near
# open borders, so edge noise is suppressed without discarding valid nearby
# artifacts.
#
# This is a hypothesis to evaluate visually; no accuracy claim is made. Like all
# earlier methods it is detection-only and never edits geometry.


def _mad(values: List[float], med: float) -> float:
    """Median absolute deviation about ``med`` (0.0 for empty input)."""
    if not values:
        return 0.0
    return _median([abs(v - med) for v in values])


def robust_normalize_scores(scores: Dict[int, float],
                            epsilon: float = 1e-8,
                            clamp_negative: bool = True,
                            ) -> Dict[int, float]:
    """Robustly normalize a score map using median / MAD.

        robust_z_i = (score_i - median(score)) / (1.4826 * MAD + epsilon)

    The ``1.4826`` factor makes MAD a consistent estimator of the standard
    deviation for normal data, so the result is a robust analogue of a z-score
    that resists outliers dominating the scale. Negative values (below the
    median) are clamped to 0 by default, since only high scores indicate
    candidate artifacts.
    """
    if not scores:
        return {}
    values = list(scores.values())
    med = _median(values)
    denom = 1.4826 * _mad(values, med) + epsilon
    out: Dict[int, float] = {}
    for i, v in scores.items():
        z = (v - med) / denom
        if clamp_negative and z < 0.0:
            z = 0.0
        out[i] = z
    return out


def compute_normal_inconsistency_scores(mesh_name: str,
                                        normal_rings: int = 1,
                                        adjacency: Optional[List[List[int]]] = None,
                                        normals: Optional[List[List[float]]] = None,
                                        ) -> Dict[int, float]:
    """Score vertices by how much neighbouring surface normals disagree.

        normal_score_i = 1 - mean(clamp(dot(n_i, n_j), -1, 1) for j in N_r(i))

    High where the local surface direction varies (creases, folds, flipped or
    noisy normals); ~0 on a smooth patch where neighbours point the same way.
    Vertex normals and adjacency are each read once.

    Raises
    ------
    ValueError
        If ``normal_rings < 1``.
    """
    if normal_rings < 1:
        raise ValueError("normal_rings must be >= 1, got {0}".format(normal_rings))
    if adjacency is None:
        adjacency = mesh_utils.get_vertex_neighbors(mesh_name)
    if normals is None:
        normals = mesh_utils.get_vertex_normals(mesh_name)
    if not normals or not adjacency:
        print("[artifact_detection][M3V2] no normals/topology for '{0}'".format(mesh_name))
        return {}

    n = min(len(normals), len(adjacency))
    scores: Dict[int, float] = {}
    for i in range(n):
        if not adjacency[i]:
            scores[i] = 0.0
            continue
        members = _ring_layers(adjacency, i, normal_rings).get(normal_rings, set())
        if not members:
            scores[i] = 0.0
            continue
        ni = normals[i]
        acc = 0.0
        for j in members:
            nj = normals[j]
            dot = ni[0] * nj[0] + ni[1] * nj[1] + ni[2] * nj[2]
            if dot > 1.0:
                dot = 1.0
            elif dot < -1.0:
                dot = -1.0
            acc += dot
        scores[i] = 1.0 - (acc / len(members))
    return scores


def compute_distance_to_anatomy_scores(skin_mesh: str,
                                       anatomical_meshes: Sequence[str],
                                       distance_rings: int = 2,
                                       robust: bool = True,
                                       adjacency: Optional[List[List[int]]] = None,
                                       epsilon: float = 1e-8,
                                       ) -> Dict[str, object]:
    """Score vertices by local deviation of their distance to the anatomy.

    For each skin vertex ``i`` let ``d_i`` be the shortest distance to the union
    of valid internal meshes. The local (robust) deviation is::

        distance_score_i = |d_i - median(d_j for j in N_r(i))|            # robust=False
        distance_score_i = |d_i - local_median| / (local_MAD + epsilon)   # robust=True (default)

    A vertex that sits much closer/farther from the anatomy than its neighbours
    is flagged -- this catches BROAD drift off the underlying structures even
    where the surface is locally smooth (which the Laplacian misses). Closest
    distances use one cached ``MMeshIntersector`` per anatomical mesh (see
    :func:`mesh_utils.compute_closest_point_distances`).

    Returns
    -------
    dict
        ``{"distance_scores": {i: s}, "raw_distances": {i: d},
           "nearest_anatomy": {i: mesh}, "valid_meshes": [...],
           "missing_meshes": [...]}``.

    Raises
    ------
    ValueError
        If ``anatomical_meshes`` is empty or ``distance_rings < 1``.
    """
    if not anatomical_meshes:
        raise ValueError("anatomical_meshes must be a non-empty list of mesh names")
    if distance_rings < 1:
        raise ValueError("distance_rings must be >= 1, got {0}".format(distance_rings))

    points = mesh_utils.get_mesh_vertices(skin_mesh)
    empty = {"distance_scores": {}, "raw_distances": {}, "nearest_anatomy": {},
             "valid_meshes": [], "missing_meshes": list(anatomical_meshes)}
    if not points:
        print("[artifact_detection][M3V2] skin mesh '{0}' empty/missing".format(skin_mesh))
        return empty
    if adjacency is None:
        adjacency = mesh_utils.get_vertex_neighbors(skin_mesh)

    dist_res = mesh_utils.compute_closest_point_distances(points, list(anatomical_meshes))
    raw = dist_res["distances"]
    nearest = dist_res["nearest"]

    v = len(points)
    raw_distances = {i: raw[i] for i in range(v)}
    nearest_anatomy = {i: nearest[i] for i in range(v)}
    distance_scores: Dict[int, float] = {}
    for i in range(v):
        members = _ring_layers(adjacency, i, distance_rings).get(distance_rings, set())
        if not members:
            distance_scores[i] = 0.0
            continue
        local = [raw[j] for j in members]
        local_median = _median(local)
        dev = abs(raw[i] - local_median)
        if robust:
            local_mad = _median([abs(x - local_median) for x in local])
            distance_scores[i] = dev / (local_mad + epsilon)
        else:
            distance_scores[i] = dev

    return {"distance_scores": distance_scores, "raw_distances": raw_distances,
            "nearest_anatomy": nearest_anatomy,
            "valid_meshes": dist_res["valid_meshes"],
            "missing_meshes": dist_res["missing_meshes"]}


def compute_hybrid_artifact_scores(skin_mesh: str,
                                   anatomical_meshes: Optional[Sequence[str]] = None,
                                   laplacian_scales: Sequence[int] = (1, 2),
                                   normal_rings: int = 1,
                                   distance_rings: int = 2,
                                   laplacian_weight: float = 0.3,
                                   normal_weight: float = 0.2,
                                   distance_weight: float = 0.5,
                                   normalize_laplacian: bool = True,
                                   epsilon: float = 1e-8,
                                   ) -> Dict[str, object]:
    """Combine Laplacian, normal, and distance signals into one hybrid score.

        combined_i = w_lap * rz(laplacian_i)
                   + w_normal * rz(normal_i)
                   + w_distance * rz(distance_i)

    where ``rz`` is :func:`robust_normalize_scores`. Weights must be
    non-negative with a positive total and are normalized internally to sum to
    1. The distance signal is skipped (all zeros) when ``distance_weight == 0``
    so weight-free ablations stay fast and need no anatomy.

    Returns a dict with ``combined_scores``, the three per-signal score maps,
    ``raw_distances``, ``nearest_anatomy``, normalized ``weights``, valid/missing
    anatomy lists, and ``statistics`` (per-signal + combined summaries).

    Raises
    ------
    ValueError
        On negative weights, non-positive total weight, or if the distance
        signal is requested (``distance_weight > 0``) without anatomical meshes.
    """
    weights_in = [laplacian_weight, normal_weight, distance_weight]
    if any(w < 0 for w in weights_in):
        raise ValueError("weights must be non-negative, got {0}".format(weights_in))
    total = sum(weights_in)
    if total <= 0:
        raise ValueError("weights must have a positive total, got {0}".format(weights_in))
    w_lap, w_normal, w_distance = (w / total for w in weights_in)

    adjacency = mesh_utils.get_vertex_neighbors(skin_mesh)
    if not adjacency:
        print("[artifact_detection][M3V2] could not read topology for '{0}'".format(skin_mesh))
        return {"combined_scores": {}, "laplacian_scores": {}, "normal_scores": {},
                "distance_scores": {}, "raw_distances": {}, "nearest_anatomy": {},
                "weights": {"laplacian": w_lap, "normal": w_normal, "distance": w_distance},
                "statistics": {}, "valid_meshes": [], "missing_meshes": []}
    v = len(adjacency)

    # 1. multi-scale Laplacian (reuse V1), combined across scales by mean
    lap = compute_multiscale_laplacian_scores(
        skin_mesh, scales=laplacian_scales, normalize=normalize_laplacian,
        epsilon=epsilon, aggregation="mean")
    lap_scores: Dict[int, float] = lap["combined_scores"]  # type: ignore

    # 2. normal inconsistency
    normal_scores = compute_normal_inconsistency_scores(
        skin_mesh, normal_rings=normal_rings, adjacency=adjacency)

    # 3. distance-to-anatomy (only if it contributes)
    raw_distances: Dict[int, float] = {}
    nearest_anatomy: Dict[int, Optional[str]] = {}
    valid_meshes: List[str] = []
    missing_meshes: List[str] = []
    if w_distance > 0:
        if not anatomical_meshes:
            raise ValueError(
                "distance_weight > 0 requires anatomical_meshes (e.g. INTERNAL_MESHES)")
        dist = compute_distance_to_anatomy_scores(
            skin_mesh, anatomical_meshes, distance_rings=distance_rings,
            robust=True, adjacency=adjacency, epsilon=epsilon)
        distance_scores: Dict[int, float] = dist["distance_scores"]  # type: ignore
        raw_distances = dist["raw_distances"]                         # type: ignore
        nearest_anatomy = dist["nearest_anatomy"]                     # type: ignore
        valid_meshes = dist["valid_meshes"]                           # type: ignore
        missing_meshes = dist["missing_meshes"]                       # type: ignore
    else:
        distance_scores = {i: 0.0 for i in range(v)}

    # robust global normalization of each signal
    norm_lap = robust_normalize_scores(lap_scores, epsilon=epsilon)
    norm_normal = robust_normalize_scores(normal_scores, epsilon=epsilon)
    norm_distance = robust_normalize_scores(distance_scores, epsilon=epsilon)

    combined_scores: Dict[int, float] = {}
    for i in range(v):
        combined_scores[i] = (w_lap * norm_lap.get(i, 0.0)
                              + w_normal * norm_normal.get(i, 0.0)
                              + w_distance * norm_distance.get(i, 0.0))

    statistics = {
        "laplacian": summarize_scores(norm_lap),
        "normal": summarize_scores(norm_normal),
        "distance": summarize_scores(norm_distance),
        "combined": summarize_scores(combined_scores),
    }

    return {
        "combined_scores": combined_scores,
        "laplacian_scores": norm_lap,
        "normal_scores": norm_normal,
        "distance_scores": norm_distance,
        "raw_distances": raw_distances,
        "nearest_anatomy": nearest_anatomy,
        "weights": {"laplacian": w_lap, "normal": w_normal, "distance": w_distance},
        "statistics": statistics,
        "valid_meshes": valid_meshes,
        "missing_meshes": missing_meshes,
    }


def compute_boundary_weights(mesh_name: str,
                             max_rings: int = 3,
                             ring_weights: Sequence[float] = (0.0, 0.25, 0.6, 1.0),
                             adjacency: Optional[List[List[int]]] = None,
                             ) -> Dict[int, float]:
    """Soft per-vertex weight that ramps up with distance from a mesh boundary.

    Ring distance ``d`` (edge steps to the nearest true boundary vertex) maps to
    ``ring_weights[min(d, len-1)]``; vertices farther than ``max_rings`` get the
    last weight. With the default ``(0.0, 0.25, 0.6, 1.0)``: boundary -> 0.0,
    1 ring -> 0.25, 2 rings -> 0.6, >=3 rings -> 1.0. Multiplying scores by this
    weight softly suppresses edge-like detections without hard deletion.

    Distances come from a single multi-source BFS seeded at all boundary
    vertices (found by edge topology, never position). Returns weight 1.0
    everywhere if the mesh has no open boundary.

    Raises
    ------
    ValueError
        If ``max_rings < 0`` or ``ring_weights`` is empty.
    """
    from collections import deque

    if max_rings < 0:
        raise ValueError("max_rings must be >= 0, got {0}".format(max_rings))
    if not ring_weights:
        raise ValueError("ring_weights must be a non-empty sequence")

    if adjacency is None:
        adjacency = mesh_utils.get_vertex_neighbors(mesh_name)
    v = len(adjacency)
    last_weight = ring_weights[-1]
    if v == 0:
        return {}

    boundary = find_boundary_vertices(mesh_name)
    if not boundary:
        return {i: last_weight for i in range(v)}

    inf = max_rings + 1
    dist = [inf] * v
    dq = deque()
    for b in boundary:
        if 0 <= b < v:
            dist[b] = 0
            dq.append(b)
    while dq:
        u = dq.popleft()
        if dist[u] >= max_rings:
            continue
        for w in adjacency[u]:
            if dist[w] > dist[u] + 1:
                dist[w] = dist[u] + 1
                dq.append(w)

    weights: Dict[int, float] = {}
    for i in range(v):
        d = dist[i]
        if d > max_rings:
            weights[i] = last_weight
        else:
            weights[i] = ring_weights[min(d, len(ring_weights) - 1)]
    return weights


def _validate_hybrid_detector_params(method, percentile, threshold,
                                     min_component_size, final_growth_rings,
                                     boundary_max_rings):
    if method not in ("percentile", "zscore"):
        raise ValueError(
            "method must be 'percentile' or 'zscore', got '{0}'".format(method))
    if not (0.0 <= percentile <= 100.0):
        raise ValueError("percentile must be in [0, 100], got {0}".format(percentile))
    if threshold < 0:
        raise ValueError("threshold must be >= 0, got {0}".format(threshold))
    if min_component_size < 1:
        raise ValueError("min_component_size must be >= 1, got {0}".format(min_component_size))
    if final_growth_rings < 0:
        raise ValueError("final_growth_rings must be >= 0, got {0}".format(final_growth_rings))
    if boundary_max_rings < 0:
        raise ValueError("boundary_max_rings must be >= 0, got {0}".format(boundary_max_rings))


def detect_hybrid_artifacts(skin_mesh: str,
                            anatomical_meshes: Optional[Sequence[str]] = None,
                            method: str = "percentile",
                            percentile: float = 97.5,
                            threshold: float = 2.5,
                            laplacian_scales: Sequence[int] = (1, 2),
                            normal_rings: int = 1,
                            distance_rings: int = 2,
                            laplacian_weight: float = 0.3,
                            normal_weight: float = 0.2,
                            distance_weight: float = 0.5,
                            use_boundary_weights: bool = True,
                            boundary_max_rings: int = 3,
                            boundary_ring_weights: Sequence[float] = (0.0, 0.25, 0.6, 1.0),
                            min_component_size: int = 3,
                            final_growth_rings: int = 1,
                            select: bool = True,
                            epsilon: float = 1e-8,
                            ) -> Tuple[List[int], Dict[str, object]]:
    """M3 V2 detector: hybrid geometry + anatomy-aware artifact detection.

    Pipeline (detection only -- never edits geometry): validate -> hybrid scores
    -> soft boundary weighting -> outlier detection (reuses
    :func:`detect_outliers`) -> connected components -> drop small components ->
    grow survivors -> optional Maya selection.

    Returns ``(final_indices, report)``; ``report`` holds valid/missing anatomy,
    per-signal statistics, normalized weights, raw/boundary-suppressed counts,
    component counts, final count, and the top-scoring vertices with their
    nearest anatomical structures.
    """
    _validate_hybrid_detector_params(method, percentile, threshold,
                                     min_component_size, final_growth_rings,
                                     boundary_max_rings)

    def _empty_report():
        return {"mesh": skin_mesh, "valid_anatomy_count": 0, "missing_anatomy": [],
                "per_signal_stats": {}, "weights": {}, "raw_detected_count": 0,
                "boundary_suppressed_count": 0, "components_before_filter": 0,
                "components_after_filter": 0, "final_count_before_growth": 0,
                "final_count": 0, "top_vertices": [], "top_nearest_anatomy": {}}

    if not mesh_utils.mesh_exists(skin_mesh):
        print("[artifact_detection][M3V2] object '{0}' does not exist".format(skin_mesh))
        return [], _empty_report()

    adjacency = mesh_utils.get_vertex_neighbors(skin_mesh)
    if not adjacency:
        print("[artifact_detection][M3V2] could not read topology for '{0}'".format(skin_mesh))
        return [], _empty_report()

    # 2. hybrid scores
    hybrid = compute_hybrid_artifact_scores(
        skin_mesh, anatomical_meshes=anatomical_meshes,
        laplacian_scales=laplacian_scales, normal_rings=normal_rings,
        distance_rings=distance_rings, laplacian_weight=laplacian_weight,
        normal_weight=normal_weight, distance_weight=distance_weight,
        epsilon=epsilon)
    combined: Dict[int, float] = hybrid["combined_scores"]  # type: ignore
    if not combined:
        print("[artifact_detection][M3V2] no scores computed")
        return [], _empty_report()

    # 3. soft boundary weighting
    if use_boundary_weights:
        bw = compute_boundary_weights(skin_mesh, max_rings=boundary_max_rings,
                                      ring_weights=boundary_ring_weights,
                                      adjacency=adjacency)
        weighted = {i: combined[i] * bw.get(i, 1.0) for i in combined}
    else:
        weighted = dict(combined)

    # 4. outlier detection (raw vs boundary-weighted)
    raw_detected = set(detect_outliers(combined, method=method,
                                       threshold=threshold, percentile=percentile,
                                       min_score=0.0))
    detected = set(detect_outliers(weighted, method=method, threshold=threshold,
                                   percentile=percentile, min_score=0.0))
    boundary_suppressed_count = len(raw_detected - detected)

    # 5-6. components + drop small
    components_before = connected_vertex_components(detected, adjacency)
    filtered, comp_stats = filter_small_components(detected, adjacency, min_component_size)
    final_count_before_growth = len(filtered)

    # 7. grow survivors
    final_indices = grow_vertex_set(filtered, adjacency, rings=final_growth_rings)

    # top-scoring vertices + nearest anatomy
    top = sorted(weighted.items(), key=lambda kv: (-kv[1], kv[0]))[:10]
    top_vertices = [(i, round(s, 4)) for i, s in top]
    nearest_map = hybrid.get("nearest_anatomy", {})
    top_nearest = {i: nearest_map.get(i) for i, _ in top}

    report: Dict[str, object] = {
        "mesh": skin_mesh,
        "valid_anatomy_count": len(hybrid.get("valid_meshes", [])),
        "missing_anatomy": hybrid.get("missing_meshes", []),
        "per_signal_stats": hybrid.get("statistics", {}),
        "weights": hybrid.get("weights", {}),
        "raw_detected_count": len(raw_detected),
        "boundary_suppressed_count": boundary_suppressed_count,
        "components_before_filter": len(components_before),
        "components_after_filter": comp_stats["components_after"],
        "removed_small_component_vertices": comp_stats["removed_vertices"],
        "final_count_before_growth": final_count_before_growth,
        "final_count": len(final_indices),
        "top_vertices": top_vertices,
        "top_nearest_anatomy": top_nearest,
    }

    # concise report
    w = report["weights"]
    print("[artifact_detection][M3V2] '{0}' | weights lap={1:.2f} nrm={2:.2f} "
          "dist={3:.2f} | anatomy {4} valid, {5} missing".format(
              skin_mesh, w.get("laplacian", 0), w.get("normal", 0),
              w.get("distance", 0), report["valid_anatomy_count"],
              len(report["missing_anatomy"])))
    if report["missing_anatomy"]:
        print("  missing anatomy: {0}".format(", ".join(report["missing_anatomy"])))
    for sig in ("laplacian", "normal", "distance", "combined"):
        st = hybrid["statistics"].get(sig, {})  # type: ignore
        if st:
            print("  {0:9s}: mean={1:.3f} median={2:.3f} std={3:.3f} max={4:.3f}".format(
                sig, st["mean"], st["median"], st["std"], st["max"]))
    print("  detected {0} (boundary-suppressed {1}); components {2} -> {3} "
          "(< {4} dropped)".format(
              len(raw_detected), boundary_suppressed_count,
              report["components_before_filter"], report["components_after_filter"],
              min_component_size))
    print("  final: {0} verts -> {1} after +{2} growth ring(s)".format(
        final_count_before_growth, report["final_count"], final_growth_rings))

    # 8. selection
    if select and final_indices:
        select_vertices(skin_mesh, final_indices)
        print("[artifact_detection][M3V2] selected {0} vertices for inspection "
              "(mesh unchanged)".format(len(final_indices)))

    return final_indices, report


# =============================================================================
# M4 - SDF-Reference Laplacian Artifact Detection
# =============================================================================
# WHAT M4 CHANGES vs. M2
# ----------------------
# M2 compared the current skin against an EXTERNALLY SUPPLIED target mesh
# (``..._target``). That target was not an accurate ideal of the registered
# skin, so M2 flagged valid disagreement (eyelids, lips, nostrils, mouth, neck,
# outer borders) as artifacts.
#
# M4 keeps the reference-Laplacian idea but REPLACES the supplied target with a
# target DERIVED FROM THE UNDERLYING ANATOMY. Let ``phi(x)`` be the (unsigned)
# distance from ``x`` to the union of the internal anatomical meshes (muscles,
# fat, bone). A desired skin offset ``d0 > 0`` defines the expected anatomical
# skin surface as the iso-surface ``phi(x) = d0``. For every current skin vertex
# ``x_i`` we compute an SDF-derived target ``t_i`` by projecting ``x_i`` onto
# ``phi = d0`` WHILE KEEPING THE SKIN'S TOPOLOGY AND VERTEX CORRESPONDENCE (no
# marching cubes, no new unrelated mesh). Then, exactly as in M2::
#
#     L_current(i)    = x_i - mean(x_j for j in N(i))
#     L_sdf_target(i) = t_i - mean(t_j for j in N(i))
#     score_i         = || L_current(i) - L_sdf_target(i) ||
#
# So M2 measures disagreement with a supplied mesh; M4 measures disagreement
# with an anatomy-derived offset surface.
#
# SDF REUSE (no second, incompatible SDF)
# ---------------------------------------
# The distance field is the SAME one the registration uses. The d98 script owns
# ``compute_sdf_for_point`` (a smooth-min union over per-mesh
# ``MFnMesh.getClosestPoint`` queries) and ``get_mesh_fn``. Because this module
# cannot import the d98 *script*, the d98 wrapper injects those callables ONCE
# via :func:`configure_sdf_backend`. When the backend is configured, M4's SDF
# values match the registration field (smooth-min blend). When it is NOT
# configured (e.g. running this module standalone), M4 falls back to a plain
# HARD-MIN union built on :func:`mesh_utils.closest_point_on_mesh` -- the same
# Maya API 2.0 acceleration primitive, no ``maya.cmds`` per-vertex calls.
#
# COORDINATE SPACE: everything below is WORLD space, consistent with
# ``mesh_utils.get_mesh_vertices`` (``MSpace.kWorld``) and the injected d98 SDF.
#
# SIGNED vs UNSIGNED: the available infrastructure returns UNSIGNED distance to
# the union surface (there is no inside/outside test). Projection therefore
# assumes the skin lies OUTSIDE the anatomy (true for a registered skin) and
# moves along the outward distance gradient. This limitation is exposed on the
# query object as ``query.signed == False`` and documented in the reports.
#
# SAFETY: M4 is DETECTION ONLY. It never moves the skin vertices, never edits
# the anatomical meshes, never smooths, and never creates geometry -- except the
# explicit, opt-in :func:`create_sdf_target_debug_mesh` visualization helper.

# --- Optional dependency injection of the registration's exact SDF backend ----
# The d98 wrapper calls configure_sdf_backend(get_mesh_fn, compute_sdf_for_point)
# so M4 reuses the registration's MFnMesh provider and smooth-min union SDF
# instead of building a second, incompatible field.
_MESH_FN_PROVIDER: Optional[Callable[[str], Any]] = None
_POINT_SDF_FN: Optional[Callable[..., Tuple[float, Sequence[float],
                                            Sequence[float], Optional[str]]]] = None


def configure_sdf_backend(mesh_fn_provider: Optional[Callable[[str], Any]] = None,
                          point_sdf_fn: Optional[Callable[..., Tuple]] = None,
                          ) -> None:
    """Register the registration project's SDF/MFnMesh callables for reuse.

    Parameters
    ----------
    mesh_fn_provider:
        ``callable(mesh_name) -> MFnMesh`` (e.g. d98's ``get_mesh_fn``). If
        omitted, :func:`mesh_utils.get_mesh_fn` is used -- functionally the same
        Maya API 2.0 accessor.
    point_sdf_fn:
        ``callable(point, mesh_fns, offset, blend_k) -> (sdf_value, closest,
        outward, mesh_name)`` (e.g. d98's ``compute_sdf_for_point``). When
        provided, M4's SDF matches the registration's smooth-min union field and
        uses its analytic outward direction as the gradient. When omitted, a
        hard-min union is used instead.

    Notes
    -----
    Read-only w.r.t. the scene. Call again with ``None`` args to reset to the
    standalone fallback. This keeps every M4 public function signature exactly as
    specified while still reusing the existing SDF implementation.
    """
    global _MESH_FN_PROVIDER, _POINT_SDF_FN
    _MESH_FN_PROVIDER = mesh_fn_provider
    _POINT_SDF_FN = point_sdf_fn
    backend = "d98 smooth-min union" if point_sdf_fn is not None else "hard-min union (fallback)"
    provider = "injected" if mesh_fn_provider is not None else "mesh_utils.get_mesh_fn"
    print("[artifact_detection][M4] SDF backend configured: {0}; mesh_fn provider: {1}".format(
        backend, provider))


def _is_finite(x: float) -> bool:
    try:
        return math.isfinite(x)
    except (TypeError, ValueError):
        return False


def _vec_is_finite(v: Sequence[float]) -> bool:
    return len(v) == 3 and all(_is_finite(c) for c in v)


# =============================================================================
# M4.2  ANATOMY SDF QUERY (union distance field, world space)
# =============================================================================

class AnatomySDFQuery(object):
    """Distance query against the union of internal anatomical meshes.

    Represents ``phi(x)`` = distance from a world-space point ``x`` to the union
    of the supplied muscle / fat / bone meshes. Built ONCE (one ``MFnMesh`` per
    anatomical mesh, each carrying Maya's internal acceleration structure) and
    then queried many times without any per-vertex ``maya.cmds`` calls.

    Distance semantics
    ------------------
    ``evaluate(point)`` returns an UNSIGNED distance to the nearest surface of
    the union (``self.signed is False``). ``gradient(point)`` returns a unit
    vector pointing AWAY from the nearest surface (the analytic gradient of an
    unsigned distance field, magnitude ~1). Both are world-space.

    Backend
    -------
    If :func:`configure_sdf_backend` supplied the d98 ``compute_sdf_for_point``,
    each query delegates to it (smooth-min union, matching the registration
    field) and uses its outward direction as the gradient. Otherwise a hard-min
    union over :func:`mesh_utils.closest_point_on_mesh` is used and the gradient
    is ``normalize(point - closest)``.
    """

    def __init__(self,
                 mesh_fns: "Dict[str, Any]",
                 blend_k: float = 1.0,
                 offset: float = 0.0,
                 point_sdf_fn: Optional[Callable[..., Tuple]] = None,
                 ) -> None:
        self.mesh_fns = mesh_fns
        self.blend_k = blend_k
        self.offset = offset
        self._point_sdf_fn = point_sdf_fn
        self.space = "world"
        self.signed = False  # unsigned distance to the union surface
        self.backend = "d98-smooth-min" if point_sdf_fn is not None else "hard-min"

    def closest(self, point: Sequence[float]
                ) -> Tuple[float, List[float], List[float], Optional[str]]:
        """Return ``(distance, closest_point, outward_unit, nearest_mesh)``.

        ``distance`` is unsigned distance to the union (minus ``self.offset`` if
        an offset was baked in; M4 uses ``offset=0`` and treats the offset as the
        iso value instead). ``outward_unit`` is the world-space gradient
        direction. Returns ``inf`` distance when no mesh is hittable.
        """
        if self._point_sdf_fn is not None:
            sdf_val, cp, outward, mesh = self._point_sdf_fn(
                list(point), self.mesh_fns, self.offset, self.blend_k)
            return (sdf_val, [cp[0], cp[1], cp[2]],
                    [outward[0], outward[1], outward[2]], mesh)

        best_dist = float("inf")
        best_cp = list(point)
        best_mesh: Optional[str] = None
        for mesh_name, mesh_fn in self.mesh_fns.items():
            if mesh_fn is None:
                continue
            cp, dist = mesh_utils.closest_point_on_mesh(mesh_fn, point)
            if dist < best_dist:
                best_dist = dist
                best_cp = cp
                best_mesh = mesh_name
        if best_mesh is None:
            return float("inf"), list(point), [0.0, 0.0, 1.0], None
        outward = mesh_utils.vec_sub(point, best_cp)
        length = mesh_utils.vec_length(outward)
        if length > 1e-8:
            outward = mesh_utils.vec_scale(outward, 1.0 / length)
        else:
            outward = [0.0, 0.0, 1.0]
        return best_dist - self.offset, best_cp, outward, best_mesh

    def evaluate(self, point: Sequence[float]) -> float:
        """Return the (unsigned) union distance ``phi(point)`` in world space."""
        return self.closest(point)[0]

    def evaluate_with_gradient(self, point: Sequence[float]
                               ) -> Tuple[float, List[float]]:
        """Return ``(phi(point), analytic_gradient)`` in a single query.

        The analytic gradient is the outward unit direction; using it avoids the
        6 extra distance queries a finite-difference gradient would need per
        step.
        """
        dist, _cp, outward, _mesh = self.closest(point)
        return dist, outward

    def gradient(self, point: Sequence[float]) -> List[float]:
        """Return the analytic (outward, unit) gradient of ``phi`` at ``point``."""
        return self.closest(point)[2]

    def nearest_mesh(self, point: Sequence[float]) -> Optional[str]:
        """Return the name of the nearest anatomical mesh (or ``None``)."""
        return self.closest(point)[3]


# =============================================================================
# M4.1  RESOLVE ANATOMICAL MESHES
# =============================================================================

def resolve_anatomical_meshes(anatomical_meshes: Optional[Sequence[str]] = None,
                              ) -> Dict[str, Any]:
    """Split a requested anatomical-mesh list into valid and missing names.

    Read-only. This module deliberately does NOT hard-code the project's
    ``INTERNAL_MESHES`` list -- that lives in d98. The d98 wrapper
    (:func:`detect_sdf_reference_skin_artifacts`) passes ``INTERNAL_MESHES`` in,
    so the list has exactly one source of truth.

    Parameters
    ----------
    anatomical_meshes:
        Iterable of mesh names to validate. If ``None`` a ``ValueError`` is
        raised, because the list must be supplied by the d98 wrapper (never
        duplicated here).

    Returns
    -------
    dict
        ``{"valid": [...], "missing": [...], "requested_count": int,
        "valid_count": int, "missing_count": int}`` with the valid names in the
        requested order (deduplicated).
    """
    if anatomical_meshes is None:
        raise ValueError(
            "resolve_anatomical_meshes: anatomical_meshes is None. The mesh list "
            "must be supplied by the d98 wrapper (INTERNAL_MESHES); it is not "
            "duplicated in artifact_detection.py.")

    seen = set()
    valid: List[str] = []
    missing: List[str] = []
    for name in anatomical_meshes:
        if name in seen:
            continue
        seen.add(name)
        if mesh_utils.mesh_exists(name):
            valid.append(name)
        else:
            missing.append(name)

    print("[artifact_detection][M4] anatomical meshes: {0} valid / {1} missing "
          "(of {2} requested)".format(len(valid), len(missing), len(seen)))
    if missing:
        preview = ", ".join(missing[:8])
        if len(missing) > 8:
            preview += ", ..."
        print("[artifact_detection][M4] missing meshes: {0}".format(preview))

    return {
        "valid": valid,
        "missing": missing,
        "requested_count": len(seen),
        "valid_count": len(valid),
        "missing_count": len(missing),
    }


def build_anatomy_sdf_query(anatomical_meshes: Sequence[str],
                            blend_k: float = 1.0,
                            offset: float = 0.0,
                            ) -> AnatomySDFQuery:
    """Build a reusable :class:`AnatomySDFQuery` over the given meshes.

    Builds one ``MFnMesh`` per VALID anatomical mesh (via the injected d98
    provider if configured, else :func:`mesh_utils.get_mesh_fn`) so acceleration
    data is created once per mesh and shared across all subsequent queries.
    Reuses the injected d98 ``compute_sdf_for_point`` when available so the
    distance field matches the registration exactly; otherwise uses a hard-min
    union of the same API 2.0 closest-point primitive.

    Parameters
    ----------
    anatomical_meshes:
        Mesh names (already resolved or raw; missing ones are skipped with a
        warning). ``offset`` is normally left at 0 for M4 -- the desired skin
        offset ``d0`` is passed to the projection as the iso value, not baked
        into the field.

    Returns
    -------
    AnatomySDFQuery
        A query whose ``evaluate``/``gradient`` return UNSIGNED distance and the
        outward unit direction, in WORLD space. Raises ``ValueError`` if no valid
        mesh could be loaded.
    """
    provider = _MESH_FN_PROVIDER if _MESH_FN_PROVIDER is not None else mesh_utils.get_mesh_fn

    mesh_fns: Dict[str, Any] = {}
    skipped: List[str] = []
    for name in anatomical_meshes:
        fn = provider(name)
        if fn is not None:
            mesh_fns[name] = fn
        else:
            skipped.append(name)

    if not mesh_fns:
        raise ValueError(
            "build_anatomy_sdf_query: could not load any anatomical MFnMesh "
            "(requested {0}); is Maya running and are the meshes present?".format(
                len(list(anatomical_meshes))))

    if skipped:
        print("[artifact_detection][M4] SDF query skipped {0} unloadable mesh(es)".format(
            len(skipped)))

    query = AnatomySDFQuery(mesh_fns, blend_k=blend_k, offset=offset,
                            point_sdf_fn=_POINT_SDF_FN)
    print("[artifact_detection][M4] built anatomy SDF query over {0} mesh(es) "
          "[backend={1}, signed={2}, space={3}]".format(
              len(mesh_fns), query.backend, query.signed, query.space))
    return query


# =============================================================================
# M4.3  SDF GRADIENT (finite-difference fallback)
# =============================================================================

def estimate_sdf_gradient(query: AnatomySDFQuery,
                          point: Sequence[float],
                          step: float = 0.01,
                          ) -> List[float]:
    """Estimate ``grad phi(point)`` by central finite differences (world space).

    The :class:`AnatomySDFQuery` already exposes an ANALYTIC gradient (the
    outward direction), which the projection uses by default. This function is
    the finite-difference fallback / cross-check requested by the spec, for
    backends that do not provide an analytic gradient::

        grad_x = (phi(x+h, y, z) - phi(x-h, y, z)) / (2h)

    (and likewise for y, z), all sampled in the SAME world space.

    Parameters
    ----------
    query:
        Distance query providing ``evaluate(point)``.
    point:
        World-space ``[x, y, z]``.
    step:
        Finite-difference half-step ``h`` (world units). Must be ``> 0``.

    Returns
    -------
    list[float]
        The estimated gradient ``[gx, gy, gz]``. Returns ``[0, 0, 0]`` if any
        sample is non-finite (a near-zero / degenerate gradient the caller must
        guard against).

    Raises
    ------
    ValueError
        If ``step <= 0``.
    """
    if step <= 0:
        raise ValueError("estimate_sdf_gradient: step must be > 0 (got {0})".format(step))

    px, py, pz = point[0], point[1], point[2]
    grad = [0.0, 0.0, 0.0]
    for axis in range(3):
        plus = [px, py, pz]
        minus = [px, py, pz]
        plus[axis] += step
        minus[axis] -= step
        d_plus = query.evaluate(plus)
        d_minus = query.evaluate(minus)
        if not (_is_finite(d_plus) and _is_finite(d_minus)):
            return [0.0, 0.0, 0.0]
        grad[axis] = (d_plus - d_minus) / (2.0 * step)
    return grad


# =============================================================================
# M4.4  PROJECT A POINT ONTO THE ISO-SURFACE phi(x) = target_offset
# =============================================================================

def project_point_to_iso_surface(point: Sequence[float],
                                 sdf_query: AnatomySDFQuery,
                                 target_offset: float,
                                 max_iterations: int = 10,
                                 tolerance: float = 1e-4,
                                 max_step: Optional[float] = None,
                                 gradient_step: Optional[float] = None,
                                 epsilon: float = 1e-8,
                                 ) -> Tuple[List[float], Dict[str, Any]]:
    """Iteratively project a world-space point onto ``phi(x) = target_offset``.

    Uses the damped Newton / gradient-descent-on-distance update::

        t_next = t - ((phi(t) - d0) / (||grad_phi(t)||^2 + epsilon)) * grad_phi(t)

    where ``d0 = target_offset``. The analytic (outward) gradient from
    ``sdf_query`` is used unless ``gradient_step`` is given, in which case a
    finite-difference gradient (:func:`estimate_sdf_gradient`) is used instead.

    Stops when ``abs(phi(point) - target_offset) <= tolerance``. Guards against a
    near-zero gradient, NaN/inf SDF values, and excessive movement (via
    ``max_step`` per iteration).

    Parameters
    ----------
    point:
        World-space start ``[x, y, z]`` (typically a current skin vertex).
    sdf_query:
        The anatomy distance query.
    target_offset:
        Desired distance ``d0`` from the anatomy (the iso value).
    max_iterations:
        Maximum Newton iterations.
    tolerance:
        Convergence tolerance on ``|phi - d0|`` (world units).
    max_step:
        Optional per-iteration clamp on the step length (world units).
    gradient_step:
        If given (``> 0``), use a central finite-difference gradient with this
        half-step; otherwise use the analytic gradient.
    epsilon:
        Small constant guarding the ``1 / ||grad||^2`` denominator.

    Returns
    -------
    (projected_point, convergence_info)
        ``projected_point`` is the world-space result (the ORIGINAL point,
        unchanged, on invalid input). ``convergence_info`` has keys:
        ``converged`` (bool), ``iterations`` (int), ``initial_distance`` (float),
        ``final_distance`` (float), ``total_displacement`` (float, ``||t - x||``)
        and ``failure_reason`` (str or ``None``).
    """
    origin = [float(point[0]), float(point[1]), float(point[2])]
    p = list(origin)

    initial_distance = sdf_query.evaluate(p)
    info: Dict[str, Any] = {
        "converged": False,
        "iterations": 0,
        "initial_distance": initial_distance,
        "final_distance": initial_distance,
        "total_displacement": 0.0,
        "failure_reason": None,
    }

    if not _is_finite(initial_distance) or initial_distance == float("inf"):
        info["failure_reason"] = "invalid_initial_sdf"
        return origin, info

    use_fd = gradient_step is not None
    if use_fd and gradient_step <= 0:
        raise ValueError("project_point_to_iso_surface: gradient_step must be > 0")

    for it in range(1, max_iterations + 1):
        info["iterations"] = it

        if use_fd:
            dist = sdf_query.evaluate(p)
            grad = estimate_sdf_gradient(sdf_query, p, step=gradient_step)
        else:
            dist, grad = sdf_query.evaluate_with_gradient(p)

        info["final_distance"] = dist

        if not _is_finite(dist):
            info["failure_reason"] = "nan_or_inf_sdf"
            break

        residual = dist - target_offset
        if abs(residual) <= tolerance:
            info["converged"] = True
            break

        if not _vec_is_finite(grad):
            info["failure_reason"] = "nan_or_inf_gradient"
            break

        grad_norm2 = grad[0] * grad[0] + grad[1] * grad[1] + grad[2] * grad[2]
        if grad_norm2 <= epsilon or math.sqrt(grad_norm2) < 1e-6:
            info["failure_reason"] = "near_zero_gradient"
            break

        scale = residual / (grad_norm2 + epsilon)
        step_vec = [-scale * grad[0], -scale * grad[1], -scale * grad[2]]
        step_len = mesh_utils.vec_length(step_vec)

        if not _is_finite(step_len):
            info["failure_reason"] = "nan_or_inf_step"
            break

        if max_step is not None and step_len > max_step and step_len > 1e-12:
            shrink = max_step / step_len
            step_vec = [step_vec[0] * shrink, step_vec[1] * shrink, step_vec[2] * shrink]

        p = [p[0] + step_vec[0], p[1] + step_vec[1], p[2] + step_vec[2]]

        if not _vec_is_finite(p):
            info["failure_reason"] = "nan_or_inf_position"
            p = list(origin)
            info["final_distance"] = initial_distance
            break

    if not info["converged"] and info["failure_reason"] is None:
        info["failure_reason"] = "max_iterations"

    info["total_displacement"] = mesh_utils.vec_length(mesh_utils.vec_sub(p, origin))
    return p, info


# =============================================================================
# M4.5  SDF-DERIVED TARGET POSITIONS (topology / correspondence preserved)
# =============================================================================

def compute_sdf_target_positions(skin_mesh: str,
                                 anatomical_meshes: Optional[Sequence[str]],
                                 target_offset: float,
                                 indices: Optional[List[int]] = None,
                                 max_iterations: int = 10,
                                 tolerance: float = 1e-4,
                                 max_projection_distance: Optional[float] = None,
                                 gradient_step: Optional[float] = None,
                                 epsilon: float = 1e-8,
                                 sdf_query: Optional[AnatomySDFQuery] = None,
                                 ) -> Dict[str, Any]:
    """Project skin vertices onto ``phi = target_offset`` (positions in memory).

    For each requested skin vertex ``x_i`` this computes an SDF-derived target
    ``t_i`` while PRESERVING the skin's vertex indexing and adjacency (no new
    mesh, no marching cubes). Skin positions are read ONCE and the anatomy SDF
    query is built ONCE (or reused if ``sdf_query`` is supplied).

    The skin mesh is NEVER modified and NO geometry is created. Vertices whose
    projection fails are listed in ``failed_indices`` and their ``target_positions``
    entry is set to the CURRENT position (a safe no-op), so downstream code can
    exclude them rather than consume an invalid projection.

    Parameters
    ----------
    skin_mesh:
        Skin mesh to read (not modified).
    anatomical_meshes:
        Anatomical mesh names (from the d98 wrapper). Ignored when ``sdf_query``
        is supplied; required otherwise.
    target_offset:
        Desired distance ``d0`` from the anatomy (world units).
    indices:
        Vertices to project; ``None`` projects all.
    max_iterations, tolerance, gradient_step, epsilon:
        Passed to :func:`project_point_to_iso_surface`.
    max_projection_distance:
        If given, also used as the per-iteration ``max_step`` clamp AND as an
        absolute cap: a converged vertex whose net displacement exceeds it is
        moved to ``failed_indices`` (implausible projection).
    sdf_query:
        Optional pre-built query to reuse (avoids rebuilding acceleration data).

    Returns
    -------
    dict
        Keys: ``target_positions`` ({index: [x,y,z]}), ``initial_sdf_values``,
        ``final_sdf_values``, ``projection_displacements``, ``converged_indices``,
        ``failed_indices``, ``nearest_anatomy`` ({index: mesh_name}),
        ``valid_anatomical_meshes``, ``missing_anatomical_meshes``, and
        ``statistics``.
    """
    verts = mesh_utils.get_mesh_vertices(skin_mesh)
    if not verts:
        print("[artifact_detection][M4] skin mesh '{0}' missing/empty".format(skin_mesh))
        return _empty_target_data()

    n = len(verts)
    if indices is None:
        target_indices = list(range(n))
    else:
        target_indices = [i for i in indices if 0 <= i < n]

    if sdf_query is None:
        resolved = resolve_anatomical_meshes(anatomical_meshes)
        valid = resolved["valid"]
        missing = resolved["missing"]
        if not valid:
            print("[artifact_detection][M4] no valid anatomical meshes; cannot project")
            data = _empty_target_data()
            data["missing_anatomical_meshes"] = missing
            return data
        sdf_query = build_anatomy_sdf_query(valid)
    else:
        valid = list(sdf_query.mesh_fns.keys())
        missing = []

    max_step = max_projection_distance

    target_positions: Dict[int, List[float]] = {}
    initial_sdf: Dict[int, float] = {}
    final_sdf: Dict[int, float] = {}
    displacements: Dict[int, float] = {}
    nearest_anatomy: Dict[int, Optional[str]] = {}
    converged_indices: List[int] = []
    failed_indices: List[int] = []

    for i in target_indices:
        x_i = verts[i]
        t_i, conv = project_point_to_iso_surface(
            x_i, sdf_query, target_offset,
            max_iterations=max_iterations, tolerance=tolerance,
            max_step=max_step, gradient_step=gradient_step, epsilon=epsilon)

        initial_sdf[i] = conv["initial_distance"]
        final_sdf[i] = conv["final_distance"]
        disp = conv["total_displacement"]
        displacements[i] = disp

        too_far = (max_projection_distance is not None
                   and _is_finite(disp) and disp > max_projection_distance)

        if conv["converged"] and not too_far and _vec_is_finite(t_i):
            target_positions[i] = t_i
            converged_indices.append(i)
            nearest_anatomy[i] = sdf_query.nearest_mesh(t_i)
        else:
            # Failed / implausible: keep the CURRENT position as a safe no-op and
            # flag the vertex so scoring can exclude it (never use an invalid
            # projection silently).
            target_positions[i] = list(x_i)
            failed_indices.append(i)
            nearest_anatomy[i] = None

    disp_vals = [displacements[i] for i in converged_indices if _is_finite(displacements[i])]
    stats = {
        "requested": len(target_indices),
        "converged": len(converged_indices),
        "failed": len(failed_indices),
        "convergence_rate": (len(converged_indices) / len(target_indices)
                             if target_indices else 0.0),
        "displacement_mean": (sum(disp_vals) / len(disp_vals)) if disp_vals else 0.0,
        "displacement_median": _median(disp_vals),
        "displacement_max": max(disp_vals) if disp_vals else 0.0,
        "target_offset": target_offset,
        "signed_distance": bool(sdf_query.signed),
        "backend": sdf_query.backend,
    }

    return {
        "target_positions": target_positions,
        "initial_sdf_values": initial_sdf,
        "final_sdf_values": final_sdf,
        "projection_displacements": displacements,
        "converged_indices": sorted(converged_indices),
        "failed_indices": sorted(failed_indices),
        "nearest_anatomy": nearest_anatomy,
        "valid_anatomical_meshes": valid,
        "missing_anatomical_meshes": missing,
        "statistics": stats,
    }


def _empty_target_data() -> Dict[str, Any]:
    return {
        "target_positions": {},
        "initial_sdf_values": {},
        "final_sdf_values": {},
        "projection_displacements": {},
        "converged_indices": [],
        "failed_indices": [],
        "nearest_anatomy": {},
        "valid_anatomical_meshes": [],
        "missing_anatomical_meshes": [],
        "statistics": {"requested": 0, "converged": 0, "failed": 0,
                       "convergence_rate": 0.0, "displacement_mean": 0.0,
                       "displacement_median": 0.0, "displacement_max": 0.0},
    }


# =============================================================================
# M4.6  VALIDATE THE SDF TARGET
# =============================================================================

def validate_sdf_target(skin_mesh: str,
                        target_data: Dict[str, Any],
                        max_failed_fraction: float = 0.05,
                        max_projection_distance: Optional[float] = None,
                        ) -> Dict[str, Any]:
    """Validate an SDF target produced by :func:`compute_sdf_target_positions`.

    Checks that target positions exist for the expected vertices, reports the
    convergence rate, and flags excessive / implausible displacements. This does
    NOT assert anatomical correctness -- convergence to ``phi = d0`` only means
    the projection reached the requested iso-distance, not that the resulting
    surface is anatomically right (see the module limitations).

    Parameters
    ----------
    skin_mesh:
        Skin mesh the target was built from (read-only).
    target_data:
        The dict returned by :func:`compute_sdf_target_positions`.
    max_failed_fraction:
        Validation fails (``ok=False``) if the failed fraction exceeds this.
    max_projection_distance:
        If given, count converged vertices whose displacement exceeds it as
        implausible and include them in the report / warnings.

    Returns
    -------
    dict
        ``{"ok": bool, "vertex_count", "requested", "converged", "failed",
        "convergence_rate", "failed_fraction", "implausible_count",
        "displacement_max", "warnings": [...]}``.
    """
    stats = target_data.get("statistics", {})
    requested = stats.get("requested", 0)
    converged = stats.get("converged", 0)
    failed = stats.get("failed", 0)
    displacements = target_data.get("projection_displacements", {})

    warnings: List[str] = []
    failed_fraction = (failed / requested) if requested else 1.0

    implausible = 0
    if max_projection_distance is not None:
        implausible = sum(1 for d in displacements.values()
                          if _is_finite(d) and d > max_projection_distance)
        if implausible:
            warnings.append(
                "{0} vertex/vertices exceed max_projection_distance={1}".format(
                    implausible, max_projection_distance))

    n_expected = mesh_utils.get_vertex_count(skin_mesh)
    missing_targets = requested - len(target_data.get("target_positions", {}))
    if missing_targets > 0:
        warnings.append("{0} requested vertices have no target position".format(
            missing_targets))

    ok = (requested > 0) and (failed_fraction <= max_failed_fraction)
    if not ok and requested > 0:
        warnings.append(
            "failed fraction {0:.1%} exceeds allowed {1:.1%}".format(
                failed_fraction, max_failed_fraction))
    if requested == 0:
        warnings.append("no vertices were requested / projected")

    report = {
        "ok": ok,
        "vertex_count": n_expected,
        "requested": requested,
        "converged": converged,
        "failed": failed,
        "convergence_rate": stats.get("convergence_rate", 0.0),
        "failed_fraction": failed_fraction,
        "implausible_count": implausible,
        "displacement_max": stats.get("displacement_max", 0.0),
        "warnings": warnings,
    }
    print("[artifact_detection][M4] target validation: {0} ({1}/{2} converged, "
          "{3:.1%} failed){4}".format(
              "OK" if ok else "WARN", converged, requested, failed_fraction,
              "" if not warnings else " | " + "; ".join(warnings)))
    return report


# =============================================================================
# M4.7  SDF-REFERENCE LAPLACIAN SCORES
# =============================================================================

def compute_sdf_reference_laplacian_scores(skin_mesh: str,
                                           anatomical_meshes: Optional[Sequence[str]],
                                           target_offset: float,
                                           indices: Optional[List[int]] = None,
                                           normalize: bool = True,
                                           max_iterations: int = 10,
                                           tolerance: float = 1e-4,
                                           max_projection_distance: Optional[float] = None,
                                           gradient_step: Optional[float] = None,
                                           epsilon: float = 1e-8,
                                           sdf_query: Optional[AnatomySDFQuery] = None,
                                           target_data: Optional[Dict[str, Any]] = None,
                                           debug_laplacians: bool = False,
                                           ) -> Dict[str, Any]:
    """Score vertices by current-vs-SDF-target local-shape disagreement (M4).

    For each scored vertex ``i`` with shared 1-ring ``N(i)``::

        L_current(i)    = x_i - mean(x_j for j in N(i))
        L_sdf_target(i) = t_i - mean(t_j for j in N(i))
        score_i         = || L_current(i) - L_sdf_target(i) ||

    The FULL Laplacian VECTORS are compared (not magnitudes), so anatomy present
    in both the current skin and the SDF target cancels and only genuine local
    deviation survives. If ``normalize`` is True the score is divided by the
    current mesh's local edge scale ``mean(||x_i - x_j||) + epsilon``.

    Vertices whose target projection FAILED are excluded from scoring (they are
    reported in ``target_data['failed_indices']``); the skin is never modified.

    Parameters
    ----------
    skin_mesh, anatomical_meshes, target_offset, indices, max_iterations,
    tolerance, max_projection_distance, gradient_step, epsilon:
        As in :func:`compute_sdf_target_positions`.
    normalize:
        Scale-relative normalization (default True).
    sdf_query, target_data:
        Optional pre-built query / pre-computed target to reuse.
    debug_laplacians:
        If True, also return the per-vertex current/target Laplacian vectors.

    Returns
    -------
    dict
        ``{"scores", "raw_scores", "current_laplacians", "target_laplacians",
        "target_data", "score_statistics"}``. ``scores`` are the (optionally
        normalized) scores actually used; ``raw_scores`` are always the
        un-normalized magnitudes.
    """
    current = mesh_utils.get_mesh_vertices(skin_mesh)
    if not current:
        print("[artifact_detection][M4] skin mesh '{0}' missing/empty".format(skin_mesh))
        return {"scores": {}, "raw_scores": {}, "current_laplacians": {},
                "target_laplacians": {}, "target_data": _empty_target_data(),
                "score_statistics": summarize_scores({})}

    neighbors = mesh_utils.get_vertex_neighbors(skin_mesh)
    if not neighbors:
        print("[artifact_detection][M4] could not read topology for '{0}'".format(skin_mesh))
        return {"scores": {}, "raw_scores": {}, "current_laplacians": {},
                "target_laplacians": {}, "target_data": _empty_target_data(),
                "score_statistics": summarize_scores({})}

    if target_data is None:
        target_data = compute_sdf_target_positions(
            skin_mesh, anatomical_meshes, target_offset, indices=indices,
            max_iterations=max_iterations, tolerance=tolerance,
            max_projection_distance=max_projection_distance,
            gradient_step=gradient_step, epsilon=epsilon, sdf_query=sdf_query)

    target_positions = target_data["target_positions"]
    failed = set(target_data["failed_indices"])

    # Score only vertices with a VALID target projection.
    scored_indices = [i for i in target_positions.keys() if i not in failed]

    scores: Dict[int, float] = {}
    raw_scores: Dict[int, float] = {}
    current_laps: Dict[int, List[float]] = {}
    target_laps: Dict[int, List[float]] = {}

    for i in scored_indices:
        nbrs = neighbors[i]
        if not nbrs:
            scores[i] = 0.0
            raw_scores[i] = 0.0
            continue

        # A neighbour without a valid target cannot contribute a target centroid;
        # skip such vertices rather than mixing current + target coordinates.
        if any(j not in target_positions for j in nbrs):
            continue

        cen_cur = mesh_utils.vec_mean([current[j] for j in nbrs])
        cen_tgt = mesh_utils.vec_mean([target_positions[j] for j in nbrs])
        lap_cur = mesh_utils.vec_sub(current[i], cen_cur)
        lap_tgt = mesh_utils.vec_sub(target_positions[i], cen_tgt)
        diff = mesh_utils.vec_sub(lap_cur, lap_tgt)
        raw = mesh_utils.vec_length(diff)
        raw_scores[i] = raw

        score = raw
        if normalize:
            local_scale = sum(
                mesh_utils.vec_length(mesh_utils.vec_sub(current[i], current[j]))
                for j in nbrs) / len(nbrs)
            score = raw / (local_scale + epsilon)
        scores[i] = score

        if debug_laplacians:
            current_laps[i] = lap_cur
            target_laps[i] = lap_tgt

    return {
        "scores": scores,
        "raw_scores": raw_scores,
        "current_laplacians": current_laps,
        "target_laplacians": target_laps,
        "target_data": target_data,
        "score_statistics": summarize_scores(scores),
    }


# =============================================================================
# M4.8  SKIN BOUNDARY VERTICES (true topology + buffer rings)
# =============================================================================

def find_skin_boundary_vertices(mesh_name: str,
                                buffer_rings: int = 0,
                                ) -> List[int]:
    """Return true topological boundary vertices, optionally grown by rings.

    Reuses :func:`mesh_utils.get_boundary_vertices` (edges bordering a single
    face) so the openings that caused M2's false positives -- eye openings, lips
    / mouth opening, nostrils, the neck opening, and the outer mesh border -- are
    identified exactly, not guessed by bounding box. ``buffer_rings`` grows the
    boundary set inward by that many topological rings so vertices NEAR an
    opening can also be excluded.

    Parameters
    ----------
    mesh_name:
        Mesh to inspect (not modified).
    buffer_rings:
        Extra rings of neighbours to include around each boundary vertex.

    Returns
    -------
    list[int]
        Sorted boundary (and buffer) vertex indices.
    """
    boundary = mesh_utils.get_boundary_vertices(mesh_name)
    if not boundary:
        return []
    if buffer_rings > 0:
        neighbors = mesh_utils.get_vertex_neighbors(mesh_name)
        grown = mesh_utils.grow_indices(neighbors, list(boundary), rings=buffer_rings)
        return sorted(grown)
    return sorted(boundary)


def _connected_components(indices: Sequence[int],
                          neighbors: Sequence[Sequence[int]],
                          ) -> List[List[int]]:
    """Split ``indices`` into connected components using mesh adjacency."""
    index_set = set(indices)
    visited = set()
    components: List[List[int]] = []
    for start in indices:
        if start in visited:
            continue
        stack = [start]
        visited.add(start)
        comp = []
        while stack:
            v = stack.pop()
            comp.append(v)
            if 0 <= v < len(neighbors):
                for nb in neighbors[v]:
                    if nb in index_set and nb not in visited:
                        visited.add(nb)
                        stack.append(nb)
        components.append(sorted(comp))
    return components


# =============================================================================
# M4.9  DETECTION PIPELINE
# =============================================================================

def detect_sdf_reference_artifacts(skin_mesh: str,
                                   anatomical_meshes: Optional[Sequence[str]],
                                   target_offset: float,
                                   method: str = "percentile",
                                   percentile: float = 97.5,
                                   threshold: float = 2.5,
                                   normalize: bool = True,
                                   exclude_boundaries: bool = True,
                                   boundary_buffer_rings: int = 1,
                                   min_component_size: int = 3,
                                   final_growth_rings: int = 1,
                                   reapply_boundary_exclusion_after_growth: bool = True,
                                   max_iterations: int = 10,
                                   tolerance: float = 1e-4,
                                   max_projection_distance: Optional[float] = None,
                                   min_score: float = 0.0,
                                   gradient_step: Optional[float] = None,
                                   epsilon: float = 1e-8,
                                   select: bool = True,
                                   ) -> Tuple[List[int], Dict[str, Any]]:
    """M4 end-to-end SDF-reference artifact detection (DETECTION ONLY).

    Pipeline: validate inputs -> resolve anatomy -> build SDF query -> project
    skin onto ``phi = target_offset`` -> validate convergence/displacement ->
    reference-Laplacian scores -> outliers via :func:`detect_outliers` -> remove
    true boundary + buffer rings -> connected components -> drop small components
    -> grow survivors -> (optionally) re-exclude boundaries after growth ->
    select. The skin and anatomy meshes are NEVER modified and NO geometry is
    created.

    Returns
    -------
    (final_indices, report)
        ``final_indices`` are the selected artifact vertices; ``report`` is a
        detailed dict (see the keys built below).
    """
    warnings: List[str] = []

    report: Dict[str, Any] = {
        "skin_mesh": skin_mesh,
        "target_offset": target_offset,
        "method": method,
        "normalize": normalize,
        "warnings": warnings,
    }

    if not mesh_utils.mesh_exists(skin_mesh):
        print("[artifact_detection][M4] skin mesh '{0}' does not exist".format(skin_mesh))
        warnings.append("skin mesh missing")
        return [], report

    if target_offset <= 0:
        warnings.append("target_offset <= 0 is non-physical for a skin offset")

    resolved = resolve_anatomical_meshes(anatomical_meshes)
    report["valid_anatomical_count"] = resolved["valid_count"]
    report["missing_anatomical_count"] = resolved["missing_count"]
    report["missing_anatomical_meshes"] = resolved["missing"]
    if not resolved["valid"]:
        warnings.append("no valid anatomical meshes")
        return [], report

    sdf_query = build_anatomy_sdf_query(resolved["valid"])
    report["sdf_signed"] = sdf_query.signed
    report["sdf_backend"] = sdf_query.backend
    if not sdf_query.signed:
        warnings.append("SDF is UNSIGNED; projection assumes skin lies outside "
                        "anatomy and moves along the outward gradient")

    score_data = compute_sdf_reference_laplacian_scores(
        skin_mesh, resolved["valid"], target_offset, normalize=normalize,
        max_iterations=max_iterations, tolerance=tolerance,
        max_projection_distance=max_projection_distance,
        gradient_step=gradient_step, epsilon=epsilon, sdf_query=sdf_query)

    target_data = score_data["target_data"]
    tstats = target_data["statistics"]

    validation = validate_sdf_target(
        skin_mesh, target_data,
        max_projection_distance=max_projection_distance)
    warnings.extend(validation["warnings"])

    report["projection_converged"] = tstats["converged"]
    report["projection_requested"] = tstats["requested"]
    report["projection_convergence_pct"] = 100.0 * tstats["convergence_rate"]
    report["projection_failed"] = tstats["failed"]
    report["projection_displacement_mean"] = tstats["displacement_mean"]
    report["projection_displacement_median"] = tstats["displacement_median"]
    report["projection_displacement_max"] = tstats["displacement_max"]

    scores = score_data["scores"]
    sstats = score_data["score_statistics"]
    report["score_mean"] = sstats["mean"]
    report["score_median"] = sstats["median"]
    report["score_std"] = sstats["std"]
    report["score_max"] = sstats["max"]
    report["scored_vertices"] = sstats["count"]

    if sstats["count"] == 0:
        warnings.append("no scores computed (all projections failed?)")
        return [], report

    detected = detect_outliers(scores, method=method, threshold=threshold,
                               percentile=percentile, min_score=min_score)
    report["raw_outlier_count"] = len(detected)

    neighbors = mesh_utils.get_vertex_neighbors(skin_mesh)

    # 8. Remove true boundary vertices + requested buffer rings.
    if exclude_boundaries:
        boundary = set(find_skin_boundary_vertices(
            skin_mesh, buffer_rings=boundary_buffer_rings))
        before_boundary = len(detected)
        detected = [i for i in detected if i not in boundary]
        report["boundary_vertex_count"] = len(boundary)
        report["boundary_excluded_count"] = before_boundary - len(detected)
    else:
        report["boundary_vertex_count"] = 0
        report["boundary_excluded_count"] = 0

    # 9-10. Connected components; drop those smaller than min_component_size.
    components = _connected_components(detected, neighbors)
    report["components_before_filter"] = len(components)
    kept = [c for c in components if len(c) >= min_component_size]
    report["components_after_filter"] = len(kept)
    filtered = sorted(idx for c in kept for idx in c)
    report["count_after_component_filter"] = len(filtered)

    # 11. Grow survivors.
    report["count_before_growth"] = len(filtered)
    if final_growth_rings > 0 and filtered:
        grown = mesh_utils.grow_indices(neighbors, filtered, rings=final_growth_rings)
    else:
        grown = sorted(filtered)

    # 12. Re-apply boundary exclusion after growth (growth can re-touch a rim).
    if exclude_boundaries and reapply_boundary_exclusion_after_growth and grown:
        boundary = set(find_skin_boundary_vertices(
            skin_mesh, buffer_rings=boundary_buffer_rings))
        grown = [i for i in grown if i not in boundary]
    report["count_after_growth"] = len(grown)

    final_indices = sorted(grown)
    report["final_count"] = len(final_indices)

    # Highest-scoring vertices among the final set (fallback: overall top).
    ranked_source = final_indices if final_indices else list(scores.keys())
    ranked = sorted(ranked_source, key=lambda i: scores.get(i, 0.0), reverse=True)
    report["highest_scoring_indices"] = ranked[:10]

    crit = ("z-score >= {0}".format(threshold) if method == "zscore"
            else "top {0:.1f}% (percentile >= {1})".format(100.0 - percentile, percentile))
    print("[artifact_detection][M4] '{0}' d0={1}: scored {2} verts | "
          "mean={3:.4f} median={4:.4f} std={5:.4f} max={6:.4f}".format(
              skin_mesh, target_offset, sstats["count"], sstats["mean"],
              sstats["median"], sstats["std"], sstats["max"]))
    print("[artifact_detection][M4] projection {0}/{1} converged ({2:.1f}%), "
          "{3} failed; disp mean={4:.3f} max={5:.3f}".format(
              tstats["converged"], tstats["requested"],
              report["projection_convergence_pct"], tstats["failed"],
              tstats["displacement_mean"], tstats["displacement_max"]))
    print("[artifact_detection][M4] flagged {0} by {1}; boundary-excluded {2}; "
          "{3}->{4} components; final {5} verts (+{6} ring(s))".format(
              report["raw_outlier_count"], crit,
              report["boundary_excluded_count"],
              report["components_before_filter"], report["components_after_filter"],
              report["final_count"], final_growth_rings))

    if select and final_indices:
        select_vertices(skin_mesh, final_indices)
        print("[artifact_detection][M4] selected {0} vertices for inspection "
              "(mesh unchanged)".format(len(final_indices)))

    return final_indices, report


# =============================================================================
# M4.10  OPTIONAL DEBUG TARGET MESH (VISUALIZATION ONLY -- opt in)
# =============================================================================

def create_sdf_target_debug_mesh(skin_mesh: str,
                                 target_positions: Dict[int, List[float]],
                                 name: str = "skin_sdf_target_debug",
                                 overwrite: bool = False,
                                 ) -> Optional[str]:
    """Create a DUPLICATE of the skin displaced to the SDF target (DEBUG ONLY).

    VISUALIZATION / DEBUG helper. Duplicates ``skin_mesh`` (so the debug mesh has
    IDENTICAL topology and vertex order), then writes the projected
    ``target_positions`` onto the duplicate. The ORIGINAL skin is never touched.
    This is the ONLY M4 function that creates geometry and it is never called
    automatically.

    Parameters
    ----------
    skin_mesh:
        Source skin mesh to duplicate (not modified).
    target_positions:
        ``{index: [x, y, z]}`` from :func:`compute_sdf_target_positions`. Indices
        absent from the dict keep their duplicated (current) position.
    name:
        Name for the debug duplicate.
    overwrite:
        If a node called ``name`` already exists, this function REFUSES unless
        ``overwrite=True`` (in which case the existing node is deleted first).

    Returns
    -------
    str or None
        The created node name, or ``None`` on failure / refusal / outside Maya.
    """
    try:
        import maya.cmds as cmds  # lazy: keep module importable outside Maya
    except ImportError:
        print("[artifact_detection][M4] not running inside Maya; cannot create debug mesh")
        return None

    if not mesh_utils.mesh_exists(skin_mesh):
        print("[artifact_detection][M4] cannot duplicate: '{0}' missing".format(skin_mesh))
        return None

    if cmds.objExists(name):
        if not overwrite:
            print("[artifact_detection][M4] '{0}' already exists; pass overwrite=True "
                  "to replace it (refusing by default)".format(name))
            return None
        cmds.delete(name)

    dup = cmds.duplicate(skin_mesh, name=name)
    debug_name = dup[0] if dup else None
    if not debug_name:
        print("[artifact_detection][M4] duplicate failed")
        return None

    verts = mesh_utils.get_mesh_vertices(debug_name)
    for i, pos in target_positions.items():
        if 0 <= i < len(verts) and _vec_is_finite(pos):
            verts[i] = list(pos)
    mesh_utils.set_mesh_vertices(debug_name, verts)

    print("[artifact_detection][M4] created DEBUG target mesh '{0}' ({1} verts "
          "displaced). Original '{2}' unchanged.".format(
              debug_name, len(target_positions), skin_mesh))
    return debug_name


# =============================================================================
# M4.11  COMPARE M2 vs M4
# =============================================================================

def compare_m2_and_m4(skin_mesh: str,
                      provided_target_mesh: str,
                      anatomical_meshes: Optional[Sequence[str]],
                      target_offset: float,
                      percentile: float = 97.5,
                      method: str = "percentile",
                      threshold: float = 2.5,
                      normalize: bool = True,
                      exclude_boundaries: bool = True,
                      boundary_buffer_rings: int = 1,
                      min_component_size: int = 3,
                      final_growth_rings: int = 1,
                      max_iterations: int = 10,
                      tolerance: float = 1e-4,
                      max_projection_distance: Optional[float] = None,
                      ) -> Dict[str, Any]:
    """Run M2 and M4 (WITHOUT selecting) and compare their detected sets.

    Neither detector selects in the viewport here (so the two selections do not
    fight); use :func:`select_vertices` afterwards to view whichever set you
    want. Neither mesh is modified.

    Returns
    -------
    dict
        ``{"m2_indices", "m4_indices", "overlap", "only_m2", "only_m4",
        "m2_count", "m4_count", "overlap_count", "only_m2_count",
        "only_m4_count", "jaccard", "m4_report"}``.
    """
    m2_indices, _m2_stats = detect_reference_irregular_region(
        skin_mesh, provided_target_mesh, method=method, threshold=threshold,
        percentile=percentile, normalize=normalize,
        rings=final_growth_rings, select=False)

    m4_indices, m4_report = detect_sdf_reference_artifacts(
        skin_mesh, anatomical_meshes, target_offset, method=method,
        percentile=percentile, threshold=threshold, normalize=normalize,
        exclude_boundaries=exclude_boundaries,
        boundary_buffer_rings=boundary_buffer_rings,
        min_component_size=min_component_size,
        final_growth_rings=final_growth_rings, max_iterations=max_iterations,
        tolerance=tolerance, max_projection_distance=max_projection_distance,
        select=False)

    s2, s4 = set(m2_indices), set(m4_indices)
    overlap = sorted(s2 & s4)
    only_m2 = sorted(s2 - s4)
    only_m4 = sorted(s4 - s2)
    union = s2 | s4
    jaccard = (len(overlap) / len(union)) if union else 0.0

    print("[artifact_detection][M2vM4] M2={0}  M4={1}  overlap={2}  "
          "only-M2={3}  only-M4={4}  Jaccard={5:.3f}".format(
              len(s2), len(s4), len(overlap), len(only_m2), len(only_m4), jaccard))

    return {
        "m2_indices": sorted(s2),
        "m4_indices": sorted(s4),
        "overlap": overlap,
        "only_m2": only_m2,
        "only_m4": only_m4,
        "m2_count": len(s2),
        "m4_count": len(s4),
        "overlap_count": len(overlap),
        "only_m2_count": len(only_m2),
        "only_m4_count": len(only_m4),
        "jaccard": jaccard,
        "m4_report": m4_report,
    }


# =============================================================================
# M4.  SKIN-TO-ANATOMY SDF DISTRIBUTION (choose target_offset from scene scale)
# =============================================================================

def summarize_skin_sdf_values(skin_mesh: str,
                              anatomical_meshes: Optional[Sequence[str]],
                              sample_indices: Optional[List[int]] = None,
                              sdf_query: Optional[AnatomySDFQuery] = None,
                              ) -> Dict[str, Any]:
    """Summarize current skin-vertex distances to the anatomy (no projection).

    Reports the distribution of ``phi(x_i)`` over the (sampled) skin vertices so
    a sensible global ``target_offset`` can be chosen FROM THE SCENE'S OWN SCALE
    rather than guessed. Read-only; no geometry is created.

    Parameters
    ----------
    skin_mesh:
        Skin mesh to read.
    anatomical_meshes:
        Anatomical mesh names (ignored if ``sdf_query`` supplied).
    sample_indices:
        Optional subset of skin vertices to sample (``None`` = all).
    sdf_query:
        Optional pre-built query to reuse.

    Returns
    -------
    dict
        ``{"count", "mean", "median", "std", "min", "max",
        "percentiles": {5, 25, 50, 75, 95}, "signed", "backend"}`` (all world
        units). Empty-safe.
    """
    verts = mesh_utils.get_mesh_vertices(skin_mesh)
    if not verts:
        print("[artifact_detection][M4] skin mesh '{0}' missing/empty".format(skin_mesh))
        return {"count": 0}

    n = len(verts)
    if sample_indices is None:
        idxs = list(range(n))
    else:
        idxs = [i for i in sample_indices if 0 <= i < n]

    if sdf_query is None:
        resolved = resolve_anatomical_meshes(anatomical_meshes)
        if not resolved["valid"]:
            print("[artifact_detection][M4] no valid anatomical meshes")
            return {"count": 0}
        sdf_query = build_anatomy_sdf_query(resolved["valid"])

    values: List[float] = []
    for i in idxs:
        d = sdf_query.evaluate(verts[i])
        if _is_finite(d) and d != float("inf"):
            values.append(d)

    if not values:
        print("[artifact_detection][M4] no finite SDF values sampled")
        return {"count": 0}

    count = len(values)
    mean = sum(values) / count
    ordered = sorted(values)
    result = {
        "count": count,
        "mean": mean,
        "median": _median(values),
        "std": _std(values, mean),
        "min": ordered[0],
        "max": ordered[-1],
        "percentiles": {
            5: _percentile(ordered, 5),
            25: _percentile(ordered, 25),
            50: _percentile(ordered, 50),
            75: _percentile(ordered, 75),
            95: _percentile(ordered, 95),
        },
        "signed": bool(sdf_query.signed),
        "backend": sdf_query.backend,
    }
    p = result["percentiles"]
    print("[artifact_detection][M4] skin->anatomy distance over {0} verts | "
          "mean={1:.3f} median={2:.3f} std={3:.3f} min={4:.3f} max={5:.3f}".format(
              count, result["mean"], result["median"], result["std"],
              result["min"], result["max"]))
    print("[artifact_detection][M4]   percentiles p5={0:.3f} p25={1:.3f} "
          "p50={2:.3f} p75={3:.3f} p95={4:.3f} (unsigned distance, world units)".format(
              p[5], p[25], p[50], p[75], p[95]))
    print("[artifact_detection][M4]   suggested global target_offset candidates: "
          "median={0:.3f}, p25={1:.3f}, p75={2:.3f}".format(p[50], p[25], p[75]))
    return result
