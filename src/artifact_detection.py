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

from typing import Dict, List, Optional, Tuple

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
