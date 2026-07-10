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
