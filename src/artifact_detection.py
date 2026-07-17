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

from typing import Dict, List, Optional, Sequence, Set, Tuple

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
