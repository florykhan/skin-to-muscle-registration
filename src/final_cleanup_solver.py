"""
final_cleanup_solver.py
========================
Unified, automatic, iterative cleanup solver for post-registration skin.

This module is the FINAL correction stage the M1-M5 detection research and the
V2 anatomy-supported intersection repair were building toward. It is
orchestration + a small amount of new force/region logic; it does NOT
reimplement anything that already exists and is considered correct:

* detection            -- :func:`artifact_detection.detect_unified_artifacts` (M5,
  UNCHANGED). M3/M4 provenance (m3_only / m4_only / overlap) is read from the M5
  report and used to choose a correction force per vertex; detection thresholds
  are never touched here.
* anatomy-derived target -- :func:`artifact_detection.compute_sdf_target_positions`
  (the SAME SDF projection M4 already uses) supplies the "gentle attraction
  toward anatomy" signal for m4_only / overlap vertices. Computed ONCE per run
  (anatomy does not move), not re-solved every outer iteration.
* exact clearance       -- :func:`anatomy_constraint.enforce_anatomy_clearance` /
  :func:`anatomy_constraint.compute_clearance_floors` (baseline-aware; a vertex
  that already sat closer than the nominal floor is never forced outward).
* the ground-truth safety test -- :func:`anatomy_constraint.analyze_skin_anatomy_intersections`
  (real triangle/triangle skin-vs-anatomy intersection; UNCHANGED).
* repair                -- :func:`anatomy_constraint.resolve_skin_anatomy_intersections`
  (the broad, anatomy-supported, harmonic-displacement-field patch repair; V1
  per-vertex normal pushing is NOT used here). This function already partitions
  a repair request into topology-connected components internally and solves
  each with its own local step size, which is exactly the "no single global
  alpha" property this solver requires -- reused as-is, not rebuilt.
* mesh I/O / metrics    -- :mod:`mesh_utils`, :mod:`metrics_utils`.

What IS new here (the actual gap this module fills):

* a graded active region (artifact core -> fairing band -> transition band ->
  untouched exterior) instead of a single hard-edged vertex set;
* a per-vertex, provenance-weighted combination of THREE forces (local fairing,
  anatomy-target attraction, rest-shape restraint) instead of one indiscriminate
  Laplacian pass over the whole M5 region;
* a per-vertex trust-region step cap, so one tight vertex can never throttle the
  rest of the region the way a single shared line-search scalar did in
  :func:`smoothing_utils.constrained_smooth_mesh_region`;
* a "propose everywhere, repair only what actually broke, accept" outer loop
  instead of "reject the whole batch if anything is unsafe";
* multi-signal convergence with patience, so ``max_iterations`` is a safety cap,
  not the definition of "done" -- running longer than necessary should reach the
  same fixed point, not keep shrinking the face;
* a PHASE 0 (:func:`_run_initial_feasibility`) that repairs every pre-existing
  surface intersection and captures the resulting anatomy-VALID mesh as the
  rest/positional reference BEFORE any fairing/M4/shape force is computed.
  Earlier versions of this module captured the rest reference from the raw,
  unrepaired registered mesh; since that mesh could itself be anatomically
  invalid wherever M5/the intersection scan had flagged a problem, the
  shape-restraint force was pulling freshly-repaired geometry straight back
  toward the position that had just been repaired away -- a fair -> repair ->
  fair -> repair cycle that never converges to a FAIR surface, only to a
  perpetually-defended one. Phase 0 is what makes "rest position" and "valid
  position" the same thing for every vertex the solver touches;
* region-dependent shape restraint (:func:`shape_weight_from_anchor`), M4
  force decay once its improvement plateaus (:func:`decay_m4_weight`), and
  oscillation detection (:func:`update_oscillation_tracker`) that damps a
  vertex's forces once it is repeatedly repaired, instead of letting it fight
  the same battle for hundreds of iterations;
* roughness is now a real ACCEPTANCE/convergence target (bounded by a ceiling
  derived from the mesh's own original roughness), not just a number that gets
  logged while the loop happily continues regardless of what it does;
* every accepted iteration validates and absorbs the FULL footprint the broad
  repair actually touched (which can extend past the pre-repair active set via
  its own ``repair_blend_rings`` growth), not just the vertices the solver
  already considered active.

Historical milestones (M1-M5), the legacy V1/V2 repair modes, and the
strict / repair_after_batch / off smoothing safety modes in
:mod:`smoothing_utils` are all left exactly as they were, for reproducibility
and A/B comparison. This module adds a NEW, separate path; it changes nothing
in any other file.

Coordinate space is WORLD space throughout, consistent with the rest of the
project. This module imports cleanly outside Maya: all Maya access is isolated
in :mod:`mesh_utils` (read/write) and behind the ``anatomy_backend`` duck-typed
interface already defined by :class:`anatomy_constraint.MayaAnatomyBackend`, so
the pure region/force/convergence logic below can be (and is, see
``.cs_test/test_final_cleanup_solver.py``) unit-tested without a running Maya
session.
"""

from __future__ import print_function

import csv
import datetime
import json
import math
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import mesh_utils
import metrics_utils
import artifact_detection
import anatomy_constraint

try:
    import maya_io
except ImportError:  # pragma: no cover - maya_io present in this project
    maya_io = None


# =============================================================================
# DEFAULTS
# =============================================================================

DEFAULT_CLEANUP_GROWTH_RINGS = 6
DEFAULT_TRANSITION_RINGS = 4
DEFAULT_BOUNDARY_BUFFER_RINGS = 1

DEFAULT_W_FAIR = 0.5
DEFAULT_W_M4 = 0.25
DEFAULT_W_SHAPE = 0.15
DEFAULT_M4_ONLY_FAIR_SCALE = 0.3   # m4_only vertices still get a little fairing
DEFAULT_OVERLAP_M4_SCALE = 0.6     # overlap vertices get damped M4 pull (both forces already act)

DEFAULT_MAX_STEP_EDGE_RATIO = 0.5

DEFAULT_CLEARANCE_POLICY = "preserve_valid_baseline"
DEFAULT_CLEARANCE_TOLERANCE = 1e-4
DEFAULT_MAX_CONSTRAINT_ITERATIONS = 8
DEFAULT_INTERSECTION_TOLERANCE = 1e-6
DEFAULT_REPAIR_PROFILE = "broad_fair"

DEFAULT_DYNAMIC_GROWTH_ZSCORE = 2.0
DEFAULT_MAX_ACTIVE_GROWTH_FRACTION = 0.5

DEFAULT_MAX_ITERATIONS = 100
DEFAULT_CONVERGENCE_PATIENCE = 5
DEFAULT_MEAN_DISPLACEMENT_TOLERANCE = 1e-4
DEFAULT_MAX_DISPLACEMENT_TOLERANCE = 5e-4
DEFAULT_ROUGHNESS_REL_IMPROVEMENT_TOLERANCE = 0.01
DEFAULT_M4_REL_IMPROVEMENT_TOLERANCE = 0.01

# --- Phase 0 (initial feasibility) ------------------------------------------
DEFAULT_MAX_FEASIBILITY_PASSES = 5

# --- region-dependent shape restraint ----------------------------------------
# Shape restraint is WEAKEST at the artifact core (most freedom to fair local
# defects, including ones introduced by Phase 0's own repair) and ramps up to
# full strength across the transition band (holding facial identity near the
# fixed exterior), using the SAME anchor value build_active_region already
# computes -- no separate region bookkeeping needed.
DEFAULT_CORE_SHAPE_SCALE = 0.3

# --- M4 force decay ------------------------------------------------------
# M4 is guidance, not a final target: once its relative IMPROVEMENT plateaus
# for m4_decay_patience consecutive iterations, ratchet w_m4_effective down by
# m4_decay_rate. This is PERSISTENT, MONOTONIC state (see decay_m4_weight) --
# the plateau streak resets only on genuine continued improvement, never on a
# regression, because once we deliberately decay M4 some regression in raw M4
# disagreement is an EXPECTED, ACCEPTABLE side effect (fairing/shape are now
# allowed to dominate), not a sign that more M4 correction is still needed. An
# earlier version recomputed the "effective" weight from scratch each
# iteration as a function of a reset-prone counter, which caused it to snap
# back to full strength the moment the first decay step visibly worked -- see
# the module docstring.
DEFAULT_M4_DECAY_PATIENCE = 2
DEFAULT_M4_DECAY_RATE = 0.7
DEFAULT_M4_MIN_WEIGHT_FRACTION = 0.0
# Per-vertex M4 freeze: once an individual vertex's OWN distance to its M4
# target is within this fraction of target_offset, stop pulling it further --
# regardless of the global decay state. Some M4 vertices converge long before
# others; this is the "per-vertex M4 activity" refinement without a full
# per-component redesign.
DEFAULT_M4_FREEZE_RELATIVE_TOLERANCE = 0.05

# --- roughness as a real acceptance/convergence target ------------------------
DEFAULT_MAX_ROUGHNESS_RATIO = 1.5   # vs. the ORIGINAL pre-cleanup roughness
DEFAULT_REPAIR_DISPLACEMENT_TOLERANCE = 1e-3
DEFAULT_FORCE_STABLE_TOLERANCE = 0.03  # relative change of the RAW (pre-clamp) combined force

# --- rest-reference low-pass filtering ----------------------------------------
# Even a FEASIBLE Phase-0 output can still contain the broad bumps its own
# repair introduced. A direct x_rest - x shape-restraint term would then pull
# vertices back toward those bumps, opposing fairing at exactly the vertices
# that need it most. Low-pass filtering ONLY the reference (never the actual
# working geometry, never anatomy-checked -- it is a soft target, not enforced
# geometry) lets shape-restraint hold LOW-FREQUENCY facial form while leaving
# HIGH-FREQUENCY repair bumps free to fair away. Scoped to the fairing region
# only (core_shape_scale already gives the core the most freedom; the
# transition band's reference is deliberately left unsmoothed so it keeps
# anchoring identity precisely).
DEFAULT_REST_REFERENCE_SMOOTHING_ITERATIONS = 12
DEFAULT_REST_REFERENCE_SMOOTHING_STRENGTH = 0.3

# --- oscillation (fair -> repair -> fair -> repair ping-pong) detection -------
DEFAULT_OSCILLATION_WINDOW = 5
DEFAULT_OSCILLATION_REPEAT_THRESHOLD = 3
DEFAULT_OSCILLATION_DAMPING = 0.2

# --- roughness-targeted fairing weights (run_final_fairing only) --------------
# Uniform weight 1.0 was spreading fairing onto already-smooth skin (p50 got
# worse) while the remaining ridges (p90/p95/p99) barely moved. Map local
# Laplacian magnitude onto a mild [0.75, 2.0] weight, then diffuse it over a
# few one-rings so a ridge is a coherent patch, not an isolated spike.
DEFAULT_ROUGHNESS_WEIGHTING = True
DEFAULT_ROUGHNESS_WEIGHT_MIN = 0.75
DEFAULT_ROUGHNESS_WEIGHT_MAX = 2.0
DEFAULT_ROUGHNESS_WEIGHT_PERCENTILE_START = 75.0
DEFAULT_ROUGHNESS_WEIGHT_SMOOTHING_RINGS = 3
_ROUGHNESS_WEIGHT_UPPER_PERCENTILES = (90.0, 95.0, 99.0)

PHASE_ANATOMY_CORRECTION = "anatomy_correction"
PHASE_BALANCED_CLEANUP = "balanced_cleanup"
PHASE_FAIRING_FINISH = "fairing_finish"

STOP_MAX_ITERATIONS = "max_iterations"
STOP_CONVERGED = "convergence_tolerance"
STOP_NO_ARTIFACT_REGION = "no artifact region detected"
STOP_DRY_RUN = "dry run (no geometry modified)"
STOP_VALIDATION_FAILED = "post-hoc validation found residual intersections"

_BACKUP_SUFFIX = "_precleanupsolver"


# =============================================================================
# 1. PURE REGION / FORCE HELPERS
#    No Maya access. Operate on plain [[x, y, z], ...] lists and adjacency
#    lists, exactly like smoothing_utils.laplacian_smooth. Unit-tested in
#    .cs_test/test_final_cleanup_solver.py without a Maya session.
# =============================================================================

def _bfs_ring_distance(seeds: Sequence[int],
                       neighbors: List[List[int]],
                       blocked: Set[int],
                       max_rings: int,
                       ) -> Dict[int, int]:
    """Topological ring distance from ``seeds`` (distance 0), never crossing
    ``blocked`` vertices (protected boundaries). Stops at ``max_rings``."""
    dist: Dict[int, int] = {i: 0 for i in seeds if i not in blocked}
    frontier = set(dist.keys())
    for r in range(1, max(0, int(max_rings)) + 1):
        nxt: Set[int] = set()
        for u in frontier:
            if u < 0 or u >= len(neighbors):
                continue
            for w in neighbors[u]:
                if w in dist or w in blocked:
                    continue
                dist[w] = r
                nxt.add(w)
        if not nxt:
            break
        frontier = nxt
    return dist


def build_active_region(core: Sequence[int],
                        neighbors: List[List[int]],
                        boundary: Set[int],
                        cleanup_growth_rings: int,
                        transition_rings: int,
                        ) -> Dict[str, Any]:
    """Build the graded active region: fairing band -> transition band.

    ``core`` (the M5 / intersection seed set) is grown by ``cleanup_growth_rings``
    topological rings into the full-strength FAIRING region, then by a further
    ``transition_rings`` into a TRANSITION band whose ``anchor`` weight ramps
    smoothly from 0 (fairing edge) to 1 (outer edge, effectively frozen).
    Vertices in ``boundary`` (protected facial openings / true topological
    edges, already ring-buffered by the caller) are never entered.

    Returns
    -------
    dict
        ``{"fairing": [...], "transition": [...], "active": [...],
           "anchor": {i: float in [0, 1]}}``, all sorted / deterministic.
    """
    boundary = set(boundary or [])
    core = sorted(set(int(i) for i in core) - boundary)
    if not core:
        return {"fairing": [], "transition": [], "active": [], "anchor": {}}

    total_rings = max(0, int(cleanup_growth_rings)) + max(0, int(transition_rings))
    dist = _bfs_ring_distance(core, neighbors, boundary, total_rings)

    fairing = sorted(i for i, r in dist.items() if r <= cleanup_growth_rings)
    transition = sorted(i for i, r in dist.items() if cleanup_growth_rings < r)

    anchor: Dict[int, float] = {i: 0.0 for i in fairing}
    tr = max(1, int(transition_rings))
    for i in transition:
        t = (dist[i] - cleanup_growth_rings) / float(tr)
        t = min(1.0, max(0.0, t))
        anchor[i] = t * t * (3.0 - 2.0 * t)  # smoothstep, 0 -> 1

    active = sorted(set(fairing) | set(transition))
    return {"fairing": fairing, "transition": transition, "active": active, "anchor": anchor}


def classify_provenance(active: Sequence[int],
                        m3_only: Set[int],
                        m4_only: Set[int],
                        overlap: Set[int],
                        ) -> Dict[int, str]:
    """Tag every active vertex with WHY it is being corrected.

    ``"overlap"`` / ``"m4_only"`` / ``"m3_only"`` come straight from the M5
    report (see :func:`artifact_detection.detect_unified_artifacts`).
    ``"grown"`` is a vertex pulled in only by regional growth (fairing-band
    expansion, transition band, or dynamic active-set growth / intersection
    repair) with no M5 evidence of its own -- treated like m3_only (pure
    fairing, no anatomy-target pull), since it has no M4 target to attract to.
    """
    prov: Dict[int, str] = {}
    for i in active:
        if i in overlap:
            prov[i] = "overlap"
        elif i in m4_only:
            prov[i] = "m4_only"
        elif i in m3_only:
            prov[i] = "m3_only"
        else:
            prov[i] = "grown"
    return prov


def provenance_weights(active: Sequence[int],
                       provenance: Dict[int, str],
                       w_fair: float,
                       w_m4: float,
                       m4_only_fair_scale: float = DEFAULT_M4_ONLY_FAIR_SCALE,
                       overlap_m4_scale: float = DEFAULT_OVERLAP_M4_SCALE,
                       ) -> Dict[int, Tuple[float, float]]:
    """Return ``{i: (fair_weight, m4_weight)}`` from each vertex's provenance.

    m3_only / grown  -> fairing only (no M4 target pull; nothing to attract to
                        conceptually for m3_only, and "grown" halo vertices have
                        no M5 evidence at all).
    m4_only          -> mostly M4-target attraction, a SMALL fairing term so the
                        patch does not develop a crease at its own edge.
    overlap          -> both, with the M4 term damped so it does not simply
                        double the m4_only pull where geometry and anatomy
                        evidence already agree.
    """
    weights: Dict[int, Tuple[float, float]] = {}
    for i in active:
        p = provenance.get(i, "grown")
        if p == "m4_only":
            weights[i] = (w_fair * m4_only_fair_scale, w_m4)
        elif p == "overlap":
            weights[i] = (w_fair, w_m4 * overlap_m4_scale)
        else:  # "m3_only" or "grown"
            weights[i] = (w_fair, 0.0)
    return weights


def local_edge_lengths(positions: List[List[float]],
                       neighbors: List[List[int]],
                       indices: Sequence[int],
                       ) -> Dict[int, float]:
    """Mean 1-ring edge length per vertex (trust-region scale reference)."""
    out: Dict[int, float] = {}
    for i in indices:
        nbrs = neighbors[i] if 0 <= i < len(neighbors) else []
        if not nbrs:
            out[i] = 0.0
            continue
        out[i] = sum(mesh_utils.vec_length(mesh_utils.vec_sub(positions[i], positions[j]))
                     for j in nbrs) / len(nbrs)
    return out


def fairing_force(positions: List[List[float]],
                  neighbors: List[List[int]],
                  indices: Sequence[int],
                  fair_weight: Dict[int, float],
                  ) -> Dict[int, List[float]]:
    """Umbrella-Laplacian pull toward the local neighbour average.

    ``w * (mean_{j in N(i)} x_j - x_i)`` -- the same local-fairness force every
    other smoother in this project uses (see :func:`smoothing_utils.laplacian_smooth`),
    just computed as an explicit, weighted, per-vertex FORCE here instead of an
    unconditional replace-with-average step, so it can be combined with the
    other two forces before any trust-region clamp or anatomy projection.
    """
    out: Dict[int, List[float]] = {}
    for i in indices:
        w = fair_weight.get(i, 0.0)
        nbrs = neighbors[i] if 0 <= i < len(neighbors) else []
        if not nbrs or w == 0.0:
            out[i] = [0.0, 0.0, 0.0]
            continue
        avg = mesh_utils.vec_mean([positions[j] for j in nbrs])
        out[i] = mesh_utils.vec_scale(mesh_utils.vec_sub(avg, positions[i]), w)
    return out


def taubin_force(positions: List[List[float]],
                 neighbors: List[List[int]],
                 indices: Sequence[int],
                 lamb: float,
                 mu: float,
                 weight: Dict[int, float],
                 ) -> Dict[int, List[float]]:
    """Net displacement of one Taubin lambda/mu pass pair, as a FORCE (target
    minus current) rather than a position replacement.

    ``smoothing_utils.taubin_smooth`` is not used here: it unconditionally
    overwrites positions in two full passes and has no notion of a per-vertex
    weight or a "proposal to be combined with other forces" -- it is not
    force-composable, and this solver's whole safety architecture (shape
    restraint, per-vertex trust region, anatomy projection) depends on every
    term being a combinable force. Taubin's lambda/mu pair is mathematically
    just two umbrella-Laplacian half-steps with opposite-signed strength, and
    :func:`fairing_force` already computes exactly that per-vertex, per-weight
    umbrella-Laplacian step -- so this function reproduces the identical
    Taubin math by calling it twice (feeding the first half-step's result into
    the second), rather than introducing a second smoothing implementation.

    ``weight`` scales BOTH half-steps per vertex (the same role
    :func:`fairing_force`'s own weight plays for plain Laplacian), so
    oscillation damping etc. apply uniformly regardless of ``method``.
    """
    w1 = {i: lamb * weight.get(i, 0.0) for i in indices}
    mid = apply_step(positions, fairing_force(positions, neighbors, indices, w1))
    w2 = {i: mu * weight.get(i, 0.0) for i in indices}
    end = apply_step(mid, fairing_force(mid, neighbors, indices, w2))
    return {i: mesh_utils.vec_sub(end[i], positions[i]) for i in indices}


def shape_weight_from_anchor(anchor_i: float, core_shape_scale: float) -> float:
    """Region-dependent shape-restraint multiplier from a transition anchor.

    ``core_shape_scale`` at ``anchor_i == 0`` (artifact core / fairing region,
    where local defects -- including Phase 0's own repair bumps -- need the
    MOST freedom to fair), ramping linearly to ``1.0`` at ``anchor_i == 1``
    (outer transition edge, where facial identity should be held firmly).
    Uses the SAME anchor value :func:`build_active_region` already computes;
    no separate region bookkeeping is needed.
    """
    a = min(1.0, max(0.0, float(anchor_i)))
    scale = min(1.0, max(0.0, float(core_shape_scale)))
    return scale + (1.0 - scale) * a


def shape_force(positions: List[List[float]],
                rest_positions: List[List[float]],
                indices: Sequence[int],
                w_shape,
                ) -> Dict[int, List[float]]:
    """Restoring pull toward the registered ("rest") position.

    ``w_shape * (x_rest_i - x_i)`` -- this is what keeps repeated fairing from
    converging toward a fully flattened harmonic surface: the operator's fixed
    point is now a compromise between "locally smooth" and "close to the
    original registered shape", not smoothness alone.

    ``w_shape`` may be a single float (applied to every index) or a
    ``{i: weight}`` dict (region-dependent restraint via
    :func:`shape_weight_from_anchor`).

    IMPORTANT: ``rest_positions`` must be an anatomy-FEASIBLE reference (the
    Phase-0-repaired skin), never the raw pre-repair registered mesh -- pulling
    toward a position that itself violates anatomy is exactly what turns
    intersection repair into a fight the repair can never win (see the module
    docstring's Phase 0 section).
    """
    per_vertex = w_shape if isinstance(w_shape, dict) else None
    out: Dict[int, List[float]] = {}
    for i in indices:
        w = per_vertex.get(i, 0.0) if per_vertex is not None else w_shape
        if w == 0.0:
            out[i] = [0.0, 0.0, 0.0]
            continue
        out[i] = mesh_utils.vec_scale(
            mesh_utils.vec_sub(rest_positions[i], positions[i]), w)
    return out


def decay_m4_weight(current_w_m4_effective: float,
                    decay_rate: float,
                    floor: float,
                    ) -> float:
    """ONE ratchet step of PERSISTENT, MONOTONIC M4-weight decay.

    ``current_w_m4_effective`` is solver state carried across iterations (the
    caller stores it, decides WHEN to call this -- see
    :func:`should_decay_m4`), never recomputed from scratch. Each call can
    only move the weight DOWN (or leave it, at ``floor``); it never increases.
    This is deliberate: once M4 has been judged to have plateaued and its
    influence reduced, that reduction should persist even if the resulting
    (expected, intentional) shift in M4 disagreement looks like "change" to a
    naive plateau detector -- see :func:`should_decay_m4` and the module
    docstring for why an earlier version snapped back to full strength.
    """
    return max(float(floor), float(current_w_m4_effective) * float(decay_rate))


def m4_is_improving(prev_disagreement: float,
                    disagreement: float,
                    rel_improvement_tolerance: float,
                    ) -> bool:
    """True iff M4 disagreement genuinely DECREASED by at least the tolerance
    fraction. Deliberately SIGNED (unlike the roughness/motion patience
    checks, which care about "changed at all"): once M4 is intentionally
    decayed, a resulting INCREASE in disagreement is an expected, acceptable
    side effect, not a sign more correction is needed -- it must not reset the
    plateau streak the same way a genuine improvement resetting it does.
    """
    rel = (float(prev_disagreement) - float(disagreement)) / (float(prev_disagreement) + 1e-9)
    return rel >= float(rel_improvement_tolerance)


def should_decay_m4(m4_plateau_streak: int, decay_patience: int) -> bool:
    """True on every ``decay_patience``-th consecutive plateaued iteration
    (2, 4, 6, ... for ``decay_patience=2``), so decay keeps ratcheting down
    the longer the plateau persists, per :func:`decay_m4_weight`."""
    streak = int(m4_plateau_streak)
    patience = max(1, int(decay_patience))
    return streak > 0 and streak % patience == 0


def classify_phase(w_m4_effective: float,
                   w_m4: float,
                   m4_floor: float,
                   proposal_face_count: int,
                   repair_mean: float,
                   repair_mean_tolerance: float,
                   ) -> str:
    """State-derived (never a fixed iteration number) coarse phase label.

    ``anatomy_correction`` while M4 is still at (or near) full strength;
    ``balanced_cleanup`` once M4 has started decaying but the surface is not
    yet quiet (still occasional proposal intersections / repair work);
    ``fairing_finish`` once M4 has hit its floor AND the surface has gone
    quiet (proposal intersections and repair work both near zero) -- fairing
    and the feasible-rest reference are effectively the only remaining active
    forces, with anatomy purely a one-sided feasibility backstop.
    """
    m4_at_floor = w_m4_effective <= m4_floor + 1e-9
    quiet = (proposal_face_count <= 2) and (repair_mean <= repair_mean_tolerance)
    if m4_at_floor and quiet:
        return PHASE_FAIRING_FINISH
    if w_m4_effective < w_m4 - 1e-9:
        return PHASE_BALANCED_CLEANUP
    return PHASE_ANATOMY_CORRECTION


def force_opposition(force_a: Dict[int, List[float]],
                     force_b: Dict[int, List[float]],
                     indices: Sequence[int],
                     ) -> Dict[str, float]:
    """Mean dot product / cosine similarity between two per-vertex force (or
    displacement) fields, over vertices where BOTH are non-negligible.

    A magnitude alone cannot say whether two forces cancel -- ``mean_cosine``
    near ``-1`` means they point opposite ways (one is undoing the other);
    near ``0`` means they are roughly orthogonal (not really interacting);
    near ``+1`` means they reinforce. ``opposing_fraction`` is the share of
    compared vertices where they point more than 90 degrees apart.
    """
    dots: List[float] = []
    coss: List[float] = []
    for i in indices:
        va = force_a.get(i)
        vb = force_b.get(i)
        if va is None or vb is None:
            continue
        la = mesh_utils.vec_length(va)
        lb = mesh_utils.vec_length(vb)
        if la < 1e-9 or lb < 1e-9:
            continue
        d = mesh_utils.vec_dot(va, vb)
        dots.append(d)
        coss.append(d / (la * lb))
    return {
        "count": len(dots),
        "mean_dot": (sum(dots) / len(dots)) if dots else 0.0,
        "mean_cosine": (sum(coss) / len(coss)) if coss else 0.0,
        "opposing_fraction": (sum(1 for c in coss if c < 0) / len(coss)) if coss else 0.0,
    }


def is_better_state(roughness: float,
                    m4_disagreement: float,
                    best_roughness: float,
                    best_m4: float,
                    eps: float = 1e-9,
                    ) -> bool:
    """Best-feasible-state ranking rule: lower roughness wins; ties (within
    ``eps``) are broken by lower M4 disagreement. Every state compared here is
    already anatomy-valid by construction (the caller only ever calls this on
    ACCEPTED post-repair-and-rollback states), so intersection-freedom is not
    part of the comparison -- it is a precondition, not a tiebreaker.
    """
    if roughness < best_roughness - eps:
        return True
    return roughness <= best_roughness + eps and m4_disagreement < best_m4


def provenance_buckets(indices: Sequence[int],
                       provenance: Dict[int, str],
                       oscillating: Optional[Set[int]] = None,
                       ) -> Dict[str, List[int]]:
    """Group ``indices`` by provenance tag, with oscillating/damped vertices
    broken out into their own bucket regardless of their original tag (they
    are behaving differently now, by design)."""
    osc = set(oscillating or [])
    buckets: Dict[str, List[int]] = {}
    for i in indices:
        tag = "oscillating" if i in osc else provenance.get(i, "grown")
        buckets.setdefault(tag, []).append(i)
    return buckets


def _smooth_rest_reference(positions: List[List[float]],
                           neighbors: List[List[int]],
                           indices: Sequence[int],
                           iterations: int,
                           strength: float,
                           ) -> List[List[float]]:
    """Low-pass filter the REST/POSITIONAL REFERENCE (never the actual working
    geometry, never anatomy-checked -- see the module docstring's rest-
    reference section) over ``indices`` only; vertices outside ``indices``
    (including true anatomy/topology, irrelevant here since this never touches
    real geometry) act as fixed anchors, exactly the same pattern as
    :func:`fairing_force` / ``smoothing_utils.laplacian_smooth`` elsewhere in
    this project. Reuses the existing pure fairing/step primitives rather than
    introducing a second smoothing implementation.
    """
    out = [list(p) for p in positions]
    w = {i: float(strength) for i in indices}
    for _ in range(max(0, int(iterations))):
        f = fairing_force(out, neighbors, indices, w)
        out = apply_step(out, f)
    return out


def update_oscillation_tracker(history: List[Set[int]],
                               repaired_now: Sequence[int],
                               window: int,
                               threshold: int,
                               ) -> Tuple[List[Set[int]], Set[int]]:
    """Track a rolling window of per-iteration repaired-vertex sets and flag
    vertices repaired in ``>= threshold`` of the last ``window`` iterations
    (a fair -> repair -> fair -> repair ping-pong signature).

    Returns ``(updated_history, oscillating_vertices)``. Pure / no Maya access
    -- unit-tested directly with synthetic repair histories.
    """
    hist = list(history) + [set(repaired_now)]
    if len(hist) > max(1, int(window)):
        hist = hist[-int(window):]
    counts: Dict[int, int] = {}
    for s in hist:
        for v in s:
            counts[v] = counts.get(v, 0) + 1
    oscillating = {v for v, c in counts.items() if c >= int(threshold)}
    return hist, oscillating


def m4_force(positions: List[List[float]],
            targets: Dict[int, List[float]],
            indices: Sequence[int],
            m4_weight: Dict[int, float],
            ) -> Dict[int, List[float]]:
    """Damped attraction toward the anatomy-derived M4 target (where available).

    ``w * (t_i^M4 - x_i)``. Only applied where ``targets`` has a converged M4
    projection for ``i`` (see :func:`artifact_detection.compute_sdf_target_positions`);
    a directional/reference signal, never a hard snap -- ``m4_weight`` is a
    fraction well under 1 so this does not pull the skin onto the anatomy
    surface itself (see the module docstring / DEFAULT_W_M4).
    """
    out: Dict[int, List[float]] = {}
    for i in indices:
        w = m4_weight.get(i, 0.0)
        t = targets.get(i)
        if w == 0.0 or t is None:
            out[i] = [0.0, 0.0, 0.0]
            continue
        out[i] = mesh_utils.vec_scale(mesh_utils.vec_sub(t, positions[i]), w)
    return out


def combine_step(fair_f: Dict[int, List[float]],
                 shape_f: Dict[int, List[float]],
                 m4_f: Dict[int, List[float]],
                 indices: Sequence[int],
                 anchor: Dict[int, float],
                 max_step_edge_ratio: Optional[float],
                 local_edge: Dict[int, float],
                 ) -> Dict[int, List[float]]:
    """Sum the three forces, damp by ``(1 - anchor)``, clamp to a PER-VERTEX
    trust region.

    This is the direct replacement for the single shared line-search alpha in
    :func:`smoothing_utils.constrained_smooth_mesh_region`: every vertex here
    gets its OWN cap (``max_step_edge_ratio`` times ITS OWN local edge length,
    further damped by ITS OWN transition-band anchor weight), so one tight
    vertex can only ever limit itself, never the rest of the active region.
    """
    step: Dict[int, List[float]] = {}
    for i in indices:
        fx = fair_f.get(i, [0.0, 0.0, 0.0])
        sx = shape_f.get(i, [0.0, 0.0, 0.0])
        mx = m4_f.get(i, [0.0, 0.0, 0.0])
        raw = [fx[k] + sx[k] + mx[k] for k in range(3)]
        damp = 1.0 - float(anchor.get(i, 0.0))
        if damp <= 0.0:
            step[i] = [0.0, 0.0, 0.0]
            continue
        raw = [c * damp for c in raw]
        if max_step_edge_ratio is not None:
            cap = float(max_step_edge_ratio) * local_edge.get(i, 0.0)
            if cap > 0.0:
                mag = mesh_utils.vec_length(raw)
                if mag > cap and mag > 1e-12:
                    raw = mesh_utils.vec_scale(raw, cap / mag)
        step[i] = raw
    return step


def apply_step(positions: List[List[float]],
              step: Dict[int, List[float]],
              ) -> List[List[float]]:
    """Return a NEW position list with ``step`` added at its keys only."""
    out = [list(p) for p in positions]
    for i, d in step.items():
        out[i] = mesh_utils.vec_add(out[i], d)
    return out


def laplacian_magnitudes(positions: List[List[float]],
                         neighbors: List[List[int]],
                         indices: Sequence[int],
                         ) -> Dict[int, float]:
    """Per-vertex umbrella-Laplacian magnitude -- the SAME formula as
    :func:`mean_laplacian_magnitude` / M1's ``compute_laplacian_scores``,
    evaluated on an in-memory position array. Vertices with no neighbours
    are omitted (they have no local fairness signal).
    """
    out: Dict[int, float] = {}
    for i in indices:
        nbrs = neighbors[i] if 0 <= i < len(neighbors) else []
        if not nbrs:
            continue
        avg = mesh_utils.vec_mean([positions[j] for j in nbrs])
        out[i] = mesh_utils.vec_length(mesh_utils.vec_sub(positions[i], avg))
    return out


def mean_laplacian_magnitude(positions: List[List[float]],
                             neighbors: List[List[int]],
                             indices: Sequence[int],
                             ) -> float:
    """Mean umbrella-Laplacian magnitude over ``indices`` -- the SAME formula
    as :func:`artifact_detection.compute_laplacian_scores` (M1), evaluated on
    an in-memory position array instead of a live mesh.

    This is a deliberate, minimal re-expression, not a duplicated detector:
    the solver holds state in-memory across many iterations and must not
    round-trip every one of them through Maya just to measure roughness, so
    it cannot call the Maya-backed M1 function directly here.
    """
    vals = list(laplacian_magnitudes(positions, neighbors, indices).values())
    return (sum(vals) / len(vals)) if vals else 0.0


def _percentile(sorted_vals: Sequence[float], pct: float) -> float:
    """Linear-interpolation percentile of an already-sorted ascending list."""
    if not sorted_vals:
        return 0.0
    if pct <= 0:
        return float(sorted_vals[0])
    if pct >= 100:
        return float(sorted_vals[-1])
    k = (len(sorted_vals) - 1) * (float(pct) / 100.0)
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    if lo == hi:
        return float(sorted_vals[int(k)])
    return float(sorted_vals[lo]) + (float(sorted_vals[hi]) - float(sorted_vals[lo])) * (k - lo)


def roughness_percentile_summary(values: Sequence[float]) -> Dict[str, float]:
    """p50 / p90 / p95 / p99 / max of a Laplacian-magnitude sample."""
    vals = [float(v) for v in values]
    if not vals:
        return {"p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0, "mean": 0.0}
    s = sorted(vals)
    return {
        "mean": sum(s) / len(s),
        "p50": _percentile(s, 50.0),
        "p90": _percentile(s, 90.0),
        "p95": _percentile(s, 95.0),
        "p99": _percentile(s, 99.0),
        "max": s[-1],
    }


def _piecewise_lerp(x: float, xs: Sequence[float], ys: Sequence[float]) -> float:
    if not xs:
        return 1.0
    if x <= xs[0]:
        return float(ys[0])
    if x >= xs[-1]:
        return float(ys[-1])
    for i in range(1, len(xs)):
        if x <= xs[i]:
            span = xs[i] - xs[i - 1]
            t = 0.0 if span <= 1e-15 else (x - xs[i - 1]) / span
            return float(ys[i - 1]) + t * (float(ys[i]) - float(ys[i - 1]))
    return float(ys[-1])


def _roughness_weight_knots(sorted_vals: Sequence[float],
                            percentile_start: float,
                            weight_min: float,
                            weight_max: float,
                            ) -> Tuple[List[float], List[float]]:
    """Roughness-value knots and matching fairing-weight knots.

    Continuous map (defaults ``weight_min=0.75``, ``weight_max=2.0``,
    ``percentile_start=75``)::

        min .. p75   ->  0.75 .. 1.00   (already-smooth skin: slightly less)
        p75  .. p90  ->  1.00 .. 1.25
        p90  .. p95  ->  1.25 .. 1.50
        p95  .. p99  ->  1.50 .. 1.75
        p99  .. max  ->  1.75 .. 2.00
    """
    if not sorted_vals:
        return [0.0], [1.0]
    if float(sorted_vals[-1]) - float(sorted_vals[0]) <= 1e-12:
        return [float(sorted_vals[0])], [1.0]
    w_min = float(weight_min)
    w_max = max(1.0, float(weight_max))
    span = w_max - 1.0
    r_knots = [
        float(sorted_vals[0]),
        _percentile(sorted_vals, float(percentile_start)),
        _percentile(sorted_vals, _ROUGHNESS_WEIGHT_UPPER_PERCENTILES[0]),
        _percentile(sorted_vals, _ROUGHNESS_WEIGHT_UPPER_PERCENTILES[1]),
        _percentile(sorted_vals, _ROUGHNESS_WEIGHT_UPPER_PERCENTILES[2]),
        float(sorted_vals[-1]),
    ]
    w_knots = [
        w_min,
        1.0,
        1.0 + 0.25 * span,
        1.0 + 0.50 * span,
        1.0 + 0.75 * span,
        w_max,
    ]
    xs: List[float] = [r_knots[0]]
    ys: List[float] = [w_knots[0]]
    for r, w in zip(r_knots[1:], w_knots[1:]):
        if r > xs[-1] + 1e-15:
            xs.append(r)
            ys.append(w)
        # Duplicate roughness knot: keep the earlier (lower) weight so a
        # collapsed tail cannot jump straight to weight_max.
    return xs, ys


def roughness_to_fairing_weight(magnitude: float,
                                r_knots: Sequence[float],
                                w_knots: Sequence[float],
                                ) -> float:
    return _piecewise_lerp(float(magnitude), r_knots, w_knots)


def _smooth_weight_field(weights: Dict[int, float],
                         neighbors: List[List[int]],
                         rings: int,
                         ) -> Dict[int, float]:
    """One-ring Jacobi averages of a scalar weight field (``rings`` passes).

    Restricted to ``weights`` keys, so the field cannot leak onto frozen
    exterior vertices. Including self in the average keeps an isolated peak
    from collapsing in one step while still spreading it to its 1-ring.
    """
    w = {i: float(v) for i, v in weights.items()}
    allowed = set(w.keys())
    n_pass = max(0, int(rings))
    for _ in range(n_pass):
        nxt = {}
        for i, wi in w.items():
            nbrs = [j for j in (neighbors[i] if 0 <= i < len(neighbors) else [])
                    if j in allowed]
            if not nbrs:
                nxt[i] = wi
                continue
            nxt[i] = (wi + sum(w[j] for j in nbrs)) / float(1 + len(nbrs))
        w = nxt
    return w


def build_roughness_fairing_weights(positions: List[List[float]],
                                    neighbors: List[List[int]],
                                    indices: Sequence[int],
                                    percentile_start: float = DEFAULT_ROUGHNESS_WEIGHT_PERCENTILE_START,
                                    weight_min: float = DEFAULT_ROUGHNESS_WEIGHT_MIN,
                                    weight_max: float = DEFAULT_ROUGHNESS_WEIGHT_MAX,
                                    smoothing_rings: int = DEFAULT_ROUGHNESS_WEIGHT_SMOOTHING_RINGS,
                                    ) -> Tuple[Dict[int, float], Dict[str, Any]]:
    """Map local Laplacian roughness onto a spatially-smooth fairing weight.

    Returns ``(weights, info)`` where ``info`` holds the frozen roughness
    knots (so later-grown vertices can be mapped the same way) and the
    pre-damping weight statistics.
    """
    idx = list(indices)
    scores = laplacian_magnitudes(positions, neighbors, idx)
    vals = sorted(scores.values())
    if len(vals) < 2 or (vals[-1] - vals[0]) <= 1e-12:
        weights = {i: 1.0 for i in idx}
        info = {
            "r_knots": [vals[0] if vals else 0.0],
            "w_knots": [1.0],
            "uniform": True,
            "min": 1.0, "mean": 1.0, "max": 1.0,
            "boosted_count": 0,
        }
        return weights, info
    r_knots, w_knots = _roughness_weight_knots(
        vals, percentile_start, weight_min, weight_max)
    raw = {}
    for i in idx:
        mag = scores.get(i)
        raw[i] = 1.0 if mag is None else roughness_to_fairing_weight(mag, r_knots, w_knots)
    weights = _smooth_weight_field(raw, neighbors, smoothing_rings)
    lo = float(min(weight_min, 1.0))
    hi = float(max(weight_max, 1.0))
    for i in list(weights.keys()):
        if weights[i] < lo:
            weights[i] = lo
        elif weights[i] > hi:
            weights[i] = hi
    wvals = list(weights.values())
    boosted = sum(1 for v in wvals if v > 1.0 + 1e-6)
    info = {
        "r_knots": list(r_knots),
        "w_knots": list(w_knots),
        "uniform": False,
        "min": min(wvals) if wvals else 1.0,
        "mean": (sum(wvals) / len(wvals)) if wvals else 1.0,
        "max": max(wvals) if wvals else 1.0,
        "boosted_count": boosted,
        "smoothing_rings": int(smoothing_rings),
        "percentile_start": float(percentile_start),
    }
    return weights, info


def mean_target_disagreement(positions: List[List[float]],
                             targets: Dict[int, List[float]],
                             indices: Sequence[int],
                             ) -> float:
    """Mean distance from current position to the (static) M4 target, over
    whichever of ``indices`` have a converged target."""
    if not indices or not targets:
        return 0.0
    vals = [mesh_utils.vec_length(mesh_utils.vec_sub(positions[i], targets[i]))
            for i in indices if i in targets]
    return (sum(vals) / len(vals)) if vals else 0.0


def _stats(values: Sequence[float]) -> Dict[str, float]:
    vals = list(values)
    if not vals:
        return {"count": 0, "mean": 0.0, "max": 0.0}
    return {"count": len(vals), "mean": sum(vals) / len(vals), "max": max(vals)}


# =============================================================================
# 2. MAYA-TOUCHING ORCHESTRATION HELPERS
#    (thin; each delegates the actual work to artifact_detection /
#    anatomy_constraint / mesh_utils)
# =============================================================================

def _boundary_set(skin_mesh: str, neighbors: List[List[int]], rings: int) -> Set[int]:
    """Protected facial openings / true mesh edges, ring-buffered once."""
    boundary = mesh_utils.get_boundary_vertices(skin_mesh)
    if rings and rings > 0 and boundary:
        boundary = set(mesh_utils.grow_indices(neighbors, sorted(boundary), rings=int(rings)))
    return set(boundary or [])


def _m5_provenance_sets(report: Dict[str, Any]) -> Tuple[Set[int], Set[int], Set[int]]:
    return (set(report.get("m3_only_indices") or []),
            set(report.get("m4_only_indices") or []),
            set(report.get("overlap_indices") or []))


def _build_anatomy_backend(anatomical_meshes: Sequence[str]) -> Tuple[Any, List[str]]:
    """Build a :class:`anatomy_constraint.MayaAnatomyBackend` over the given
    mesh names, reusing :func:`mesh_utils.get_mesh_fn` for each (the SAME
    Maya API 2.0 accessor everything else in this project uses)."""
    mesh_fns = {}
    for name in anatomical_meshes:
        fn = mesh_utils.get_mesh_fn(name)
        if fn is not None:
            mesh_fns[name] = fn
    if not mesh_fns:
        raise ValueError(
            "final_cleanup_solver: could not load any anatomical mesh from "
            "anatomical_meshes ({0} requested). Is Maya running and are the "
            "meshes present?".format(len(list(anatomical_meshes))))
    backend = anatomy_constraint.MayaAnatomyBackend(mesh_fns)
    return backend, list(mesh_fns.keys())


def _scoped_intersection_core(skin_mesh: str,
                              positions: List[List[float]],
                              scope: Sequence[int],
                              backend: Any,
                              neighbors: List[List[int]],
                              skin_topology: Dict[str, Any],
                              boundary_buffer_rings: int,
                              intersection_tolerance: float,
                              ) -> Tuple[List[int], Dict[str, Any]]:
    """Real triangle/triangle intersection scan restricted to ``scope`` (plus
    its own incident faces), and the intersecting-face vertex core clipped
    back to ``scope``. Scoping to the active region is CORRECT and COMPLETE
    here: only active vertices ever move between calls, so any new
    intersecting face must be incident to one of them.
    """
    report = anatomy_constraint.analyze_skin_anatomy_intersections(
        skin_mesh, skin_indices=scope, backend=backend, positions=positions,
        neighbors=neighbors, skin_topology=skin_topology,
        boundary_buffer_rings=boundary_buffer_rings,
        intersection_tolerance=intersection_tolerance,
        detailed=False, verbose=False)
    core = anatomy_constraint.intersection_vertices_from_report(
        report, skin_topology, allowed=set(scope))
    return core, report


def _anatomy_project_clearance(positions: List[List[float]],
                               indices: Sequence[int],
                               backend: Any,
                               floors: Dict[int, float],
                               clearance_tolerance: float,
                               max_constraint_iterations: int,
                               normals: Optional[List[List[float]]],
                               ) -> Tuple[List[List[float]], Dict[int, float]]:
    """Per-vertex exact-clearance projection (cheap, local; NOT the triangle
    test). Reuses :func:`anatomy_constraint.enforce_anatomy_clearance` exactly
    as the existing constrained smoother does."""
    out = [list(p) for p in positions]
    moved: Dict[int, float] = {}
    for i in indices:
        floor_i = floors.get(i, 0.0)
        if floor_i <= 0.0:
            continue
        q = backend.exact_closest(out[i])
        if anatomy_constraint.is_clearance_satisfied(q["distance"], floor_i, clearance_tolerance):
            continue
        nrm = normals[i] if (normals and 0 <= i < len(normals)) else None
        sol = anatomy_constraint.enforce_anatomy_clearance(
            out[i], backend, floor_i,
            max_constraint_iterations=max_constraint_iterations,
            clearance_tolerance=clearance_tolerance, skin_normal=nrm)
        moved[i] = mesh_utils.vec_length(mesh_utils.vec_sub(sol["position"], out[i]))
        out[i] = sol["position"]
    return out, moved


def _repair_local(positions: List[List[float]],
                  core_now: Sequence[int],
                  backend: Any,
                  skin_mesh: str,
                  neighbors: List[List[int]],
                  normals: Optional[List[List[float]]],
                  skin_topology: Dict[str, Any],
                  min_clearance: float,
                  clearance_policy: str,
                  boundary_buffer_rings: int,
                  intersection_tolerance: float,
                  repair_kwargs: Dict[str, Any],
                  ) -> Tuple[List[List[float]], Optional[Dict[str, Any]]]:
    """Repair ONLY the vertices in ``core_now`` (the actually-offending
    component(s)), via the existing broad anatomy-supported patch repair.
    ``resolve_skin_anatomy_intersections`` partitions ``core_now`` into
    topology-connected components internally and solves each with its own
    local step -- exactly the "no single global alpha" property this solver
    needs, reused rather than rebuilt.
    """
    if not core_now:
        return positions, None
    result = anatomy_constraint.resolve_skin_anatomy_intersections(
        positions, list(core_now), backend, skin_mesh=skin_mesh,
        neighbors=neighbors, normals=normals, skin_topology=skin_topology,
        min_clearance=min_clearance, clearance_policy=clearance_policy,
        boundary_buffer_rings=boundary_buffer_rings,
        intersection_tolerance=intersection_tolerance, verbose=False,
        **repair_kwargs)
    return result["positions"], result


# =============================================================================
# 2b. PHASE 0 -- INITIAL FEASIBILITY
#    Establishes an anatomy-VALID starting point BEFORE the iterative fairing
#    loop, and before the "rest"/positional reference is captured. This is
#    NOT an ordinary fairing iteration: it exists specifically so the solver
#    never asks the rest-shape restraint to pull repaired geometry back toward
#    a position that was itself anatomically invalid (see the module
#    docstring). The broad anatomy-supported patch repair (unchanged) already
#    includes its own bounded local relax pass (``post_repair_relax``); Phase
#    0 just gives that machinery room to work BEFORE fairing forces start
#    layering on top of it.
# =============================================================================

def _run_initial_feasibility(positions: List[List[float]],
                             skin_mesh: str,
                             backend: Any,
                             neighbors: List[List[int]],
                             normals: Optional[List[List[float]]],
                             skin_topology: Dict[str, Any],
                             min_clearance: float,
                             clearance_policy: str,
                             boundary_buffer_rings: int,
                             intersection_tolerance: float,
                             repair_kwargs: Dict[str, Any],
                             max_passes: int,
                             verbose: bool,
                             ) -> Dict[str, Any]:
    """Repair every PRE-EXISTING real surface intersection (whole mesh) before
    any fairing/M4/shape force is ever computed.

    Bounded by ``max_passes``: each pass repairs whatever the CURRENT
    intersecting core is (the broad repair's own patch/component machinery is
    reused as-is), then re-scans the WHOLE mesh. Stops early once clean.
    Non-convergence after ``max_passes`` is reported, not silently hidden --
    the outer iterative loop's own per-iteration repair will keep trying on
    whatever remains, but Phase 0 is where the bulk of the pre-existing
    workload should be absorbed.

    Returns
    -------
    dict
        ``{"positions", "touched_vertices", "pre_intersections", "post_intersections",
           "passes", "resolved", "displacement", "roughness_before", "roughness_after"}``.
    """
    work = [list(p) for p in positions]
    n = len(work)
    all_indices = list(range(n))

    pre_report = anatomy_constraint.analyze_skin_anatomy_intersections(
        skin_mesh, skin_indices=all_indices, backend=backend, positions=work,
        neighbors=neighbors, skin_topology=skin_topology,
        boundary_buffer_rings=boundary_buffer_rings,
        intersection_tolerance=intersection_tolerance, detailed=False, verbose=False)
    pre_core = anatomy_constraint.intersection_vertices_from_report(pre_report, skin_topology)
    roughness_before = mean_laplacian_magnitude(work, neighbors, pre_core) if pre_core else 0.0

    touched: Set[int] = set()
    last_report = pre_report
    passes = 0
    for p in range(1, max(1, int(max_passes)) + 1):
        passes = p
        core_now = anatomy_constraint.intersection_vertices_from_report(last_report, skin_topology)
        if not core_now:
            break
        result = anatomy_constraint.resolve_skin_anatomy_intersections(
            work, core_now, backend, skin_mesh=skin_mesh, neighbors=neighbors,
            normals=normals, skin_topology=skin_topology,
            min_clearance=min_clearance, clearance_policy=clearance_policy,
            boundary_buffer_rings=boundary_buffer_rings,
            intersection_tolerance=intersection_tolerance, verbose=False,
            **repair_kwargs)
        work = result["positions"]
        touched |= set(result.get("patch_vertices") or core_now)
        last_report = anatomy_constraint.analyze_skin_anatomy_intersections(
            skin_mesh, skin_indices=all_indices, backend=backend, positions=work,
            neighbors=neighbors, skin_topology=skin_topology,
            boundary_buffer_rings=boundary_buffer_rings,
            intersection_tolerance=intersection_tolerance, detailed=False, verbose=False)
        if verbose:
            print("  [feasibility pass {0}] faces {1} -> {2}".format(
                p, pre_report.get("intersecting_skin_face_count", 0) if p == 1 else "...",
                last_report.get("intersecting_skin_face_count", 0)))
        if last_report.get("intersection_pair_count", 0) <= 0:
            break

    resolved = last_report.get("intersection_pair_count", 0) <= 0
    disp = [mesh_utils.vec_length(mesh_utils.vec_sub(work[i], positions[i])) for i in touched]
    roughness_after = mean_laplacian_magnitude(work, neighbors, sorted(touched)) if touched else 0.0

    return {
        "positions": work,
        "touched_vertices": sorted(touched),
        "pre_intersections": {"face_count": pre_report.get("intersecting_skin_face_count", 0),
                             "pair_count": pre_report.get("intersection_pair_count", 0)},
        "post_intersections": {"face_count": last_report.get("intersecting_skin_face_count", 0),
                              "pair_count": last_report.get("intersection_pair_count", 0)},
        "passes": passes,
        "resolved": bool(resolved),
        "displacement": _stats(disp),
        "roughness_before": roughness_before,
        "roughness_after": roughness_after,
    }


def _absorb_vertices(new_vertices: Sequence[int],
                     fairing_region: Set[int],
                     boundary: Set[int],
                     neighbors: List[List[int]],
                     transition_rings: int,
                     provenance: Dict[int, str],
                     weights: Dict[int, Tuple[float, float]],
                     floors: Dict[int, float],
                     positions: List[List[float]],
                     anatomy_backend: Any,
                     min_clearance: float,
                     clearance_policy: str,
                     clearance_tolerance: float,
                     w_fair: float,
                     tag: str,
                     ) -> Tuple[Set[int], Set[int], Dict[str, float]]:
    """Fold newly-discovered vertices (dynamic growth, redetection, or a
    repair patch that grew past the current active set) into the active
    region, with the same bookkeeping every growth path needs: rebuild
    fairing/transition/anchor, tag provenance, assign a default (fairing-only)
    weight, and compute their clearance floor. Shared so this bookkeeping is
    written once instead of three times.

    Returns ``(new_fairing_region, new_active, new_anchor)``.
    """
    new_vertices = sorted(set(new_vertices) - boundary)
    if not new_vertices:
        region = build_active_region(sorted(fairing_region), neighbors, boundary,
                                     0, transition_rings)
        return fairing_region, set(region["active"]), region["anchor"]
    fairing_region = set(fairing_region) | set(new_vertices)
    region = build_active_region(sorted(fairing_region), neighbors, boundary,
                                 0, transition_rings)
    active = set(region["active"])
    for j in new_vertices:
        provenance.setdefault(j, tag)
        weights.setdefault(j, (w_fair, 0.0))
    unfloored = sorted(active - set(floors.keys()))
    if unfloored:
        new_floors, _od, _pu = anatomy_constraint.compute_clearance_floors(
            positions, unfloored, anatomy_backend, min_clearance,
            clearance_policy=clearance_policy, clearance_tolerance=clearance_tolerance)
        floors.update(new_floors)
    for j in active - set(weights.keys()):
        provenance.setdefault(j, tag)
        weights.setdefault(j, (w_fair * 0.5, 0.0))
    return fairing_region, active, region["anchor"]


def _compute_m4_targets(positions: List[List[float]],
                        indices: Sequence[int],
                        sdf_query: Any,
                        target_offset: float,
                        max_projection_distance: Optional[float] = None,
                        ) -> Dict[int, List[float]]:
    """Static anatomy-derived correction targets for ``indices`` (computed
    ONCE; anatomy does not move during cleanup). Reuses M4's own per-point
    projector (:func:`artifact_detection.project_point_to_iso_surface`)
    directly against the in-memory ``positions`` array.

    Deliberately does NOT call :func:`artifact_detection.compute_sdf_target_positions`
    (the mesh-level M4 wrapper): that function reads its OWN copy of the
    vertex positions straight from the live Maya mesh, which would either (a)
    use the pre-Phase-0 (still anatomy-invalid) positions as the Newton
    search's starting point, or (b) require writing ``positions`` to the scene
    first to get the right starting point -- which would corrupt ``apply=False``
    dry runs by leaving the scene modified. Calling the per-point primitive
    directly avoids both problems while reusing the exact same projection math.
    """
    if not indices or sdf_query is None:
        return {}
    targets: Dict[int, List[float]] = {}
    for i in indices:
        t_i, conv = artifact_detection.project_point_to_iso_surface(
            positions[i], sdf_query, target_offset,
            max_step=max_projection_distance)
        if not conv.get("converged"):
            continue
        if (max_projection_distance is not None
                and conv.get("total_displacement", 0.0) > max_projection_distance):
            continue
        targets[i] = t_i
    return targets


# =============================================================================
# 3. LOGGING (same conventions as cleanup_pipeline.py)
# =============================================================================

def _unique_path(path: str) -> str:
    if not os.path.exists(path):
        return path
    stem, ext = os.path.splitext(path)
    n = 2
    while os.path.exists("{0}_v{1:02d}{2}".format(stem, n, ext)):
        n += 1
    return "{0}_v{1:02d}{2}".format(stem, n, ext)


def _write_json(report: Dict[str, Any], path: str) -> Optional[str]:
    path = _unique_path(path)
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print("[final_cleanup] wrote JSON log -> {0}".format(path))
        return path
    except (OSError, TypeError) as exc:
        print("[final_cleanup] WARNING: could not write JSON log ({0})".format(exc))
        return None


_CSV_COLUMNS = [
    "iteration", "phase", "active_count", "w_m4_effective", "frozen_m4_vertex_count",
    "fair_disp_mean", "fair_disp_max",
    "m4_disp_mean", "m4_disp_max",
    "shape_disp_mean", "shape_disp_max",
    "raw_proposal_mean", "raw_proposal_max",
    "clearance_disp_mean", "clearance_disp_max",
    "repair_disp_mean", "repair_disp_max",
    "net_disp_mean", "net_disp_max",
    "roughness_before", "roughness_after", "roughness_ceiling",
    "core_roughness", "transition_roughness",
    "m4_disagreement_before", "m4_disagreement_after",
    "proposal_face_count", "intersection_pair_count", "repaired_component_count",
    "min_exact_distance", "no_forbidden_intersections", "oscillating_count",
    "fair_vs_m4_cosine", "fair_vs_shape_cosine", "fair_vs_net_cosine",
    "best_so_far_roughness", "best_so_far_iteration",
    "patience_motion", "patience_roughness", "patience_m4",
    "patience_proposal_clean", "patience_repair", "patience_force_stable",
]


def _write_csv(rows: List[Dict[str, Any]], path: str,
               columns: Optional[List[str]] = None) -> Optional[str]:
    cols = columns if columns is not None else _CSV_COLUMNS
    path = _unique_path(path)
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k, "") for k in cols})
        print("[final_cleanup] wrote CSV log  -> {0}".format(path))
        return path
    except OSError as exc:
        print("[final_cleanup] WARNING: could not write CSV log ({0})".format(exc))
        return None


def _iteration_csv_row(rec: Dict[str, Any]) -> Dict[str, Any]:
    row = {"iteration": rec["iteration"], "phase": rec.get("phase", ""),
          "active_count": rec["active_count"],
          "w_m4_effective": round(rec.get("w_m4_effective", 0.0), 6),
          "frozen_m4_vertex_count": rec.get("frozen_m4_vertex_count", 0)}
    for prefix, key in (("fair", "fairing_displacement"), ("m4", "m4_displacement"),
                        ("shape", "shape_displacement"), ("clearance", "clearance_displacement"),
                        ("repair", "repair_displacement"), ("net", "net_displacement")):
        row[prefix + "_disp_mean"] = round(rec[key]["mean"], 8)
        row[prefix + "_disp_max"] = round(rec[key]["max"], 8)
    raw = rec.get("raw_proposal_displacement") or {"mean": 0.0, "max": 0.0}
    row["raw_proposal_mean"] = round(raw["mean"], 8)
    row["raw_proposal_max"] = round(raw["max"], 8)
    row["roughness_before"] = round(rec["roughness"]["before"], 8)
    row["roughness_after"] = round(rec["roughness"]["after"], 8)
    row["roughness_ceiling"] = round(rec.get("roughness_ceiling", 0.0), 8)
    row["core_roughness"] = round(rec.get("core_roughness", 0.0), 8)
    row["transition_roughness"] = round(rec.get("transition_roughness", 0.0), 8)
    row["m4_disagreement_before"] = round(rec["m4_disagreement"]["before"], 8)
    row["m4_disagreement_after"] = round(rec["m4_disagreement"]["after"], 8)
    row["proposal_face_count"] = rec["intersections"]["face_count"]
    row["intersection_pair_count"] = rec["intersections"]["pair_count"]
    row["repaired_component_count"] = rec["intersections"]["repaired_component_count"]
    row["min_exact_distance"] = round(rec["min_exact_distance"], 6)
    row["no_forbidden_intersections"] = rec["no_forbidden_intersections"]
    row["oscillating_count"] = rec.get("oscillating_count", 0)
    opp = rec.get("force_opposition") or {}
    row["fair_vs_m4_cosine"] = round((opp.get("fair_vs_m4") or {}).get("mean_cosine", 0.0), 4)
    row["fair_vs_shape_cosine"] = round((opp.get("fair_vs_shape") or {}).get("mean_cosine", 0.0), 4)
    row["fair_vs_net_cosine"] = round((opp.get("fair_vs_net") or {}).get("mean_cosine", 0.0), 4)
    best = rec.get("best_so_far") or {}
    row["best_so_far_roughness"] = round(best.get("roughness", 0.0), 8)
    row["best_so_far_iteration"] = best.get("iteration", 0)
    row["patience_motion"] = rec["patience"]["motion"]
    row["patience_roughness"] = rec["patience"]["roughness"]
    row["patience_m4"] = rec["patience"]["m4"]
    row["patience_proposal_clean"] = rec["patience"].get("proposal_clean", 0)
    row["patience_repair"] = rec["patience"].get("repair", 0)
    row["patience_force_stable"] = rec["patience"].get("force_stable", 0)
    return row


_CSV_COLUMNS_FAIRING = [
    "iteration", "active_count",
    "fair_disp_mean", "fair_disp_max",
    "shape_disp_mean", "shape_disp_max",
    "raw_proposal_mean", "raw_proposal_max",
    "clearance_disp_mean", "clearance_disp_max",
    "repair_disp_mean", "repair_disp_max",
    "net_disp_mean", "net_disp_max",
    "roughness_before", "roughness_after",
    "core_roughness_before", "core_roughness_after",
    "proposal_face_count", "intersection_pair_count", "repaired_component_count",
    "min_exact_distance", "no_forbidden_intersections", "oscillating_count",
    "fair_vs_shape_cosine",
    "best_so_far_roughness", "best_so_far_iteration",
    "patience_motion", "patience_roughness",
    "patience_proposal_clean", "patience_repair", "patience_force_stable",
]


def _iteration_csv_row_fairing(rec: Dict[str, Any]) -> Dict[str, Any]:
    row = {"iteration": rec["iteration"], "active_count": rec["active_count"]}
    for prefix, key in (("fair", "fairing_displacement"), ("shape", "shape_displacement"),
                        ("clearance", "clearance_displacement"),
                        ("repair", "repair_displacement"), ("net", "net_displacement")):
        row[prefix + "_disp_mean"] = round(rec[key]["mean"], 8)
        row[prefix + "_disp_max"] = round(rec[key]["max"], 8)
    raw = rec.get("raw_proposal_displacement") or {"mean": 0.0, "max": 0.0}
    row["raw_proposal_mean"] = round(raw["mean"], 8)
    row["raw_proposal_max"] = round(raw["max"], 8)
    row["roughness_before"] = round(rec["roughness"]["before"], 8)
    row["roughness_after"] = round(rec["roughness"]["after"], 8)
    row["core_roughness_before"] = round(rec.get("core_roughness", {}).get("before", 0.0), 8)
    row["core_roughness_after"] = round(rec.get("core_roughness", {}).get("after", 0.0), 8)
    row["proposal_face_count"] = rec["intersections"]["face_count"]
    row["intersection_pair_count"] = rec["intersections"]["pair_count"]
    row["repaired_component_count"] = rec["intersections"]["repaired_component_count"]
    row["min_exact_distance"] = round(rec["min_exact_distance"], 6)
    row["no_forbidden_intersections"] = rec["no_forbidden_intersections"]
    row["oscillating_count"] = rec.get("oscillating_count", 0)
    row["fair_vs_shape_cosine"] = round((rec.get("force_opposition") or {}).get("mean_cosine", 0.0), 4)
    best = rec.get("best_so_far") or {}
    row["best_so_far_roughness"] = round(best.get("roughness", 0.0), 8)
    row["best_so_far_iteration"] = best.get("iteration", 0)
    row["patience_motion"] = rec["patience"]["motion"]
    row["patience_roughness"] = rec["patience"]["roughness"]
    row["patience_proposal_clean"] = rec["patience"].get("proposal_clean", 0)
    row["patience_repair"] = rec["patience"].get("repair", 0)
    row["patience_force_stable"] = rec["patience"].get("force_stable", 0)
    return row


# =============================================================================
# 4. MAIN ENTRY POINT
# =============================================================================

def run_cleanup_solver(skin_mesh: str,
                       anatomical_meshes: Sequence[str],
                       target_offset: Optional[float] = None,
                       min_clearance: Optional[float] = None,
                       indices: Optional[Sequence[int]] = None,
                       m5_report: Optional[Dict[str, Any]] = None,
                       anatomy_backend: Optional[Any] = None,
                       sdf_query: Optional[Any] = None,
                       m5_detector_kwargs: Optional[Dict[str, Any]] = None,
                       # --- active region -------------------------------------
                       cleanup_growth_rings: int = DEFAULT_CLEANUP_GROWTH_RINGS,
                       transition_rings: int = DEFAULT_TRANSITION_RINGS,
                       boundary_buffer_rings: int = DEFAULT_BOUNDARY_BUFFER_RINGS,
                       # --- forces ----------------------------------------------
                       w_fair: float = DEFAULT_W_FAIR,
                       w_m4: float = DEFAULT_W_M4,
                       w_shape: float = DEFAULT_W_SHAPE,
                       m4_only_fair_scale: float = DEFAULT_M4_ONLY_FAIR_SCALE,
                       overlap_m4_scale: float = DEFAULT_OVERLAP_M4_SCALE,
                       max_step_edge_ratio: float = DEFAULT_MAX_STEP_EDGE_RATIO,
                       # --- anatomy feasibility -----------------------------
                       clearance_policy: str = DEFAULT_CLEARANCE_POLICY,
                       clearance_tolerance: float = DEFAULT_CLEARANCE_TOLERANCE,
                       max_constraint_iterations: int = DEFAULT_MAX_CONSTRAINT_ITERATIONS,
                       intersection_tolerance: float = DEFAULT_INTERSECTION_TOLERANCE,
                       repair_profile: str = DEFAULT_REPAIR_PROFILE,
                       repair_kwargs: Optional[Dict[str, Any]] = None,
                       # --- Phase 0: initial feasibility ----------------------
                       run_initial_feasibility: bool = True,
                       max_feasibility_passes: int = DEFAULT_MAX_FEASIBILITY_PASSES,
                       # --- region-dependent shape restraint -------------------
                       core_shape_scale: float = DEFAULT_CORE_SHAPE_SCALE,
                       # --- M4 force decay (persistent, monotonic; see decay_m4_weight) ---
                       m4_decay_patience: int = DEFAULT_M4_DECAY_PATIENCE,
                       m4_decay_rate: float = DEFAULT_M4_DECAY_RATE,
                       m4_min_weight_fraction: float = DEFAULT_M4_MIN_WEIGHT_FRACTION,
                       m4_freeze_relative_tolerance: float = DEFAULT_M4_FREEZE_RELATIVE_TOLERANCE,
                       # --- rest-reference low-pass filtering ------------------
                       rest_reference_smoothing_iterations: int = DEFAULT_REST_REFERENCE_SMOOTHING_ITERATIONS,
                       rest_reference_smoothing_strength: float = DEFAULT_REST_REFERENCE_SMOOTHING_STRENGTH,
                       # --- roughness as an acceptance target ------------------
                       max_roughness_ratio: float = DEFAULT_MAX_ROUGHNESS_RATIO,
                       repair_displacement_tolerance: float = DEFAULT_REPAIR_DISPLACEMENT_TOLERANCE,
                       force_stable_tolerance: float = DEFAULT_FORCE_STABLE_TOLERANCE,
                       # --- best-feasible-state tracking -----------------------
                       track_best_state: bool = True,
                       # --- oscillation detection -------------------------------
                       oscillation_window: int = DEFAULT_OSCILLATION_WINDOW,
                       oscillation_repeat_threshold: int = DEFAULT_OSCILLATION_REPEAT_THRESHOLD,
                       oscillation_damping: float = DEFAULT_OSCILLATION_DAMPING,
                       # --- dynamic active set --------------------------------
                       dynamic_active_set: bool = True,
                       dynamic_growth_zscore: float = DEFAULT_DYNAMIC_GROWTH_ZSCORE,
                       max_active_growth_fraction: float = DEFAULT_MAX_ACTIVE_GROWTH_FRACTION,
                       redetect_every: Optional[int] = None,
                       # --- iteration / convergence --------------------------
                       max_iterations: int = DEFAULT_MAX_ITERATIONS,
                       convergence_patience: int = DEFAULT_CONVERGENCE_PATIENCE,
                       mean_displacement_tolerance: float = DEFAULT_MEAN_DISPLACEMENT_TOLERANCE,
                       max_displacement_tolerance: float = DEFAULT_MAX_DISPLACEMENT_TOLERANCE,
                       roughness_rel_improvement_tolerance: float = DEFAULT_ROUGHNESS_REL_IMPROVEMENT_TOLERANCE,
                       m4_rel_improvement_tolerance: float = DEFAULT_M4_REL_IMPROVEMENT_TOLERANCE,
                       # --- output / safety -----------------------------------
                       apply: bool = True,
                       verbose: bool = True,
                       report_interval: int = 5,
                       select_final: bool = True,
                       select_on_failure: bool = True,
                       create_backup: bool = True,
                       backup_suffix: str = _BACKUP_SUFFIX,
                       log_path: Optional[str] = None,
                       save_json: bool = True,
                       save_csv: bool = True,
                       ) -> Dict[str, Any]:
    """Run the unified fairing / anatomy-projection / repair solver to
    convergence (see the module docstring for the full design).

    Self-contained: needs only ``skin_mesh`` + ``anatomical_meshes`` (a fresh
    Maya session has no pre-existing Python state to rely on). ``target_offset``
    (M4's D0) and ``min_clearance`` (the hard anatomy floor) are both computed
    automatically from the scene's own distance distribution when omitted --
    see :func:`artifact_detection.summarize_skin_sdf_values`. Pass ``apply=False``
    for a dry run: every step still runs, in memory, but the scene is never
    written and no backup is made.

    Parameters
    ----------
    skin_mesh, anatomical_meshes:
        The registered skin and the internal anatomy mesh names.
    target_offset:
        M4's anatomy-offset iso-value ``d0``. Auto-picked as the scene's median
        skin->anatomy distance when ``None``.
    min_clearance:
        Hard anatomy floor (same role as the registration's
        ``collision_min_distance``). Defaults to ``target_offset`` when ``None``
        -- pass the project's actual collision floor explicitly for a faithful
        run (the d98 wrapper does this).
    indices, m5_report:
        Optional pre-computed M5 region / report, to avoid re-running M5 (e.g.
        for an A/B comparison against an identical baseline detection). If both
        are ``None``, M5 is run once here.
    anatomy_backend, sdf_query:
        Optional pre-built :class:`anatomy_constraint.MayaAnatomyBackend` /
        :class:`artifact_detection.AnatomySDFQuery` to reuse (e.g. the exact
        smooth-min field the registration itself uses). Built fresh over
        ``anatomical_meshes`` via :func:`mesh_utils.get_mesh_fn` when omitted,
        so this function works standalone.
    cleanup_growth_rings, transition_rings:
        Size of the full-strength fairing band and the damped transition band
        grown around the M5/intersection core (see :func:`build_active_region`).
    w_fair, w_m4, w_shape:
        Base weights for the three correction forces (see :func:`combine_step`
        and the module docstring's provenance rules).
    max_step_edge_ratio:
        Per-vertex trust-region cap, as a fraction of THAT vertex's own mean
        edge length. There is no shared/global step scalar anywhere in this
        solver.
    dynamic_active_set:
        If True, a vertex just outside the current fairing region is absorbed
        into it once its local roughness exceeds a scene-wide threshold
        (bounded by ``max_active_growth_fraction`` of the initial core size).
    redetect_every:
        If set (and ``apply=True``), re-run M5 every N iterations and union any
        newly detected vertices into the active region. Ignored when
        ``apply=False`` (M5 needs to read the live mesh).
    run_initial_feasibility, max_feasibility_passes:
        Phase 0 (see the module docstring): repair every pre-existing surface
        intersection and use the result as the rest/positional reference,
        before any iteration runs. Disabling this reproduces the earlier
        (buggy) behaviour of pulling toward the raw, possibly-invalid,
        registered mesh -- keep it on unless you are specifically A/B testing
        Phase 0 itself.
    core_shape_scale:
        Shape-restraint multiplier at the artifact core (``1.0`` at the outer
        transition edge); lower values give local defects more freedom to fair
        instead of being pulled back toward their (possibly still slightly
        rough) Phase-0-repaired position.
    m4_decay_patience, m4_decay_rate, m4_min_weight_fraction:
        M4 is guidance, not a final target: once its relative improvement has
        plateaued for ``m4_decay_patience`` iterations, ``w_m4`` decays
        geometrically by ``m4_decay_rate`` per further plateaued iteration,
        floored at ``w_m4 * m4_min_weight_fraction``.
    max_roughness_ratio:
        Convergence additionally requires final roughness to be at or below
        ``max_roughness_ratio`` times the ORIGINAL (pre-cleanup) roughness --
        "anatomically safe" is necessary but not sufficient for convergence.
    oscillation_window, oscillation_repeat_threshold, oscillation_damping:
        A vertex needing repair in ``oscillation_repeat_threshold`` of the last
        ``oscillation_window`` iterations is a fair<->repair ping-pong; its
        fairing/M4 weights are damped by ``oscillation_damping`` (once) so it
        holds its last safe position while the rest of the region keeps
        converging, instead of fighting to ``max_iterations``.
    m4_freeze_relative_tolerance:
        Per-vertex M4 freeze: once a vertex's OWN distance to its M4 target is
        within this fraction of ``target_offset``, its individual M4 weight
        drops to 0 regardless of the global decay state -- some M4 vertices
        converge long before others.
    rest_reference_smoothing_iterations, rest_reference_smoothing_strength:
        Low-pass filter the shape-restraint REFERENCE (never the actual
        working geometry) over the fairing region after Phase 0, so
        shape-restraint holds low-frequency facial form without also
        preserving Phase 0's own high-frequency repair bumps. Set iterations
        to 0 to disable and isolate this as a variable.
    force_stable_tolerance:
        Convergence also requires the RAW (pre-trust-region-clamp) combined
        force to have stabilized (relative change below this tolerance), not
        just the net accepted displacement -- a state where forces cancel to
        a small net while the underlying pull is still large/changing is not
        genuine convergence.
    track_best_state:
        If True (default), the best anatomically-valid state seen (lowest
        roughness, ties broken by lower M4 disagreement) is tracked and used
        as the final result if the LAST iteration ends up worse -- the
        adaptive schedule should not discard a better state it passed through.
    max_iterations, convergence_patience, *_tolerance:
        Hard cap and the multi-signal convergence gate (see the module
        docstring). ``max_iterations`` is a safety cap, not the target.
        Convergence now requires ALL of: no forbidden intersections, the final
        proposal being clean (no NEW intersections to repair), repair
        displacement near zero, net displacement near zero, the raw combined
        force stabilized, roughness relative-improvement near zero AND under
        the roughness ceiling, and M4 relative-improvement near zero -- each
        sustained for ``convergence_patience`` consecutive iterations.

    Returns
    -------
    dict
        Full run report: ``converged``, ``iterations``, ``stop_reason``,
        before/after intersection / roughness / M4-disagreement / displacement
        metrics, the per-iteration log, and (on failure) the unresolved faces /
        anatomy meshes / components for manual inspection.
    """
    t_run0 = time.time()

    if not mesh_utils.mesh_exists(skin_mesh):
        raise ValueError("final_cleanup_solver: skin_mesh '{0}' does not exist".format(skin_mesh))
    if not anatomical_meshes:
        raise ValueError("final_cleanup_solver: anatomical_meshes is required "
                         "(list of internal mesh names)")
    if max_iterations < 1:
        raise ValueError("max_iterations must be >= 1, got {0}".format(max_iterations))
    if convergence_patience < 1:
        raise ValueError("convergence_patience must be >= 1, got {0}".format(convergence_patience))

    positions0 = mesh_utils.get_mesh_vertices(skin_mesh)
    if not positions0:
        raise ValueError("final_cleanup_solver: skin_mesh '{0}' has no readable "
                         "vertices".format(skin_mesh))
    neighbors = mesh_utils.get_vertex_neighbors(skin_mesh)
    normals = mesh_utils.get_vertex_normals(skin_mesh)
    mesh_fn = mesh_utils.get_mesh_fn(skin_mesh)
    skin_topology = mesh_utils.get_triangle_topology(mesh_fn) if mesh_fn is not None else None
    if not skin_topology or not skin_topology.get("triangles"):
        raise ValueError("final_cleanup_solver: could not read triangle topology "
                         "for '{0}'".format(skin_mesh))

    boundary = _boundary_set(skin_mesh, neighbors, boundary_buffer_rings)

    if anatomy_backend is None:
        anatomy_backend, valid_anatomical_meshes = _build_anatomy_backend(anatomical_meshes)
    else:
        valid_anatomical_meshes = list(anatomy_backend.mesh_names())

    if target_offset is None:
        summary = artifact_detection.summarize_skin_sdf_values(skin_mesh, valid_anatomical_meshes)
        if not summary.get("count"):
            raise ValueError("final_cleanup_solver: could not auto-compute "
                             "target_offset (no finite SDF samples); pass it explicitly")
        target_offset = summary["median"]
        if verbose:
            print("[final_cleanup] auto target_offset (D0) = {0:.4f} "
                  "(scene median skin->anatomy distance)".format(target_offset))

    if min_clearance is None:
        min_clearance = target_offset
        if verbose:
            print("[final_cleanup] min_clearance not given; using target_offset "
                  "({0:.4f}) as the anatomy floor".format(min_clearance))

    # --- initial M5 detection (unless the caller supplied one) ---------------
    if m5_report is None and indices is None:
        t0 = time.time()
        m5_indices, m5_report = artifact_detection.detect_unified_artifacts(
            skin_mesh, valid_anatomical_meshes, target_offset, select=False,
            **dict(m5_detector_kwargs or {}))
        m5_seconds = time.time() - t0
        if verbose:
            print("[final_cleanup] M5 baseline: {0} vertices ({1:.2f}s)".format(
                len(m5_indices), m5_seconds))
    elif m5_report is not None:
        m5_indices = sorted(set(m5_report.get("final_indices") or []))
    else:
        m5_indices = sorted(set(int(i) for i in indices))
        m5_report = {"final_indices": m5_indices, "m3_indices": [], "m4_indices": [],
                     "overlap_indices": [], "m3_only_indices": list(m5_indices),
                     "m4_only_indices": []}

    m3_only0, m4_only0, overlap0 = _m5_provenance_sets(m5_report)

    if create_backup and apply and m5_indices and maya_io is not None:
        backup_name = skin_mesh + backup_suffix
        if not mesh_utils.mesh_exists(backup_name):
            maya_io.duplicate_mesh(skin_mesh, suffix=backup_suffix)
        elif verbose:
            print("[final_cleanup] backup '{0}' already exists; keeping it".format(backup_name))

    repair_kw = dict(repair_kwargs or {})
    repair_kw.setdefault("repair_profile", repair_profile)

    # --- one whole-mesh intersection scan seeds the core with any PRE-EXISTING
    #     surface intersections M5's score-based detectors would not flag ------
    isect0_core, isect0_report = _scoped_intersection_core(
        skin_mesh, positions0, list(range(len(positions0))), anatomy_backend,
        neighbors, skin_topology, boundary_buffer_rings, intersection_tolerance)

    # --- PHASE 0: initial feasibility -----------------------------------------
    # Repair every PRE-EXISTING real intersection BEFORE the rest/positional
    # reference is captured and before any fairing/M4 force is computed. This
    # is what stops the iterative loop from ever asking the shape-restraint
    # force to pull repaired geometry back toward a position that was itself
    # anatomically invalid (see the module docstring). NOT counted as an
    # ordinary fairing iteration; reported separately.
    if run_initial_feasibility and isect0_core:
        if verbose:
            print("[final_cleanup] --- Phase 0: initial feasibility "
                  "({0} pre-existing intersecting face(s)) ---".format(
                      isect0_report.get("intersecting_skin_face_count", 0)))
        feasibility = _run_initial_feasibility(
            positions0, skin_mesh, anatomy_backend, neighbors, normals,
            skin_topology, min_clearance, clearance_policy, boundary_buffer_rings,
            intersection_tolerance, repair_kw, max_feasibility_passes, verbose)
        feasible_positions = feasibility["positions"]
        phase0_touched = set(feasibility["touched_vertices"])
        if verbose:
            print("[final_cleanup] Phase 0: {0} pass(es), resolved={1}, "
                  "faces {2}->{3}, touched={4}, disp mean/max={5:.4f}/{6:.4f}".format(
                      feasibility["passes"], feasibility["resolved"],
                      feasibility["pre_intersections"]["face_count"],
                      feasibility["post_intersections"]["face_count"],
                      len(phase0_touched), feasibility["displacement"]["mean"],
                      feasibility["displacement"]["max"]))
    else:
        feasible_positions = [list(p) for p in positions0]
        phase0_touched = set()
        feasibility = {
            "positions": feasible_positions, "touched_vertices": [],
            "pre_intersections": {"face_count": isect0_report.get("intersecting_skin_face_count", 0),
                                 "pair_count": isect0_report.get("intersection_pair_count", 0)},
            "post_intersections": {"face_count": isect0_report.get("intersecting_skin_face_count", 0),
                                  "pair_count": isect0_report.get("intersection_pair_count", 0)},
            "passes": 0, "resolved": not bool(isect0_core),
            "displacement": {"count": 0, "mean": 0.0, "max": 0.0},
            "roughness_before": 0.0, "roughness_after": 0.0,
        }

    core0 = sorted((set(m5_indices) | set(isect0_core) | phase0_touched) - boundary)
    if not core0:
        if verbose:
            print("[final_cleanup] no artifact / intersection region detected; nothing to do")
        return {
            "skin_mesh": skin_mesh, "converged": True, "iterations": 0,
            "stop_reason": STOP_NO_ARTIFACT_REGION, "dry_run": not apply,
            "target_offset": target_offset, "min_clearance": min_clearance,
            "m5_report": m5_report, "runtime_seconds": time.time() - t_run0,
        }

    region = build_active_region(core0, neighbors, boundary,
                                 cleanup_growth_rings, transition_rings)
    fairing_region: Set[int] = set(region["fairing"])
    active: Set[int] = set(region["active"])
    anchor: Dict[int, float] = region["anchor"]
    provenance = classify_provenance(sorted(active), m3_only0, m4_only0, overlap0)
    weights: Dict[int, Tuple[float, float]] = provenance_weights(
        sorted(active), provenance, w_fair, w_m4, m4_only_fair_scale, overlap_m4_scale)

    # --- static M4 targets (anatomy does not move; computed once, against the
    #     FEASIBLE starting positions so the Newton search begins from valid
    #     geometry) --------------------------------------------------------
    if sdf_query is None:
        resolved = artifact_detection.resolve_anatomical_meshes(valid_anatomical_meshes)
        sdf_query = (artifact_detection.build_anatomy_sdf_query(resolved["valid"])
                    if resolved["valid"] else None)
    m4_needed = sorted(i for i in active if weights[i][1] > 0.0)
    m4_targets: Dict[int, List[float]] = _compute_m4_targets(
        feasible_positions, m4_needed, sdf_query, target_offset)

    # --- baseline-aware clearance floors, computed against the FEASIBLE state -
    floors, _orig_dist, policy_used = anatomy_constraint.compute_clearance_floors(
        feasible_positions, sorted(active), anatomy_backend, min_clearance,
        clearance_policy=clearance_policy, clearance_tolerance=clearance_tolerance)

    # --- dynamic-growth roughness threshold (one-time, whole-mesh reference) -
    roughness_threshold = float("inf")
    if dynamic_active_set:
        whole_scores = artifact_detection.compute_laplacian_scores(skin_mesh)
        whole_stats = artifact_detection.summarize_scores(whole_scores)
        roughness_threshold = whole_stats["mean"] + dynamic_growth_zscore * whole_stats["std"]

    # --- THE REST/POSITIONAL REFERENCE IS THE FEASIBLE STATE, NOT THE RAW
    #     ORIGINAL REGISTERED MESH. Pulling shape-restraint toward positions
    #     that were themselves anatomically invalid is what turned repair into
    #     a fight it could never win (root cause; see the module docstring).
    #     The reference is ADDITIONALLY low-pass filtered over the fairing
    #     region only, so shape-restraint holds low-frequency facial form
    #     without also preserving Phase 0's own high-frequency repair bumps
    #     (see DEFAULT_REST_REFERENCE_SMOOTHING_ITERATIONS). This never
    #     touches the actual working geometry ``x`` and is never anatomy-
    #     checked -- it is a soft target, not enforced geometry.
    x_rest = [list(p) for p in feasible_positions]
    if rest_reference_smoothing_iterations > 0 and fairing_region:
        x_rest = _smooth_rest_reference(
            x_rest, neighbors, sorted(fairing_region),
            rest_reference_smoothing_iterations, rest_reference_smoothing_strength)
    x = [list(p) for p in feasible_positions]   # current accepted state (in-memory working copy)

    # Roughness measured on the ORIGINAL registered skin (before Phase 0 too)
    # -- this is the honest "did we end up smoother than we started" baseline
    # and defines the roughness ceiling below, never something the iterative
    # loop is allowed to silently exceed forever.
    roughness_original = mean_laplacian_magnitude(positions0, neighbors, sorted(fairing_region))
    roughness_ceiling = roughness_original * max_roughness_ratio
    roughness0 = mean_laplacian_magnitude(x, neighbors, sorted(fairing_region))  # post-Phase-0 start
    core_roughness0 = mean_laplacian_magnitude(x, neighbors, core0)
    m4_disagreement0 = mean_target_disagreement(x, m4_targets, sorted(m4_targets.keys()))
    min_dist0 = min((anatomy_backend.exact_closest(x[i])["distance"] for i in sorted(active)),
                    default=float("inf"))

    iterations_log: List[Dict[str, Any]] = []
    csv_rows: List[Dict[str, Any]] = []
    patience = {"motion": 0, "roughness": 0, "m4": 0, "proposal_clean": 0, "repair": 0,
               "force_stable": 0}
    prev_roughness = roughness0
    prev_m4_disagreement = m4_disagreement0
    prev_raw_force_mean = float("inf")
    stop_reason: Optional[str] = None
    converged = False
    total_active_growth = 0
    max_growth_budget = int(math.ceil(max_active_growth_fraction * len(core0)))
    repair_history: List[Set[int]] = []
    damped_oscillating: Set[int] = set()
    oscillating_details: List[Dict[str, Any]] = []

    # --- persistent, monotonic M4 decay state (see decay_m4_weight) ----------
    w_m4_effective = w_m4
    m4_floor = w_m4 * m4_min_weight_fraction
    m4_freeze_distance = m4_freeze_relative_tolerance * float(target_offset)
    m4_plateau_streak = 0
    frozen_m4_vertices: Set[int] = set()

    # --- best-feasible-state tracking (never discard a better state later
    #     iterations regress from; ranked by roughness, then M4 disagreement --
    #     every ACCEPTED state here is already anatomy-valid by construction) -
    best_state: Optional[List[List[float]]] = None
    best_roughness = float("inf")
    best_m4 = float("inf")
    best_iteration = 0

    it = 0

    if verbose:
        print("[final_cleanup] core={0} fairing={1} transition={2} active={3} "
              "(provenance: m3_only={4} m4_only={5} overlap={6} grown={7}) "
              "roughness_ceiling={8:.5f}".format(
                  len(core0), len(fairing_region), len(active) - len(fairing_region),
                  len(active),
                  sum(1 for p in provenance.values() if p == "m3_only"),
                  sum(1 for p in provenance.values() if p == "m4_only"),
                  sum(1 for p in provenance.values() if p == "overlap"),
                  sum(1 for p in provenance.values() if p == "grown"),
                  roughness_ceiling))

    for it in range(1, int(max_iterations) + 1):
        x_before = [list(p) for p in x]
        active_list = sorted(active)

        local_edge = local_edge_lengths(x, neighbors, active_list)
        fair_w = {i: weights[i][0] for i in active_list}
        # Region-dependent shape restraint: weakest at the core (freedom to
        # fair local defects, including Phase 0's own repair bumps), ramping
        # to full strength across the transition band.
        shape_w = {i: w_shape * shape_weight_from_anchor(anchor.get(i, 0.0), core_shape_scale)
                  for i in active_list}
        # M4 is guidance, not a final target. Global component: the
        # PERSISTENT, MONOTONIC w_m4_effective (ratcheted below; never simply
        # recomputed from the current patience count -- see decay_m4_weight
        # and the module docstring for why an earlier version undid its own
        # decay). Per-vertex component: a vertex already close to its OWN M4
        # target is frozen (weight 0) regardless of the global state, since
        # some M4 vertices converge long before others.
        frozen_m4_vertices = {
            i for i in active_list
            if weights[i][1] > 0.0 and i in m4_targets
            and mesh_utils.vec_length(mesh_utils.vec_sub(x[i], m4_targets[i])) <= m4_freeze_distance
        }
        m4_w = {i: (0.0 if i in frozen_m4_vertices else weights[i][1] * (
                    (w_m4_effective / w_m4) if w_m4 else 0.0))
               for i in active_list}

        fair_f = fairing_force(x, neighbors, active_list, fair_w)
        shape_f = shape_force(x, x_rest, active_list, shape_w)
        m4_f = m4_force(x, m4_targets, active_list, m4_w)

        # RAW combined force, BEFORE the trust-region clamp / anchor damping --
        # this is what "genuinely settled" means for the force_stable
        # convergence signal, and what the fair-vs-M4/fair-vs-shape opposition
        # diagnostics below are measured on.
        raw_combined = {i: [fair_f[i][k] + shape_f[i][k] + m4_f[i][k] for k in range(3)]
                        for i in active_list}
        raw_force_mags = [mesh_utils.vec_length(v) for v in raw_combined.values()]
        raw_force_mean = (sum(raw_force_mags) / len(raw_force_mags)) if raw_force_mags else 0.0
        opp_fair_m4 = force_opposition(fair_f, m4_f, active_list)
        opp_fair_shape = force_opposition(fair_f, shape_f, active_list)

        step = combine_step(fair_f, shape_f, m4_f, active_list, anchor,
                            max_step_edge_ratio, local_edge)
        x_proposed = apply_step(x, step)  # RAW proposal; may temporarily violate anatomy

        x_projected, clearance_moved = _anatomy_project_clearance(
            x_proposed, active_list, anatomy_backend, floors, clearance_tolerance,
            max_constraint_iterations, normals)

        core_now, isect_report = _scoped_intersection_core(
            skin_mesh, x_projected, active_list, anatomy_backend, neighbors,
            skin_topology, boundary_buffer_rings, intersection_tolerance)

        repair_result = None
        repair_disp: List[float] = []
        repair_footprint: Set[int] = set()
        if core_now:
            x_accepted, repair_result = _repair_local(
                x_projected, core_now, anatomy_backend, skin_mesh, neighbors,
                normals, skin_topology, min_clearance, clearance_policy,
                boundary_buffer_rings, intersection_tolerance, repair_kw)
            if repair_result:
                # The broad repair's own patch growth (repair_blend_rings, ~6
                # rings by default) is NOT clipped to this solver's active set
                # -- verify and absorb the FULL footprint it actually touched,
                # not just the pre-repair active vertices (see the module
                # docstring: "repair patch validation footprint").
                repair_footprint = set(repair_result.get("patch_vertices") or core_now)
                repair_disp = [mesh_utils.vec_length(mesh_utils.vec_sub(x_accepted[i], x_projected[i]))
                              for i in sorted(repair_footprint)]
        else:
            x_accepted = x_projected

        # Final local verification: the ACCEPTED state must be intersection-free
        # over EVERY vertex this iteration could have touched -- the pre-repair
        # active set AND whatever the repair patch grew into. If repair could
        # not fully clear it (rare -- repair has its own bounded pass count),
        # roll back ONLY the still-offending vertices, never the whole iteration.
        verify_scope = sorted(set(active_list) | repair_footprint)
        residual_core, residual_report = _scoped_intersection_core(
            skin_mesh, x_accepted, verify_scope, anatomy_backend, neighbors,
            skin_topology, boundary_buffer_rings, intersection_tolerance)
        if residual_core:
            for i in residual_core:
                x_accepted[i] = list(x_before[i])
            residual_core, residual_report = _scoped_intersection_core(
                skin_mesh, x_accepted, verify_scope, anatomy_backend, neighbors,
                skin_topology, boundary_buffer_rings, intersection_tolerance)

        net_disp = [mesh_utils.vec_length(mesh_utils.vec_sub(x_accepted[i], x_before[i]))
                   for i in active_list]
        net_step = {i: mesh_utils.vec_sub(x_accepted[i], x_before[i]) for i in active_list}
        x = x_accepted

        # Absorb anything the repair patch touched beyond the pre-repair active
        # set into the region going forward, so it gets faired (not left as a
        # frozen, untracked bump) and validated on every later iteration too.
        grown_by_repair = repair_footprint - active
        if grown_by_repair:
            fairing_region, active, anchor = _absorb_vertices(
                grown_by_repair, fairing_region, boundary, neighbors, transition_rings,
                provenance, weights, floors, x, anatomy_backend, min_clearance,
                clearance_policy, clearance_tolerance, w_fair, tag="repaired")

        roughness_after = mean_laplacian_magnitude(x, neighbors, sorted(fairing_region))
        core_roughness_after = mean_laplacian_magnitude(x, neighbors, core0)
        transition_roughness_after = mean_laplacian_magnitude(
            x, neighbors, sorted(active - fairing_region))
        m4_after = mean_target_disagreement(x, m4_targets, sorted(m4_targets.keys()))
        min_dist_now = min((anatomy_backend.exact_closest(x[i])["distance"] for i in active_list),
                           default=float("inf"))

        mean_net = sum(net_disp) / len(net_disp) if net_disp else 0.0
        max_net = max(net_disp) if net_disp else 0.0
        roughness_rel = abs(prev_roughness - roughness_after) / (prev_roughness + 1e-9)
        no_forbidden = not bool(residual_core)
        repair_mean = _stats(repair_disp)["mean"]
        repair_max = _stats(repair_disp)["max"]
        proposal_clean = (isect_report.get("intersecting_skin_face_count", 0) == 0)
        force_rel = (abs(prev_raw_force_mean - raw_force_mean) / (prev_raw_force_mean + 1e-9)
                    if math.isfinite(prev_raw_force_mean) else 1.0)

        tiny_motion = (mean_net < mean_displacement_tolerance
                      and max_net < max_displacement_tolerance)
        patience["motion"] = patience["motion"] + 1 if tiny_motion else 0
        patience["roughness"] = (patience["roughness"] + 1
                                 if roughness_rel < roughness_rel_improvement_tolerance else 0)
        # M4 patience is SIGNED (see m4_is_improving): a regression caused by
        # our OWN deliberate decay must not reset it, or decay undoes itself
        # the moment it works -- this was the actual root cause of w_m4
        # bouncing 0.25 -> 0.175 -> 0.25 (see the module docstring). No M4
        # targets at all counts as "plateaued" (increment), matching how the
        # region behaved before M4 decay/freeze existed -- NOT as "always
        # still improving" (which would prevent the M4 signal from ever
        # reaching convergence_patience in a pure-M3 region).
        m4_plateaued = (not m4_targets) or not m4_is_improving(
            prev_m4_disagreement, m4_after, m4_rel_improvement_tolerance)
        patience["m4"] = patience["m4"] + 1 if m4_plateaued else 0
        patience["proposal_clean"] = patience["proposal_clean"] + 1 if proposal_clean else 0
        patience["repair"] = (patience["repair"] + 1
                              if repair_mean < repair_displacement_tolerance else 0)
        patience["force_stable"] = patience["force_stable"] + 1 if force_rel < force_stable_tolerance else 0

        # Ratchet the M4 decay: PERSISTS across iterations, only ever
        # decreases. Fires every m4_decay_patience-th consecutive plateaued
        # iteration, so it keeps ratcheting down the longer the plateau lasts.
        if should_decay_m4(patience["m4"], m4_decay_patience):
            w_m4_effective = decay_m4_weight(w_m4_effective, m4_decay_rate, m4_floor)

        phase = classify_phase(w_m4_effective, w_m4, m4_floor,
                               isect_report.get("intersecting_skin_face_count", 0),
                               repair_mean, repair_displacement_tolerance)

        # --- oscillation detection: a vertex repeatedly needing repair across
        # iterations is a fair<->repair ping-pong, not progress. Once flagged,
        # damp its fairing/M4 pull (once) so it holds its last safe position
        # while the REST of the region keeps converging, instead of silently
        # looping to max_iterations.
        repair_history, oscillating = update_oscillation_tracker(
            repair_history, core_now, oscillation_window, oscillation_repeat_threshold)
        newly_oscillating = oscillating - damped_oscillating
        newly_oscillating_info = []
        for j in newly_oscillating:
            if j in weights:
                fw, mw = weights[j]
                weights[j] = (fw * oscillation_damping, mw * oscillation_damping)
            info = {"vertex": j, "iteration": it, "provenance": provenance.get(j, "grown"),
                   "nearest_anatomy": anatomy_backend.exact_closest(x[j]).get("mesh")}
            newly_oscillating_info.append(info)
            oscillating_details.append(info)
            damped_oscillating.add(j)

        # --- large, unexplained late repair events should be attributable ----
        repair_spike = None
        if repair_disp and repair_result:
            trailing = [r["repair_displacement"]["max"] for r in iterations_log[-5:]]
            trailing_ref = (sum(trailing) / len(trailing)) if trailing else 0.0
            if repair_max > max(3.0 * trailing_ref, 0.5):
                comps = repair_result.get("repair_components") or []
                spike_provenance = sorted({provenance.get(v, "grown")
                                          for c in comps for v in c.get("core_vertices", [])})
                repair_spike = {
                    "iteration": it, "repair_max": repair_max, "trailing_reference": trailing_ref,
                    "offending_anatomy_meshes": repair_result.get("offending_anatomy_meshes") or [],
                    "provenance_involved": spike_provenance,
                    "component_count": len(comps),
                }

        # --- best-feasible-state tracking: every ACCEPTED state here is
        # already anatomy-valid by construction, so ranking is just
        # (roughness, then M4 disagreement) -- never silently discard a
        # better state a later, worse iteration regresses from.
        if track_best_state and is_better_state(roughness_after, m4_after, best_roughness, best_m4):
            best_state = [list(p) for p in x]
            best_roughness = roughness_after
            best_m4 = m4_after
            best_iteration = it

        record = {
            "iteration": it,
            "phase": phase,
            "active_count": len(active_list),
            "w_m4_effective": w_m4_effective,
            "frozen_m4_vertex_count": len(frozen_m4_vertices),
            "fairing_displacement": _stats([mesh_utils.vec_length(v) for v in fair_f.values()]),
            "m4_displacement": _stats([mesh_utils.vec_length(v) for v in m4_f.values()]),
            "shape_displacement": _stats([mesh_utils.vec_length(v) for v in shape_f.values()]),
            "raw_proposal_displacement": {"mean": raw_force_mean,
                                         "max": max(raw_force_mags) if raw_force_mags else 0.0},
            "clearance_displacement": _stats(list(clearance_moved.values())),
            "repair_displacement": _stats(repair_disp),
            "net_displacement": {"mean": mean_net, "max": max_net},
            "roughness": {"before": prev_roughness, "after": roughness_after},
            "roughness_ceiling": roughness_ceiling,
            "core_roughness": core_roughness_after,
            "transition_roughness": transition_roughness_after,
            "m4_disagreement": {"before": prev_m4_disagreement, "after": m4_after},
            "intersections": {
                "face_count": isect_report.get("intersecting_skin_face_count", 0),
                "pair_count": isect_report.get("intersection_pair_count", 0),
                "repaired_component_count": (repair_result or {}).get("repair_component_count", 0),
            },
            "force_opposition": {"fair_vs_m4": opp_fair_m4, "fair_vs_shape": opp_fair_shape,
                                "fair_vs_net": force_opposition(fair_f, net_step, active_list)},
            "min_exact_distance": min_dist_now,
            "no_forbidden_intersections": no_forbidden,
            "oscillating_count": len(damped_oscillating),
            "newly_oscillating": newly_oscillating_info,
            "repair_spike": repair_spike,
            "best_so_far": {"roughness": best_roughness, "iteration": best_iteration},
            "patience": dict(patience),
        }

        if verbose and (it == 1 or it % max(1, report_interval) == 0):
            buckets = provenance_buckets(active_list, provenance, damped_oscillating)
            bucket_diag = {}
            for tag, idxs in buckets.items():
                bucket_diag[tag] = {
                    "count": len(idxs),
                    "fair_mean": _stats([mesh_utils.vec_length(fair_f[i]) for i in idxs
                                       if i in fair_f])["mean"],
                    "m4_mean": _stats([mesh_utils.vec_length(m4_f[i]) for i in idxs
                                     if i in m4_f])["mean"],
                    "shape_mean": _stats([mesh_utils.vec_length(shape_f[i]) for i in idxs
                                        if i in shape_f])["mean"],
                    "net_mean": _stats([mesh_utils.vec_length(net_step[i]) for i in idxs
                                      if i in net_step])["mean"],
                }
            record["provenance_diagnostics"] = bucket_diag

        iterations_log.append(record)
        csv_rows.append(_iteration_csv_row(record))

        if verbose and (it == 1 or it % max(1, report_interval) == 0):
            print("  [iter {0:4d}] phase={1} active={2} net disp mean/max={3:.5f}/{4:.5f} "
                  "roughness {5:.5f}->{6:.5f} (ceiling {7:.5f}) m4 {8:.5f}->{9:.5f} "
                  "w_m4={10:.4f}(frozen={11}) proposal_faces={12} repair_disp={13:.5f} "
                  "osc={14} min_dist={15:.4f} patience(m/r/4/p/x/f)={16}/{17}/{18}/{19}/{20}/{21}".format(
                      it, phase, len(active_list), mean_net, max_net, prev_roughness,
                      roughness_after, roughness_ceiling, prev_m4_disagreement, m4_after,
                      w_m4_effective, len(frozen_m4_vertices), record["intersections"]["face_count"],
                      repair_mean, len(damped_oscillating), min_dist_now,
                      patience["motion"], patience["roughness"], patience["m4"],
                      patience["proposal_clean"], patience["repair"], patience["force_stable"]))
            print("    fair.m4 cosine={0:.3f} fair.shape cosine={1:.3f} fair.net cosine={2:.3f}".format(
                opp_fair_m4["mean_cosine"], opp_fair_shape["mean_cosine"],
                record["force_opposition"]["fair_vs_net"]["mean_cosine"]))
            if repair_spike:
                print("    REPAIR SPIKE: max={0:.4f} (vs trailing ~{1:.4f}) meshes={2} provenance={3}".format(
                    repair_spike["repair_max"], repair_spike["trailing_reference"],
                    repair_spike["offending_anatomy_meshes"], repair_spike["provenance_involved"]))

        prev_roughness = roughness_after
        prev_m4_disagreement = m4_after
        prev_raw_force_mean = raw_force_mean

        if (no_forbidden and roughness_after <= roughness_ceiling
                and patience["motion"] >= convergence_patience
                and patience["roughness"] >= convergence_patience
                and patience["m4"] >= convergence_patience
                and patience["proposal_clean"] >= convergence_patience
                and patience["repair"] >= convergence_patience
                and patience["force_stable"] >= convergence_patience):
            converged = True
            stop_reason = STOP_CONVERGED
            break

        # --- dynamic active-set growth (bounded, roughness-triggered) --------
        if dynamic_active_set and total_active_growth < max_growth_budget:
            frontier = sorted((set(mesh_utils.grow_indices(neighbors, sorted(active), rings=1))
                              - active) - boundary)
            newly: List[int] = []
            for j in frontier:
                if total_active_growth >= max_growth_budget:
                    break
                if mean_laplacian_magnitude(x, neighbors, [j]) > roughness_threshold:
                    newly.append(j)
                    total_active_growth += 1
            if newly:
                fairing_region, active, anchor = _absorb_vertices(
                    newly, fairing_region, boundary, neighbors, transition_rings,
                    provenance, weights, floors, x, anatomy_backend, min_clearance,
                    clearance_policy, clearance_tolerance, w_fair, tag="grown")
                if verbose:
                    print("  [iter {0:4d}] dynamic growth: +{1} vertices "
                          "(budget {2}/{3})".format(it, len(newly), total_active_growth,
                                                    max_growth_budget))

        # --- periodic M5 redetection (live mesh required) ---------------------
        if redetect_every and apply and (it % int(redetect_every) == 0):
            mesh_utils.set_mesh_vertices(skin_mesh, x)
            new_idx, new_report = artifact_detection.detect_unified_artifacts(
                skin_mesh, valid_anatomical_meshes, target_offset, select=False,
                **dict(m5_detector_kwargs or {}))
            extra = sorted((set(new_idx) - active) - boundary)
            if extra:
                fairing_region, active, anchor = _absorb_vertices(
                    extra, fairing_region, boundary, neighbors, transition_rings,
                    provenance, weights, floors, x, anatomy_backend, min_clearance,
                    clearance_policy, clearance_tolerance, w_fair, tag="m3_only")
                m3o, m4o, ov = _m5_provenance_sets(new_report)
                for j in extra:
                    if j in ov:
                        provenance[j] = "overlap"
                        weights[j] = (w_fair, w_m4 * overlap_m4_scale)
                    elif j in m4o:
                        provenance[j] = "m4_only"
                        weights[j] = (w_fair * m4_only_fair_scale, w_m4)
                more_m4 = sorted(j for j in extra if weights[j][1] > 0.0)
                if more_m4:
                    m4_targets.update(_compute_m4_targets(x, more_m4, sdf_query, target_offset))
                if verbose:
                    print("  [iter {0:4d}] M5 redetect: +{1} vertices".format(it, len(extra)))
    else:
        stop_reason = STOP_MAX_ITERATIONS

    if stop_reason is None:
        stop_reason = STOP_MAX_ITERATIONS

    # --- best-feasible-state reversion: never ship a worse result than one the
    # solver already passed through and accepted (every accepted state here is
    # anatomy-valid by construction, so this is a pure roughness/M4 safety net
    # against a still-adapting schedule regressing on its very last iteration).
    used_best_state = False
    if track_best_state and best_state is not None and best_roughness < prev_roughness - 1e-6:
        x = best_state
        used_best_state = True
        if verbose:
            print("[final_cleanup] reverting to best-feasible-state from iteration {0} "
                  "(roughness {1:.5f} vs final {2:.5f})".format(
                      best_iteration, best_roughness, prev_roughness))

    # --- write once, at the end (matches constrained_smooth_mesh_region) -----
    if apply:
        mesh_utils.set_mesh_vertices(skin_mesh, x)

    final_active = sorted(active)
    final_core, final_scoped_report = _scoped_intersection_core(
        skin_mesh, x, final_active, anatomy_backend, neighbors, skin_topology,
        boundary_buffer_rings, intersection_tolerance)
    whole_report = anatomy_constraint.analyze_skin_anatomy_intersections(
        skin_mesh, skin_indices=list(range(len(x))), backend=anatomy_backend, positions=x,
        neighbors=neighbors, skin_topology=skin_topology,
        boundary_buffer_rings=boundary_buffer_rings,
        intersection_tolerance=intersection_tolerance, detailed=False, verbose=False)

    fully_clean = (not final_core) and whole_report.get("intersection_pair_count", 0) <= 0
    converged = bool(converged and fully_clean)
    if not converged and stop_reason == STOP_CONVERGED:
        stop_reason = STOP_VALIDATION_FAILED

    total_disp = metrics_utils.displacement_stats(positions0, x, indices=sorted(set(core0) | active))
    whole_disp = metrics_utils.displacement_stats(positions0, x)

    unresolved = None
    if not converged:
        unresolved = {
            "unresolved_faces": whole_report.get("intersecting_skin_faces") or [],
            "unresolved_face_count": whole_report.get("intersecting_skin_face_count", 0),
            "unresolved_vertices": final_core,
            "responsible_anatomy_meshes": whole_report.get("intersecting_anatomy_meshes") or [],
        }
        if verbose:
            print("[final_cleanup] NOT converged: {0} unresolved face(s), meshes={1}".format(
                unresolved["unresolved_face_count"], unresolved["responsible_anatomy_meshes"]))

    if apply and select_final and converged and final_active:
        mesh_utils.select_vertices(skin_mesh, final_active, replace=True)
    elif apply and not converged and select_on_failure:
        sel = final_core or final_active
        if sel:
            mesh_utils.select_vertices(skin_mesh, sel, replace=True)
            if verbose:
                print("[final_cleanup] selected {0} vertex(es) for manual inspection".format(len(sel)))

    result: Dict[str, Any] = {
        "skin_mesh": skin_mesh,
        "converged": converged,
        "iterations": it,
        "stop_reason": stop_reason,
        "dry_run": not apply,
        "target_offset": target_offset,
        "min_clearance": min_clearance,
        "clearance_policy": policy_used,
        "initial_core_count": len(core0),
        "fairing_count": len(fairing_region),
        "transition_count": len(final_active) - len(fairing_region),
        "active_count": len(final_active),
        # The actual index list (not just a count), so a later finishing/
        # fairing stage can reuse EXACTLY this region instead of re-deriving
        # one from a fresh M5 detection (see run_final_fairing).
        "active_indices": final_active,
        "fairing_indices": sorted(fairing_region),
        "dynamic_growth_added": total_active_growth,
        "provenance_counts": {
            "m3_only": sum(1 for p in provenance.values() if p == "m3_only"),
            "m4_only": sum(1 for p in provenance.values() if p == "m4_only"),
            "overlap": sum(1 for p in provenance.values() if p == "overlap"),
            "grown": sum(1 for p in provenance.values() if p == "grown"),
        },
        "intersections": {
            "before": {"face_count": isect0_report.get("intersecting_skin_face_count", 0),
                      "pair_count": isect0_report.get("intersection_pair_count", 0)},
            "after": {"face_count": whole_report.get("intersecting_skin_face_count", 0),
                     "pair_count": whole_report.get("intersection_pair_count", 0)},
        },
        "feasibility": {k: v for k, v in feasibility.items() if k != "positions"},
        "roughness": {
            "before": roughness_original,           # ORIGINAL registered skin (honest headline metric)
            "after": mean_laplacian_magnitude(x, neighbors, sorted(fairing_region)),
            "post_feasibility": roughness0,          # start of the iterative loop (after Phase 0 only)
            "ceiling": roughness_ceiling,
            "core": mean_laplacian_magnitude(x, neighbors, core0),
            "transition": mean_laplacian_magnitude(x, neighbors, sorted(active - fairing_region)),
        },
        "m4_disagreement": {"before": m4_disagreement0, "after": prev_m4_disagreement},
        "m4_decay": {"final_w_m4_effective": w_m4_effective, "w_m4": w_m4, "floor": m4_floor,
                    "frozen_vertex_count": len(frozen_m4_vertices)},
        "min_exact_anatomy_distance": {"before": min_dist0,
                                      "after": min((anatomy_backend.exact_closest(x[i])["distance"]
                                                  for i in final_active), default=float("inf"))},
        "oscillating_vertices": {"count": len(damped_oscillating),
                                "vertices": sorted(damped_oscillating),
                                "details": oscillating_details},
        "best_state": {"used": used_best_state, "roughness": best_roughness,
                      "m4_disagreement": best_m4, "iteration": best_iteration},
        "final_phase": iterations_log[-1]["phase"] if iterations_log else None,
        "total_displacement": total_disp,
        "whole_mesh_displacement": whole_disp,
        "max_shape_deviation": {
            "vertex": whole_disp.get("max_index", -1),
            "distance": whole_disp.get("max", 0.0),
            "provenance": provenance.get(whole_disp.get("max_index", -1), "exterior"),
            "oscillating": whole_disp.get("max_index", -1) in damped_oscillating,
        },
        "unresolved": unresolved,
        "m5_report": m5_report,
        "iterations_log": iterations_log,
        "runtime_seconds": time.time() - t_run0,
        "config": {
            "cleanup_growth_rings": cleanup_growth_rings, "transition_rings": transition_rings,
            "w_fair": w_fair, "w_m4": w_m4, "w_shape": w_shape,
            "core_shape_scale": core_shape_scale,
            "m4_decay_patience": m4_decay_patience, "m4_decay_rate": m4_decay_rate,
            "m4_min_weight_fraction": m4_min_weight_fraction,
            "m4_freeze_relative_tolerance": m4_freeze_relative_tolerance,
            "rest_reference_smoothing_iterations": rest_reference_smoothing_iterations,
            "rest_reference_smoothing_strength": rest_reference_smoothing_strength,
            "max_roughness_ratio": max_roughness_ratio,
            "force_stable_tolerance": force_stable_tolerance,
            "track_best_state": track_best_state,
            "oscillation_window": oscillation_window,
            "oscillation_repeat_threshold": oscillation_repeat_threshold,
            "max_step_edge_ratio": max_step_edge_ratio,
            "dynamic_active_set": dynamic_active_set, "redetect_every": redetect_every,
            "max_iterations": max_iterations, "convergence_patience": convergence_patience,
        },
    }

    if verbose:
        print_final_cleanup_report(result)

    if save_json or save_csv:
        log_dir = log_path if log_path else "cleanup_logs"
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base = "final_cleanup_{0}".format(timestamp)
        log_files = {}
        if save_json:
            p = _write_json(result, os.path.join(log_dir, base + ".json"))
            if p:
                log_files["json"] = p
        if save_csv:
            p = _write_csv(csv_rows, os.path.join(log_dir, base + ".csv"))
            if p:
                log_files["csv"] = p
        result["log_files"] = log_files

    return result


def print_final_cleanup_report(result: Dict[str, Any]) -> None:
    """Human-readable summary of a :func:`run_cleanup_solver` result."""
    print("\n" + "=" * 60)
    print("FINAL CLEANUP SOLVER {0}".format(
        "COMPLETE" if result.get("converged") else "STOPPED (not converged)"))
    print("=" * 60)
    print("converged:  {0}".format(result.get("converged")))
    print("iterations: {0}".format(result.get("iterations")))
    print("final phase: {0}".format(result.get("final_phase")))
    print("stop reason: {0}".format(result.get("stop_reason")))
    if result.get("dry_run"):
        print("(dry run -- scene NOT modified)")
    feas = result.get("feasibility") or {}
    if feas.get("passes", 0) > 0:
        print("Phase 0 (initial feasibility): {0} pass(es), resolved={1}".format(
            feas.get("passes", 0), feas.get("resolved")))
        print("  intersections: {0} -> {1} faces".format(
            feas.get("pre_intersections", {}).get("face_count", 0),
            feas.get("post_intersections", {}).get("face_count", 0)))
        print("  displacement: mean={0:.5f} max={1:.5f} ({2} vertices touched)".format(
            feas.get("displacement", {}).get("mean", 0.0),
            feas.get("displacement", {}).get("max", 0.0),
            feas.get("displacement", {}).get("count", 0)))
    isect = result.get("intersections") or {}
    b, a = isect.get("before", {}), isect.get("after", {})
    print("intersections (faces): {0} -> {1}".format(
        b.get("face_count", 0), a.get("face_count", 0)))
    rough = result.get("roughness") or {}
    print("roughness (mean Laplacian magnitude, fairing region): {0:.5f} -> {1:.5f}  "
          "(ceiling {2:.5f}; post-Phase-0 start was {3:.5f})".format(
        rough.get("before", 0.0), rough.get("after", 0.0),
        rough.get("ceiling", 0.0), rough.get("post_feasibility", 0.0)))
    print("  by region: core={0:.5f} transition={1:.5f}".format(
        rough.get("core", 0.0), rough.get("transition", 0.0)))
    if rough.get("after", 0.0) > rough.get("ceiling", float("inf")):
        print("  WARNING: final roughness exceeds the ceiling -- the surface is "
              "not yet FAIR, not just anatomically unsafe.")
    m4 = result.get("m4_disagreement") or {}
    print("M4 target disagreement: {0:.5f} -> {1:.5f}".format(
        m4.get("before", 0.0), m4.get("after", 0.0)))
    dec = result.get("m4_decay") or {}
    print("M4 effective weight: {0:.4f} -> {1:.4f} (floor {2:.4f}); {3} vertex(es) "
         "individually frozen at the end".format(
        dec.get("w_m4", 0.0), dec.get("final_w_m4_effective", 0.0), dec.get("floor", 0.0),
        dec.get("frozen_vertex_count", 0)))
    dist = result.get("min_exact_anatomy_distance") or {}
    print("min exact anatomy distance: {0:.4f} -> {1:.4f}".format(
        dist.get("before", float("inf")), dist.get("after", float("inf"))))
    disp = result.get("total_displacement") or {}
    print("mean final iteration displacement: see iterations_log[-1]['net_displacement']")
    print("displacement over touched region: mean={0:.5f} max={1:.5f}".format(
        disp.get("mean", 0.0), disp.get("max", 0.0)))
    whole = result.get("whole_mesh_displacement") or {}
    msd = result.get("max_shape_deviation") or {}
    print("max shape deviation (whole mesh vs. registered): {0:.5f} at vertex {1} "
         "(provenance={2}, oscillating={3})".format(
        whole.get("max", 0.0), msd.get("vertex", -1), msd.get("provenance"),
        msd.get("oscillating")))
    prov = result.get("provenance_counts") or {}
    print("provenance: m3_only={0} m4_only={1} overlap={2} grown={3}".format(
        prov.get("m3_only", 0), prov.get("m4_only", 0), prov.get("overlap", 0),
        prov.get("grown", 0)))
    osc = result.get("oscillating_vertices") or {}
    if osc.get("count", 0) > 0:
        by_prov: Dict[str, int] = {}
        for d in osc.get("details") or []:
            by_prov[d["provenance"]] = by_prov.get(d["provenance"], 0) + 1
        print("oscillation detected & damped: {0} vertex(es) (repeatedly needed repair; "
              "held at last safe position -- provenance breakdown: {1})".format(
                  osc.get("count", 0), by_prov))
    best = result.get("best_state") or {}
    if best.get("used"):
        print("BEST-FEASIBLE-STATE REVERSION: final result is from iteration {0} "
             "(roughness {1:.5f}), not the last iteration -- it regressed".format(
             best.get("iteration"), best.get("roughness")))
    if result.get("iterations_log"):
        last = result["iterations_log"][-1]
        print("last-iteration repair displacement: mean={0:.5f} max={1:.5f} "
              "| proposal faces={2}".format(
                  last["repair_displacement"]["mean"], last["repair_displacement"]["max"],
                  last["intersections"]["face_count"]))
        opp = last.get("force_opposition") or {}
        if opp:
            print("last-iteration force opposition (cosine similarity, -1=opposing, "
                 "+1=reinforcing): fair.m4={0:.3f} fair.shape={1:.3f} fair.net={2:.3f}".format(
                 (opp.get("fair_vs_m4") or {}).get("mean_cosine", 0.0),
                 (opp.get("fair_vs_shape") or {}).get("mean_cosine", 0.0),
                 (opp.get("fair_vs_net") or {}).get("mean_cosine", 0.0)))
        raw = last.get("raw_proposal_displacement") or {}
        if raw:
            print("last-iteration raw proposal (pre-clamp) mean={0:.5f} vs net accepted "
                 "mean={1:.5f}".format(raw.get("mean", 0.0), last["net_displacement"]["mean"]))
    unresolved = result.get("unresolved")
    if unresolved:
        print("UNRESOLVED: {0} face(s); anatomy meshes: {1}".format(
            unresolved.get("unresolved_face_count", 0),
            unresolved.get("responsible_anatomy_meshes", [])))
    print("runtime: {0:.2f}s".format(result.get("runtime_seconds", 0.0)))
    print("=" * 60)


# =============================================================================
# 5. FINAL FAIRING STAGE -- pure finishing pass, deliberately smaller than
#    run_cleanup_solver.
#
# Contains ONLY: surface fairing, low-frequency/current-shape preservation,
# anatomy feasibility, and local intersection repair. Deliberately does NOT
# contain: M4 target attraction, M5 provenance-based correction forces,
# original-registered-skin attraction, or any new detection logic -- by the
# time this runs, the unified solver has already done that work. This is a
# SEPARATE, smaller function reusing the same pure primitives and the same
# per-iteration propose -> clamp -> project -> check -> repair pipeline as
# run_cleanup_solver (no second safety implementation, no global alpha), not
# a parallel pipeline and not a change to run_cleanup_solver's own behaviour.
#
# Starting/reference geometry is the CURRENT Maya mesh (read fresh at call
# time), never the original registered skin and never run_cleanup_solver's
# Phase-0 reference -- see run_final_fairing's docstring.
# =============================================================================

def run_final_fairing(skin_mesh: str,
                      anatomical_meshes: Sequence[str],
                      indices: Optional[Sequence[int]] = None,
                      min_clearance: Optional[float] = None,
                      target_offset: Optional[float] = None,
                      anatomy_backend: Optional[Any] = None,
                      m5_detector_kwargs: Optional[Dict[str, Any]] = None,
                      # --- region ---------------------------------------------
                      transition_rings: int = DEFAULT_TRANSITION_RINGS,
                      boundary_buffer_rings: int = DEFAULT_BOUNDARY_BUFFER_RINGS,
                      # --- fairing operator -------------------------------------
                      method: str = "taubin",
                      taubin_lambda: float = 0.33,
                      taubin_mu: float = -0.34,
                      fair_strength: float = 0.5,
                      w_shape: float = DEFAULT_W_SHAPE,
                      core_shape_scale: float = DEFAULT_CORE_SHAPE_SCALE,
                      rest_reference_smoothing_iterations: int = 8,
                      rest_reference_smoothing_strength: float = DEFAULT_REST_REFERENCE_SMOOTHING_STRENGTH,
                      max_step_edge_ratio: float = DEFAULT_MAX_STEP_EDGE_RATIO,
                      # --- anatomy feasibility -----------------------------
                      clearance_policy: str = DEFAULT_CLEARANCE_POLICY,
                      clearance_tolerance: float = DEFAULT_CLEARANCE_TOLERANCE,
                      max_constraint_iterations: int = DEFAULT_MAX_CONSTRAINT_ITERATIONS,
                      intersection_tolerance: float = DEFAULT_INTERSECTION_TOLERANCE,
                      repair_profile: str = DEFAULT_REPAIR_PROFILE,
                      repair_kwargs: Optional[Dict[str, Any]] = None,
                      run_initial_feasibility: bool = True,
                      max_feasibility_passes: int = DEFAULT_MAX_FEASIBILITY_PASSES,
                      # --- oscillation detection -------------------------------
                      oscillation_window: int = DEFAULT_OSCILLATION_WINDOW,
                      oscillation_repeat_threshold: int = DEFAULT_OSCILLATION_REPEAT_THRESHOLD,
                      oscillation_damping: float = DEFAULT_OSCILLATION_DAMPING,
                      # --- roughness-targeted fairing weights ------------------
                      roughness_weighting: bool = DEFAULT_ROUGHNESS_WEIGHTING,
                      roughness_weight_min: float = DEFAULT_ROUGHNESS_WEIGHT_MIN,
                      roughness_weight_max: float = DEFAULT_ROUGHNESS_WEIGHT_MAX,
                      roughness_weight_percentile_start: float = DEFAULT_ROUGHNESS_WEIGHT_PERCENTILE_START,
                      roughness_weight_smoothing_rings: int = DEFAULT_ROUGHNESS_WEIGHT_SMOOTHING_RINGS,
                      # --- best-feasible-state tracking -----------------------
                      track_best_state: bool = True,
                      # --- iteration / convergence --------------------------
                      max_iterations: int = 20,
                      convergence_patience: int = DEFAULT_CONVERGENCE_PATIENCE,
                      mean_displacement_tolerance: float = DEFAULT_MEAN_DISPLACEMENT_TOLERANCE,
                      max_displacement_tolerance: float = DEFAULT_MAX_DISPLACEMENT_TOLERANCE,
                      roughness_rel_improvement_tolerance: float = DEFAULT_ROUGHNESS_REL_IMPROVEMENT_TOLERANCE,
                      repair_displacement_tolerance: float = DEFAULT_REPAIR_DISPLACEMENT_TOLERANCE,
                      force_stable_tolerance: float = DEFAULT_FORCE_STABLE_TOLERANCE,
                      # --- output / safety -----------------------------------
                      apply: bool = True,
                      verbose: bool = True,
                      report_interval: int = 5,
                      select_final: bool = True,
                      select_on_failure: bool = True,
                      create_backup: bool = True,
                      backup_suffix: str = "_prefairing",
                      log_path: Optional[str] = None,
                      save_json: bool = True,
                      save_csv: bool = True,
                      ) -> Dict[str, Any]:
    """FINAL FAIRING: a pure finishing pass over the CURRENT Maya mesh.

    Run this AFTER :func:`run_cleanup_solver` (or its d98 wrapper
    ``run_final_skin_cleanup``) has already produced an anatomically valid,
    largely-M4-corrected skin. This stage does ONE thing: reduce remaining
    high-frequency roughness, using the CURRENT mesh (read fresh from Maya at
    call time) as both the working geometry and the shape-preservation
    reference -- never the original registered skin, never re-running Phase 0
    against it, never M4.

    Parameters
    ----------
    skin_mesh, anatomical_meshes:
        The skin (read fresh -- this IS the "current mesh" requirement) and
        internal anatomy mesh names.
    indices:
        The region to fair. Pass ``cleanup_result["active_indices"]`` from a
        prior :func:`run_cleanup_solver` call to reuse EXACTLY its broad
        active/fairing region (recommended -- this is what avoids a hard
        transition at a raw-M5-sized boundary). If omitted, a region is
        derived from a FRESH M5 detection instead (the same detector, not new
        logic, but it will not reproduce any repair/dynamic-growth history
        from a previous run) -- a documented convenience fallback, not the
        precise behaviour.
    min_clearance, target_offset:
        Same roles as in :func:`run_cleanup_solver`. ``target_offset`` is only
        used to auto-pick ``min_clearance`` (or, on the M5-fallback region
        path, for M5's own M4 sub-stage) -- it never drives an M4 force here.
    method:
        ``"taubin"`` (default, shrinkage-resistant, reuses :func:`taubin_force`
        -- see the module note above on why ``smoothing_utils.taubin_smooth``
        itself is not called directly) or ``"laplacian"`` (plain
        :func:`fairing_force`, for A/B comparison).
    transition_rings:
        A FRESH soft transition band grown around ``indices`` (treated as
        already-grown "fairing" core, i.e. ``cleanup_growth_rings=0`` in
        :func:`build_active_region`), so the region's own outer edge still has
        a graded handoff to the fixed exterior rather than a hard cut.
    rest_reference_smoothing_iterations:
        Re-applies the same low-pass filter as Phase 0's rest reference (see
        :func:`_smooth_rest_reference`) to the CURRENT mesh before using it as
        this stage's shape-restraint target, so restraint holds low-frequency
        form without also preserving any texture still present in the current
        state.

    This stage deliberately has NO M4 term, NO M5-provenance-based weighting,
    and NO redetection. Base fairing weights default to a roughness-targeted
    field (:func:`build_roughness_fairing_weights`) so remaining ridges get
    more emphasis than already-smooth skin; pass ``roughness_weighting=False``
    to restore the old uniform 1.0. Oscillation detection still multiplies
    those base weights in place (it never overwrites the targeting).
    Per-vertex trust region, real triangle/triangle checking, and LOCAL
    repair (never a global alpha) are unchanged -- see :func:`combine_step`,
    :func:`_scoped_intersection_core`, :func:`_repair_local`.

    Returns
    -------
    dict
        ``converged``, ``iterations``, ``stop_reason``, before/after
        roughness (whole fairing region AND the caller's core region
        separately), intersections, displacement, and ``best_state`` (see
        :func:`is_better_state` -- the best anatomy-valid state seen is used
        if the last iteration regressed).
    """
    t_run0 = time.time()

    if method not in ("taubin", "laplacian"):
        raise ValueError("method must be 'taubin' or 'laplacian', got {0!r}".format(method))
    if not mesh_utils.mesh_exists(skin_mesh):
        raise ValueError("run_final_fairing: skin_mesh '{0}' does not exist".format(skin_mesh))
    if not anatomical_meshes:
        raise ValueError("run_final_fairing: anatomical_meshes is required")
    if max_iterations < 1:
        raise ValueError("max_iterations must be >= 1, got {0}".format(max_iterations))
    if convergence_patience < 1:
        raise ValueError("convergence_patience must be >= 1, got {0}".format(convergence_patience))

    # The CURRENT mesh -- read fresh, this IS the starting/reference geometry.
    positions0 = mesh_utils.get_mesh_vertices(skin_mesh)
    if not positions0:
        raise ValueError("run_final_fairing: skin_mesh '{0}' has no readable vertices".format(skin_mesh))
    neighbors = mesh_utils.get_vertex_neighbors(skin_mesh)
    normals = mesh_utils.get_vertex_normals(skin_mesh)
    mesh_fn = mesh_utils.get_mesh_fn(skin_mesh)
    skin_topology = mesh_utils.get_triangle_topology(mesh_fn) if mesh_fn is not None else None
    if not skin_topology or not skin_topology.get("triangles"):
        raise ValueError("run_final_fairing: could not read triangle topology for '{0}'".format(skin_mesh))

    boundary = _boundary_set(skin_mesh, neighbors, boundary_buffer_rings)

    if anatomy_backend is None:
        anatomy_backend, valid_anatomical_meshes = _build_anatomy_backend(anatomical_meshes)
    else:
        valid_anatomical_meshes = list(anatomy_backend.mesh_names())

    if min_clearance is None:
        if target_offset is None:
            summary = artifact_detection.summarize_skin_sdf_values(skin_mesh, valid_anatomical_meshes)
            if not summary.get("count"):
                raise ValueError("run_final_fairing: could not auto-compute min_clearance "
                                 "(no finite SDF samples); pass it explicitly")
            target_offset = summary["median"]
        min_clearance = target_offset
        if verbose:
            print("[final_fairing] min_clearance not given; using {0:.4f} "
                  "(scene median skin->anatomy distance)".format(min_clearance))

    repair_kw = dict(repair_kwargs or {})
    repair_kw.setdefault("repair_profile", repair_profile)

    # --- region: caller-supplied broad region (recommended), or a fresh M5 --
    if indices is not None:
        base_region = sorted(set(int(i) for i in indices) - boundary)
        region_source = "caller"
    else:
        if verbose:
            print("[final_fairing] no indices given; falling back to a fresh M5 detection "
                 "-- pass indices=cleanup_result['active_indices'] from run_final_skin_cleanup "
                 "to reuse ITS EXACT region instead (recommended)")
        if target_offset is None:
            summary = artifact_detection.summarize_skin_sdf_values(skin_mesh, valid_anatomical_meshes)
            if not summary.get("count"):
                raise ValueError("run_final_fairing: could not auto-compute target_offset for "
                                 "the M5 fallback; pass indices explicitly instead")
            target_offset = summary["median"]
        seed_indices, _m5_report = artifact_detection.detect_unified_artifacts(
            skin_mesh, valid_anatomical_meshes, target_offset, select=False,
            **dict(m5_detector_kwargs or {}))
        base_region = sorted(set(seed_indices) - boundary)
        region_source = "m5_fallback"

    if not base_region:
        if verbose:
            print("[final_fairing] empty region; nothing to do")
        return {"skin_mesh": skin_mesh, "converged": True, "iterations": 0,
               "stop_reason": STOP_NO_ARTIFACT_REGION, "dry_run": not apply,
               "region_source": region_source, "runtime_seconds": time.time() - t_run0}

    if create_backup and apply and maya_io is not None:
        backup_name = skin_mesh + backup_suffix
        if not mesh_utils.mesh_exists(backup_name):
            maya_io.duplicate_mesh(skin_mesh, suffix=backup_suffix)
        elif verbose:
            print("[final_fairing] backup '{0}' already exists; keeping it".format(backup_name))

    # --- defensive, cheap initial feasibility check --------------------------
    # This stage ASSUMES the current mesh is already anatomy-valid (that is
    # run_cleanup_solver's job). This is a one-time whole-mesh scan (cheap) and,
    # only if it finds something, ONE bounded repair pass reusing Phase 0's own
    # machinery verbatim -- not new logic, just a safety net against the
    # current mesh not actually being what it's assumed to be.
    isect0_core, isect0_report = _scoped_intersection_core(
        skin_mesh, positions0, list(range(len(positions0))), anatomy_backend,
        neighbors, skin_topology, boundary_buffer_rings, intersection_tolerance)
    if run_initial_feasibility and isect0_core:
        if verbose:
            print("[final_fairing] WARNING: current mesh has {0} pre-existing intersecting "
                 "face(s); repairing before fairing begins".format(
                     isect0_report.get("intersecting_skin_face_count", 0)))
        feasibility = _run_initial_feasibility(
            positions0, skin_mesh, anatomy_backend, neighbors, normals, skin_topology,
            min_clearance, clearance_policy, boundary_buffer_rings, intersection_tolerance,
            repair_kw, max_feasibility_passes, verbose)
        start_positions = feasibility["positions"]
        base_region = sorted(set(base_region) | set(feasibility["touched_vertices"]))
    else:
        start_positions = [list(p) for p in positions0]
        feasibility = None

    # --- graded region: the caller's broad region at full strength, with a
    # FRESH soft transition band around it (never a hard cut at its edge) -----
    region = build_active_region(base_region, neighbors, boundary, 0, transition_rings)
    fairing_region: Set[int] = set(region["fairing"])
    active: Set[int] = set(region["active"])
    anchor: Dict[int, float] = region["anchor"]

    # --- shape-restraint reference: the CURRENT mesh, low-pass filtered over
    # the fairing region (never the original registered skin, never Phase 0's
    # reference from a previous, now-stale, run) --------------------------
    x_rest = [list(p) for p in start_positions]
    if rest_reference_smoothing_iterations > 0 and fairing_region:
        x_rest = _smooth_rest_reference(
            x_rest, neighbors, sorted(fairing_region),
            rest_reference_smoothing_iterations, rest_reference_smoothing_strength)
    x = [list(p) for p in start_positions]

    floors, _od, policy_used = anatomy_constraint.compute_clearance_floors(
        x, sorted(active), anatomy_backend, min_clearance,
        clearance_policy=clearance_policy, clearance_tolerance=clearance_tolerance)

    roughness_start = mean_laplacian_magnitude(x, neighbors, sorted(fairing_region))
    core_roughness_start = mean_laplacian_magnitude(x, neighbors, base_region)
    roughness_pct_before = roughness_percentile_summary(
        list(laplacian_magnitudes(x, neighbors, sorted(fairing_region)).values()))
    min_dist0 = min((anatomy_backend.exact_closest(x[i])["distance"] for i in sorted(active)),
                    default=float("inf"))

    # Base fairing weight per vertex. Oscillation detection only ever MULTIPLIES
    # these in place -- it must not overwrite roughness targeting with 1.0.
    weight_knots: Optional[Dict[str, Any]] = None
    if roughness_weighting and active:
        weights, weight_info = build_roughness_fairing_weights(
            x, neighbors, sorted(active),
            percentile_start=roughness_weight_percentile_start,
            weight_min=roughness_weight_min,
            weight_max=roughness_weight_max,
            smoothing_rings=roughness_weight_smoothing_rings)
        weight_knots = weight_info
        if verbose:
            print("[final_fairing] roughness-targeted weights: min={0:.3f} mean={1:.3f} "
                  "max={2:.3f} boosted={3}/{4} (>{5:g}th pct, max={6:g}, smooth={7} rings)"
                  .format(weight_info["min"], weight_info["mean"], weight_info["max"],
                          weight_info["boosted_count"], len(weights),
                          roughness_weight_percentile_start, roughness_weight_max,
                          roughness_weight_smoothing_rings))
    else:
        weights = {i: 1.0 for i in active}
        wvals = list(weights.values())
        weight_info = {
            "uniform": True,
            "min": min(wvals) if wvals else 1.0,
            "mean": (sum(wvals) / len(wvals)) if wvals else 1.0,
            "max": max(wvals) if wvals else 1.0,
            "boosted_count": 0,
        }
    base_weight_stats = {
        "min": weight_info["min"],
        "mean": weight_info["mean"],
        "max": weight_info["max"],
        "boosted_count": weight_info["boosted_count"],
        "enabled": bool(roughness_weighting),
    }

    iterations_log: List[Dict[str, Any]] = []
    csv_rows: List[Dict[str, Any]] = []
    patience = {"motion": 0, "roughness": 0, "proposal_clean": 0, "repair": 0, "force_stable": 0}
    prev_roughness = roughness_start
    prev_core_roughness = core_roughness_start
    prev_raw_force_mean = float("inf")
    repair_history: List[Set[int]] = []
    damped_oscillating: Set[int] = set()
    oscillating_details: List[Dict[str, Any]] = []
    best_state: Optional[List[List[float]]] = None
    best_roughness = float("inf")
    best_iteration = 0
    stop_reason: Optional[str] = None
    converged = False
    it = 0

    if verbose:
        print("[final_fairing] region base={0} (source={1}) fairing={2} transition={3} "
             "active={4} method={5}".format(
                 len(base_region), region_source, len(fairing_region),
                 len(active) - len(fairing_region), len(active), method))

    for it in range(1, int(max_iterations) + 1):
        x_before = [list(p) for p in x]
        active_list = sorted(active)

        local_edge = local_edge_lengths(x, neighbors, active_list)
        fair_w = {i: weights.get(i, 1.0) for i in active_list}
        shape_w = {i: w_shape * shape_weight_from_anchor(anchor.get(i, 0.0), core_shape_scale)
                  for i in active_list}

        if method == "taubin":
            fair_f = taubin_force(x, neighbors, active_list, taubin_lambda, taubin_mu, fair_w)
        else:
            fair_f = fairing_force(
                x, neighbors, active_list,
                {i: fair_strength * fair_w[i] for i in active_list})
        shape_f = shape_force(x, x_rest, active_list, shape_w)
        zero_f = {i: [0.0, 0.0, 0.0] for i in active_list}

        raw_combined = {i: [fair_f[i][k] + shape_f[i][k] for k in range(3)] for i in active_list}
        raw_mags = [mesh_utils.vec_length(v) for v in raw_combined.values()]
        raw_force_mean = (sum(raw_mags) / len(raw_mags)) if raw_mags else 0.0
        opp_fair_shape = force_opposition(fair_f, shape_f, active_list)

        step = combine_step(fair_f, shape_f, zero_f, active_list, anchor,
                            max_step_edge_ratio, local_edge)
        x_proposed = apply_step(x, step)  # RAW proposal; may temporarily violate anatomy

        x_projected, clearance_moved = _anatomy_project_clearance(
            x_proposed, active_list, anatomy_backend, floors, clearance_tolerance,
            max_constraint_iterations, normals)

        core_now, isect_report = _scoped_intersection_core(
            skin_mesh, x_projected, active_list, anatomy_backend, neighbors,
            skin_topology, boundary_buffer_rings, intersection_tolerance)

        repair_result = None
        repair_disp: List[float] = []
        repair_footprint: Set[int] = set()
        if core_now:
            x_accepted, repair_result = _repair_local(
                x_projected, core_now, anatomy_backend, skin_mesh, neighbors,
                normals, skin_topology, min_clearance, clearance_policy,
                boundary_buffer_rings, intersection_tolerance, repair_kw)
            if repair_result:
                repair_footprint = set(repair_result.get("patch_vertices") or core_now)
                repair_disp = [mesh_utils.vec_length(mesh_utils.vec_sub(x_accepted[i], x_projected[i]))
                              for i in sorted(repair_footprint)]
        else:
            x_accepted = x_projected

        verify_scope = sorted(set(active_list) | repair_footprint)
        residual_core, _residual_report = _scoped_intersection_core(
            skin_mesh, x_accepted, verify_scope, anatomy_backend, neighbors,
            skin_topology, boundary_buffer_rings, intersection_tolerance)
        if residual_core:
            for i in residual_core:
                x_accepted[i] = list(x_before[i])
            residual_core, _residual_report = _scoped_intersection_core(
                skin_mesh, x_accepted, verify_scope, anatomy_backend, neighbors,
                skin_topology, boundary_buffer_rings, intersection_tolerance)

        net_disp = [mesh_utils.vec_length(mesh_utils.vec_sub(x_accepted[i], x_before[i]))
                   for i in active_list]
        x = x_accepted

        grown_by_repair = repair_footprint - active
        if grown_by_repair:
            fairing_region |= grown_by_repair
            region = build_active_region(sorted(fairing_region), neighbors, boundary,
                                         0, transition_rings)
            active = set(region["active"])
            anchor = region["anchor"]
            for j in grown_by_repair:
                if j not in weights:
                    mag = laplacian_magnitudes(x, neighbors, [j]).get(j)
                    if mag is not None and weight_knots and not weight_knots.get("uniform"):
                        weights[j] = roughness_to_fairing_weight(
                            mag, weight_knots["r_knots"], weight_knots["w_knots"])
                    else:
                        weights[j] = 1.0
            for j in active - set(weights.keys()):
                mag = laplacian_magnitudes(x, neighbors, [j]).get(j)
                if mag is not None and weight_knots and not weight_knots.get("uniform"):
                    weights.setdefault(j, roughness_to_fairing_weight(
                        mag, weight_knots["r_knots"], weight_knots["w_knots"]))
                else:
                    weights.setdefault(j, 0.5)
            new_active = sorted(active - set(floors.keys()))
            if new_active:
                new_floors, _od2, _pu2 = anatomy_constraint.compute_clearance_floors(
                    x, new_active, anatomy_backend, min_clearance,
                    clearance_policy=clearance_policy, clearance_tolerance=clearance_tolerance)
                floors.update(new_floors)

        roughness_after = mean_laplacian_magnitude(x, neighbors, sorted(fairing_region))
        core_roughness_after = mean_laplacian_magnitude(x, neighbors, base_region)
        min_dist_now = min((anatomy_backend.exact_closest(x[i])["distance"] for i in active_list),
                           default=float("inf"))

        mean_net = sum(net_disp) / len(net_disp) if net_disp else 0.0
        max_net = max(net_disp) if net_disp else 0.0
        roughness_rel = abs(prev_roughness - roughness_after) / (prev_roughness + 1e-9)
        no_forbidden = not bool(residual_core)
        repair_mean = _stats(repair_disp)["mean"]
        proposal_clean = (isect_report.get("intersecting_skin_face_count", 0) == 0)
        force_rel = (abs(prev_raw_force_mean - raw_force_mean) / (prev_raw_force_mean + 1e-9)
                    if math.isfinite(prev_raw_force_mean) else 1.0)

        tiny_motion = (mean_net < mean_displacement_tolerance
                      and max_net < max_displacement_tolerance)
        patience["motion"] = patience["motion"] + 1 if tiny_motion else 0
        patience["roughness"] = (patience["roughness"] + 1
                                 if roughness_rel < roughness_rel_improvement_tolerance else 0)
        patience["proposal_clean"] = patience["proposal_clean"] + 1 if proposal_clean else 0
        patience["repair"] = (patience["repair"] + 1
                              if repair_mean < repair_displacement_tolerance else 0)
        patience["force_stable"] = (patience["force_stable"] + 1
                                    if force_rel < force_stable_tolerance else 0)

        repair_history, oscillating = update_oscillation_tracker(
            repair_history, core_now, oscillation_window, oscillation_repeat_threshold)
        newly_oscillating = oscillating - damped_oscillating
        newly_oscillating_info = []
        for j in newly_oscillating:
            weights[j] = weights.get(j, 1.0) * oscillation_damping
            info = {"vertex": j, "iteration": it,
                   "nearest_anatomy": anatomy_backend.exact_closest(x[j]).get("mesh")}
            newly_oscillating_info.append(info)
            oscillating_details.append(info)
            damped_oscillating.add(j)

        if track_best_state and is_better_state(roughness_after, 0.0, best_roughness, 0.0):
            best_state = [list(p) for p in x]
            best_roughness = roughness_after
            best_iteration = it

        record = {
            "iteration": it,
            "active_count": len(active_list),
            "fairing_displacement": _stats([mesh_utils.vec_length(v) for v in fair_f.values()]),
            "shape_displacement": _stats([mesh_utils.vec_length(v) for v in shape_f.values()]),
            "raw_proposal_displacement": {"mean": raw_force_mean,
                                         "max": max(raw_mags) if raw_mags else 0.0},
            "clearance_displacement": _stats(list(clearance_moved.values())),
            "repair_displacement": _stats(repair_disp),
            "net_displacement": {"mean": mean_net, "max": max_net},
            "roughness": {"before": prev_roughness, "after": roughness_after},
            "core_roughness": {"before": prev_core_roughness, "after": core_roughness_after},
            "intersections": {
                "face_count": isect_report.get("intersecting_skin_face_count", 0),
                "pair_count": isect_report.get("intersection_pair_count", 0),
                "repaired_component_count": (repair_result or {}).get("repair_component_count", 0),
            },
            "force_opposition": opp_fair_shape,
            "min_exact_distance": min_dist_now,
            "no_forbidden_intersections": no_forbidden,
            "oscillating_count": len(damped_oscillating),
            "newly_oscillating": newly_oscillating_info,
            "best_so_far": {"roughness": best_roughness, "iteration": best_iteration},
            "patience": dict(patience),
        }
        iterations_log.append(record)
        csv_rows.append(_iteration_csv_row_fairing(record))

        if verbose and (it == 1 or it % max(1, report_interval) == 0):
            print("  [iter {0:4d}] active={1} net disp mean/max={2:.5f}/{3:.5f} "
                 "roughness {4:.5f}->{5:.5f} core {6:.5f}->{7:.5f} proposal_faces={8} "
                 "repair_disp={9:.5f} fair.shape cosine={10:.3f} osc={11} min_dist={12:.4f} "
                 "patience(m/r/p/x/f)={13}/{14}/{15}/{16}/{17}".format(
                     it, len(active_list), mean_net, max_net, prev_roughness, roughness_after,
                     record["core_roughness"]["before"], core_roughness_after,
                     record["intersections"]["face_count"], repair_mean,
                     opp_fair_shape["mean_cosine"], len(damped_oscillating), min_dist_now,
                     patience["motion"], patience["roughness"], patience["proposal_clean"],
                     patience["repair"], patience["force_stable"]))

        prev_roughness = roughness_after
        prev_core_roughness = core_roughness_after
        prev_raw_force_mean = raw_force_mean

        if (no_forbidden
                and patience["motion"] >= convergence_patience
                and patience["roughness"] >= convergence_patience
                and patience["proposal_clean"] >= convergence_patience
                and patience["repair"] >= convergence_patience
                and patience["force_stable"] >= convergence_patience):
            converged = True
            stop_reason = STOP_CONVERGED
            break
    else:
        stop_reason = STOP_MAX_ITERATIONS

    if stop_reason is None:
        stop_reason = STOP_MAX_ITERATIONS

    used_best_state = False
    if track_best_state and best_state is not None and best_roughness < prev_roughness - 1e-6:
        x = best_state
        used_best_state = True
        if verbose:
            print("[final_fairing] reverting to best-feasible-state from iteration {0} "
                 "(roughness {1:.5f} vs final {2:.5f})".format(
                     best_iteration, best_roughness, prev_roughness))

    if apply:
        mesh_utils.set_mesh_vertices(skin_mesh, x)

    final_active = sorted(active)
    final_core, _fsr = _scoped_intersection_core(
        skin_mesh, x, final_active, anatomy_backend, neighbors, skin_topology,
        boundary_buffer_rings, intersection_tolerance)
    whole_report = anatomy_constraint.analyze_skin_anatomy_intersections(
        skin_mesh, skin_indices=list(range(len(x))), backend=anatomy_backend, positions=x,
        neighbors=neighbors, skin_topology=skin_topology,
        boundary_buffer_rings=boundary_buffer_rings,
        intersection_tolerance=intersection_tolerance, detailed=False, verbose=False)

    fully_clean = (not final_core) and whole_report.get("intersection_pair_count", 0) <= 0
    converged = bool(converged and fully_clean)
    if not converged and stop_reason == STOP_CONVERGED:
        stop_reason = STOP_VALIDATION_FAILED

    final_roughness = mean_laplacian_magnitude(x, neighbors, sorted(fairing_region))
    final_core_roughness = mean_laplacian_magnitude(x, neighbors, base_region)
    roughness_pct_after = roughness_percentile_summary(
        list(laplacian_magnitudes(x, neighbors, sorted(fairing_region)).values()))
    total_disp = metrics_utils.displacement_stats(positions0, x, indices=sorted(set(base_region) | active))
    whole_disp = metrics_utils.displacement_stats(positions0, x)

    # Is smoothing itself weak, or is anatomy constraint absorbing most of the
    # proposed motion? Aggregated from the SAME per-iteration numbers already
    # in iterations_log (nothing new computed here) so this is a pure summary,
    # not a new measurement.
    _proposal_means = [r["raw_proposal_displacement"]["mean"] for r in iterations_log]
    _net_means = [r["net_displacement"]["mean"] for r in iterations_log]
    _clr_counts = [r["clearance_displacement"]["count"] for r in iterations_log]
    _clr_means = [r["clearance_displacement"]["mean"] for r in iterations_log
                 if r["clearance_displacement"]["count"]]
    anatomy_constraint_summary = {
        "mean_raw_proposal_displacement": (
            sum(_proposal_means) / len(_proposal_means)) if _proposal_means else 0.0,
        "mean_net_accepted_displacement": (
            sum(_net_means) / len(_net_means)) if _net_means else 0.0,
        "mean_vertices_clearance_constrained_per_iteration": (
            sum(_clr_counts) / len(_clr_counts)) if _clr_counts else 0.0,
        "total_clearance_constraint_events": sum(_clr_counts),
        "mean_clearance_correction_when_applied": (
            sum(_clr_means) / len(_clr_means)) if _clr_means else 0.0,
    }

    unresolved = None
    if not converged:
        unresolved = {
            "unresolved_faces": whole_report.get("intersecting_skin_faces") or [],
            "unresolved_face_count": whole_report.get("intersecting_skin_face_count", 0),
            "unresolved_vertices": final_core,
            "responsible_anatomy_meshes": whole_report.get("intersecting_anatomy_meshes") or [],
        }
        if verbose:
            print("[final_fairing] NOT converged: {0} unresolved face(s), meshes={1}".format(
                unresolved["unresolved_face_count"], unresolved["responsible_anatomy_meshes"]))

    if apply and select_final and converged and final_active:
        mesh_utils.select_vertices(skin_mesh, final_active, replace=True)
    elif apply and not converged and select_on_failure:
        sel = final_core or final_active
        if sel:
            mesh_utils.select_vertices(skin_mesh, sel, replace=True)
            if verbose:
                print("[final_fairing] selected {0} vertex(es) for manual inspection".format(len(sel)))

    result: Dict[str, Any] = {
        "skin_mesh": skin_mesh,
        "converged": converged,
        "iterations": it,
        "stop_reason": stop_reason,
        "dry_run": not apply,
        "method": method,
        "region_source": region_source,
        "min_clearance": min_clearance,
        "clearance_policy": policy_used,
        "base_region_count": len(base_region),
        "fairing_count": len(fairing_region),
        "transition_count": len(final_active) - len(fairing_region),
        "active_count": len(final_active),
        "active_indices": final_active,
        "feasibility": ({k: v for k, v in feasibility.items() if k != "positions"}
                       if feasibility else None),
        "intersections": {
            "before": {"face_count": isect0_report.get("intersecting_skin_face_count", 0),
                      "pair_count": isect0_report.get("intersection_pair_count", 0)},
            "after": {"face_count": whole_report.get("intersecting_skin_face_count", 0),
                     "pair_count": whole_report.get("intersection_pair_count", 0)},
        },
        "roughness": {"before": roughness_start, "after": final_roughness,
                     "core_before": core_roughness_start, "core_after": final_core_roughness,
                     "percentiles_before": roughness_pct_before,
                     "percentiles_after": roughness_pct_after},
        "fairing_weight": dict(base_weight_stats),
        "anatomy_constraint_summary": anatomy_constraint_summary,
        "min_exact_anatomy_distance": {"before": min_dist0,
                                      "after": min((anatomy_backend.exact_closest(x[i])["distance"]
                                                  for i in final_active), default=float("inf"))},
        "oscillating_vertices": {"count": len(damped_oscillating),
                                "vertices": sorted(damped_oscillating),
                                "details": oscillating_details},
        "best_state": {"used": used_best_state, "roughness": best_roughness, "iteration": best_iteration},
        "total_displacement": total_disp,
        "whole_mesh_displacement": whole_disp,
        "max_shape_deviation": {"vertex": whole_disp.get("max_index", -1),
                               "distance": whole_disp.get("max", 0.0)},
        "unresolved": unresolved,
        "iterations_log": iterations_log,
        "runtime_seconds": time.time() - t_run0,
        "config": {
            "method": method, "taubin_lambda": taubin_lambda, "taubin_mu": taubin_mu,
            "fair_strength": fair_strength, "w_shape": w_shape,
            "core_shape_scale": core_shape_scale, "transition_rings": transition_rings,
            "max_step_edge_ratio": max_step_edge_ratio,
            "oscillation_window": oscillation_window,
            "oscillation_repeat_threshold": oscillation_repeat_threshold,
            "track_best_state": track_best_state,
            "max_iterations": max_iterations, "convergence_patience": convergence_patience,
            "roughness_weighting": bool(roughness_weighting),
            "roughness_weight_min": roughness_weight_min,
            "roughness_weight_max": roughness_weight_max,
            "roughness_weight_percentile_start": roughness_weight_percentile_start,
            "roughness_weight_smoothing_rings": roughness_weight_smoothing_rings,
        },
    }

    if verbose:
        print_final_fairing_report(result)

    if save_json or save_csv:
        log_dir = log_path if log_path else "cleanup_logs"
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base = "final_fairing_{0}".format(timestamp)
        log_files = {}
        if save_json:
            p = _write_json(result, os.path.join(log_dir, base + ".json"))
            if p:
                log_files["json"] = p
        if save_csv:
            p = _write_csv(csv_rows, os.path.join(log_dir, base + ".csv"), columns=_CSV_COLUMNS_FAIRING)
            if p:
                log_files["csv"] = p
        result["log_files"] = log_files

    return result


def print_final_fairing_report(result: Dict[str, Any]) -> None:
    """Human-readable summary of a :func:`run_final_fairing` result."""
    print("\n" + "=" * 60)
    print("FINAL FAIRING {0}".format(
        "COMPLETE" if result.get("converged") else "STOPPED (not converged)"))
    print("=" * 60)
    print("method: {0}".format(result.get("method")))
    print("region source: {0} ({1} base vertices)".format(
        result.get("region_source"), result.get("base_region_count")))
    print("converged:  {0}".format(result.get("converged")))
    print("iterations: {0}".format(result.get("iterations")))
    print("stop reason: {0}".format(result.get("stop_reason")))
    if result.get("dry_run"):
        print("(dry run -- scene NOT modified)")
    feas = result.get("feasibility")
    if feas and feas.get("passes", 0) > 0:
        print("pre-fairing feasibility repair: {0} pass(es), resolved={1}".format(
            feas.get("passes", 0), feas.get("resolved")))
    isect = result.get("intersections") or {}
    b, a = isect.get("before", {}), isect.get("after", {})
    print("intersections (faces): {0} -> {1}".format(
        b.get("face_count", 0), a.get("face_count", 0)))
    rough = result.get("roughness") or {}
    print("roughness (fairing region): {0:.5f} -> {1:.5f}".format(
        rough.get("before", 0.0), rough.get("after", 0.0)))
    print("roughness (core region):    {0:.5f} -> {1:.5f}".format(
        rough.get("core_before", 0.0), rough.get("core_after", 0.0)))
    pb = rough.get("percentiles_before") or {}
    pa = rough.get("percentiles_after") or {}
    if pb or pa:
        print("roughness percentiles:")
        print("  before  p50={0:.5f}  p90={1:.5f}  p95={2:.5f}  p99={3:.5f}  max={4:.5f}".format(
            pb.get("p50", 0.0), pb.get("p90", 0.0), pb.get("p95", 0.0),
            pb.get("p99", 0.0), pb.get("max", 0.0)))
        print("  after   p50={0:.5f}  p90={1:.5f}  p95={2:.5f}  p99={3:.5f}  max={4:.5f}".format(
            pa.get("p50", 0.0), pa.get("p90", 0.0), pa.get("p95", 0.0),
            pa.get("p99", 0.0), pa.get("max", 0.0)))
    fw = result.get("fairing_weight") or {}
    print("base fairing weight: min={0:.3f} mean={1:.3f} max={2:.3f}  "
          "boosted={3}  (roughness_weighting={4})".format(
              fw.get("min", 1.0), fw.get("mean", 1.0), fw.get("max", 1.0),
              fw.get("boosted_count", 0), fw.get("enabled", False)))
    acs = result.get("anatomy_constraint_summary") or {}
    if acs:
        print("smoothing vs. anatomy constraint (mean per iteration): "
             "proposed={0:.5f}  accepted={1:.5f}  ({2:.0f}% absorbed by anatomy)".format(
             acs.get("mean_raw_proposal_displacement", 0.0),
             acs.get("mean_net_accepted_displacement", 0.0),
             100.0 * (1.0 - acs.get("mean_net_accepted_displacement", 0.0)
                     / max(1e-9, acs.get("mean_raw_proposal_displacement", 0.0)))))
        print("  vertices clearance-constrained: {0:.1f}/iteration ({1} total events, "
             "mean correction {2:.5f} when applied)".format(
             acs.get("mean_vertices_clearance_constrained_per_iteration", 0.0),
             acs.get("total_clearance_constraint_events", 0),
             acs.get("mean_clearance_correction_when_applied", 0.0)))
    dist = result.get("min_exact_anatomy_distance") or {}
    print("min exact anatomy distance: {0:.4f} -> {1:.4f}".format(
        dist.get("before", float("inf")), dist.get("after", float("inf"))))
    disp = result.get("total_displacement") or {}
    print("displacement over touched region: mean={0:.5f} max={1:.5f}".format(
        disp.get("mean", 0.0), disp.get("max", 0.0)))
    whole = result.get("whole_mesh_displacement") or {}
    msd = result.get("max_shape_deviation") or {}
    print("max shape deviation (whole mesh vs. mesh AT START of this stage): "
         "{0:.5f} at vertex {1}".format(whole.get("max", 0.0), msd.get("vertex", -1)))
    osc = result.get("oscillating_vertices") or {}
    if osc.get("count", 0) > 0:
        print("oscillation detected & damped: {0} vertex(es)".format(osc.get("count", 0)))
    best = result.get("best_state") or {}
    if best.get("used"):
        print("BEST-FEASIBLE-STATE REVERSION: final result is from iteration {0} "
             "(roughness {1:.5f}), not the last iteration".format(
             best.get("iteration"), best.get("roughness")))
    if result.get("iterations_log"):
        last = result["iterations_log"][-1]
        print("last-iteration repair displacement: mean={0:.5f} max={1:.5f} | "
             "proposal faces={2}".format(
                 last["repair_displacement"]["mean"], last["repair_displacement"]["max"],
                 last["intersections"]["face_count"]))
    unresolved = result.get("unresolved")
    if unresolved:
        print("UNRESOLVED: {0} face(s); anatomy meshes: {1}".format(
            unresolved.get("unresolved_face_count", 0),
            unresolved.get("responsible_anatomy_meshes", [])))
    print("runtime: {0:.2f}s".format(result.get("runtime_seconds", 0.0)))
    print("=" * 60)


def compare_cleanup_result(before_positions: List[List[float]],
                          after_positions: List[List[float]],
                          indices: Optional[Sequence[int]] = None,
                          label: str = "final_cleanup",
                          ) -> Dict[str, Any]:
    """Thin convenience wrapper: displacement of the final mesh vs. the
    original (or vs. a saved snapshot), reusing :mod:`metrics_utils` (no new
    displacement math)."""
    return metrics_utils.print_displacement_report(
        before_positions, after_positions, indices=indices, label=label)
