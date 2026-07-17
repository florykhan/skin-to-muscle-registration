AUTO_SAVE_FOLDER = "s37 target_shrinkwrap_saves V2 added braking zone continued from 400 iter"

"""
Target-Based Shrink-Wrap Skin Registration for Maya
=====================================================
New approach:
1. Start with EXPANDED skin mesh (current skin mesh)
2. Use a SCALED-DOWN TARGET mesh as the desired position (close to internal tissues)
3. Each vertex moves toward its corresponding target vertex (1-to-1 correspondence)
4. Collision detection still uses closest-point to internal meshes
5. Collision push direction is OPPOSITE of attraction direction
6. Outline vertices (no nearby tissue) fall back to closest-point attraction
7. Landmark contour constraints work on top of everything

USAGE:
======
exec(open(r"C:\keyLabLocal_D\skin-muscle-registration-model\d98-target_shrinkwrap_registration.py").read())

# Step 1: Create target mesh (scale down skin mesh in Maya, name it TARGET_MESH)
# Step 2: Setup
setup_target_registration()

# Step 3: Run shrink-wrap
run_shrinkwrap(num_iterations=400)
"""

import maya.cmds as cmds
import maya.api.OpenMaya as om
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
import math
import time
import os
import sys
import json

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("[WARNING] matplotlib not available - plots disabled")


# =============================================================================
# OPTIONAL HELPER MODULES (post-registration cleanup toolkit)
# =============================================================================
# The reusable cleanup helpers live in the SAME src/ folder as this file:
#   mesh_utils, smoothing_utils, region_selection, metrics_utils, maya_io
#
# IMPORTANT (why auto-detect from __file__ does NOT work):
# The MayaCode VS Code extension does not run this file in place. It copies the
# editor text into a TEMP file (e.g. <tmp>/MayaCode.py) and runs
#   python("execfile('<tmp>/MayaCode.py')")
# So when this code executes, __file__ is either undefined or points at the temp
# folder -- the real src/ path is gone and the helper .py files are not there.
# The same is true for exec(open(...).read()) (no __file__ at all).
#
# THE FIX: tell d98 where src/ is, ONCE, using either of these (checked in order):
#   1) Set the SMR_SRC_DIR environment variable (e.g. in Maya's userSetup.py) --
#      best for a machine-wide, file-independent setup.
#   2) Set HELPER_SRC_DIR below -- travels with this file, so you still only ever
#      send d98 to Maya (never the helper files).
# After that, only sending d98 is required; helpers import/reload from disk.
#
# Set this to your src/ folder. It is set below to this repo's src/ on this Mac.
# If you run Maya on a different machine, change it to that machine's src/ path,
# e.g. r"C:\keyLabLocal_D\skin-muscle-registration-model\src".
HELPER_SRC_DIR = r"/Users/florykhan/Documents/Projects/Research Projects/skin-to-muscle-registration/src"

# Verbose diagnostics for the bootstrap (set False once it works if you like).
HELPER_DEBUG = True

# Order matters: modules that import siblings must come AFTER them. In
# particular artifact_detection imports mesh_utils + region_selection, so it is
# loaded last.
HELPER_MODULES = ("mesh_utils", "smoothing_utils", "region_selection",
                  "metrics_utils", "maya_io", "artifact_detection")
HELPERS_AVAILABLE = False

# Placeholders so these names always exist (reassigned to real modules on load).
mesh_utils = None
smoothing_utils = None
region_selection = None
metrics_utils = None
maya_io = None
artifact_detection = None


def _dir_has_helpers(d):
    """True only if EVERY helper module .py file exists in directory ``d``."""
    try:
        return bool(d) and all(
            os.path.exists(os.path.join(d, name + ".py")) for name in HELPER_MODULES)
    except Exception:
        return False


def _candidate_src_dirs():
    """Ordered list of directories that might contain the helper modules."""
    cands = []
    # 1) explicit constant
    if HELPER_SRC_DIR:
        cands.append(HELPER_SRC_DIR)
    # 2) environment variable
    env = os.environ.get("SMR_SRC_DIR")
    if env:
        cands.append(env)
    # 3) __file__ (only helps for execfile-style senders that set it to a real path)
    if "__file__" in globals():
        try:
            cands.append(os.path.dirname(os.path.abspath(globals()["__file__"])))
        except Exception:
            pass
    # 4) the running frame's code filename (helps if compiled with a real path)
    try:
        fn = sys._getframe().f_code.co_filename
        if fn and os.path.exists(fn):
            cands.append(os.path.dirname(os.path.abspath(fn)))
    except Exception:
        pass
    # 5) anything already on sys.path, plus the current working directory
    cands.extend(sys.path)
    try:
        cands.append(os.getcwd())
    except Exception:
        pass
    return cands


def _locate_helper_src_dir():
    for d in _candidate_src_dirs():
        if _dir_has_helpers(d):
            return d
    return None


def _load_cleanup_helpers(verbose=None):
    """Locate src/, import/reload the helper modules, and report precisely.

    Returns True on success. On failure it prints a full traceback and the exact
    module that failed, plus the resolved path and sys.path, so the reason is
    never hidden by a broad except. The core registration workflow does not
    depend on this succeeding.
    """
    global HELPERS_AVAILABLE
    import importlib
    import traceback

    if verbose is None:
        verbose = HELPER_DEBUG

    src = _locate_helper_src_dir()

    if verbose:
        print("[helpers] resolving cleanup toolkit ...")
        print("  HELPER_SRC_DIR       = {0!r}".format(HELPER_SRC_DIR))
        print("  SMR_SRC_DIR (env)    = {0!r}".format(os.environ.get("SMR_SRC_DIR")))
        print("  __file__ defined?    = {0}".format("__file__" in globals()))
        print("  co_filename          = {0!r}".format(sys._getframe().f_code.co_filename))
        print("  resolved src dir     = {0!r}".format(src))
        print("  resolved dir exists? = {0}".format(bool(src) and os.path.isdir(src)))

    if not src:
        HELPERS_AVAILABLE = False
        print("[helpers] Could NOT locate the src/ folder containing: {0}".format(
            ", ".join(m + ".py" for m in HELPER_MODULES)))
        print("          MayaCode runs this file from a temp copy, so the real path")
        print("          is unknown. Set HELPER_SRC_DIR (top of this file) or the")
        print("          SMR_SRC_DIR environment variable to your src/ folder, then")
        print("          re-send d98. Example:")
        print("            HELPER_SRC_DIR = r\"C:\\path\\to\\skin-to-muscle-registration\\src\"")
        print("  sys.path (first 10 entries):")
        for p in sys.path[:10]:
            print("     {0}".format(p))
        return False

    if src not in sys.path:
        sys.path.insert(0, src)

    for name in HELPER_MODULES:
        try:
            if name in sys.modules:
                # Reload so edits to helpers take effect when you re-send d98.
                importlib.reload(sys.modules[name])
            else:
                importlib.import_module(name)
            # Expose the module as a top-level name so the orchestrator wrappers
            # (cleanup_selected_region, etc.) can call e.g. smoothing_utils.*.
            globals()[name] = sys.modules[name]
        except Exception:
            HELPERS_AVAILABLE = False
            print("[helpers] FAILED to import helper module '{0}' from {1}".format(name, src))
            print("[helpers] --- full traceback ---")
            traceback.print_exc()
            print("[helpers] ----------------------")
            return False

    HELPERS_AVAILABLE = True
    print("[helpers] cleanup toolkit loaded from: {0}".format(src))
    return True


_load_cleanup_helpers()


# =============================================================================
# CONFIGURATION
# =============================================================================

SKIN_MESH = "skin_cloth_copy_v5_pull_back"
TARGET_MESH = "skin_cloth_copy_v5_pull_back_target"  # scaled-down target mesh

# Internal meshes (same as a99 V14)
INTERNAL_MESHES = [
    "middleres_skull_copy",
    "middleres_Jaw_copy",
    "polySurface38", "polySurface39",
    "polySurface42", "polySurface43", "polySurface48",
    "middleres_DepressorSupercilii_l1",
    "middleres_LevatorLabiiSuperioris_l1",
    "middleres_ZygomaticusMajor_l1",
    "middleres_ZygomaticusMinor_l1",
    "middleres_DepressorLabiiInferioris_l1",
    "middleres_OrbicularisOculi_l_copy",
    "middleres_DepressorSupercilii_r1",
    "middleres_LevatorLabiiSuperioris_r1",
    "middleres_ZygomaticusMajor_r1",
    "middleres_ZygomaticusMinor_r1",
    "middleres_DepressorLabiiInferioris_r1",
    "middleres_OrbicularisOculi_r_copy",
    "middleres_Frontalis_copy",
    "middleres_Procerus1",
    "middleres_Mentalis1",
    "middleres_NasalisTransverse1",
    "middleres_OrbicularisOris_copy_up_V1",
    "middleres_OrbicularisOris_copy_down_V1",
    "middleres_MasseterSuperior|polySurface41|polySurface41",
    "middleres_LateralCartige2",
    "middleres_AlarCartidge1",
    "middleres_NostrilWing1",
    "fat_2", "fat_4", "fat_7", "fat_8", "fat_9",
    "fat_10", "fat_11", "fat_12",
    "fat_element_v1", "fat_element_v2", "fat_element_v3",
    "fat_element_v2_mirrored", "fat_element_v2_mirrored1",
    "fat_element_v2_mirrored3",
]

# Muscle meshes for landmark mapping
UPPER_LIP_MUSCLE = "middleres_OrbicularisOris_copy_up_V1"
LOWER_LIP_MUSCLE = "middleres_OrbicularisOris_copy_down_V1"

# Vertex preset names
UPPER_LIP_PRESET = "upper_lip_area"
LOWER_LIP_PRESET = "lower_lip_area"
NOSE_PRESET = "nose_area"
EYE_LEFT_PRESET = "eye_left_area"
EYE_RIGHT_PRESET = "eye_right_area"

# =============================================================================
# SDF PARAMETERS
# =============================================================================

@dataclass
class SDFParams:
    """Parameters for SDF construction and target-based shrink-wrap."""
    # SDF offset = simulated skin/fat thickness (used for collision only)
    skin_offset: float = 1.5
    # Smoothing passes on the SDF (higher = smoother target surface, hides mesh edges)
    sdf_smoothing_passes: int = 6
    # Smoothing strength per pass
    sdf_smoothing_strength: float = 0.5
    # Soft boundary blending sharpness
    sdf_blend_k: float = 1.0
    # Outline detection: if target vertex is farther than this from ANY internal mesh,
    # treat it as an outline vertex and fall back to closest-point attraction
    outline_distance_threshold: float = 15.0

@dataclass
class ShrinkwrapParams:
    """Parameters controlling the shrink-wrap simulation."""
    # Attraction toward SDF iso-surface
    attraction_strength: float = 0.8
    # Strain stiffness (preserve edge lengths)
    strain_stiffness: float = 1.0
    # Bending stiffness (Laplacian smoothing toward neighbors)
    bending_stiffness: float = 0.3
    # Laplacian smoothing strength (applied every N iterations)
    smoothing_strength: float = 0.15
    smoothing_interval: int = 5
    # Damping on velocity
    damping: float = 0.7
    # Time step
    dt: float = 0.03
    # How many iterations total
    num_iterations: int = 400
    # Viewport update interval
    update_interval: int = 2
    # Max vertex movement per iteration
    max_movement: float = 0.5
    # Landmark constraint strength multiplier
    landmark_strength: float = 5.0
    # Landmark influence radius (in world units)
    landmark_influence_radius: float = 6.0
    # Shrink factor for original edge lengths (< 1 allows compression)
    shrink_factor: float = 0.95
    # Number of projection passes per iteration
    # (project vertices closer to SDF surface)
    projection_interval: int = 0  # DISABLED — using attraction force only
    projection_strength: float = 0.3
    # --- STOPPING CRITERIA ---
    # Per-vertex: settle when within this distance of target for N iterations
    settle_threshold: float = 0.1
    settle_patience: int = 10
    # Whole algorithm: stop when avg movement < threshold for N checks
    convergence_threshold: float = 0.0005
    convergence_patience: int = 20
    convergence_check_interval: int = 5
    # Hard collision: minimum distance from internal mesh surfaces
    collision_min_distance: float = 0.8

# =============================================================================
# AUTO-SAVE SETTINGS
# =============================================================================


SAVE_EVERY_N_ITERATIONS = 200
FAST_DEV_TOTAL_ITERATIONS = 400

# =============================================================================
# GLOBAL STATE
# =============================================================================

REG_DATA = {
    'initialized': False,
    'skin_mesh': None,
    'target_mesh': None,
    'mesh_fns': {},           # mesh_name -> MFnMesh
    'original_vertices': None,
    'target_vertices': None,  # positions from scaled-down target mesh (1-to-1 correspondence)
    'outline_vertices': set(),  # vertex indices that have no nearby internal tissue
    'attraction_directions': {},  # v_idx -> unit direction from skin to target
    'sdf_params': None,
    'shrinkwrap_params': None,
    'vertex_count': 0,
    # Per-vertex SDF data (still used for collision detection)
    'sdf_cache': {},          # vertex_idx -> {'dist': float, 'closest': [x,y,z], 'normal': [x,y,z], 'mesh': str}
    # Landmark data
    'landmarks': {
        'enabled': False,
        'correspondences': {},  # skin_vtx_idx -> {'target': [x,y,z], 'name': str, 'strength': float}
        'influence_map': {},    # vtx_idx -> {'landmark_vtx': int, 'weight': float, 'target': [x,y,z]}
    },
    'neighbors': None,
    'original_lengths': None,
    # Per-vertex attraction weight (0.0 = no attraction, 1.0 = full, default=1.0 if not set)
    'weight_map': {},         # vertex_idx -> float (0.0 to 1.0)
}

# Per-vertex settling state
VERTEX_SETTLED = {}    # vertex_idx -> count of consecutive iterations near target
VERTEX_FROZEN = set()  # vertices that have permanently settled

# =============================================================================
# VECTOR MATH
# =============================================================================

def vec_length(v):
    return math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])

def vec_normalize(v):
    l = vec_length(v)
    return [v[0]/l, v[1]/l, v[2]/l] if l > 1e-8 else [0, 0, 0]

def vec_sub(a, b):
    return [a[0]-b[0], a[1]-b[1], a[2]-b[2]]

def vec_add(a, b):
    return [a[0]+b[0], a[1]+b[1], a[2]+b[2]]

def vec_scale(v, s):
    return [v[0]*s, v[1]*s, v[2]*s]

def vec_dot(a, b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

def vec_cross(a, b):
    return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]

def vec_mean(vectors):
    n = len(vectors)
    if n == 0: return [0, 0, 0]
    return [sum(v[0] for v in vectors)/n,
            sum(v[1] for v in vectors)/n,
            sum(v[2] for v in vectors)/n]

def vec_lerp(a, b, t):
    """Linear interpolation: a*(1-t) + b*t"""
    return [a[0]*(1-t)+b[0]*t, a[1]*(1-t)+b[1]*t, a[2]*(1-t)+b[2]*t]

# =============================================================================
# MAYA MESH UTILITIES
# =============================================================================

def get_mesh_fn(mesh_name):
    try:
        sel = om.MSelectionList()
        sel.add(mesh_name)
        dag = sel.getDagPath(0)
        if dag.apiType() == om.MFn.kTransform:
            dag.extendToShape()
        return om.MFnMesh(dag)
    except:
        return None

def get_mesh_vertices(mesh_name):
    mesh_fn = get_mesh_fn(mesh_name)
    if mesh_fn is None: return []
    pts = mesh_fn.getPoints(om.MSpace.kWorld)
    return [[p.x, p.y, p.z] for p in pts]

def set_mesh_vertices(mesh_name, vertices):
    mesh_fn = get_mesh_fn(mesh_name)
    if mesh_fn is None: return
    pts = om.MPointArray()
    for v in vertices:
        pts.append(om.MPoint(v[0], v[1], v[2]))
    mesh_fn.setPoints(pts, om.MSpace.kWorld)
    mesh_fn.updateSurface()

def get_vertex_neighbors(mesh_name):
    mesh_fn = get_mesh_fn(mesh_name)
    if mesh_fn is None: return []
    n = mesh_fn.numVertices
    neighbors = [set() for _ in range(n)]
    edge_iter = om.MItMeshEdge(mesh_fn.object())
    while not edge_iter.isDone():
        v0, v1 = edge_iter.vertexId(0), edge_iter.vertexId(1)
        neighbors[v0].add(v1)
        neighbors[v1].add(v0)
        edge_iter.next()
    return [list(s) for s in neighbors]

def get_closest_point_on_mesh(point, mesh_fn):
    """Returns (closest_point, distance, face_normal)."""
    try:
        qp = om.MPoint(point[0], point[1], point[2])
        cp, face_id = mesh_fn.getClosestPoint(qp, om.MSpace.kWorld)
        dist = math.sqrt((point[0]-cp.x)**2 + (point[1]-cp.y)**2 + (point[2]-cp.z)**2)
        # Get face normal (use getPolygonNormal — safe, unlike getFaceVertexNormal
        # which can crash Maya if the vertex doesn't belong to the face)
        try:
            normal = mesh_fn.getPolygonNormal(face_id, om.MSpace.kWorld)
            fn = [normal.x, normal.y, normal.z]
        except:
            fn = vec_normalize(vec_sub(point, [cp.x, cp.y, cp.z]))
        return [cp.x, cp.y, cp.z], dist, fn
    except:
        return point[:], float('inf'), [0, 0, 1]

# =============================================================================
# SDF COMPUTATION (Unified distance field from all internal meshes)
# =============================================================================

def compute_sdf_for_point(point, mesh_fns, offset=0.0, blend_k=None):
    """
    Compute the distance from a point to the unified internal surface.
    
    Uses SOFT BOUNDARY (smooth-min) to blend distances from multiple meshes,
    creating smooth transitions where meshes meet instead of sharp creases.
    
    SDF value = smooth_min(distances) - offset.
    Positive = vertex is farther than 'offset' from internal surface
    Negative = vertex is closer than 'offset' to internal surface
    
    Returns: (sdf_value, blended_closest_point, outward_direction, closest_mesh_name)
    """
    # Get blend_k from params if not specified
    if blend_k is None:
        if REG_DATA.get('sdf_params'):
            blend_k = REG_DATA['sdf_params'].sdf_blend_k
        else:
            blend_k = 1.5
    
    # Collect distances from all meshes
    all_hits = []  # (distance, closest_point, mesh_name)
    for mesh_name, mesh_fn in mesh_fns.items():
        if mesh_fn is None:
            continue
        cp, dist, fn = get_closest_point_on_mesh(point, mesh_fn)
        if dist < float('inf'):
            all_hits.append((dist, cp, mesh_name))
    
    if not all_hits:
        return float('inf'), point[:], [0, 0, 1], None
    
    # Sort by distance and keep top N closest for blending
    all_hits.sort(key=lambda x: x[0])
    top_n = min(5, len(all_hits))
    hits = all_hits[:top_n]
    
    # Hard boundary mode (k=0 or only 1 hit)
    if blend_k <= 0 or top_n == 1:
        min_dist = hits[0][0]
        best_closest = hits[0][1]
        best_mesh = hits[0][2]
    else:
        # SOFT BOUNDARY: exponential smooth-min
        # smooth_min = -ln(sum(exp(-k * d_i))) / k
        # weights: w_i = exp(-k * d_i) / sum(exp(-k * d_j))
        #
        # Shift distances by min to prevent overflow: exp(-k*(d-d_min))
        d_min = hits[0][0]
        
        exp_vals = []
        for dist, cp, name in hits:
            # Clamp to prevent overflow
            exponent = -blend_k * (dist - d_min)
            exp_vals.append(math.exp(max(-50, min(50, exponent))))
        
        total_exp = sum(exp_vals)
        if total_exp < 1e-30:
            total_exp = 1e-30
        
        # Weights for blending closest points
        weights = [e / total_exp for e in exp_vals]
        
        # Smooth-min distance
        min_dist = d_min - math.log(total_exp) / blend_k
        
        # Blended closest point (weighted average)
        best_closest = [0, 0, 0]
        for i, (dist, cp, name) in enumerate(hits):
            w = weights[i]
            best_closest[0] += cp[0] * w
            best_closest[1] += cp[1] * w
            best_closest[2] += cp[2] * w
        
        best_mesh = hits[0][2]  # report the dominant mesh
    
    # Outward direction: from blended closest point toward the query point
    to_point = vec_sub(point, best_closest)
    outward = vec_normalize(to_point)
    if vec_length(outward) < 1e-8:
        outward = [0, 0, 1]
    
    # SDF value
    sdf_value = min_dist - offset
    
    return sdf_value, best_closest, outward, best_mesh


def compute_sdf_all_vertices(vertices, mesh_fns, offset=0.0):
    """
    Compute SDF for all skin vertices at once.
    Returns dict: vertex_idx -> {dist, closest, normal, mesh}
    """
    sdf_cache = {}
    total = len(vertices)
    report_interval = max(1, total // 10)
    # Refresh Maya's event loop periodically to prevent crash/hang
    refresh_interval = max(1, total // 20)
    
    print(f"  Computing SDF for {total:,} vertices...")
    print(f"  (querying {len(mesh_fns)} internal meshes per vertex = {total * len(mesh_fns):,} total queries)")
    t0 = time.time()
    
    for i, v in enumerate(vertices):
        sdf_val, cp, normal, mesh_name = compute_sdf_for_point(v, mesh_fns, offset)
        sdf_cache[i] = {
            'dist': sdf_val,
            'closest': cp,
            'normal': normal,
            'mesh': mesh_name,
        }
        # Let Maya breathe — prevents event loop starvation and crash
        if (i + 1) % refresh_interval == 0:
            cmds.refresh()
        
        if (i + 1) % report_interval == 0:
            pct = 100 * (i + 1) / total
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            remaining = (total - i - 1) / rate if rate > 0 else 0
            print(f"    {pct:5.1f}% ({i+1:,}/{total:,}) | "
                  f"Elapsed: {elapsed:.1f}s | ETA: {remaining:.1f}s")
    
    elapsed = time.time() - t0
    print(f"  SDF computed in {elapsed:.1f}s")
    return sdf_cache


def smooth_sdf_cache(sdf_cache, neighbors, passes=3, strength=0.5):
    """
    Smooth the SDF values using Laplacian smoothing on the mesh graph.
    This creates a smoother implicit surface target.
    """
    print(f"  Smoothing SDF ({passes} passes, strength={strength})...")
    
    for p in range(passes):
        new_cache = {}
        for v_idx, data in sdf_cache.items():
            nbrs = neighbors[v_idx]
            if not nbrs:
                new_cache[v_idx] = dict(data)
                continue
            
            # Average SDF distance from neighbors
            nbr_avg_dist = sum(sdf_cache[n]['dist'] for n in nbrs) / len(nbrs)
            
            # Average closest normal from neighbors (for smoothed direction)
            nbr_normals = [sdf_cache[n]['normal'] for n in nbrs]
            avg_normal = vec_normalize(vec_mean(nbr_normals))
            if vec_length(avg_normal) < 1e-8:
                avg_normal = data['normal']
            
            # Blend
            smoothed_dist = data['dist'] * (1 - strength) + nbr_avg_dist * strength
            smoothed_normal = vec_lerp(data['normal'], avg_normal, strength * 0.5)
            smoothed_normal = vec_normalize(smoothed_normal)
            if vec_length(smoothed_normal) < 1e-8:
                smoothed_normal = data['normal']
            
            new_cache[v_idx] = {
                'dist': smoothed_dist,
                'closest': data['closest'],  # keep original closest point
                'normal': smoothed_normal,
                'mesh': data['mesh'],
            }
        
        sdf_cache = new_cache
    
    print(f"  SDF smoothing complete")
    return sdf_cache

# =============================================================================
# VERTEX PRESET LOADING
# =============================================================================

VERTEX_PRESETS_FOLDER = r"C:\keyLabLocal_D\skin-muscle-registration-model\vertex_presets"

def load_vertex_preset(preset_name):
    filepath = os.path.join(VERTEX_PRESETS_FOLDER, f"{preset_name}.json")
    if not os.path.exists(filepath):
        print(f"  WARNING: Preset '{preset_name}' not found at {filepath}")
        return None
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Normalize key name: some presets use 'vertex_indices', others use 'indices'
    if 'vertex_indices' in data and 'indices' not in data:
        data['indices'] = data['vertex_indices']
    elif 'indices' in data and 'vertex_indices' not in data:
        data['vertex_indices'] = data['indices']
    
    # Ensure 'vertex_count' exists
    if 'vertex_count' not in data:
        indices = data.get('indices', data.get('vertex_indices', []))
        data['vertex_count'] = len(indices)
        data['indices'] = indices
    
    return data

# =============================================================================
# LANDMARK SYSTEM (Mouth corners, mouth shape, eye circles, nose)
# =============================================================================

def detect_feature_vertices(vertices, vertex_indices, feature_type='tip_and_corners'):
    """
    Detect key feature vertices from a set of vertex indices.
    Returns dict of feature_name -> vertex_idx
    """
    if not vertex_indices:
        return {}
    
    positions = [(idx, vertices[idx]) for idx in vertex_indices]
    
    if feature_type == 'tip_and_corners':
        # Lip-style: find center-front (max Z) and left/right corners (min/max X)
        sorted_z = sorted(positions, key=lambda p: p[1][2], reverse=True)
        sorted_x = sorted(positions, key=lambda p: p[1][0])
        
        tip_idx = sorted_z[0][0]
        left_idx = sorted_x[0][0]   # most negative X
        right_idx = sorted_x[-1][0]  # most positive X
        
        return {'tip': tip_idx, 'left_corner': left_idx, 'right_corner': right_idx}
    
    elif feature_type == 'eye_circle':
        # Eye: find top, bottom, inner corner, outer corner
        sorted_y = sorted(positions, key=lambda p: p[1][1])
        sorted_x = sorted(positions, key=lambda p: p[1][0])
        
        bottom_idx = sorted_y[0][0]
        top_idx = sorted_y[-1][0]
        
        # Inner = more toward center X (closer to 0)
        # Outer = further from center X
        center_x = sum(p[1][0] for p in positions) / len(positions)
        inner_sorted = sorted(positions, key=lambda p: abs(p[1][0] - center_x * 0.5))
        outer_sorted = sorted(positions, key=lambda p: -abs(p[1][0]))
        
        return {'top': top_idx, 'bottom': bottom_idx,
                'inner': sorted_x[-1][0] if center_x < 0 else sorted_x[0][0],
                'outer': sorted_x[0][0] if center_x < 0 else sorted_x[-1][0]}
    
    elif feature_type == 'nose':
        # Nose: find tip (max Z, most forward), bridge top (max Y), nostrils (min Y of nose + left/right)
        sorted_z = sorted(positions, key=lambda p: p[1][2], reverse=True)
        sorted_y = sorted(positions, key=lambda p: p[1][1], reverse=True)
        
        tip_idx = sorted_z[0][0]
        bridge_idx = sorted_y[0][0]
        
        # Nostrils: bottom portion, then left/right
        bottom_half = sorted(positions, key=lambda p: p[1][1])[:len(positions)//3]
        if bottom_half:
            sorted_bx = sorted(bottom_half, key=lambda p: p[1][0])
            left_nostril = sorted_bx[0][0]
            right_nostril = sorted_bx[-1][0]
        else:
            left_nostril = right_nostril = tip_idx
        
        return {'tip': tip_idx, 'bridge': bridge_idx,
                'left_nostril': left_nostril, 'right_nostril': right_nostril}
    
    return {}


def setup_landmarks():
    """
    Set up anatomical landmark constraints.
    
    Landmarks anchor specific skin vertices to corresponding positions
    on the internal muscle/bone surfaces:
    - Mouth corners (left/right commissure) -> orbicularis oris
    - Mouth shape (upper/lower lip center) -> orbicularis oris
    - Eye circles (top/bottom/inner/outer) -> orbicularis oculi
    - Nose shape (tip, bridge, nostrils) -> nasal cartilage/bone
    
    Call after setup_sdf_registration().
    """
    global REG_DATA
    
    if not REG_DATA['initialized']:
        print("ERROR: Run setup_sdf_registration() first!")
        return False
    
    print("\n" + "=" * 60)
    print("SETTING UP ANATOMICAL LANDMARKS")
    print("=" * 60)
    
    vertices = get_mesh_vertices(REG_DATA['skin_mesh'])
    neighbors = REG_DATA['neighbors']
    mesh_fns = REG_DATA['mesh_fns']
    params = REG_DATA['shrinkwrap_params']
    
    correspondences = {}
    
    # --- 1. MOUTH LANDMARKS ---
    upper_preset = load_vertex_preset(UPPER_LIP_PRESET)
    lower_preset = load_vertex_preset(LOWER_LIP_PRESET)
    
    upper_muscle_fn = get_mesh_fn(UPPER_LIP_MUSCLE)
    lower_muscle_fn = get_mesh_fn(LOWER_LIP_MUSCLE)
    
    if upper_preset and lower_preset:
        upper_features = detect_feature_vertices(vertices, upper_preset['indices'], 'tip_and_corners')
        lower_features = detect_feature_vertices(vertices, lower_preset['indices'], 'tip_and_corners')
        
        mouth_landmarks = {
            'upper_lip_center': (upper_features.get('tip'), upper_muscle_fn, UPPER_LIP_MUSCLE),
            'lower_lip_center': (lower_features.get('tip'), lower_muscle_fn, LOWER_LIP_MUSCLE),
            'left_mouth_corner': (upper_features.get('left_corner'), upper_muscle_fn, UPPER_LIP_MUSCLE),
            'right_mouth_corner': (upper_features.get('right_corner'), upper_muscle_fn, UPPER_LIP_MUSCLE),
            'lower_left_corner': (lower_features.get('left_corner'), lower_muscle_fn, LOWER_LIP_MUSCLE),
            'lower_right_corner': (lower_features.get('right_corner'), lower_muscle_fn, LOWER_LIP_MUSCLE),
        }
        
        for name, (vtx_idx, muscle_fn, muscle_name) in mouth_landmarks.items():
            if vtx_idx is None or muscle_fn is None:
                print(f"  SKIP '{name}': not found")
                continue
            skin_pos = vertices[vtx_idx]
            target, dist, _ = get_closest_point_on_mesh(skin_pos, muscle_fn)
            correspondences[vtx_idx] = {
                'target': target, 'name': name,
                'strength': params.landmark_strength,
            }
            print(f"  Landmark '{name}': vtx[{vtx_idx}] -> {muscle_name} (dist={dist:.2f})")
        
        # Add mouth SHAPE constraint: all lip preset vertices get gentle attraction
        # to their closest point on the lip muscles
        print(f"\n  Setting mouth shape constraints...")
        for i, vtx_idx in enumerate(upper_preset['indices']):
            if vtx_idx not in correspondences and upper_muscle_fn:
                skin_pos = vertices[vtx_idx]
                target, dist, _ = get_closest_point_on_mesh(skin_pos, upper_muscle_fn)
                correspondences[vtx_idx] = {
                    'target': target, 'name': 'upper_lip_shape',
                    'strength': params.landmark_strength * 0.3,
                }
            if (i + 1) % 200 == 0:
                cmds.refresh()
        for i, vtx_idx in enumerate(lower_preset['indices']):
            if vtx_idx not in correspondences and lower_muscle_fn:
                skin_pos = vertices[vtx_idx]
                target, dist, _ = get_closest_point_on_mesh(skin_pos, lower_muscle_fn)
                correspondences[vtx_idx] = {
                    'target': target, 'name': 'lower_lip_shape',
                    'strength': params.landmark_strength * 0.3,
                }
            if (i + 1) % 200 == 0:
                cmds.refresh()
        print(f"  Mouth shape: {len(upper_preset['indices'])} upper + {len(lower_preset['indices'])} lower vertices")
    
    # --- 2. EYE LANDMARKS ---
    for eye_side, preset_name, muscle_name in [
        ('left', EYE_LEFT_PRESET, 'middleres_OrbicularisOculi_l_copy'),
        ('right', EYE_RIGHT_PRESET, 'middleres_OrbicularisOculi_r_copy'),
    ]:
        preset = load_vertex_preset(preset_name)
        muscle_fn = get_mesh_fn(muscle_name)
        
        if preset and muscle_fn:
            eye_features = detect_feature_vertices(vertices, preset['indices'], 'eye_circle')
            
            for feat_name, vtx_idx in eye_features.items():
                lm_name = f'{eye_side}_eye_{feat_name}'
                if vtx_idx is not None and vtx_idx not in correspondences:
                    skin_pos = vertices[vtx_idx]
                    target, dist, _ = get_closest_point_on_mesh(skin_pos, muscle_fn)
                    correspondences[vtx_idx] = {
                        'target': target, 'name': lm_name,
                        'strength': params.landmark_strength * 0.8,
                    }
                    print(f"  Landmark '{lm_name}': vtx[{vtx_idx}] (dist={dist:.2f})")
            
            # Eye shape constraint: all eye preset verts get gentle pull
            print(f"  Setting {eye_side} eye shape constraints...")
            shape_count = 0
            for i, vtx_idx in enumerate(preset['indices']):
                if vtx_idx not in correspondences:
                    skin_pos = vertices[vtx_idx]
                    target, dist, _ = get_closest_point_on_mesh(skin_pos, muscle_fn)
                    correspondences[vtx_idx] = {
                        'target': target, 'name': f'{eye_side}_eye_shape',
                        'strength': params.landmark_strength * 0.2,
                    }
                    shape_count += 1
                if (i + 1) % 200 == 0:
                    cmds.refresh()
            print(f"    {shape_count} shape vertices")
    
    # --- 3. NOSE LANDMARKS ---
    nose_preset = load_vertex_preset(NOSE_PRESET)
    nose_meshes = ['middleres_LateralCartige2', 'middleres_AlarCartidge1',
                   'middleres_NostrilWing1', 'middleres_NasalisTransverse1']
    
    if nose_preset:
        nose_features = detect_feature_vertices(vertices, nose_preset['indices'], 'nose')
        
        for feat_name, vtx_idx in nose_features.items():
            if vtx_idx is None or vtx_idx in correspondences:
                continue
            skin_pos = vertices[vtx_idx]
            
            # Find closest among nose-related internal meshes
            best_target = skin_pos[:]
            best_dist = float('inf')
            for nm in nose_meshes:
                if nm in mesh_fns and mesh_fns[nm] is not None:
                    t, d, _ = get_closest_point_on_mesh(skin_pos, mesh_fns[nm])
                    if d < best_dist:
                        best_dist = d
                        best_target = t
            
            lm_name = f'nose_{feat_name}'
            correspondences[vtx_idx] = {
                'target': best_target, 'name': lm_name,
                'strength': params.landmark_strength * 0.6,
            }
            print(f"  Landmark '{lm_name}': vtx[{vtx_idx}] (dist={best_dist:.2f})")
        
        # Nose shape constraint
        print(f"  Setting nose shape constraints...")
        shape_count = 0
        for i, vtx_idx in enumerate(nose_preset['indices']):
            if vtx_idx not in correspondences:
                skin_pos = vertices[vtx_idx]
                best_target = skin_pos[:]
                best_dist = float('inf')
                for nm in nose_meshes:
                    if nm in mesh_fns and mesh_fns[nm] is not None:
                        t, d, _ = get_closest_point_on_mesh(skin_pos, mesh_fns[nm])
                        if d < best_dist:
                            best_dist = d
                            best_target = t
                if best_dist < float('inf'):
                    correspondences[vtx_idx] = {
                        'target': best_target, 'name': 'nose_shape',
                        'strength': params.landmark_strength * 0.15,
                    }
                    shape_count += 1
            if (i + 1) % 100 == 0:
                cmds.refresh()
        print(f"    {shape_count} shape vertices")
    
    # --- Build influence map (BFS from each landmark) ---
    influence_map = {}
    landmark_vtx_set = set(correspondences.keys())
    
    # Only build influence for strong landmarks (not shape constraints)
    strong_landmarks = {k: v for k, v in correspondences.items()
                       if v['strength'] >= params.landmark_strength * 0.5}
    
    for lm_vtx, lm_data in strong_landmarks.items():
        lm_pos = vertices[lm_vtx]
        visited = {lm_vtx}
        current_ring = {lm_vtx}
        ring_num = 0
        
        while current_ring and ring_num < 10:
            next_ring = set()
            for v_idx in current_ring:
                for n_idx in neighbors[v_idx]:
                    if n_idx in visited or n_idx in landmark_vtx_set:
                        continue
                    d = vec_length(vec_sub(vertices[n_idx], lm_pos))
                    if d < params.landmark_influence_radius:
                        visited.add(n_idx)
                        next_ring.add(n_idx)
                        t = d / params.landmark_influence_radius
                        weight = 0.5 * (1 + math.cos(math.pi * t)) * 0.4
                        
                        if n_idx not in influence_map or weight > influence_map[n_idx]['weight']:
                            influence_map[n_idx] = {
                                'landmark_vtx': lm_vtx,
                                'weight': weight,
                                'target': lm_data['target'],
                            }
            current_ring = next_ring
            ring_num += 1
    
    REG_DATA['landmarks']['correspondences'] = correspondences
    REG_DATA['landmarks']['influence_map'] = influence_map
    REG_DATA['landmarks']['enabled'] = True
    
    total_lm = len([v for v in correspondences.values() if 'shape' not in v['name']])
    total_shape = len([v for v in correspondences.values() if 'shape' in v['name']])
    print(f"\n  Strong landmarks: {total_lm}")
    print(f"  Shape constraint vertices: {total_shape}")
    print(f"  Influenced vertices: {len(influence_map)}")
    print("=" * 60 + "\n")
    return True


def compute_landmark_force(vertex_idx, vertex_pos, params):
    """Compute force pulling vertex toward its landmark target."""
    landmarks = REG_DATA['landmarks']
    if not landmarks['enabled']:
        return [0, 0, 0]
    
    # Direct landmark
    if vertex_idx in landmarks['correspondences']:
        lm = landmarks['correspondences'][vertex_idx]
        to_target = vec_sub(lm['target'], vertex_pos)
        dist = vec_length(to_target)
        if dist < 1e-6:
            return [0, 0, 0]
        direction = vec_scale(to_target, 1.0 / dist)
        # Strength scales with distance (stronger when far)
        force_mag = lm['strength'] * min(dist, 5.0)
        return vec_scale(direction, force_mag)
    
    # Influenced vertex
    if vertex_idx in landmarks['influence_map']:
        inf = landmarks['influence_map'][vertex_idx]
        to_target = vec_sub(inf['target'], vertex_pos)
        dist = vec_length(to_target)
        if dist < 1e-6:
            return [0, 0, 0]
        direction = vec_scale(to_target, 1.0 / dist)
        force_mag = inf['weight'] * params.landmark_strength * min(dist, 3.0)
        return vec_scale(direction, force_mag)
    
    return [0, 0, 0]

# =============================================================================
# MESH EXPANSION (Inflate skin outward before shrink-wrapping)
# =============================================================================

def expand_skin_mesh(amount=5.0, mesh_name=None):
    """
    Expand (inflate) the skin mesh outward along vertex normals.
    
    This is the FIRST STEP in the shrink-wrap workflow:
    1. expand_skin_mesh(5.0)   -- push skin outward
    2. setup_sdf_registration() -- compute SDF from expanded position
    3. run_shrinkwrap(400)      -- shrink inward to SDF surface
    
    The expansion pushes every vertex outward along its averaged vertex normal,
    ensuring the skin starts OUTSIDE all internal meshes.
    
    Args:
        amount: Distance to push outward (in scene units). 
                Typical values: 3.0 to 10.0
                - Too small: skin may not clear all internal meshes
                - Too large: shrink-wrap takes more iterations
        mesh_name: Mesh to expand (default: SKIN_MESH)
    
    Usage:
        >>> expand_skin_mesh(5.0)       # Push 5 units outward along normals
        >>> expand_skin_mesh(8.0)       # Push 8 units outward (more clearance)
    """
    mesh = mesh_name or SKIN_MESH
    
    if not cmds.objExists(mesh):
        print(f"ERROR: Mesh '{mesh}' not found!")
        return
    
    print(f"\n{'='*60}")
    print(f"EXPANDING SKIN MESH: {mesh}")
    print(f"{'='*60}")
    print(f"  Push amount: {amount} units along vertex normals")
    
    mesh_fn = get_mesh_fn(mesh)
    if mesh_fn is None:
        print("ERROR: Could not get mesh function!")
        return
    
    # Get current vertices
    vertices = get_mesh_vertices(mesh)
    num_verts = len(vertices)
    
    # Get vertex normals (averaged face normals per vertex)
    normals = mesh_fn.getVertexNormals(False, om.MSpace.kWorld)  # False = don't angle-weight
    
    # Push each vertex outward along its normal
    new_vertices = []
    for i in range(num_verts):
        n = normals[i]
        normal = [n.x, n.y, n.z]
        nl = vec_length(normal)
        if nl > 1e-8:
            normal = vec_scale(normal, 1.0 / nl)
        else:
            normal = [0, 0, 0]
        
        new_pos = vec_add(vertices[i], vec_scale(normal, amount))
        new_vertices.append(new_pos)
    
    # Apply
    set_mesh_vertices(mesh, new_vertices)
    cmds.refresh()
    
    # Report
    avg_move = sum(vec_length(vec_sub(new_vertices[i], vertices[i])) 
                   for i in range(num_verts)) / num_verts
    print(f"  Vertices expanded: {num_verts:,}")
    print(f"  Average displacement: {avg_move:.2f} units")
    print(f"\n  Next step: setup_sdf_registration()")
    print(f"{'='*60}\n")


def undo_expansion():
    """
    Undo the expansion by resetting to original positions.
    Only works if setup_sdf_registration() was called BEFORE expanding,
    or if you manually stored the original vertices.
    """
    if REG_DATA.get('original_vertices'):
        set_mesh_vertices(REG_DATA['skin_mesh'], REG_DATA['original_vertices'])
        cmds.refresh()
        print("Skin reset to pre-expansion positions.")
    else:
        print("No stored original vertices. Use Maya's Ctrl+Z to undo.")


def expand_and_setup(expand_amount=5.0, sdf_params=None, shrinkwrap_params=None):
    """
    Convenience: Expand skin mesh THEN set up SDF registration.
    
    This is the recommended one-step initialization:
    1. Stores original vertex positions
    2. Expands the mesh outward along normals
    3. Computes SDF from the expanded position
    4. Sets up landmark constraints
    
    Args:
        expand_amount: How far to push vertices outward (default 5.0)
        sdf_params: Optional SDFParams override
        shrinkwrap_params: Optional ShrinkwrapParams override
    
    Usage:
        >>> expand_and_setup(5.0)
        >>> run_shrinkwrap(400)
    """
    global REG_DATA
    
    # Store original positions BEFORE expansion
    original = get_mesh_vertices(SKIN_MESH)
    
    # Expand
    expand_skin_mesh(expand_amount)
    
    # Setup SDF from expanded position
    result = setup_sdf_registration(sdf_params, shrinkwrap_params)
    
    # Override original_vertices with pre-expansion positions
    # so reset_skin() goes back to the true original
    if result:
        REG_DATA['pre_expansion_vertices'] = original
        print(f"\n  [NOTE] Pre-expansion positions stored. Use reset_to_original() to go back.")
    
    return result


def reset_to_original():
    """Reset skin to the position BEFORE expansion (true original)."""
    if REG_DATA.get('pre_expansion_vertices'):
        set_mesh_vertices(REG_DATA['skin_mesh'], REG_DATA['pre_expansion_vertices'])
        cmds.refresh()
        print("Skin reset to pre-expansion (true original) positions.")
    elif REG_DATA.get('original_vertices'):
        set_mesh_vertices(REG_DATA['skin_mesh'], REG_DATA['original_vertices'])
        cmds.refresh()
        print("Skin reset to stored original positions.")
    else:
        print("No original positions stored.")


# =============================================================================
# SETUP
# =============================================================================

def setup_target_registration(sdf_params=None, shrinkwrap_params=None, target_mesh=None):
    """
    Initialize the Target-Based shrink-wrap registration system.
    
    Steps:
    1. Validate skin mesh and target mesh
    2. Build mesh function objects for internal meshes
    3. Load target vertex positions (1-to-1 correspondence)
    4. Detect outline vertices (far from internal tissue)
    5. Pre-compute attraction directions
    6. Compute SDF cache (for collision detection only)
    """
    global REG_DATA
    
    print("\n" + "=" * 70)
    print("TARGET-BASED SHRINK-WRAP REGISTRATION SETUP")
    print("=" * 70)
    
    if sdf_params is None:
        sdf_params = SDFParams()
    if shrinkwrap_params is None:
        shrinkwrap_params = ShrinkwrapParams()
    
    if target_mesh is None:
        target_mesh = TARGET_MESH
    
    # Validate skin mesh
    if not cmds.objExists(SKIN_MESH):
        print(f"\nERROR: Skin mesh '{SKIN_MESH}' not found!")
        return False
    
    # Validate target mesh
    if not cmds.objExists(target_mesh):
        print(f"\nERROR: Target mesh '{target_mesh}' not found!")
        print(f"  Create it by duplicating + scaling down '{SKIN_MESH}'")
        return False
    
    # Validate internal meshes
    valid_internal = [m for m in INTERNAL_MESHES if cmds.objExists(m)]
    missing = [m for m in INTERNAL_MESHES if not cmds.objExists(m)]
    
    # Get counts
    skin_vtx_count = cmds.polyEvaluate(SKIN_MESH, vertex=True)
    target_vtx_count = cmds.polyEvaluate(target_mesh, vertex=True)
    
    print(f"\nSkin mesh: {SKIN_MESH}")
    print(f"  Vertices: {skin_vtx_count:,}")
    print(f"Target mesh: {target_mesh}")
    print(f"  Vertices: {target_vtx_count:,}")
    
    if skin_vtx_count != target_vtx_count:
        print(f"\nERROR: Vertex count MISMATCH! Skin={skin_vtx_count}, Target={target_vtx_count}")
        print(f"  Target mesh must be a copy of skin mesh (same topology)!")
        return False
    
    print(f"  ✓ Vertex counts match — 1-to-1 correspondence confirmed")
    print(f"\nInternal meshes: {len(valid_internal)} / {len(INTERNAL_MESHES)}")
    if missing:
        print(f"  Missing: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    
    # Build mesh functions
    mesh_fns = {}
    for m in valid_internal:
        fn = get_mesh_fn(m)
        if fn:
            mesh_fns[m] = fn
    
    # Get vertices and topology
    original_vertices = get_mesh_vertices(SKIN_MESH)
    target_vertices = get_mesh_vertices(target_mesh)
    num_verts = len(original_vertices)
    neighbors = get_vertex_neighbors(SKIN_MESH)
    
    # Original edge lengths
    original_lengths = {}
    for i, nbrs in enumerate(neighbors):
        for j in nbrs:
            key = (min(i, j), max(i, j))
            if key not in original_lengths:
                original_lengths[key] = vec_length(vec_sub(
                    original_vertices[j], original_vertices[i])) * shrinkwrap_params.shrink_factor
    
    # --- DETECT OUTLINE VERTICES ---
    # Check distance from each target vertex to nearest internal mesh
    # If too far, mark as outline (will use closest-point fallback)
    print(f"\n--- DETECTING OUTLINE VERTICES ---")
    print(f"  Threshold: {sdf_params.outline_distance_threshold} units")
    outline_vertices = set()
    attraction_directions = {}
    
    t0 = time.time()
    report_interval = max(1, num_verts // 10)
    refresh_interval = max(1, num_verts // 20)
    
    for i in range(num_verts):
        target_pos = target_vertices[i]
        skin_pos = original_vertices[i]
        
        # Find minimum distance from target vertex to any internal mesh
        min_dist = float('inf')
        for mesh_name, mesh_fn in mesh_fns.items():
            cp, dist, _ = get_closest_point_on_mesh(target_pos, mesh_fn)
            if dist < min_dist:
                min_dist = dist
        
        if min_dist > sdf_params.outline_distance_threshold:
            outline_vertices.add(i)
        
        # Pre-compute attraction direction (skin → target)
        to_target = vec_sub(target_pos, skin_pos)
        dist_to_target = vec_length(to_target)
        if dist_to_target > 1e-6:
            attraction_directions[i] = vec_scale(to_target, 1.0 / dist_to_target)
        else:
            attraction_directions[i] = [0, 0, 1]
        
        if (i + 1) % refresh_interval == 0:
            cmds.refresh()
        if (i + 1) % report_interval == 0:
            pct = 100 * (i + 1) / num_verts
            print(f"    {pct:5.1f}% ({i+1:,}/{num_verts:,})")
    
    elapsed = time.time() - t0
    print(f"  Outline vertices: {len(outline_vertices):,} / {num_verts:,}")
    print(f"  Normal vertices: {num_verts - len(outline_vertices):,}")
    print(f"  Detection time: {elapsed:.1f}s")
    
    # Pre-compute original distances (skin → target per vertex)
    original_distances = {}
    for i in range(num_verts):
        original_distances[i] = vec_length(vec_sub(target_vertices[i], original_vertices[i]))
    
    # Store data
    REG_DATA['initialized'] = True
    REG_DATA['skin_mesh'] = SKIN_MESH
    REG_DATA['target_mesh'] = target_mesh
    REG_DATA['mesh_fns'] = mesh_fns
    REG_DATA['original_vertices'] = original_vertices
    REG_DATA['target_vertices'] = target_vertices
    REG_DATA['original_distances'] = original_distances
    REG_DATA['outline_vertices'] = outline_vertices
    REG_DATA['attraction_directions'] = attraction_directions
    REG_DATA['sdf_params'] = sdf_params
    REG_DATA['shrinkwrap_params'] = shrinkwrap_params
    REG_DATA['vertex_count'] = num_verts
    REG_DATA['neighbors'] = neighbors
    REG_DATA['original_lengths'] = original_lengths
    
    # Compute SDF cache (for collision detection only)
    print(f"\n--- SDF CONSTRUCTION (for collision detection) ---")
    sdf_cache = compute_sdf_all_vertices(original_vertices, mesh_fns, sdf_params.skin_offset)
    sdf_cache = smooth_sdf_cache(sdf_cache, neighbors,
                                  sdf_params.sdf_smoothing_passes,
                                  sdf_params.sdf_smoothing_strength)
    REG_DATA['sdf_cache'] = sdf_cache
    
    # Report
    avg_dist = sum(vec_length(vec_sub(target_vertices[i], original_vertices[i]))
                   for i in range(num_verts)) / num_verts
    print(f"\n  Average skin→target distance: {avg_dist:.2f}")
    
    print("\n" + "=" * 70)
    print("SETUP COMPLETE")
    print("=" * 70)
    print(f"\nCommands:")
    print(f"  apply_all_contour_landmarks()   - Load landmark constraints")
    print(f"  run_shrinkwrap(400)             - Run shrink-wrap simulation")
    print(f"  reset_skin()                    - Reset to original")
    print("=" * 70 + "\n")
    
    return True

# =============================================================================
# FORCE COMPUTATIONS
# =============================================================================

def compute_target_attraction(vertex_idx, vertex_pos, sdf_cache, mesh_fns, params, offset):
    """
    Compute attraction force toward the TARGET MESH vertex position.
    
    For NORMAL vertices: attract toward the corresponding vertex on the target mesh.
    For OUTLINE vertices (no nearby tissue): fall back to closest-point SDF attraction.
    
    Returns: (force, dist_to_target, attraction_direction)
      - attraction_direction: unit vector from current pos toward target (used for collision)
    """
    target_vertices = REG_DATA.get('target_vertices')
    outline_vertices = REG_DATA.get('outline_vertices', set())
    
    # --- OUTLINE VERTEX: fall back to closest-point SDF attraction ---
    if vertex_idx in outline_vertices or target_vertices is None:
        # Use old SDF closest-point method
        if vertex_idx in sdf_cache:
            closest_pt = sdf_cache[vertex_idx]['closest']
        else:
            _, closest_pt, _, _ = compute_sdf_for_point(vertex_pos, mesh_fns, offset)
        
        from_surface = vec_sub(vertex_pos, closest_pt)
        raw_dist = vec_length(from_surface)
        if raw_dist < 1e-6:
            return [0, 0, 0], 0.0, [0, 0, 1]
        
        outward_dir = vec_scale(from_surface, 1.0 / raw_dist)
        target_pos = vec_add(closest_pt, vec_scale(outward_dir, offset))
        to_target = vec_sub(target_pos, vertex_pos)
        dist_to_target = vec_length(to_target)
        
        if dist_to_target < 0.01:
            return [0, 0, 0], dist_to_target, outward_dir
        
        direction = vec_scale(to_target, 1.0 / dist_to_target)
        speed = 0.5 if dist_to_target > 1.0 else 0.25
        force = vec_scale(direction, params.attraction_strength * speed * dist_to_target)
        return force, dist_to_target, direction
    
    # --- NORMAL VERTEX: attract toward target mesh vertex ---
    target_pos = target_vertices[vertex_idx]
    original_dir = REG_DATA.get('attraction_directions', {}).get(vertex_idx, [0, 0, 1])
    original_dist = REG_DATA.get('original_distances', {}).get(vertex_idx, 1.0)
    
    to_target = vec_sub(target_pos, vertex_pos)
    dist_to_target = vec_length(to_target)
    
    if dist_to_target < 0.01:
        # Already at target
        return [0, 0, 0], dist_to_target, original_dir
    
    direction = vec_scale(to_target, 1.0 / dist_to_target)
    
    # --- OVERSHOOT DETECTION ---
    # Check if vertex has passed BEYOND the target (dot product with original direction < 0)
    # This means the vertex overshot and is now behind the target
    dot = vec_dot(to_target, original_dir)
    if dot < 0 and original_dist > 0.1:
        # Vertex overshot! Pull it back gently toward target
        speed = 0.1
        force = vec_scale(direction, params.attraction_strength * speed * min(dist_to_target, 1.0))
        return force, dist_to_target, direction
    
    # --- BRAKING ZONE ---
    # Calculate progress: how much of the journey is complete (0.0 = start, 1.0 = at target)
    if original_dist > 0.1:
        progress = 1.0 - (dist_to_target / original_dist)
    else:
        progress = 1.0  # already very close
    
    # Adaptive speed with BRAKING in final approach
    if progress > 0.95:
        # Last 5%: very slow (braking zone)
        speed = 0.05
    elif progress > 0.90:
        # Last 10%: slow
        speed = 0.1
    elif progress > 0.80:
        # Last 20%: moderate
        speed = 0.25
    elif dist_to_target > 10.0:
        speed = 2.0
    elif dist_to_target > 5.0:
        speed = 1.5
    elif dist_to_target > 1.0:
        speed = 1.0
    elif dist_to_target > 0.2:
        speed = 0.5
    else:
        speed = 0.25
    
    force = vec_scale(direction, params.attraction_strength * speed * dist_to_target)
    
    return force, dist_to_target, direction


def compute_strain_force(vertex_idx, vertices, neighbors, original_lengths, params):
    """Preserve original edge lengths."""
    force = [0, 0, 0]
    vertex = vertices[vertex_idx]
    for n_idx in neighbors[vertex_idx]:
        edge = vec_sub(vertices[n_idx], vertex)
        cur_len = vec_length(edge)
        key = (min(vertex_idx, n_idx), max(vertex_idx, n_idx))
        orig_len = original_lengths.get(key, cur_len)
        if cur_len > 1e-8:
            direction = vec_scale(edge, 1.0 / cur_len)
            strain = cur_len - orig_len
            force = vec_add(force, vec_scale(direction, params.strain_stiffness * strain))
    return force


def compute_bending_force(vertex_idx, vertices, neighbors, params):
    """Laplacian bending resistance."""
    nbrs = neighbors[vertex_idx]
    if not nbrs:
        return [0, 0, 0]
    avg = vec_mean([vertices[n] for n in nbrs])
    lap = vec_sub(avg, vertices[vertex_idx])
    return vec_scale(lap, params.bending_stiffness)

# =============================================================================
# SHRINK-WRAP SIMULATION
# =============================================================================

def refresh_sdf(vertices, mesh_fns, offset, neighbors, smoothing_passes, smoothing_strength):
    """Re-compute SDF for current vertex positions."""
    sdf_cache = compute_sdf_all_vertices(vertices, mesh_fns, offset)
    sdf_cache = smooth_sdf_cache(sdf_cache, neighbors, smoothing_passes, smoothing_strength)
    return sdf_cache


def project_to_sdf_surface(vertices, sdf_cache, mesh_fns, offset, strength=0.3):
    """
    Project vertices toward the SDF iso-surface (distance = offset from internal).
    Uses closest-point direction, not cached normals.
    Respects per-vertex weight_map.
    """
    wmap = REG_DATA.get('weight_map', {})
    new_verts = []
    for i, v in enumerate(vertices):
        if i in sdf_cache:
            closest_pt = sdf_cache[i]['closest']
        else:
            _, closest_pt, _, _ = compute_sdf_for_point(v, mesh_fns, offset)
        
        # Direction from closest surface to vertex
        from_surface = vec_sub(v, closest_pt)
        raw_dist = vec_length(from_surface)
        
        if raw_dist < 1e-6:
            new_verts.append(v)
            continue
        
        outward = vec_scale(from_surface, 1.0 / raw_dist)
        
        # Target position = closest_pt + offset * outward
        target = vec_add(closest_pt, vec_scale(outward, offset))
        
        # Move toward target
        to_target = vec_sub(target, v)
        if vec_length(to_target) < 0.01:
            new_verts.append(v)
            continue
        
        # Apply per-vertex weight (weighted areas project less)
        vtx_strength = strength * wmap.get(i, 1.0)
        
        new_verts.append(vec_add(v, vec_scale(to_target, vtx_strength)))
    return new_verts


def apply_laplacian_smoothing(vertices, neighbors, strength=0.1):
    """Apply Laplacian smoothing to all vertices."""
    new_verts = []
    for i, v in enumerate(vertices):
        nbrs = neighbors[i]
        if not nbrs:
            new_verts.append(v)
            continue
        avg = vec_mean([vertices[n] for n in nbrs])
        new_verts.append(vec_lerp(v, avg, strength))
    return new_verts


def clamp_movement(old_verts, new_verts, max_move):
    """Clamp per-vertex movement to prevent explosions."""
    result = []
    for old, new in zip(old_verts, new_verts):
        diff = vec_sub(new, old)
        dist = vec_length(diff)
        if dist > max_move:
            diff = vec_scale(diff, max_move / dist)
            result.append(vec_add(old, diff))
        else:
            result.append(new)
    return result


def auto_save_scene(iteration):
    """Save Maya scene at checkpoint."""
    try:
        current_file = cmds.file(query=True, sceneName=True)
        
        # Use FIXED base directory — never derive from current scene name
        # (otherwise it recursively nests if scene was already renamed)
        base_dir = r"C:\keyLabLocal_D\c99-maya-ziva-workshop\experiment_files"
        save_dir = os.path.join(base_dir, AUTO_SAVE_FOLDER)
        
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"iter_{iteration:04d}.mb")
        
        # Save a copy WITHOUT renaming the current scene
        cmds.file(rename=save_path)
        cmds.file(save=True, type="mayaBinary")
        
        # RESTORE the original scene name so we don't nest paths
        if current_file:
            cmds.file(rename=current_file)
        
        print(f"  [Auto-save] {save_path}")
    except Exception as e:
        print(f"  [Auto-save ERROR] {e}")


def run_shrinkwrap(num_iterations=None, refresh_interval=100, params=None):
    """
    Run the SDF shrink-wrap simulation.
    
    The skin mesh is iteratively attracted toward the SDF iso-surface
    (the smoothed, offset distance field from all internal meshes).
    
    Args:
        num_iterations: Number of iterations (default from params)
        refresh_interval: Recompute SDF every N iterations (0 = never)
        params: Override ShrinkwrapParams
    """
    import maya.mel as mel
    
    global REG_DATA
    
    if not REG_DATA['initialized']:
        print("ERROR: Run setup_sdf_registration() first!")
        return
    
    if params is not None:
        REG_DATA['shrinkwrap_params'] = params
    sw_params = REG_DATA['shrinkwrap_params']
    sdf_params = REG_DATA['sdf_params']
    
    if num_iterations is not None:
        sw_params.num_iterations = num_iterations
    
    skin_mesh = REG_DATA['skin_mesh']
    mesh_fns = REG_DATA['mesh_fns']
    neighbors = REG_DATA['neighbors']
    original_lengths = REG_DATA['original_lengths']
    sdf_cache = REG_DATA['sdf_cache']
    
    vertices = get_mesh_vertices(skin_mesh)
    num_verts = len(vertices)
    velocities = [[0, 0, 0] for _ in range(num_verts)]
    
    print("\n" + "=" * 70)
    print("SDF SHRINK-WRAP SIMULATION")
    print("=" * 70)
    print(f"  Vertices: {num_verts:,}")
    print(f"  Iterations: {sw_params.num_iterations}")
    print(f"  SDF offset: {sdf_params.skin_offset}")
    print(f"  Attraction: {sw_params.attraction_strength}")
    print(f"  Strain: {sw_params.strain_stiffness}")
    print(f"  Bending: {sw_params.bending_stiffness}")
    print(f"  Landmark strength: {sw_params.landmark_strength}")
    print(f"  Landmarks enabled: {REG_DATA['landmarks']['enabled']}")
    lm_count = len(REG_DATA['landmarks']['correspondences'])
    print(f"  Landmark vertices: {lm_count}")
    print(f"  SDF refresh interval: {refresh_interval}")
    print(f"\n>>> PRESS ESC TO CANCEL <<<")
    print("=" * 70 + "\n")
    
    # Setup progress bar
    gMainProgressBar = mel.eval('$tmp = $gMainProgressBar')
    cmds.progressBar(gMainProgressBar, edit=True, beginProgress=True,
                     isInterruptable=True,
                     status='SDF Shrink-Wrap... Press ESC to cancel',
                     maxValue=sw_params.num_iterations)
    
    start_time = time.time()
    cancelled = False
    converged = False
    stable_count = 0  # consecutive convergence checks below threshold
    collision_watchlist = set()  # vertices to re-check for penetration
    
    # Reset settling state
    global VERTEX_SETTLED, VERTEX_FROZEN
    VERTEX_SETTLED = {}
    VERTEX_FROZEN = set()
    
    # MEMORY OPTIMIZATION: Disable undo queue during simulation
    # Maya accumulates every vertex edit as an undo step → massive memory growth
    import gc
    undo_was_on = cmds.undoInfo(query=True, state=True)
    if undo_was_on:
        cmds.undoInfo(state=False)  # disable undo
        cmds.flushUndo()            # clear existing undo history
        print("  [Memory] Undo disabled + flushed for simulation")
    
    try:
        for iteration in range(sw_params.num_iterations):
            # Per-vertex attraction directions (for collision push-back)
            attraction_dirs = {}
            
            # Force Maya to process events (so ESC actually works)
            cmds.refresh()
            
            # Check ESC
            if cmds.progressBar(gMainProgressBar, query=True, isCancelled=True):
                print("\n!!! CANCELLED BY USER (ESC) !!!")
                cancelled = True
                break
            
            # Check FILE-BASED STOP: create a file called "STOP" to halt
            stop_file = os.path.join(os.path.dirname(__file__) if '__file__' in dir() else
                          r"C:\keyLabLocal_D\skin-muscle-registration-model", "STOP")
            if os.path.exists(stop_file):
                try:
                    os.remove(stop_file)
                except:
                    pass
                print("\n!!! STOPPED BY FILE FLAG (STOP file detected) !!!")
                cancelled = True
                break
            
            cmds.progressBar(gMainProgressBar, edit=True, step=1)
            
            old_vertices = [v[:] for v in vertices]
            new_vertices = []
            
            # Refresh SDF periodically
            if refresh_interval > 0 and iteration > 0 and iteration % refresh_interval == 0:
                print(f"  [Refreshing SDF at iteration {iteration}...]")
                sdf_cache = refresh_sdf(vertices, mesh_fns, sdf_params.skin_offset,
                                        neighbors,
                                        sdf_params.sdf_smoothing_passes,
                                        sdf_params.sdf_smoothing_strength)
                REG_DATA['sdf_cache'] = sdf_cache
            
            # Per-vertex force computation
            for v_idx in range(num_verts):
                vertex = vertices[v_idx]
                
                # Skip frozen (permanently settled) vertices
                if v_idx in VERTEX_FROZEN:
                    new_vertices.append(vertex)
                    continue
                
                # Target-based attraction (toward corresponding target vertex)
                f_attraction, dist_to_target, attract_dir = compute_target_attraction(
                    v_idx, vertex, sdf_cache, mesh_fns,
                    sw_params, sdf_params.skin_offset)
                
                # Store attraction direction for collision response
                attraction_dirs[v_idx] = attract_dir
                sdf_dist = dist_to_target  # for reporting
                
                # Strain (preserve edge lengths)
                f_strain = compute_strain_force(v_idx, vertices, neighbors,
                                                original_lengths, sw_params)
                
                # Bending (Laplacian smoothness)
                f_bending = compute_bending_force(v_idx, vertices, neighbors, sw_params)
                
                # Landmark constraint
                f_landmark = compute_landmark_force(v_idx, vertex, sw_params)
                
                # Sum forces
                f_total = vec_add(vec_add(vec_add(f_attraction, f_strain), f_bending), f_landmark)
                
                # Apply per-vertex weight to TOTAL force (not just attraction)
                # weight=0.2 means this vertex resists movement (deforms less)
                weight = REG_DATA.get('weight_map', {}).get(v_idx, 1.0)
                if weight < 1.0:
                    f_total = vec_scale(f_total, weight)
                
                # Update velocity with damping
                vel = velocities[v_idx]
                vel = vec_add(vec_scale(vel, sw_params.damping),
                              vec_scale(f_total, sw_params.dt))
                
                # BRAKING: Extra velocity damping when close to target
                # Prevents overshoot from accumulated momentum
                target_verts = REG_DATA.get('target_vertices')
                orig_dists = REG_DATA.get('original_distances', {})
                if target_verts and v_idx not in REG_DATA.get('outline_vertices', set()):
                    orig_d = orig_dists.get(v_idx, 1.0)
                    if orig_d > 0.1:
                        progress = 1.0 - (dist_to_target / orig_d)
                        if progress > 0.95:
                            vel = vec_scale(vel, 0.1)   # heavy brake
                        elif progress > 0.90:
                            vel = vec_scale(vel, 0.3)   # moderate brake
                        elif progress > 0.80:
                            vel = vec_scale(vel, 0.5)   # light brake
                
                velocities[v_idx] = vel
                
                new_pos = vec_add(vertex, vec_scale(vel, sw_params.dt))
                
                # OVERSHOOT CLAMP: don't move further than remaining distance to target
                if target_verts and v_idx not in REG_DATA.get('outline_vertices', set()):
                    step_dist = vec_length(vec_sub(new_pos, vertex))
                    if step_dist > dist_to_target and dist_to_target > 0.01:
                        # Clamp to target position
                        new_pos = target_verts[v_idx][:]
                        velocities[v_idx] = [0, 0, 0]
                
                new_vertices.append(new_pos)
                
                # --- Per-vertex settling check ---
                movement = vec_length(vec_sub(new_pos, vertex))
                if movement < sw_params.settle_threshold and abs(sdf_dist) < sw_params.settle_threshold:
                    VERTEX_SETTLED[v_idx] = VERTEX_SETTLED.get(v_idx, 0) + 1
                    if VERTEX_SETTLED[v_idx] >= sw_params.settle_patience:
                        VERTEX_FROZEN.add(v_idx)
                        velocities[v_idx] = [0, 0, 0]
                else:
                    VERTEX_SETTLED[v_idx] = 0  # reset counter
            
            # Clamp movement
            vertices = clamp_movement(old_vertices, new_vertices, sw_params.max_movement)
            
            # Periodic Laplacian smoothing
            if (sw_params.smoothing_interval > 0 and
                iteration > 0 and iteration % sw_params.smoothing_interval == 0):
                vertices = apply_laplacian_smoothing(vertices, neighbors,
                                                      sw_params.smoothing_strength)
            
            # Periodic SDF projection (shrink-wrap snap)
            if (sw_params.projection_interval > 0 and
                iteration > 0 and iteration % sw_params.projection_interval == 0):
                vertices = project_to_sdf_surface(vertices, sdf_cache, mesh_fns,
                                                   sdf_params.skin_offset,
                                                   sw_params.projection_strength)
            
            # HARD COLLISION CONSTRAINT: After all forces, ensure no penetration
            # Two-tier check: full sweep every 3 iters, watchlist-only otherwise
            collision_min = sw_params.collision_min_distance
            penetrating = 0
            
            if iteration % 2 == 0:
                check_set = range(num_verts)
            else:
                check_set = list(collision_watchlist)
            
            new_watchlist = set()
            pushed_vertices = set()  # track which verts were teleported
            for v_idx in check_set:
                sdf_val, cp, outward, _ = compute_sdf_for_point(
                    vertices[v_idx], mesh_fns, 0)
                
                if sdf_val < collision_min:
                    # Push direction: SDF outward (perpendicular away from surface)
                    safe_pos = vec_add(cp, vec_scale(outward, collision_min))
                    vertices[v_idx] = safe_pos
                    velocities[v_idx] = [0, 0, 0]
                    penetrating += 1
                    pushed_vertices.add(v_idx)
                    new_watchlist.add(v_idx)
                    
                    if neighbors and v_idx < len(neighbors):
                        for n_idx in neighbors[v_idx]:
                            new_watchlist.add(n_idx)
                    
                    if v_idx in VERTEX_FROZEN:
                        VERTEX_FROZEN.discard(v_idx)
                        VERTEX_SETTLED[v_idx] = 0
            
            collision_watchlist = new_watchlist
            
            # POST-COLLISION SMOOTHING: Blend pushed vertices with their neighbors
            # to prevent isolated vertices from creating stretched triangles.
            # Only affects vertices that were just teleported + their 1-ring neighbors.
            if pushed_vertices and neighbors:
                smooth_set = set(pushed_vertices)
                for v_idx in pushed_vertices:
                    if v_idx < len(neighbors):
                        for n_idx in neighbors[v_idx]:
                            smooth_set.add(n_idx)
                
                # Light Laplacian smoothing on affected region only
                for v_idx in smooth_set:
                    if v_idx >= len(neighbors) or not neighbors[v_idx]:
                        continue
                    nbrs = neighbors[v_idx]
                    avg = vec_mean([vertices[n] for n in nbrs])
                    # Gentle blend: 80% current + 20% neighbor average
                    vertices[v_idx] = vec_lerp(vertices[v_idx], avg, 0.2)
            
            # MAX EDGE-STRETCH CLAMP: Prevent any edge from exceeding 2x original length
            if original_lengths:
                max_stretch = 2.0
                for v_idx in range(num_verts):
                    if v_idx >= len(neighbors):
                        continue
                    for n_idx in neighbors[v_idx]:
                        if n_idx <= v_idx:  # process each edge once
                            continue
                        edge_key = (min(v_idx, n_idx), max(v_idx, n_idx))
                        if edge_key not in original_lengths:
                            continue
                        orig_len = original_lengths[edge_key]
                        edge_vec = vec_sub(vertices[n_idx], vertices[v_idx])
                        curr_len = vec_length(edge_vec)
                        
                        if curr_len > orig_len * max_stretch and curr_len > 1e-6:
                            # Pull both endpoints toward each other
                            excess = curr_len - orig_len * max_stretch
                            direction = vec_scale(edge_vec, 1.0 / curr_len)
                            pull = excess * 0.3  # 30% correction per iteration
                            vertices[v_idx] = vec_add(vertices[v_idx], 
                                                       vec_scale(direction, pull))
                            vertices[n_idx] = vec_add(vertices[n_idx], 
                                                       vec_scale(direction, -pull))
            
            # Update viewport
            if iteration % sw_params.update_interval == 0:
                set_mesh_vertices(skin_mesh, vertices)
                cmds.refresh()
                
                elapsed = time.time() - start_time
                rate = (iteration + 1) / elapsed if elapsed > 0 else 0
                remaining = (sw_params.num_iterations - iteration - 1) / rate if rate > 0 else 0
                
                # Compute avg movement
                total_move = sum(vec_length(vec_sub(vertices[i], old_vertices[i]))
                                 for i in range(num_verts))
                avg_move = total_move / num_verts
                
                # Compute avg SDF distance
                sdf_dists = []
                sample = range(0, num_verts, max(1, num_verts // 100))
                for si in sample:
                    sd, _, _, _ = compute_sdf_for_point(vertices[si], mesh_fns,
                                                         sdf_params.skin_offset)
                    sdf_dists.append(sd)
                avg_sdf = sum(sdf_dists) / len(sdf_dists) if sdf_dists else 0
                
                frozen_count = len(VERTEX_FROZEN)
                settling_count = sum(1 for c in VERTEX_SETTLED.values() if c > 0)
                
                print(f"[{iteration+1:4d}/{sw_params.num_iterations}] "
                      f"Time: {elapsed:6.1f}s | "
                      f"ETA: {remaining:5.0f}s | "
                      f"Move: {avg_move:.6f} | "
                      f"SDF: {avg_sdf:.3f} | "
                      f"Settled: {frozen_count:,} | "
                      f"Penetr: {penetrating}")
                
                # --- Early stopping convergence check ---
                if (iteration > 0 and 
                    iteration % sw_params.convergence_check_interval == 0):
                    if avg_move < sw_params.convergence_threshold:
                        stable_count += 1
                        if stable_count >= sw_params.convergence_patience:
                            print(f"\n{'='*70}")
                            print(f"CONVERGED! Early stopping at iteration {iteration+1}")
                            print(f"  Avg movement ({avg_move:.6f}) < threshold "
                                  f"({sw_params.convergence_threshold}) for "
                                  f"{stable_count} consecutive checks")
                            print(f"  Settled vertices: {frozen_count:,}/{num_verts:,} "
                                  f"({100*frozen_count/num_verts:.1f}%)")
                            print(f"{'='*70}")
                            converged = True
                            break
                    else:
                        stable_count = 0
            
            # Auto-save
            if (iteration + 1) % SAVE_EVERY_N_ITERATIONS == 0:
                auto_save_scene(iteration + 1)
        
        # Final update
        set_mesh_vertices(skin_mesh, vertices)
        cmds.refresh()
        
        elapsed = time.time() - start_time
        frozen_count = len(VERTEX_FROZEN)
        
        if cancelled:
            print(f"\nStopped at iteration {iteration+1}. Time: {elapsed:.1f}s")
        elif converged:
            print(f"\nConverged at iteration {iteration+1}. Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
        else:
            print(f"\n{'='*70}")
            print(f"COMPLETE! Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
            print(f"{'='*70}")
        
        print(f"  Settled vertices: {frozen_count:,}/{num_verts:,} ({100*frozen_count/num_verts:.1f}%)")
        unsettled = num_verts - frozen_count
        if unsettled > 0 and not converged:
            print(f"  Still moving: {unsettled:,} vertices")
            print(f"  Tip: Run run_shrinkwrap() again to continue from current state")
        
    finally:
        cmds.progressBar(gMainProgressBar, edit=True, endProgress=True)
        # Re-enable undo
        if undo_was_on:
            cmds.undoInfo(state=True)
            print("  [Memory] Undo re-enabled")
        gc.collect()

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def reset_skin():
    """Reset skin to original positions."""
    if REG_DATA['original_vertices']:
        set_mesh_vertices(REG_DATA['skin_mesh'], REG_DATA['original_vertices'])
        cmds.refresh()
        print("Skin reset to original positions.")
    else:
        print("ERROR: No original vertices stored. Run setup first.")


def visualize_sdf_distances(mesh_name=None, use_cache=True):
    """
    Color the mesh by SDF distance.
    Red = penetrating (SDF < 0), Green = at surface (SDF ~ 0), Blue = outside (SDF > 0)
    
    Args:
        mesh_name: Mesh to colorize (default: skin mesh)
        use_cache: If True, use cached SDF values (fast). If False, recompute (slow but current).
    """
    if not REG_DATA['initialized']:
        print("ERROR: Run setup first!")
        return
    
    mesh = mesh_name or REG_DATA['skin_mesh']
    vertices = get_mesh_vertices(mesh)
    mesh_fns = REG_DATA['mesh_fns']
    offset = REG_DATA['sdf_params'].skin_offset
    sdf_cache = REG_DATA.get('sdf_cache', {})
    num_verts = len(vertices)
    
    print(f"Computing SDF distances for colorization ({num_verts:,} vertices)...")
    
    # --- Step 1: Create or select a color set ---
    color_set_name = "sdf_distance_colors"
    existing_sets = cmds.polyColorSet(mesh, query=True, allColorSets=True) or []
    if color_set_name not in existing_sets:
        cmds.polyColorSet(mesh, create=True, colorSet=color_set_name, representation='RGB')
    cmds.polyColorSet(mesh, currentColorSet=True, colorSet=color_set_name)
    
    # --- Step 2: Compute colors ---
    r_list, g_list, b_list = [], [], []
    for i, v in enumerate(vertices):
        if use_cache and i in sdf_cache:
            sdf_val = sdf_cache[i]['dist']
        else:
            sdf_val, _, _, _ = compute_sdf_for_point(v, mesh_fns, offset)
            if (i + 1) % 500 == 0:
                cmds.refresh()
        
        # Map SDF to color: Red (inside) -> Green (surface) -> Blue (outside)
        if sdf_val < 0:
            t = min(1.0, abs(sdf_val) / 5.0)
            r_list.append(1.0); g_list.append(1.0 - t); b_list.append(1.0 - t)
        elif sdf_val < 1.0:
            t = sdf_val
            r_list.append(1.0 - t); g_list.append(1.0); b_list.append(1.0 - t)
        else:
            t = min(1.0, sdf_val / 10.0)
            r_list.append(1.0 - t); g_list.append(1.0 - t); b_list.append(1.0)
    
    # --- Step 3: Batch-apply all vertex colors via API (much faster) ---
    print(f"  Applying vertex colors...")
    mesh_fn = get_mesh_fn(mesh)
    color_array = om.MColorArray()
    vertex_ids = om.MIntArray()
    for i in range(num_verts):
        color_array.append(om.MColor([r_list[i], g_list[i], b_list[i]]))
        vertex_ids.append(i)
    mesh_fn.setVertexColors(color_array, vertex_ids)
    
    # --- Step 4: Force vertex color display ---
    cmds.setAttr(f"{mesh}.displayColors", 1)
    
    # Force the viewport to show vertex colors (override material)
    try:
        # Get the active viewport panel
        panel = cmds.getPanel(withFocus=True)
        if panel and cmds.getPanel(typeOf=panel) == 'modelPanel':
            # Set display mode to show vertex colors
            cmds.modelEditor(panel, edit=True, displayTextures=False)
    except:
        pass
    
    cmds.refresh()
    
    print(f"  Colors applied! ({num_verts:,} vertices)")
    print(f"  Color key: RED = penetrating | GREEN = at SDF surface | BLUE = outside")
    print(f"")
    print(f"  If you don't see colors, try:")
    print(f"    1. Select the mesh")
    print(f"    2. In viewport menu: Lighting > Use All Lights (or Use Default Lighting)")
    print(f"    3. Make sure: Display > Polygons > Vertex Colors is checked")
    print(f"  To remove colors later:")
    print(f"    cmds.setAttr('{mesh}.displayColors', 0)")


def show_landmark_info():
    """Print current landmark information."""
    if not REG_DATA['landmarks']['enabled']:
        print("Landmarks not set up.")
        return
    
    corr = REG_DATA['landmarks']['correspondences']
    inf = REG_DATA['landmarks']['influence_map']
    
    print(f"\n{'='*50}")
    print(f"LANDMARK INFO")
    print(f"{'='*50}")
    
    # Group by type
    groups = {}
    for vtx, data in corr.items():
        name = data['name']
        base = name.rsplit('_', 1)[0] if '_shape' in name else name
        if base not in groups:
            groups[base] = []
        groups[base].append((vtx, data))
    
    for group_name, items in sorted(groups.items()):
        if len(items) <= 5:
            for vtx, data in items:
                print(f"  {data['name']}: vtx[{vtx}] strength={data['strength']:.2f}")
        else:
            print(f"  {group_name}: {len(items)} vertices (strength={items[0][1]['strength']:.2f})")
    
    print(f"\n  Total constraint vertices: {len(corr)}")
    print(f"  Influenced vertices: {len(inf)}")
    print(f"{'='*50}\n")


def set_sdf_offset(offset):
    """Change the SDF skin/fat thickness offset and recompute."""
    REG_DATA['sdf_params'].skin_offset = offset
    print(f"SDF offset set to {offset}. Run setup_sdf_registration() to recompute.")


# =============================================================================
# MANUAL LANDMARK TOOLS (hand-assign skin ↔ target pairs)
# =============================================================================

def add_manual_landmark(skin_vtx_idx, target_pos, strength=1.0, name="manual"):
    """
    Add a single manual landmark constraint: skin vertex → target position.
    
    Args:
        skin_vtx_idx: Vertex index on the skin mesh
        target_pos: Target position [x, y, z] (on muscle mesh surface)
        strength: Constraint strength (default 1.0)
        name: Label for this landmark
        
    Usage:
        add_manual_landmark(1234, [5.2, 10.3, -2.1], strength=1.0, name="mouth_corner_L")
    """
    landmarks = REG_DATA['landmarks']
    landmarks['enabled'] = True
    landmarks['correspondences'][skin_vtx_idx] = {
        'target': list(target_pos),
        'name': name,
        'strength': strength,
    }
    print(f"  Added landmark: vtx[{skin_vtx_idx}] → {target_pos} "
          f"(strength={strength}, name='{name}')")


def pair_from_selection(strength=1.0, name="manual_pair"):
    """
    Pair landmarks from current Maya selection.
    
    WORKFLOW:
      1. Select skin vertices that form a contour (e.g., mouth outline)
      2. Then Ctrl+select the SAME NUMBER of target vertices on the muscle mesh
      3. Run: pair_from_selection()
    
    The first half of selection = skin vertices, second half = target vertices.
    They are paired BY ORDER (first skin ↔ first target, etc.)
    
    Alternatively, select exactly 2 vertices: first = skin, second = target mesh.
    """
    sel = cmds.ls(selection=True, flatten=True)
    if not sel:
        print("ERROR: Nothing selected!")
        print("  Select skin vertices first, then target vertices (same count).")
        return
    
    # Parse vertex indices and mesh names
    vtx_data = []
    for s in sel:
        # Parse "meshName.vtx[123]"
        if '.vtx[' in s:
            mesh_name = s.split('.')[0]
            idx = int(s.split('[')[1].rstrip(']'))
            pos = cmds.pointPosition(s, world=True)
            vtx_data.append((mesh_name, idx, pos))
    
    if len(vtx_data) < 2:
        print(f"ERROR: Need at least 2 vertices selected, got {len(vtx_data)}")
        return
    
    if len(vtx_data) % 2 != 0:
        print(f"ERROR: Need EVEN number of vertices (half skin, half target). Got {len(vtx_data)}")
        return
    
    skin_mesh = REG_DATA.get('skin_mesh', '')
    
    # Split into skin and target halves
    half = len(vtx_data) // 2
    
    # Try to auto-detect: which are skin, which are target
    # Check if first half is on skin mesh
    skin_verts = []
    target_verts = []
    
    first_mesh = vtx_data[0][0]
    second_mesh = vtx_data[half][0]
    
    if first_mesh == skin_mesh or (first_mesh != second_mesh):
        skin_verts = vtx_data[:half]
        target_verts = vtx_data[half:]
    else:
        # All on same mesh - assume first half = skin, second half = target
        skin_verts = vtx_data[:half]
        target_verts = vtx_data[half:]
    
    landmarks = REG_DATA['landmarks']
    landmarks['enabled'] = True
    
    print(f"\nPairing {half} landmark pairs:")
    for i in range(half):
        s_mesh, s_idx, s_pos = skin_verts[i]
        t_mesh, t_idx, t_pos = target_verts[i]
        
        pair_name = f"{name}_{i}"
        landmarks['correspondences'][s_idx] = {
            'target': list(t_pos),
            'name': pair_name,
            'strength': strength,
        }
        print(f"  {pair_name}: skin vtx[{s_idx}] → target vtx[{t_idx}] on {t_mesh}")
    
    print(f"\nDone! {half} landmark pairs added with strength={strength}")
    print(f"Total landmarks: {len(landmarks['correspondences'])}")


def clear_manual_landmarks():
    """Remove all landmark constraints."""
    REG_DATA['landmarks']['correspondences'] = {}
    REG_DATA['landmarks']['influence_map'] = {}
    REG_DATA['landmarks']['enabled'] = False
    print("All landmarks cleared.")


def load_landmark_preset(preset_name, strength=1.0):
    """
    Load a landmark pair preset from JSON and apply constraints.
    
    The JSON should have format:
      {"name": "...", "pairs": [{"skin_vtx": N, "target_mesh": "...", "target_vtx": M}, ...]}
    
    Target positions are resolved LIVE from Maya (so they match current mesh positions).
    """
    filepath = os.path.join(VERTEX_PRESETS_FOLDER, f"{preset_name}.json")
    if not os.path.exists(filepath):
        print(f"  ERROR: Preset '{preset_name}' not found at {filepath}")
        return 0
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    pairs = data.get('pairs', [])
    if not pairs:
        print(f"  ERROR: No pairs in preset '{preset_name}'")
        return 0
    
    landmarks = REG_DATA['landmarks']
    landmarks['enabled'] = True
    count = 0
    
    for i, pair in enumerate(pairs):
        skin_vtx = pair['skin_vtx']
        target_mesh = pair['target_mesh']
        target_vtx = pair['target_vtx']
        
        # Resolve target position from Maya (live lookup)
        try:
            target_pos = cmds.pointPosition(f"{target_mesh}.vtx[{target_vtx}]", world=True)
        except:
            print(f"    WARNING: Could not find {target_mesh}.vtx[{target_vtx}], skipping")
            continue
        
        pair_name = f"{data.get('name', preset_name)}_{i}"
        landmarks['correspondences'][skin_vtx] = {
            'target': list(target_pos),
            'name': pair_name,
            'strength': strength,
        }
        count += 1
    
    print(f"  Loaded '{preset_name}': {count} landmark pairs (strength={strength})")
    return count


def apply_all_contour_landmarks(strength=1.0, weight=0.2):
    """
    ONE-CALL function: Load and apply ALL contour landmarks + set area weights.
    
    Loads: mouth, nose, left eye, right eye contour presets.
    Also sets attraction weight for lip/nose/eye areas.
    
    Usage (after setup_sdf_registration):
        apply_all_contour_landmarks()             # defaults: strength=1.0, weight=0.2
        apply_all_contour_landmarks(strength=2.0)  # stronger constraints
    """
    print(f"\n{'='*50}")
    print(f"APPLYING ALL CONTOUR LANDMARKS")
    print(f"{'='*50}")
    
    total = 0
    total += load_landmark_preset("landmark_mouth_contour", strength)
    total += load_landmark_preset("landmark_nose_contour", strength)
    total += load_landmark_preset("landmark_left_eye_contour", strength)
    total += load_landmark_preset("landmark_right_eye_contour", strength)
    
    print(f"\n  Total landmarks: {total}")
    
    # Also set area weights for reduced deformation
    print(f"\n  Setting area weights to {weight}:")
    set_preset_weight("upper_lip_area", weight)
    set_preset_weight("lower_lip_area", weight)
    set_preset_weight("nose_area", weight)
    set_preset_weight("eye_left_area", weight)
    set_preset_weight("eye_right_area", weight)
    
    # Create smooth gradient around weighted areas (6 rings)
    print(f"\n  Feathering weight boundaries (6 rings):")
    feather_weight_map(rings=6)
    
    print(f"\n{'='*50}")
    print(f"Done! {total} landmarks + area weights + gradient applied.")
    print(f"  Run run_shrinkwrap() to simulate.")
    print(f"{'='*50}\n")


# =============================================================================
# PER-VERTEX WEIGHT MAP (control attraction strength per vertex)
# =============================================================================

def set_vertex_weight(vertex_indices, weight):
    """
    Set attraction weight for specific vertices.
    
    Args:
        vertex_indices: list of vertex indices, or a single int
        weight: 0.0 (no attraction) to 1.0 (full attraction)
                Use 0.2 for light deformation.
                
    Usage:
        set_vertex_weight([100, 101, 102], 0.2)   # these vertices deform less
        set_vertex_weight(range(500, 600), 0.0)    # these don't deform at all
    """
    if isinstance(vertex_indices, int):
        vertex_indices = [vertex_indices]
    
    wmap = REG_DATA['weight_map']
    count = 0
    for idx in vertex_indices:
        wmap[idx] = weight
        count += 1
    
    print(f"  Set weight={weight} for {count:,} vertices")


def set_selection_weight(weight):
    """
    Set attraction weight for currently selected vertices in Maya.
    
    Usage:
        1. Select vertices in Maya viewport
        2. Run: set_selection_weight(0.2)   # these deform less
    """
    sel = cmds.ls(selection=True, flatten=True)
    if not sel:
        print("ERROR: Nothing selected!")
        return
    
    indices = []
    for s in sel:
        if '.vtx[' in s:
            idx = int(s.split('[')[1].rstrip(']'))
            indices.append(idx)
    
    if not indices:
        print("ERROR: No vertices in selection!")
        return
    
    set_vertex_weight(indices, weight)
    print(f"  Applied to {len(indices)} selected vertices")


def set_preset_weight(preset_name, weight):
    """
    Set attraction weight for vertices from a saved preset.
    
    Usage:
        set_preset_weight("upper_lip_area", 0.2)   # lip area deforms less
        set_preset_weight("nose_area", 0.2)         # nose area deforms less
        set_preset_weight("eye_left_area", 0.2)     # eye area deforms less
    """
    data = load_vertex_preset(preset_name)
    if data is None:
        print(f"ERROR: Could not load preset '{preset_name}'")
        return
    
    # Extract the actual vertex index list from the preset dict
    indices = data.get('indices', data.get('vertex_indices', []))
    if not indices:
        print(f"ERROR: Preset '{preset_name}' has no vertex indices!")
        return
    
    set_vertex_weight(indices, weight)
    print(f"  (from preset '{preset_name}')")


def clear_weight_map():
    """Reset all vertex weights to default (1.0 = full attraction)."""
    REG_DATA['weight_map'] = {}
    print("Weight map cleared — all vertices at full attraction.")


def show_weight_info():
    """Print weight map summary."""
    wmap = REG_DATA.get('weight_map', {})
    if not wmap:
        print("No custom weights set. All vertices at default weight 1.0.")
        return
    
    from collections import Counter
    weight_counts = Counter()
    for idx, w in wmap.items():
        weight_counts[w] += 1
    
    total = REG_DATA.get('vertex_count', 0)
    custom = len(wmap)
    default = total - custom
    
    print(f"\nWeight Map Summary ({custom:,} custom / {total:,} total):")
    print(f"  Default (1.0): {default:,} vertices")
    for w, count in sorted(weight_counts.items()):
        print(f"  Weight {w:.2f}: {count:,} vertices")


def feather_weight_map(rings=6):
    """
    Create smooth gradient around weighted areas using mesh topology.
    
    Expands outward from the boundary of weighted vertices for N rings,
    blending weight linearly toward 1.0. Prevents sharp creases.
    
    Example with weight=0.2 and 6 rings:
        Core area:  0.20
        Ring 1:     0.31
        Ring 2:     0.43
        Ring 3:     0.54
        Ring 4:     0.66
        Ring 5:     0.77
        Ring 6:     0.89
        Outside:    1.00 (default)
    """
    wmap = REG_DATA.get('weight_map', {})
    neighbors = REG_DATA.get('neighbors')
    
    if not wmap:
        print("No weights to feather. Set weights first.")
        return
    if not neighbors:
        print("ERROR: No neighbor data. Run setup_sdf_registration() first.")
        return
    
    num_verts = REG_DATA.get('vertex_count', 0)
    
    # Find the core weighted vertices and their base weight
    core_vertices = set(wmap.keys())
    
    # Find boundary: core vertices that have at least one non-core neighbor
    boundary = set()
    for v_idx in core_vertices:
        if v_idx < len(neighbors):
            for n_idx in neighbors[v_idx]:
                if n_idx not in core_vertices:
                    boundary.add(n_idx)
    
    # Expand rings outward
    current_ring = boundary
    total_feathered = 0
    
    for ring_num in range(1, rings + 1):
        # Weight for this ring: linearly interpolate toward 1.0
        # Ring 1 is closest to core (lowest weight), ring N is furthest (highest)
        t = ring_num / (rings + 1)
        
        next_ring = set()
        for v_idx in current_ring:
            if v_idx in core_vertices:
                continue  # don't overwrite core weights
            
            # Find the minimum weight among this vertex's weighted neighbors
            min_neighbor_weight = 1.0
            for n_idx in neighbors[v_idx] if v_idx < len(neighbors) else []:
                if n_idx in wmap:
                    min_neighbor_weight = min(min_neighbor_weight, wmap[n_idx])
            
            # Blend from neighbor's weight toward 1.0
            ring_weight = min_neighbor_weight + (1.0 - min_neighbor_weight) * t
            wmap[v_idx] = ring_weight
            total_feathered += 1
            
            # Find next ring candidates
            if v_idx < len(neighbors):
                for n_idx in neighbors[v_idx]:
                    if n_idx not in core_vertices and n_idx not in wmap:
                        next_ring.add(n_idx)
        
        print(f"  Ring {ring_num}: {len(current_ring):,} vertices, weight ~{ring_weight:.2f}")
        current_ring = next_ring
        
        if not current_ring:
            break
    
    REG_DATA['weight_map'] = wmap
    print(f"  Feathered {total_feathered:,} vertices across {rings} rings")


# =============================================================================
# FILE-BASED SDF VISUALIZATION (saves PNG plots)
# =============================================================================

PLOT_FOLDER = r"C:\keyLabLocal_D\skin-muscle-registration-model\plot"

def plot_sdf_cross_section(plane='YZ', slice_pos=0.0, grid_size=200, 
                            extent=None, save_name=None):
    """
    Plot a 2D cross-section of the SDF field and save to PNG file.
    
    Samples the SDF on a dense grid at a fixed plane position,
    creating a heatmap showing where SDF=0 (the target surface).
    
    Args:
        plane: 'YZ' (side view), 'XZ' (front view), or 'XY' (top view)
        slice_pos: Position along the fixed axis (e.g., X=0 for YZ plane)
        grid_size: Number of samples along each axis (higher = more detail)
        extent: [min1, max1, min2, max2] bounds. None = auto from skin mesh.
        save_name: Filename (in plot folder). None = auto-generate.
    
    Usage:
        >>> plot_sdf_cross_section('YZ', slice_pos=0)    # midsagittal view
        >>> plot_sdf_cross_section('XZ', slice_pos=0)    # frontal view
    """
    if not MATPLOTLIB_AVAILABLE:
        print("ERROR: matplotlib not available. Install it: pip install matplotlib")
        return
    
    if not REG_DATA['initialized']:
        print("ERROR: Run setup first!")
        return
    
    mesh_fns = REG_DATA['mesh_fns']
    offset = REG_DATA['sdf_params'].skin_offset
    vertices = get_mesh_vertices(REG_DATA['skin_mesh'])
    
    os.makedirs(PLOT_FOLDER, exist_ok=True)
    
    # Determine axis mapping
    if plane == 'YZ':
        ax0_label, ax1_label, fixed_label = 'Y', 'Z', 'X'
        ax0_idx, ax1_idx, fixed_idx = 1, 2, 0
    elif plane == 'XZ':
        ax0_label, ax1_label, fixed_label = 'X', 'Z', 'Y'
        ax0_idx, ax1_idx, fixed_idx = 0, 2, 1
    else:  # XY
        ax0_label, ax1_label, fixed_label = 'X', 'Y', 'Z'
        ax0_idx, ax1_idx, fixed_idx = 0, 1, 2
    
    # Auto extent from skin vertices
    if extent is None:
        vals0 = [v[ax0_idx] for v in vertices]
        vals1 = [v[ax1_idx] for v in vertices]
        margin = 5.0
        extent = [min(vals0) - margin, max(vals0) + margin,
                  min(vals1) - margin, max(vals1) + margin]
    
    print(f"\nPlotting SDF cross-section: {plane} plane at {fixed_label}={slice_pos}")
    print(f"  Grid: {grid_size}x{grid_size} = {grid_size**2:,} samples")
    print(f"  Extent: {ax0_label}=[{extent[0]:.1f}, {extent[1]:.1f}], "
          f"{ax1_label}=[{extent[2]:.1f}, {extent[3]:.1f}]")
    
    import numpy as np
    
    # Create grid
    ax0_vals = np.linspace(extent[0], extent[1], grid_size)
    ax1_vals = np.linspace(extent[2], extent[3], grid_size)
    sdf_grid = np.zeros((grid_size, grid_size))
    
    t0 = time.time()
    total = grid_size * grid_size
    for j, a1 in enumerate(ax1_vals):
        for i, a0 in enumerate(ax0_vals):
            # Build 3D point
            point = [0, 0, 0]
            point[ax0_idx] = float(a0)
            point[ax1_idx] = float(a1)
            point[fixed_idx] = slice_pos
            
            sdf_val, _, _, _ = compute_sdf_for_point(point, mesh_fns, offset)
            sdf_grid[j, i] = sdf_val
        
        # Progress
        if (j + 1) % max(1, grid_size // 10) == 0:
            pct = 100 * (j + 1) / grid_size
            elapsed = time.time() - t0
            print(f"    {pct:.0f}% ({elapsed:.1f}s)")
        
        # Let Maya breathe
        if (j + 1) % 5 == 0:
            cmds.refresh()
    
    elapsed = time.time() - t0
    print(f"  Grid computed in {elapsed:.1f}s")
    
    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # Heatmap
    vmin, vmax = -5, 10
    im = ax.imshow(sdf_grid, origin='lower', cmap='RdYlBu',
                   extent=[extent[0], extent[1], extent[2], extent[3]],
                   vmin=vmin, vmax=vmax, aspect='equal')
    
    # SDF=0 contour (the target surface)
    try:
        ax.contour(ax0_vals, ax1_vals, sdf_grid, levels=[0], colors='black', linewidths=2)
    except:
        pass
    
    # Overlay skin vertices near this slice
    slice_tol = 2.0
    near_verts = [(v[ax0_idx], v[ax1_idx]) for v in vertices
                   if abs(v[fixed_idx] - slice_pos) < slice_tol]
    if near_verts:
        vx = [p[0] for p in near_verts]
        vy = [p[1] for p in near_verts]
        ax.scatter(vx, vy, s=1, c='white', alpha=0.5, label=f'Skin vertices')
    
    plt.colorbar(im, ax=ax, label='SDF Distance', shrink=0.8)
    ax.set_xlabel(ax0_label)
    ax.set_ylabel(ax1_label)
    ax.set_title(f'SDF Cross-Section ({plane} plane, {fixed_label}={slice_pos})\n'
                 f'Black contour = target surface (SDF=0), offset={offset}')
    ax.legend(loc='upper right', fontsize=8)
    
    # Save
    if save_name is None:
        save_name = f"sdf_{plane}_{fixed_label}{slice_pos:.0f}.png"
    save_path = os.path.join(PLOT_FOLDER, save_name)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"  Saved: {save_path}")
    print(f"  Color key: RED = close to surface | BLUE = far from surface")
    print(f"  Black contour = SDF=0 (target iso-surface)")


def plot_sdf_per_vertex(save_name="sdf_distance_histogram.png"):
    """
    Plot a histogram of per-vertex SDF distances and save to PNG.
    Uses cached SDF values from setup.
    """
    if not MATPLOTLIB_AVAILABLE:
        print("ERROR: matplotlib not available.")
        return
    
    if not REG_DATA['initialized'] or not REG_DATA.get('sdf_cache'):
        print("ERROR: Run setup first!")
        return
    
    os.makedirs(PLOT_FOLDER, exist_ok=True)
    
    import numpy as np
    
    sdf_cache = REG_DATA['sdf_cache']
    dists = [sdf_cache[i]['dist'] for i in sorted(sdf_cache.keys())]
    dists = np.array(dists)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    ax = axes[0]
    ax.hist(dists, bins=80, color='steelblue', edgecolor='white', alpha=0.8)
    ax.axvline(x=0, color='red', linewidth=2, linestyle='--', label='SDF=0 (target surface)')
    ax.set_xlabel('SDF Distance')
    ax.set_ylabel('Vertex Count')
    ax.set_title(f'SDF Distance Distribution ({len(dists):,} vertices)')
    ax.legend()
    
    # Sorted distances
    ax = axes[1]
    sorted_d = np.sort(dists)
    ax.plot(sorted_d, np.arange(len(sorted_d)) / len(sorted_d), color='steelblue')
    ax.axvline(x=0, color='red', linewidth=2, linestyle='--', label='SDF=0')
    ax.set_xlabel('SDF Distance')
    ax.set_ylabel('Cumulative Fraction')
    ax.set_title('Cumulative Distribution')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Stats
    stats_text = (f"Min: {dists.min():.2f}  Max: {dists.max():.2f}\n"
                  f"Mean: {dists.mean():.2f}  Std: {dists.std():.2f}\n"
                  f"Inside (SDF<0): {(dists<0).sum():,}\n"
                  f"At surface (|SDF|<0.5): {(np.abs(dists)<0.5).sum():,}\n"
                  f"Outside (SDF>0): {(dists>0).sum():,}")
    fig.text(0.5, -0.02, stats_text, ha='center', fontsize=9,
             fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightyellow'))
    
    save_path = os.path.join(PLOT_FOLDER, save_name)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"  Saved: {save_path}")
    print(f"  {stats_text}")


print("\n" + "=" * 70)
print("TARGET-BASED SHRINK-WRAP REGISTRATION LOADED")
print("=" * 70)
print(f"\nRecommended workflow:")
print(f"  1. Create target mesh: duplicate '{SKIN_MESH}', scale down, name '{TARGET_MESH}'")
print(f"  2. setup_target_registration()     - Setup with target mesh")
print(f"  3. apply_all_contour_landmarks()   - Load landmark constraints")
print(f"  4. run_shrinkwrap(400)             - Run simulation")
print(f"  5. reset_skin()                    - Reset if needed")
print(f"\nManual Landmarks (pair skin ↔ muscle vertices):")
print(f"  pair_from_selection()              - Select skin verts + target verts → pair them")
print(f"  apply_all_contour_landmarks()      - Load all saved contour presets")
print(f"\nVertex Weights (control per-area deformation):")
print(f"  set_selection_weight(0.2)          - Selected verts deform less")
print(f"  set_preset_weight('nose_area',0.2) - Preset area deforms less")
print(f"  show_weight_info()                 - Show weight summary")
print("=" * 70 + "\n")


# =============================================================================
# POST-REGISTRATION CLEANUP ORCHESTRATION (thin wrappers over src/ helpers)
# =============================================================================
# These entry points are the NEW workflow for localized cleanup AFTER a normal
# registration run. They keep d98 as the orchestrator: the real work lives in
# the helper modules (smoothing_utils, region_selection, metrics_utils,
# maya_io). All of them require the cleanup toolkit to be importable
# (HELPERS_AVAILABLE); if it is not, they print a hint and do nothing.
#
# TYPICAL CLEANUP WORKFLOW FROM MAYA
# ---------------------------------
#   # 0. Send this file to Maya (VS Code Maya extension) / or:
#   #    exec(open(r"...\src\d98-target_shrinkwrap_registration.py").read())
#   #
#   # 1. (Optional) run the normal registration first:
#   #      setup_target_registration(); apply_all_contour_landmarks(); run_shrinkwrap(400)
#   #
#   # 2. In the viewport, right-click the skin mesh -> Vertex, and select a bad patch.
#   # 3. Smooth ONLY that selection (volume-preserving Taubin by default):
#          cleanup_selected_region(strength=0.3, iterations=8)
#   #
#   # 4. Or smooth an approximate named region without hand-selecting:
#          cleanup_named_region('lips', strength=0.3, iterations=8, grow=1)
#   #
#   # 5. Save your work as a NEW scene (never overwrites an existing file):
#          save_cleanup_scene(r"C:\path\to\t33_cleanup_from_t32.mb")

def _helpers_ready():
    if not HELPERS_AVAILABLE:
        print("[cleanup] helper toolkit not loaded. Ensure the src/ folder is "
              "importable (set HELPER_SRC_DIR) and re-run this script.")
        return False
    return True


def cleanup_selected_region(strength=0.3, iterations=8, method="taubin",
                            grow=0, mesh_name=None, export_csv=None):
    """Smooth ONLY the vertices currently selected on the skin mesh.

    Select a problematic patch in the Maya viewport (Vertex component mode),
    then call this. Neighboring (unselected) vertices are used as references so
    the smoothed patch blends into its surroundings instead of tearing.

    Parameters
    ----------
    strength : float      blend / lambda strength per pass.
    iterations : int      number of smoothing passes.
    method : str          "taubin" (volume preserving) or "laplacian".
    grow : int            grow the selection by N rings first (softer edges).
    mesh_name : str       defaults to SKIN_MESH.
    export_csv : str      optional path to write a per-vertex displacement CSV.
    """
    if not _helpers_ready():
        return
    mesh_name = mesh_name or SKIN_MESH
    indices = region_selection.from_current_selection(mesh_name)
    if not indices:
        print("[cleanup] no vertices selected on '{0}'. Select a patch first.".format(mesh_name))
        return
    if grow > 0:
        indices = region_selection.grow_region(mesh_name, indices, rings=grow)
    return _run_region_cleanup(mesh_name, indices, strength, iterations,
                               method, export_csv, label="selection")


def cleanup_named_region(region, strength=0.3, iterations=8, method="taubin",
                         grow=1, mesh_name=None, export_csv=None):
    """Smooth an approximate named facial region (heuristic bounding box).

    ``region`` is one of: 'chin', 'lips', 'nose', 'cheeks', 'eyes'. These are
    rough starting selections -- for precise work, select by hand and use
    :func:`cleanup_selected_region` instead.
    """
    if not _helpers_ready():
        return
    mesh_name = mesh_name or SKIN_MESH
    indices = region_selection.named_region(mesh_name, region)
    if not indices:
        return
    if grow > 0:
        indices = region_selection.grow_region(mesh_name, indices, rings=grow)
    return _run_region_cleanup(mesh_name, indices, strength, iterations,
                               method, export_csv, label=region)


def _run_region_cleanup(mesh_name, indices, strength, iterations, method,
                        export_csv, label):
    """Shared body: smooth a region, report displacement, keep the mesh name."""
    before, after = smoothing_utils.smooth_mesh_region(
        mesh_name, indices=indices, strength=strength,
        iterations=iterations, method=method, apply=True)
    if not before:
        return
    stats = metrics_utils.print_displacement_report(
        before, after, indices=indices, label="cleanup[{0}]".format(label))
    if export_csv:
        metrics_utils.export_displacement_csv(before, after, export_csv, indices=indices)
    return stats


def detect_skin_artifacts(mesh_name=None, method="percentile", threshold=2.5,
                          percentile=97.5, rings=1, select=True, min_score=0.0):
    """Auto-detect locally irregular skin vertices (detection ONLY, no edits).

    Thin wrapper over artifact_detection.detect_irregular_region. Flags spikes /
    dents / folds using the umbrella-Laplacian irregularity score, optionally
    grows the set, and selects it in the viewport for inspection. Returns
    ``(indices, score_stats)``. Does NOT move vertices or rename objects.
    """
    if not _helpers_ready():
        return
    mesh_name = mesh_name or SKIN_MESH
    return artifact_detection.detect_irregular_region(
        mesh_name, method=method, threshold=threshold, percentile=percentile,
        rings=rings, select=select, min_score=min_score)


def detect_reference_skin_artifacts(current_mesh=None, target_mesh=None,
                                    method="percentile", threshold=2.5,
                                    percentile=97.5, min_score=0.0,
                                    normalize=True, rings=1, select=True,
                                    epsilon=1e-8):
    """M2 detector: reference-based Laplacian comparison (detection ONLY).

    Compares the current skin's local shape against the target mesh so valid
    high-curvature anatomy (nose, lips, eyelids, brow, jaw) cancels out and only
    genuine local disagreement (registration artifacts) is flagged. Selects the
    result for inspection. Returns ``(indices, score_stats)``. Never edits or
    renames meshes.
    """
    if not _helpers_ready():
        return
    current_mesh = current_mesh or SKIN_MESH
    target_mesh = target_mesh or TARGET_MESH
    return artifact_detection.detect_reference_irregular_region(
        current_mesh, target_mesh, method=method, threshold=threshold,
        percentile=percentile, min_score=min_score, normalize=normalize,
        rings=rings, select=select, epsilon=epsilon)


def backup_skin_mesh(suffix="_precleanup"):
    """Duplicate the skin mesh as a backup before cleanup (name preserved)."""
    if not _helpers_ready():
        return
    return maya_io.duplicate_mesh(SKIN_MESH, suffix=suffix)


def save_cleanup_scene(path, force=False):
    """Save the scene to a NEW .mb path. Refuses to overwrite unless force=True.

    Use this to version cleanup work, e.g.
        save_cleanup_scene(r"C:\\...\\t33_cleanup_from_t32.mb")
    It will NOT overwrite t32 or any existing file unless you pass force=True.
    """
    if not _helpers_ready():
        return
    return maya_io.save_scene_as(path, force=force)


if HELPERS_AVAILABLE:
    print("Post-registration cleanup helpers:")
    print("  detect_skin_artifacts(percentile=97.5)                - M1 AUTO-detect irregular verts (no edits)")
    print("  detect_reference_skin_artifacts(percentile=97.5)      - M2 reference-based detect (fewer false +)")
    print("  cleanup_selected_region(strength=0.3, iterations=8)   - smooth viewport selection")
    print("  cleanup_named_region('lips', strength=0.3)            - smooth heuristic region")
    print("  backup_skin_mesh()                                    - duplicate skin before edits")
    print("  save_cleanup_scene(r'...\\t33_cleanup_from_t32.mb')    - save WITHOUT overwriting")
    print("=" * 70 + "\n")
