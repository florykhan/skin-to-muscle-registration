"""
maya_io.py
==========
Scene-safe I/O helpers: object checks, safe duplication, snapshots, and
overwrite-protected saving.

The single most important rule for this project's workflow is: **never overwrite
an existing scene** (especially ``t32``, the current best result). New cleanup
work must be saved as a NEW file, e.g. ``t33_cleanup_from_t32.mb``. The save
helpers here refuse to clobber an existing file unless ``force=True`` is passed
explicitly.

These utilities are deliberately conservative -- they print what they do and bail
out safely rather than silently modifying/overwriting the scene.
"""

import json
import os

try:
    import maya.cmds as cmds
    MAYA_AVAILABLE = True
except ImportError:
    cmds = None
    MAYA_AVAILABLE = False

from mesh_utils import get_mesh_vertices, set_mesh_vertices, mesh_exists


# =============================================================================
# OBJECT CHECKS
# =============================================================================

def object_exists(name):
    return mesh_exists(name)


def require_objects(names):
    """Return (ok, missing). Print a warning listing any missing objects.

    Useful as a guard at the start of a cleanup routine so it fails clearly
    instead of deep inside an operation.
    """
    missing = [n for n in names if not mesh_exists(n)]
    if missing:
        print("[maya_io] MISSING objects: {0}".format(", ".join(missing)))
    return (len(missing) == 0), missing


# =============================================================================
# SAFE DUPLICATION (backup a mesh before editing, keeping the original name)
# =============================================================================

def duplicate_mesh(mesh_name, suffix="_backup"):
    """Duplicate a mesh as a backup and return the new node name.

    The ORIGINAL keeps its name (the registration code depends on it); the copy
    gets ``<mesh_name><suffix>``. Handy to snapshot the skin mesh before a
    destructive smoothing pass so you can visually A/B or restore it.
    """
    if not MAYA_AVAILABLE:
        return None
    if not mesh_exists(mesh_name):
        print("[maya_io] cannot duplicate: '{0}' does not exist".format(mesh_name))
        return None
    new_name = mesh_name + suffix
    dup = cmds.duplicate(mesh_name, name=new_name)
    result = dup[0] if dup else None
    print("[maya_io] duplicated '{0}' -> '{1}'".format(mesh_name, result))
    return result


# =============================================================================
# VERTEX SNAPSHOTS (lightweight, no .mb needed)
# =============================================================================

def snapshot_vertices(mesh_name, path):
    """Save a mesh's vertex positions to a JSON snapshot (not a Maya scene)."""
    verts = get_mesh_vertices(mesh_name)
    if not verts:
        print("[maya_io] nothing to snapshot for '{0}'".format(mesh_name))
        return None
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"mesh": mesh_name, "vertex_count": len(verts),
                   "vertices": verts}, f)
    print("[maya_io] snapshot {0} verts -> {1}".format(len(verts), path))
    return path


def restore_vertices(mesh_name, path):
    """Restore vertex positions from a JSON snapshot onto ``mesh_name``."""
    if not os.path.exists(path):
        print("[maya_io] snapshot not found: {0}".format(path))
        return False
    with open(path, "r") as f:
        data = json.load(f)
    verts = data.get("vertices", [])
    if not verts:
        return False
    set_mesh_vertices(mesh_name, verts)
    print("[maya_io] restored {0} verts onto '{1}'".format(len(verts), mesh_name))
    return True


# =============================================================================
# OVERWRITE-PROTECTED SCENE SAVING
# =============================================================================

def save_scene_as(path, force=False, file_type="mayaBinary"):
    """Save the current Maya scene to ``path`` as a NEW file.

    Refuses to overwrite an existing file unless ``force=True``. This is the
    guardrail that protects ``t32`` and other results: a cleanup session should
    call e.g. ``save_scene_as(".../t33_cleanup_from_t32.mb")`` and will get a
    clear error rather than clobbering an existing scene.

    The current scene is renamed to ``path`` (standard Maya "Save As" behavior)
    only when the save actually happens.
    """
    if not MAYA_AVAILABLE:
        print("[maya_io] not running inside Maya; cannot save scene")
        return None

    if os.path.exists(path) and not force:
        print("[maya_io] REFUSING to overwrite existing file:\n    {0}\n"
              "         Pass force=True only if you are certain.".format(path))
        return None

    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)

    cmds.file(rename=path)
    cmds.file(save=True, type=file_type)
    print("[maya_io] saved scene -> {0}".format(path))
    return path


def next_version_path(directory, base_name, ext=".mb"):
    """Return a non-colliding path like ``<base>_v001.mb`` inside ``directory``.

    Convenience for iterative cleanup saves that never overwrite a prior file.
    """
    os.makedirs(directory, exist_ok=True)
    n = 1
    while True:
        candidate = os.path.join(directory, "{0}_v{1:03d}{2}".format(base_name, n, ext))
        if not os.path.exists(candidate):
            return candidate
        n += 1
