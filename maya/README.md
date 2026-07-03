# maya_scenes

This folder is a placeholder for the Autodesk Maya scene files used by the
registration workflow.

**The actual `.mb` scene files are intentionally not committed to this repository.**
They are large and, in some cases, require supervisor / lab approval before they can
be shared publicly. See the repository root [`.gitignore`](../.gitignore), which
excludes `.mb`/`.ma` files and generated iteration saves.

## Relevant scenes

The registration workflow refers to the following scenes (obtain them from the project
lab storage, not from Git):

- **`s37 repaired the lower lip not match issue.mb`** — the original starting scene used
  as the entry point for registration.
- **`t32 the converged result second time 1600 iterations with data collection and plot.mb`**
  — the working / converged result (~1600 iterations). This file also includes manual
  smoothing done by the supervisor.

## Using scenes with the script

1. Place (or open) the approved `.mb` scene in Maya.
2. Follow the run instructions in the root [`README.md`](../README.md) to load and run
   [`d98-target_shrinkwrap_registration.py`](../d98-target_shrinkwrap_registration.py).

If you do add scene files here for local work, they will be ignored by Git by default.
Only commit a scene file if its inclusion has been explicitly approved.
