# skin-to-muscle-registration

Automated registration of skin meshes to customized musculature for biomechanical facial models.

> **Status: early research prototype.** This is an undergraduate research project and a
> work-in-progress. The code and workflow are experimental, tuned to a specific Maya scene,
> and are not yet packaged as a general-purpose, reusable tool. Expect rough edges.

## Overview

This project explores **automated skin-to-muscle mesh registration** for biomechanical
facial models built in [Autodesk Maya](https://www.autodesk.com/products/maya/).

The goal is to take an expanded skin mesh and "shrink-wrap" it onto the underlying
anatomy (skull, jaw, and facial muscles) while:

- moving each skin vertex toward a corresponding target position (1-to-1 correspondence),
- preventing the skin from penetrating internal tissues (collision detection),
- keeping anatomical landmark contours in place via constraints.

The main runnable script is [`d98-target_shrinkwrap_registration.py`](d98-target_shrinkwrap_registration.py),
which is executed **inside Maya**.

## Repository layout

```
skin-to-muscle-registration/
├── README.md                                # this file
├── LICENSE                                  # MIT
├── .gitignore
├── requirements.txt                         # optional external Python packages
├── d98-target_shrinkwrap_registration.py    # main Maya registration script
├── summary.txt                              # working notes / experiment log
├── simulation_metrics.csv                   # metrics collected during a run
├── simulation_plot.png                      # convergence plot from a run
├── maya_scenes/                             # (mostly empty; see its README)
│   └── README.md
└── paper/                                   # write-up / report materials
    └── README.md
```

The registration script lives in the repository root for now because it is the main
runnable Maya script. It is intentionally **not** yet split into helper modules or
packages while the project is still an early prototype.

## Requirements

- **Autodesk Maya** (with its bundled Python interpreter, `maya.cmds` and
  `maya.api.OpenMaya`).
- Optional Python packages listed in [`requirements.txt`](requirements.txt), currently
  just `matplotlib` for generating convergence plots. If `matplotlib` is not available,
  the script still runs but plotting is disabled.

Because the script uses Maya's Python API, it must be run from within Maya (or `mayapy`),
not from a standalone Python environment.

## Running the script in Maya

1. Open the relevant Maya scene (the registration is tuned to a specific scene; see
   [`maya_scenes/README.md`](maya_scenes/README.md) and `summary.txt` for details).
   - The original starting scene is `s37 repaired the lower lip not match issue.mb`.
   - The working/converged result is
     `t32 the converged result second time 1600 iterations with data collection and plot.mb`
     (this file also includes manual smoothing done by the supervisor).
2. In Maya's **Script Editor** (Python tab), load and execute the script:

   ```python
   exec(open(r"path/to/d98-target_shrinkwrap_registration.py").read())
   ```

3. Run the main workflow:

   ```python
   setup_target_registration()
   apply_all_contour_landmarks()
   run_shrinkwrap(400)
   ```

   The argument to `run_shrinkwrap()` is the number of iterations. Metrics and plots may
   be written out during/after the run (see `simulation_metrics.csv` and
   `simulation_plot.png` for example outputs).

## A note on large files and data

Large Maya scene files (`.mb`), model assets, textures, and generated saves are **not**
included in this repository by default. They may be large, and some may require
**supervisor / lab approval** before they can be shared publicly.

As a rule of thumb, `.mb` files, generated iteration saves, cache files, and large model
folders should be kept **out of Git** unless their inclusion has been explicitly approved.
See [`.gitignore`](.gitignore) for the patterns that are excluded.

## License

Released under the [MIT License](LICENSE).
