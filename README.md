# 🧬 Skin-to-Muscle Registration

**Skin-to-Muscle Registration** is an undergraduate research project at **Simon Fraser University** investigating the automation of skin mesh registration for biomechanical facial models.

The project aims to replace the manual "shrink-wrap" process currently required to fit a generic epidermal skin mesh onto customized facial musculature. By combining geometry processing techniques with physically-inspired constraints inside Autodesk Maya, the goal is to produce an automated registration pipeline that significantly reduces artist effort while preserving anatomical accuracy.

---

# 🎯 Research Overview

Preparing biomechanical facial models currently requires artists to manually adjust and register an epidermal skin mesh around a personalized arrangement of muscles, bones and fat compartments.

This manual workflow is:

- Time-consuming
- Difficult to reproduce
- Dependent on expert knowledge
- Hard to scale for multiple characters

This project investigates whether this process can instead be performed automatically using computational geometry and optimization methods.

The final goal is a pipeline that can automatically transform a generic facial skin mesh into a registered skin suitable for biomechanical facial simulation.

---

# 🖼 Pipeline Overview

> **Placeholder:** Insert a high-level pipeline figure here.

```
paper/figures/pipeline.png
```

Example flow:

```
Generic Skin Mesh
        │
        ▼
Target Generation
        │
        ▼
Shrink-Wrap Registration
        │
        ▼
Collision Handling
        │
        ▼
Landmark Constraints
        │
        ▼
Registered Skin Mesh
```

---

# ✨ Current Features

Current implementation includes:

- Target-based shrink-wrap registration
- Collision-aware mesh deformation
- Signed Distance Field (SDF) collision handling
- Anatomical landmark constraints
- Lip, eye and nose contour preservation
- Vertex neighborhood smoothing
- Automatic convergence monitoring
- Simulation statistics collection
- Visualization of registration progress

---

# 📊 Registration Progress

> **Placeholder:** Replace with your convergence plot.

```
simulation_plot.png
```

The algorithm records registration metrics throughout optimization, allowing convergence analysis and comparison between different parameter settings.

---

# 📈 Evaluation Metrics

During every registration run the algorithm records:

- Average vertex displacement
- Convergence rate
- Registration error
- Iteration count
- Runtime statistics

> **Placeholder:** Insert graphs generated from `simulation_metrics.csv`.

Suggested figures:

- Convergence Curve
- Registration Error vs Iteration
- Average Vertex Movement
- Runtime per Iteration

---

# 🧱 Repository Structure

```
skin-to-muscle-registration/
│
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
│
├── d98-target_shrinkwrap_registration.py
│
├── simulation_metrics.csv
├── simulation_plot.png
├── summary.txt
│
├── maya_scenes/
│   └── README.md
│
└── paper/
    └── README.md
```

---

# ⚙️ Current Workflow

The registration pipeline is currently implemented inside **Autodesk Maya** using the Maya Python API.

Typical workflow:

```python
exec(open("d98-target_shrinkwrap_registration.py").read())

setup_target_registration()

apply_all_contour_landmarks()

run_shrinkwrap(400)
```

The pipeline performs:

1. Load the skin mesh
2. Load the target mesh
3. Build Signed Distance Field (SDF)
4. Detect anatomical landmarks
5. Run iterative shrink-wrap optimization
6. Preserve important facial contours
7. Record convergence statistics
8. Export the registered mesh

---

# 🧠 Registration Algorithm

The current implementation combines several techniques:

- Target-based vertex attraction
- Signed Distance Field (SDF) collision detection
- Laplacian smoothing
- Edge-length preservation
- Landmark constraints
- Soft-body inspired relaxation
- Automatic convergence detection

Future work will investigate additional registration methods including:

- Non-Rigid ICP
- Laplacian Surface Editing
- As-Rigid-As-Possible (ARAP) deformation
- Energy-based optimization

---

# 📷 Example Results

### Initial Skin Mesh

> Placeholder

```
paper/figures/initial_mesh.png
```

---

### Target Mesh

> Placeholder

```
paper/figures/target_mesh.png
```

---

### Registered Mesh

> Placeholder

```
paper/figures/final_registration.png
```

---

### Error Heatmap

> Placeholder

```
paper/figures/error_heatmap.png
```

---

# 🛠 Technologies

- Python
- Autodesk Maya API
- Maya Commands (cmds)
- Maya OpenMaya API
- Computational Geometry
- Mesh Processing
- Numerical Optimization
- Matplotlib

---

# 🚀 Future Work

Planned improvements include:

- Modularizing the registration pipeline
- Improved collision handling
- Faster convergence
- Better landmark correspondence
- Automatic parameter tuning
- Quantitative comparison with manual registration
- Evaluation against Non-Rigid ICP approaches
- Publishable benchmark experiments

---

# 📚 Related Research

This repository accompanies undergraduate research on automated skin registration for biomechanical facial models.

Primary reference:

> **Beneath the Skin: Interactive Biomechanical Facial Simulations via Composable Co-Expressions**

Additional references will be added throughout the project.

---

# 📄 License

This project is released under the MIT License.

See the `LICENSE` file for details.

---

# 👨‍💻 Author

**Ilian Khankhalaev**

Undergraduate Researcher  
Simon Fraser University

---

# 🙏 Acknowledgements

This project is conducted as part of an undergraduate research position at Simon Fraser University.

Special thanks to my research supervisor for guidance throughout the project.