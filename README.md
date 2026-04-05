# Best Tilting Plane

Python project for building a reduced whole-body `biorbd` model, simulating aerial twisting with prescribed arm
motions, visualizing the best tilting plane, and optimizing the arm strategy that minimizes the final number of twists.

## Project Goals

This repository is being built incrementally with:

- a floating-base whole-body model with a 6-DoF pelvis/root
- two arm DoFs per side: elevation plane and elevation
- zero-gravity predictive simulation with prescribed quintic arm kinematics
- a GUI with sliders, numeric inputs, simulation controls, 3D animation, and twist-count output
- an optional best-tilting-plane overlay inspired by Charbonneau et al.
- nonlinear optimizations based on IPOPT and CasADi to minimize the final root twist

## Development Rules

- Python 3.11
- docstrings on public modules, classes, and functions
- unit tests added with every feature
- one Git commit per feature or correction

## Environment

Create the dedicated environment with:

```bash
cd /Users/mickaelbegon/Documents/best_tilting_plane
conda env create -f environment.yml
conda activate best-tilting-plane
```

The local environment installs:

- `biorbd`
- `biobuddy` from `../GIT/biobuddy`
- `casadi` for the IPOPT-based optimizations

## Install

```bash
pip install -e .[test,opt]
```

## Test

```bash
pytest -q
```

## Launch The GUI

You can now launch the first interactive GUI with either:

```bash
python best_tilting_plane_gui.py
```

or:

```bash
python -m best_tilting_plane
```

The current GUI provides:

- sliders and numeric entry boxes for the decision variables
- auto-simulation on slider changes with debounce, plus a `Simulate` button
- an `Optimize` button with `Optimize 2D` and `Optimize DMS` modes, plus a checkbox to ignore cached optima
- a `Comparer jerk bras 1` button that opens an external comparison window for the first-arm plane `q`, `qdot`, and `qddot`
- a single integrated window with embedded 3D animation and 2D plotting
- play/pause controls and a time slider to scrub through the animation
- configurable 2D plots against time or somersault for root angles and arm deviations
- a dedicated top-view mode showing the arm motion relative to the pelvis
- a `q(root)=0` visualization mode that zeroes the first 6 DoFs for display and uses an `xOy` camera view
- automatic RK4 versus RK45 selection, with RK4 `dt=0.005 s` retained on the standard case
- a CasADi-based direct multiple-shooting prototype with piecewise-constant jerk controls on the arm planes
- an optional best-tilting-plane overlay and the angular momentum shown at the CoM in the 3D view

## Initial Roadmap

1. Scaffold the repository and quality tooling.
2. Implement and test the quintic Yeadon trajectory generator with symbolic derivatives.
3. Build the reduced whole-body `biorbd` model from De Leva anthropometry.
4. Implement the zero-gravity predictive simulation with `RK45`.
5. Add the GUI and 3D animation.
6. Add the best-tilting-plane overlay.
7. Add IPOPT-based optimization.
