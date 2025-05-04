[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/zjSXGKeR)

# Soul - Finders: Wind Turbine Modeling using BEM

**Team**: Soul - Finders

## Overview

This project implements a steady-state Blade Element Momentum (BEM) model to predict the aerodynamic performance of wind turbines, using the IEA 15 MW offshore reference turbine as a case study.

Our Python package simulates rotor behavior based on geometry, airfoil shape, and aerodynamic polars, allowing the calculation of thrust, torque, power, and efficiency. It supports visualization and data export, helping researchers and engineers optimize turbine blade designs and operational strategies.

## Quick Start Guide

### 1. Clone the Repository

```bash
git clone https://github.com/YourUsername/final-project-soul-finders.git
cd final-project-soul-finders
```

### 2. Install Dependencies

You need Python 3.8 or above and the following Python packages:

* numpy
* pandas
* matplotlib
* scipy
* pytest
* pytest-cov

Install everything by running:

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not available, manually install:

```bash
pip install numpy pandas matplotlib scipy pytest pytest-cov
```

### 3. Run the Model

```bash
cd examples
python main.py
```

### 4. Customize Inputs

Modify the `inputs/IEA-15-240-RWT` folder to simulate different turbines. You can change the airfoil shapes, polar files, and blade geometry.

### 5. Output

The results (plots and `.csv` files) will be saved in the `outputs/` directory, covering:

* Blade aerodynamic parameters for each wind speed
* Rotor thrust, torque, and power
* Power and thrust curves

---

## Project Structure

```
final-project-soul-finders/
├── examples/
│   └── main.py                   # Main entry point for running the model
├── inputs/                       # Contains turbine geometry and airfoil data
├── outputs/                      # Auto-generated results (plots, CSVs)
├── src/
│   ├── compute/                  # Core computational modules
│   │   └── __init__.py           # Includes classes and functions for BEM model
├── tests/                        # Unit tests
├── pyproject.toml                # Metadata and package configuration
├── LICENSE
├── README.md
├── project_info.md
└── .gitignore
```

---

## Architecture and Class Description

### `TurbineData` Class – *Located in* `src/compute/__init__.py`

Handles all data input and preprocessing:

* **Attributes**:

  * `geometry_path`
  * `operational_strategy_path`
* **Methods**:

  * `load_geometry()` – Reads the blade geometry (span, chord, twist, airfoil index).
  * `load_operational_strategy()` – Loads wind speed, pitch angle, and rotor speed.

### Functional Modules

* `load_airfoil_shape()` – Loads airfoil shapes from coordinates.
* `load_airfoil_polar()` – Extracts aerodynamic polar data.
* `interpolate_2d()` – Interpolates Cl/Cd values across span and angle of attack.
* `compute_a_s()` – Calculates axial and tangential induction factors.
* `calculate_rotor_parameters()` – Computes thrust, torque, and power.
* `compute_power_and_thrust_curves()` – Computes and compares predicted vs. real performance curves.
* `plot_3d_airfoil_shape()`, `plot_3d_cl_cd_vs_r_alpha()` – Visualization helpers.

### Example Run: `main.py`

This script orchestrates the modeling pipeline:

1. Loads geometry and polars
2. Computes lift/drag and induction factors
3. Calculates rotor outputs
4. Plots 3D geometry and performance curves
5. Saves data to `.csv` for further analysis

---

## Example Diagram

Below is a conceptual flowchart of our model:

```
    [Geometry + Operational Strategy + Polars]
                     |
         -----------------------------
         |                           |
    [TurbineData]            [Airfoil Data Loader]
         |                           |
         ---------[compute_a_s()]----------
                         |
         [calculate_rotor_parameters()]
                         |
       [Power / Thrust / Torque Output]
                         |
       [Visualization + CSV Export]
```

---

## Testing and Quality Assurance

All functions were tested under `src/tests` using `pytest`. Achievements include:

* **80%+ test coverage**: Verified via `pytest --cov=src tests/`
* **Pylint score > 8.0**: Verified via `pylint src/`

Example test files:

* `test_basic.py`: Unit tests for loading data
* `test_rotor_computation.py`: Validates thrust, torque, power calculations
* `test_operational_strategy.py`: Checks interpolation and strategy extraction

---

## Acknowledgements

This package is developed as part of the 46120 Scientific Programming for Wind Energy course at DTU. We thank the course staff and IEA Task 37 for the open turbine dataset.

