import sys
import os
import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')  # Prevent plot from opening
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.compute import plot_power_and_thrust_curves
from src.compute import plot_3d_cl_cd_vs_r_alpha
from src.compute import plot_3d_airfoil_shape

def test_plot_power_and_thrust_curves():
    wind_speeds = np.linspace(3, 25, 10)
    power = np.random.rand(10) * 15000  # random dummy power values
    thrust = np.random.rand(10) * 2000  # random dummy thrust values

    operational_strategy_path = "inputs/IEA-15-240-RWT/IEA_15MW_RWT_Onshore.opt"

    # Call plotting function
    plot_power_and_thrust_curves(wind_speeds, power, thrust, operational_strategy_path)

    assert True  # If no crash, test passes

def test_plot_3d_cl_cd_vs_r_alpha_runs():
    
    # Input data
    r = np.linspace(5, 100, 10)
    alpha_values = np.linspace(-10, 20, 50)
    cl_data = np.random.rand(len(r), len(alpha_values))
    cd_data = np.random.rand(len(r), len(alpha_values))
    alpha_comp = np.linspace(-10, 20, len(r))

    # Call the function and ensure it completes without error
    try:
        plot_3d_cl_cd_vs_r_alpha(r, alpha_values, cl_data, cd_data, alpha_comp)
    except Exception as e:
        pytest.fail(f"plot_3d_cl_cd_vs_r_alpha failed with error: {e}")

def test_plot_3d_airfoil_shape_runs(tmp_path):
    # Simulated blade span and chord
    r = np.linspace(0, 100, 5)
    c = np.linspace(2, 5, 5)

    # Create temporary airfoil shape files
    shape_dir = tmp_path / "airfoil_shapes"
    shape_dir.mkdir()
    for i in range(3):
        file_path = shape_dir / f"airfoil_{i}.txt"
        with open(file_path, 'w') as f:
            f.write("\n" * 8)  # Skip header lines
            for x, y in zip(np.linspace(0, 1, 10), np.linspace(0, 0.1, 10)):
                f.write(f"{x:.5f} {y:.5f}\n")

    shape_files = {"section1": str(shape_dir)}

    # Call the function
    try:
        plot_3d_airfoil_shape(r, c, shape_files)
    except Exception as e:
        pytest.fail(f"plot_3d_airfoil_shape failed with error: {e}")
