import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Prevent plot from opening
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import plot_power_and_thrust_curves

def test_plot_power_and_thrust_curves():
    wind_speeds = np.linspace(3, 25, 10)
    power = np.random.rand(10) * 15000  # random dummy power values
    thrust = np.random.rand(10) * 2000  # random dummy thrust values

    operational_strategy_path = "inputs/IEA-15-240-RWT/IEA_15MW_RWT_Onshore.opt"

    # Call plotting function
    plot_power_and_thrust_curves(wind_speeds, power, thrust, operational_strategy_path)

    assert True  # If no crash, test passes
