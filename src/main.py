"""
This module performs wind turbine analysis, including loading geometry,
computing aerodynamic parameters, and plotting results.

It uses the TurbineData class to load turbine geometry and operational
strategy data, and various utility functions to process and visualize
the data.
"""

import os
import numpy as np
from turbine import TurbineData
from __init__ import load_airfoil_shape
from __init__ import load_airfoil_polar
from __init__ import interpolate_2d
from __init__ import compute_a_s
from __init__ import sigma_calc
from __init__ import calculate_rotor_parameters
from __init__ import plot_3d_airfoil_shape
from __init__ import save_blade_results
from __init__ import save_rotor_parameters
from __init__ import plot_3d_cl_cd_vs_r_alpha
from __init__ import compute_power_and_thrust_curves, plot_power_and_thrust_curves


# Paths to input files
PATH_GEOMETRY = "./inputs/IEA-15-240-RWT/IEA-15-240-RWT_AeroDyn15_blade.dat"
PATH_OPERATIONAL_STRATEGY = "./inputs/IEA-15-240-RWT/IEA_15MW_RWT_Onshore.opt"
SHAPE_FILES = {"cord_files": "./inputs/IEA-15-240-RWT/Airfoils/cord_files"}
POLAR_FILES = {"polar_files": "./inputs/IEA-15-240-RWT/Airfoils/polar_files"}
POLAR_FILES_DIR = "./inputs/IEA-15-240-RWT/Airfoils/polar_files"

# Initialize the turbine data object
turbine_data = TurbineData(PATH_GEOMETRY, PATH_OPERATIONAL_STRATEGY)

# Load geometry data
geometry = turbine_data.load_geometry()
r = geometry["blade_span_positions"]
B = geometry["twist_angle"]
c = geometry["chord_lengths"]
Ai = geometry["airfoil_indices"]

# Load operational strategy data
operational_strategy = turbine_data.load_operational_strategy()
v = operational_strategy["wind_speed"]
p = operational_strategy["angle_pitch"]
w = operational_strategy["rotational_speed"]
P = operational_strategy["power"]
T = operational_strategy["torque"]

# Load the airfoil shape
for key, folder_path in SHAPE_FILES.items():
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path) and file_name.endswith(".txt"):
            norm_x, norm_y = load_airfoil_shape(file_path)

# Load the airfoil polar
for key, folder_path in POLAR_FILES.items():
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path) and file_name.endswith(".dat"):
            alpha, cl, cd = load_airfoil_polar(file_path)

# Plot the 3D airfoil shape
fig, ax = plot_3d_airfoil_shape(r, c, SHAPE_FILES)

# Interpolate the airfoil polar data
alpha_values = np.linspace(0, 180, 100)  # Define alpha range
cl_data, _, _ = interpolate_2d(alpha_values, POLAR_FILES_DIR, r,
                               data_type="cl")
cd_data, _, _ = interpolate_2d(alpha_values, POLAR_FILES_DIR, r,
                               data_type="cd")
alpha_grid = interpolate_2d(alpha_values, POLAR_FILES_DIR, PATH_GEOMETRY,
                            data_type="alpha")

sigma = sigma_calc(r, c)
# Compute tangential and axial induction factors
_, _, _, an, an_prime, _ = compute_a_s(r, B, alpha_values, cl_data, cd_data,
                                       sigma, v, p, w, tolerance=1e-6,
                                       max_iter=1000)

# Initialize arrays for storing results
wind_speeds = v  # Store original wind speeds
results = []

# Compute for each wind speed
for wind_speed in wind_speeds:
    cl_new, cd_new, alpha_comp, an, an_prime, u_new = compute_a_s(
        r=r, B=B, alpha_values=alpha_values, cl_data=cl_data,
        cd_data=cd_data, sigma=sigma, v=[wind_speed],  # Pass as list
        p=p, w=w, tolerance=1e-6, max_iter=1000
    )

    # Save results
    save_blade_results(r=r, u_new=wind_speed, cl_new=cl_new,
                       cd_new=cd_new, alpha_comp=alpha_comp, an=an,
                       an_prime=an_prime)

    # Store results for later use
    results.append({'wind_speed': wind_speed, 'cl': cl_new, 'cd': cd_new,
                    'alpha': alpha_comp, 'a': an, 'a_prime': an_prime})

# Plot 3D cl and cd vs r and alpha
plot_3d_cl_cd_vs_r_alpha(r, alpha_values, cl_data, cd_data, alpha_comp)

# Calculate rotor parameters
rotor_params = calculate_rotor_parameters(r, w, v, an, an_prime, rho=1.225)

# Initialize an empty list to store rotor parameters for each wind speed
rotor_params_list = []

# Compute rotor parameters for each wind speed
for wind_speed in wind_speeds:
    cl_new, cd_new, alpha_comp, an, an_prime, u_new = compute_a_s(
        r=r, B=B, alpha_values=alpha_values, cl_data=cl_data,
        cd_data=cd_data, sigma=sigma, v=[wind_speed],  # Pass as list
        p=p, w=w, tolerance=1e-6, max_iter=1000
    )
    rotor_params = calculate_rotor_parameters(r, w, [wind_speed], an,
                                              an_prime, rho=1.225)
    rotor_params_list.append(rotor_params)

# Save rotor parameters for all wind speeds
save_rotor_parameters(wind_speeds, rotor_params_list)

# Compute and Plot Power and Thrust Curves
wind_speed_range = np.linspace(3, 25, 30)  # Define wind speed range
power_curve, thrust_curve = compute_power_and_thrust_curves(
    wind_speed_range, PATH_OPERATIONAL_STRATEGY, (r, B, c, Ai),
    POLAR_FILES_DIR, PATH_GEOMETRY, rho=1.225
)
plot_power_and_thrust_curves(wind_speed_range, power_curve, thrust_curve,
                             PATH_OPERATIONAL_STRATEGY)

# Add a final newline