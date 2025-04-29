import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interp
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from turbine import TurbineData
from __init__ import load_airfoil_shape
from __init__ import load_airfoil_polar
from __init__ import interpolate_2d
from __init__ import compute_a_s
from __init__ import sigma_calc
from __init__ import calculate_rotor_parameters
from __init__ import compute_power_and_thrust_curves, plot_power_and_thrust_curves


path_geometry = "./inputs/IEA-15-240-RWT/IEA-15-240-RWT_AeroDyn15_blade.dat"
path_operational_strategy = "./inputs/IEA-15-240-RWT/IEA_15MW_RWT_Onshore.opt"
shape_files = {"cord_files": "./inputs/IEA-15-240-RWT/Airfoils/cord_files"}
polar_files = {"polar_files": "./inputs/IEA-15-240-RWT/Airfoils/polar_files"}
polar_files_dir = "./inputs/IEA-15-240-RWT/Airfoils/polar_files"
wind_speeds = {"wind_speed": "./inputs/wind_TI_0.1"}

# 1. Load the geometry of the wind turbine blade
# Create an object of the WindTurbineData class

# Initialize the turbine data object
turbine_data = TurbineData(path_geometry, path_operational_strategy)
# Load geometry data
geometry = turbine_data.load_geometry()
r, B, c, Ai = geometry['r'], geometry['B'], geometry['c'], geometry['Ai']



# Load operational strategy data
operational_strategy = turbine_data.load_operational_strategy()
v, p, w, P, T = operational_strategy['v'], operational_strategy['p'], operational_strategy['w'], operational_strategy['P'], operational_strategy['T']

# 2. Load the wind speed data



# 3. Load the airfoil shape

for key, folder_path in shape_files.items():
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path) and file_name.endswith(".txt"):
            norm_x, norm_y = load_airfoil_shape(file_path)
            

# 4. Load the airfoil polar
for key, folder_path in polar_files.items():
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path) and file_name.endswith(".dat"):
            alpha, cl, cd = load_airfoil_polar(file_path)
            
# 5. Plot the 3D airfoil shape for each chord in the shape_files
fig = plt.figure(figsize=(10, 8))  # Increase figure size
ax = fig.add_subplot(111, projection='3d')

for key, folder_path in shape_files.items():
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path) and file_name.endswith(".txt"):
            norm_x, norm_y = load_airfoil_shape(file_path)
            
            # Assuming `r` (Blade_span) is the spanwise position for each airfoil
            for span, chord in zip(r, c):
                scaled_x = norm_x * chord
                scaled_y = norm_y * chord
                z = np.full_like(scaled_x, span)  # Set z-coordinate as the spanwise position
                
                ax.plot(scaled_x, scaled_y, z)
                
ax.set_xlabel("X-axis (Chordwise)")
ax.set_ylabel("Y-axis (Thickness)")
ax.set_zlabel("Z-axis (Spanwise)")
ax.set_title("3D Airfoil Shapes Along Blade Span")

# Adjust the z-axis limits to increase the size in the z direction
z_min, z_max = min(r), max(r)
z_padding = (z_max - z_min) * 0.00000002  # Add 20% padding
ax.set_zlim(0,170)


alpha_values = np.linspace(0, 180, 100)  # Define alpha range
cl_data, _, _ = interpolate_2d(alpha_values, polar_files_dir, path_geometry, data_type="cl")
cd_data, _, _ = interpolate_2d(alpha_values, polar_files_dir, path_geometry, data_type="cd")
alpha_grid = interpolate_2d(alpha_values, polar_files_dir, path_geometry, data_type="alpha")
sigma = sigma_calc(r, c)


_, _, _, an, an_prime, _ = compute_a_s(r, B, alpha_values, cl_data, cd_data, sigma, v, p, w, rho=1.225, tolerance=1e-6, max_iter=1000)


rotor_params = calculate_rotor_parameters(r, w, v ,an,an_prime, rho=1.225)

# 6. Compute and Plot Power and Thrust Curves

# Define a wind speed range for the curve (for example 3 m/s to 25 m/s)
wind_speed_range = np.linspace(3, 25, 30)  # 30 points between 3 and 25 m/s

# Compute power and thrust curves

power_curve, thrust_curve = compute_power_and_thrust_curves(
    wind_speed_range,
    path_operational_strategy,
    (r, B, c, Ai),
    polar_files_dir,
    path_geometry,
    rho=1.225
)


# Plot the curves
plot_power_and_thrust_curves(wind_speed_range, power_curve, thrust_curve, path_operational_strategy)




