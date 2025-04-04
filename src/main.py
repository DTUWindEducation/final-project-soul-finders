from __init__ import load_geometry
from __init__ import load_operational_strategy
from __init__ import load_airfoil_shape
from __init__ import load_airfoil_polar
from __init__ import interpolate_cl_2d
from __init__ import create_cl_table
from __init__ import interpolate_cd_2d
from __init__ import create_cd_table
from __init__ import compute_a_s
from __init__ import sigma_calc
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interp
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd


path_geometry = "./inputs/IEA-15-240-RWT/IEA-15-240-RWT_AeroDyn15_blade.dat"
path_operational_strategy = "./inputs/IEA-15-240-RWT/IEA_15MW_RWT_Onshore.opt"
shape_files = {"cord_files": "./inputs/IEA-15-240-RWT/Airfoils/cord_files"}
polar_files = {"polar_files": "./inputs/IEA-15-240-RWT/Airfoils/polar_files"}

# 1. Load the geometry of the wind turbine blade
r, B, c, Ai = load_geometry(path_geometry)
print("Blade_span", r)
print("twist_angle", B)
print("chord_length", c)
print("Airfoil index", Ai)

# 2. Load the operational strategy of the wind turbine
u, a, w, P, T = load_operational_strategy(path_operational_strategy)
print("Wind speed", u)
print("Blade pitch angle", a)
print("Rotational speed", w)
print("Power", P)
print("Thrust", T)

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

plt.legend()
plt.show()


# 6 interpolate 2D airfoil polar data cl
# Directory containing the polar file

polar_files_dir = "./inputs/IEA-15-240-RWT/Airfoils/polar_files"
path_geometry = "./inputs/IEA-15-240-RWT/IEA-15-240-RWT_AeroDyn15_blade.dat"
alpha_values = np.linspace(0, 120, 100)  # Define the range of alpha values for interpolation

interpolated_cl, alpha_grid, blspn_grid = interpolate_cl_2d(alpha_values, polar_files_dir, path_geometry)

# Plot the results
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(alpha_grid, blspn_grid, interpolated_cl, cmap='viridis')

ax.set_xlabel("Angle of Attack (α) [degrees]")
ax.set_ylabel("Blade Span (r)")
ax.set_zlabel("Lift Coefficient (Cl)")
ax.set_title("Interpolated Cl as a Function of α and Blade Span")
plt.show()

# 7 interpolate 2D airfoil polar data cd
# Directory containing the polar file



interpolated_cd, alpha_grid, blspn_grid = interpolate_cd_2d(alpha_values, polar_files_dir, path_geometry)

# Plot the results
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(alpha_grid, blspn_grid, interpolated_cd, cmap='viridis')

ax.set_xlabel("Angle of Attack (α) [degrees]")
ax.set_ylabel("Blade Span (r)")
ax.set_zlabel("Drag Coefficient (Cd)")
ax.set_title("Interpolated Cd as a Function of α and Blade Span")
plt.show()

# 8. Generate tables for interpolated Cl and Cd values
# Create a DataFrame for the interpolated Cd values

# Generate the table
cd_table = create_cd_table(interpolated_cd, alpha_values, blspn_grid[:, 0])

# 9
# Create a DataFrame for the interpolated Cl values
# Generate the table
cl_table = create_cl_table(interpolated_cl, alpha_values, blspn_grid[:, 0])

output_dir = "./final-project-soul-finders/outputs"
os.makedirs(output_dir, exist_ok=True)  # Create the folder if it doesn't exist
cd_table.to_csv(os.path.join(output_dir, "interpolated_cd_table.csv"))
cl_table.to_csv(os.path.join(output_dir, "interpolated_cl_table.csv"))


sigma = sigma_calc(r, B, c)

a, a_prime = compute_a_s(r, u, w, B, interpolated_cl, interpolated_cd, sigma, tolerance=1e-6, max_iter=100)

# Print the results
print("Axial induction factor (a):", a)
print("Tangential induction factor (a'):", a_prime)