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
from __init__ import interpolated_table
from __init__ import compute_a_s
from __init__ import sigma_calc



path_geometry = "./inputs/IEA-15-240-RWT/IEA-15-240-RWT_AeroDyn15_blade.dat"
path_operational_strategy = "./inputs/IEA-15-240-RWT/IEA_15MW_RWT_Onshore.opt"
shape_files = {"cord_files": "./inputs/IEA-15-240-RWT/Airfoils/cord_files"}
polar_files = {"polar_files": "./inputs/IEA-15-240-RWT/Airfoils/polar_files"}
polar_files_dir = "./inputs/IEA-15-240-RWT/Airfoils/polar_files"

# 1. Load the geometry of the wind turbine blade
# Create an object of the WindTurbineData class

# Initialize the turbine data object
turbine_data = TurbineData(path_geometry, path_operational_strategy)
# Load geometry data
geometry = turbine_data.load_geometry()
r, B, c, Ai = geometry['r'], geometry['B'], geometry['c'], geometry['Ai']
print("twist",B)


# Load operational strategy data
operational_strategy = turbine_data.load_operational_strategy()
u, a, w, P, T = operational_strategy['u'], operational_strategy['a'], operational_strategy['w'], operational_strategy['P'], operational_strategy['T']
print("pitch",a)

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

alpha_values = np.linspace(-180, 180, 100)  # Define alpha range
cl_data, _, _ = interpolate_2d(alpha_values, polar_files_dir, path_geometry, data_type="cl")
cd_data, _, _ = interpolate_2d(alpha_values, polar_files_dir, path_geometry, data_type="cd")
sigma = sigma_calc(r, c)

an, an_prime, cl_new, cd_new, alpha_comp, B, phi_n, A_new= compute_a_s(r, u, w, a, B, alpha_values, cl_data, cd_data, sigma)

print("alpha_comp", alpha_comp)

# Convert alpha_comp to a DataFrame for saving as a table
alpha_comp_df = pd.DataFrame(alpha_comp)

# Save the DataFrame to a CSV file
output_folder = "/Users/maksiu/Desktop/DTU/1 st semester/python/final-project-soul-finders/outputs"
alpha_comp_df.to_csv(f"{output_folder}/alpha_comp1.csv", index=False)


cl_table = interpolated_table(cl_new, alpha_comp[0, :], r, data_type="cl_new")
cd_table = interpolated_table(cd_new, alpha_comp[0, :], r, data_type="cd_new")



# Save to CSV
cl_table.to_csv("cl_table.csv")
cd_table.to_csv("cd_table.csv")
output_folder = "/Users/maksiu/Desktop/DTU/1 st semester/python/final-project-soul-finders/outputs"

# Save the tables to the specified output folder
cl_table.to_csv(f"{output_folder}/cl_table.csv")
cd_table.to_csv(f"{output_folder}/cd_table.csv")

# 6. Plot the Cl and Cd values in 3D plots as subplots in the same figure
fig = plt.figure(figsize=(16, 8))

# Create 2D grids for r and alpha_comp
r_grid, alpha_grid = np.meshgrid(r, alpha_comp[0, :], indexing='ij')

# First subplot for Cl
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(alpha_grid, r_grid, cl_new, cmap='viridis')
ax1.set_xlabel("Alpha (degrees)")
ax1.set_ylabel("Blade Span (r)")
ax1.set_zlabel("Cl")
ax1.set_title("Cl Values in 3D")

# Second subplot for Cd
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(alpha_grid, r_grid, cd_new, cmap='viridis')
ax2.set_xlabel("Alpha (degrees)")
ax2.set_ylabel("Blade Span (r)")
ax2.set_zlabel("Cd")
ax2.set_title("Cd Values in 3D")

plt.tight_layout()
plt.show()


