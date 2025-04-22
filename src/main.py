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
from __init__ import create_2d_table
from __init__ import compute_a_s
from __init__ import sigma_calc
from __init__ import calculate_rotor_parameters



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



# Load operational strategy data
operational_strategy = turbine_data.load_operational_strategy()
u, a, w, P, T = operational_strategy['u'], operational_strategy['a'], operational_strategy['w'], operational_strategy['P'], operational_strategy['T']


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
#plt.show()

alpha_values = np.linspace(-180, 180, 100)  # Define alpha range
cl_data, _, _ = interpolate_2d(alpha_values, polar_files_dir, path_geometry, data_type="cl")
cd_data, _, _ = interpolate_2d(alpha_values, polar_files_dir, path_geometry, data_type="cd")
alpha_grid = interpolate_2d(alpha_values, polar_files_dir, path_geometry, data_type="alpha")
sigma = sigma_calc(r, c)

print("cl_data shape:", cl_data.shape)
print("r shape:", r.shape)
print("alpha_values shape:", alpha_values.shape)

# Call the function
cl_new, cd_new, alpha_comp, an, an_prime = compute_a_s(r, u, w, a, B, alpha_values, cl_data, cd_data, sigma)

print("\nDebug information:")
print(f"Alpha range at r={r[0]:.2f}m: {alpha_comp[0,:].min():.2f} to {alpha_comp[0,:].max():.2f}")
print(f"Alpha range at r={r[-1]:.2f}m: {alpha_comp[-1,:].min():.2f} to {alpha_comp[-1,:].max():.2f}")
print(f"Cl range: {cl_new.min():.2f} to {cl_new.max():.2f}")
print(f"Cd range: {cd_new.min():.2f} to {cd_new.max():.2f}")

#create a table for alpha values 
alpha_table = pd.DataFrame(alpha_comp, columns=[f"Alpha_{i}" for i in range(len(alpha_comp[0]))])
output_folder = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(output_folder, exist_ok=True)
alpha_table.to_csv(os.path.join(output_folder, "alpha_table.csv"), index=False)

# Generate 2D tables for Cl and Cd
# Replace these lines:
# ...existing code...
# Generate 2D tables for Cl and Cd
cl_table, cl_stats = create_2d_table(cl_new, alpha_comp, r, data_type="Cl")
cd_table, cd_stats = create_2d_table(cd_new, alpha_comp, r, data_type="Cd")



# Create output directory if it doesn't exist
output_folder = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(output_folder, exist_ok=True)

# Save the tables and stats to CSV files
cl_table.to_csv(os.path.join(output_folder, "cl_table.csv"))
cd_table.to_csv(os.path.join(output_folder, "cd_table.csv"))


print(f"Tables saved to {output_folder}")


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

rotor_params = calculate_rotor_parameters(r, an, an_prime, rho=1.225)

# Print results

print("\nRotor Parameters:")
print(f"Thrust (T): {rotor_params['thrust']/1000:.2f} kN")
print(f"Torque (M): {rotor_params['torque']/1000:.2f} kNm")
print(f"Power (P): {rotor_params['power']/1000:.2f} kW")
print(f"Thrust Coefficient (CT): {rotor_params['thrust_coefficient']:.3f}")
print(f"Power Coefficient (CP): {rotor_params['power_coefficient']:.3f}")

# Plot thrust and torque distribution
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot differential thrust
r_mid = (r[1:] + r[:-1]) / 2
ax1.plot(r_mid, rotor_params['dT']/1000, 'b-')
ax1.set_xlabel('Blade span [m]')
ax1.set_ylabel('dT [kN]')
ax1.set_title('Thrust Distribution')
ax1.grid(True)

# Plot differential torque
ax2.plot(r_mid, rotor_params['dM']/1000, 'r-')
ax2.set_xlabel('Blade span [m]')
ax2.set_ylabel('dM [kNm]')
ax2.set_title('Torque Distribution')
ax2.grid(True)

plt.tight_layout()
plt.show()
