from __init__ import load_geometry
from __init__ import load_operational_strategy
from __init__ import load_airfoil_shape
from __init__ import load_airfoil_polar
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

path_geomtery = "./inputs/IEA-15-240-RWT/IEA-15-240-RWT_AeroDyn15_blade.dat"
path_operational_strategy = "./inputs/IEA-15-240-RWT/IEA_15MW_RWT_Onshore.opt"
shape_files = {"cord_files": "./inputs/IEA-15-240-RWT/Airfoils/cord_files"}
polar_files = {"polar_files": "./inputs/IEA-15-240-RWT/Airfoils/polar_files"}

# 1. Load the geometry of the wind turbine blade
r, B, c, Ai = load_geometry(path_geomtery)
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
            
# 4. Plot the 3D airfoil shape for each chord in the shape_files
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

import os
import numpy as np
import scipy.interpolate as interp
import matplotlib.pyplot as plt

# Directory containing the polar files
data_dir = os.path.join(os.getcwd(), "inputs", "IEA-15-240-RWT", "Airfoils", "polar_files")

def load_polar_data(file_path):
    """
    Load the polar data (α, Cl, Cd) from a file, ignoring comment lines.
    """
    cleaned_data = []
    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            if line and not line.startswith("!"):  # Ignore comments and empty lines
                try:
                    cleaned_data.append([float(x) for x in line.split()])
                except ValueError:
                    continue  # Skip any malformed lines
    
    # Convert to NumPy array
    data = np.array(cleaned_data)
    
    if data.shape[1] < 3:  # Ensure there are at least three columns
        raise ValueError(f"File {file_path} does not contain expected columns (α, Cl, Cd).")
    
    alpha = data[:, 0]  # Angle of attack (α)
    cl = data[:, 1]     # Lift coefficient (Cl)
    cd = data[:, 2]     # Drag coefficient (Cd)
    
    return alpha, cl, cd

def interpolate_cl_cd(alpha_values, cl_values, cd_values, alpha_to_interpolate, method='linear'):
    """
    Create interpolation functions for Cl and Cd as a function of α.
    Supports 'linear' and 'cubic' interpolation.
    """
    cl_interp = interp.interp1d(alpha_values, cl_values, kind=method, fill_value="extrapolate")
    cd_interp = interp.interp1d(alpha_values, cd_values, kind=method, fill_value="extrapolate")
    
    # Interpolate Cl and Cd for the desired α values
    cl_interpolated = cl_interp(alpha_to_interpolate)
    cd_interpolated = cd_interp(alpha_to_interpolate)
    
    return cl_interpolated, cd_interpolated

# Generate file names from 00 to 49
file_names = [f"IEA-15-240-RWT_AeroDyn15_Polar_{i:02d}.dat" for i in range(50)]

# Set up figure for multiple subplots
fig, ax = plt.subplots(2, 1, figsize=(10, 12))  # Two subplots: one for Cl and one for Cd

# Loop through all files
for file_name in file_names:
    file_path = os.path.join(data_dir, file_name)
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Warning: {file_name} not found, skipping.")
        continue

    try:
        alpha, cl, cd = load_polar_data(file_path)

        # Define interpolation range based on available alpha values
        alpha_to_interpolate = np.linspace(min(alpha), max(alpha), 100)

        # Interpolate Cl and Cd:
        cl_values, cd_values = interpolate_cl_cd(alpha, cl, cd, alpha_to_interpolate, method='cubic')

        # Plot results
        ax[0].plot(alpha_to_interpolate, cl_values, label=f"{file_name}", linestyle='-')
        ax[1].plot(alpha_to_interpolate, cd_values, label=f"{file_name}", linestyle='--')

    except Exception as e:
        print(f"Error processing {file_name}: {e}")

# Final plot formatting
ax[0].set_title("Lift Coefficient (Cl) vs. Angle of Attack")
ax[0].set_xlabel('Angle of Attack (α) [degrees]')
ax[0].set_ylabel('Cl')
#ax[0].legend(loc='upper left', fontsize=8, ncol=2)
ax[0].grid(True)

ax[1].set_title("Drag Coefficient (Cd) vs. Angle of Attack")
ax[1].set_xlabel('Angle of Attack (α) [degrees]')
ax[1].set_ylabel('Cd')

#ax[1].legend(loc='upper left', fontsize=8, ncol=2)
ax[1].grid(True)

plt.tight_layout()
plt.show()

