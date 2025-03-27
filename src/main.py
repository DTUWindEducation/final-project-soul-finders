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








            
