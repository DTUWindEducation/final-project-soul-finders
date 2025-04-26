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
from __init__ import interpolate_wind_speed_var_parameter
from __init__ import compute_a_s
from __init__ import sigma_calc
from __init__ import calculate_rotor_parameters



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

for key, folder_path in wind_speeds.items():
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path) and file_name.endswith(".txt"):
            u = np.loadtxt(file_path, skiprows=1)
            # Process the wind speed data as needed
            # For example, you can store it in a list or array for further analysis
            # wind_speeds.append(wind_speed)


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


alpha_values = np.linspace(-180, 180, 1000)  # Define alpha range
cl_data, _, _ = interpolate_2d(alpha_values, polar_files_dir, path_geometry, data_type="cl")
cd_data, _, _ = interpolate_2d(alpha_values, polar_files_dir, path_geometry, data_type="cd")
alpha_grid = interpolate_2d(alpha_values, polar_files_dir, path_geometry, data_type="alpha")
sigma = sigma_calc(r, c)


interpolated_results = {}
a_s_results = {}

for key, folder_path in wind_speeds.items():  # key might be wind condition or category
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path) and file_name.endswith(".txt"):
            # Load the time series wind speed data
            u = np.loadtxt(file_path, skiprows=1)
            
            # Make sure wind_speed for each file is tracked
            wind_speed_for_file = np.mean(u)  # or any other method of tracking wind speed
            
            # Interpolate wind-speed-dependent parameters
            result = interpolate_wind_speed_var_parameter(u, v, p, w, P, T)

            # Save for later (optional)
            interpolated_results[file_name] = result

            # Extract parameters needed for compute_a_s
            local_wind_speeds = result['wind_speed']
            interpolated_p = result['pitch']
            interpolated_w = result['rotational_speed']

            # Compute a and a'
            cl_new, cd_new, alpha_comp, an, an_prime, dT, dM, w_new, u_new = compute_a_s(
                r, local_wind_speeds, interpolated_p, interpolated_w,
                B, alpha_values, cl_data, cd_data, sigma, rho=1.225,
                tolerance=1e-6, max_iter=100
            )

            # Store results for this file
            a_s_results[file_name] = {
                'cl': cl_new,
                'cd': cd_new,
                'alpha': alpha_comp,
                'a': an,
                'a_prime': an_prime,
                'dT': dT,
                'dM': dM,
                'w_new': w_new,
                'u_new': u_new,
                'wind_speed': wind_speed_for_file,  # Track wind speed for each file
            }

            

output_folder = "./outputs/interpolated_parameters_wind"
os.makedirs(output_folder, exist_ok=True)

# Loop through the interpolated results and save each as a CSV
for file_name, res in interpolated_results.items():
    # Create a DataFrame from the result dictionary
    df = pd.DataFrame({
        'time': res['time'],
        'wind_speed': res['wind_speed'],
        'pitch': res['pitch'],
        'rotational_speed': res['rotational_speed'],
        'power': res['power'],
        'torque': res['torque']
    })

    # Create a valid filename by replacing .txt with _interpolated.csv
    base_name = os.path.splitext(file_name)[0]
    output_path = os.path.join(output_folder, f"{base_name}_interpolated.csv")

    # Save the DataFrame to a CSV file
    df.to_csv(output_path, index=False)
    

# Create output folders if they don't exist
output_folder_cl = "./outputs/cl_table_windspeeds"
output_folder_cd = "./outputs/cd_table_windspeeds"
os.makedirs(output_folder_cl, exist_ok=True)
os.makedirs(output_folder_cd, exist_ok=True)

# Now loop through a_s_results and save Cl and Cd
for file_name, res in a_s_results.items():
    cl_array = res['cl']   # shape: (len(r), len(alpha_values))  → actually (len(r), len(alpha_comp))
    cd_array = res['cd']
    alpha_comp_array = res['alpha']  # This is what you want!

    # r is a 1D array (length len(r))
    r_array = r  # your blade span (still constant)

    # Flatten for saving
    R_flat = np.repeat(r_array, alpha_comp_array.shape[1])  # Repeat each r value for every alpha along axis-1
    Alpha_flat = alpha_comp_array.flatten()
    Cl_flat = cl_array.flatten()
    Cd_flat = cd_array.flatten()

    # Create DataFrames
    df_cl = pd.DataFrame({
        'r': R_flat,
        'alpha_comp': Alpha_flat,
        'Cl': Cl_flat
    })

    df_cd = pd.DataFrame({
        'r': R_flat,
        'alpha_comp': Alpha_flat,
        'Cd': Cd_flat
    })

    # Build output paths
    base_name = os.path.splitext(file_name)[0]
    cl_output_path = os.path.join(output_folder_cl, f"{base_name}_cl.csv")
    cd_output_path = os.path.join(output_folder_cd, f"{base_name}_cd.csv")

    # Save to CSVs
    df_cl.to_csv(cl_output_path, index=False)
    df_cd.to_csv(cd_output_path, index=False)



# Create correct 2D grids
r_grid, alpha_grid = np.meshgrid(r, alpha_comp[0,:], indexing='ij')

# First subplot for Cl
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(r_grid, alpha_grid, cl_new, cmap='viridis')
ax1.set_xlabel("Blade Span (r)")
ax1.set_ylabel("Alpha (radians)")
ax1.set_zlabel("Cl")
ax1.set_title("Cl Values in 3D")

# Second subplot for Cd
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(r_grid, alpha_grid, cd_new, cmap='viridis')
ax2.set_xlabel("Blade Span (r)")
ax2.set_ylabel("Alpha (radians)")
ax2.set_zlabel("Cd")
ax2.set_title("Cd Values in 3D")

plt.tight_layout()
plt.show
# New dictionary to store the final rotor parameters
rotor_results = {}

# Loop through each file's results
for file_name, results in a_s_results.items():
    
    # Extract all the needed variables for this file
    dT = results['dT']
    dM = results['dM']
    w_new = results['w_new']
    u_new = results['u_new']
    
    # Compute the rotor parameters
    rotor_params = calculate_rotor_parameters(r, interpolated_w , dT, dM, u_new, rho=1.225)
    
    # Save the rotor parameters
    rotor_results[file_name] = rotor_params



# Create one big figure for all wind speeds
fig = plt.figure(figsize=(14, 6))
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')

r_mid = (r[1:] + r[:-1]) / 2

for file_name, result in a_s_results.items():
    
    # Retrieve the corresponding wind speed
    wind_speed = interpolated_results[file_name]['wind_speed']
    
    # Take average wind speed or whatever makes sense
    avg_wind_speed = np.mean(wind_speed)

    # Plot dT (Thrust)
    ax1.plot(r_mid, [avg_wind_speed]*len(r_mid), result['dT']/1000, label=f'{file_name}')

    # Plot dM (Torque)
    ax2.plot(r_mid, [avg_wind_speed]*len(r_mid), result['dM']/1000, label=f'{file_name}')

# Set labels and titles
ax1.set_xlabel('Blade span [m]')
ax1.set_ylabel('Wind Speed [m/s]')
ax1.set_zlabel('dT [kN]')
ax1.set_title('Thrust Distribution (All wind speeds)')
ax1.grid(True)

ax2.set_xlabel('Blade span [m]')
ax2.set_ylabel('Wind Speed [m/s]')
ax2.set_zlabel('dM [kNm]')
ax2.set_title('Torque Distribution (All wind speeds)')
ax2.grid(True)


plt.tight_layout()
plt.show()