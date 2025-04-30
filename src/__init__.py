"""
This module contains utility functions and classes for wind turbine analysis.

It includes functions for loading airfoil data, computing aerodynamic
parameters, and saving results, as well as plotting utilities.
"""

import os
from pathlib import Path
from io import StringIO
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


# 3. Load the airfoil shape
def load_airfoil_shape(path_shape):
    """
    Load the airfoil data
    """
    path_shape = Path(path_shape)
    data = np.genfromtxt(path_shape, delimiter='', skip_header=8)
    norm_x, norm_y = data[:, 0], data[:, 1]
    return norm_x, norm_y


# 4. Load the airfoil polar
def load_airfoil_polar(path_polar):
    """
    Load the airfoil polar data
    """
    path_polar = Path(path_polar)
    try:
        # Open the file and preprocess it to extract only the data table
        with open(path_polar, 'r') as file:
            lines = file.readlines()

        # Find the line where the table starts
        start_line = 0
        for i, line in enumerate(lines):
            if line.strip().startswith("!    Alpha"):
                start_line = i + 1  # Data starts after this line
                break

        # Extract only the relevant lines (data table)
        data_lines = []
        for line in lines[start_line:]:
            # Skip empty lines and lines starting with "!" (comments)
            if line.strip() and not line.strip().startswith("!"):
                data_lines.append(line)

        # Use np.genfromtxt to parse the cleaned data
        data = np.genfromtxt(StringIO("".join(data_lines)), delimiter=None)

        # Extract Alpha, Cl, and Cd columns
        alpha, cl, cd = data[:, 0], data[:, 1], data[:, 2]
        return alpha, cl, cd
    except Exception as e:
        print(f"Error loading airfoil polar data: {e}")
        return None, None, None


def plot_3d_airfoil_shape(r, c, shape_files):
    """
    Plot 3D airfoil shapes along the blade span with a color-coded strip for the span positions.

    Args:
        r (array): Blade span positions
        c (array): Chord lengths
        shape_files (dict): Dictionary containing paths to airfoil shape files
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Define color map for different sections
    colors = plt.cm.rainbow(np.linspace(0, 1, len(r)))
    # Plot airfoils at each span position
    for key, folder_path in shape_files.items():
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path) and file_name.endswith(".txt"):
                norm_x, norm_y = load_airfoil_shape(file_path)
                # Scale and position each airfoil section
                for i, (span, chord) in enumerate(zip(r, c)):
                    scaled_x = norm_x * chord
                    scaled_y = norm_y * chord
                    z = np.full_like(scaled_x, span)
                    # Plot with color corresponding to the span position
                    ax.plot(scaled_x, scaled_y, z, color=colors[i], alpha=0.6, linewidth=1.5)

    # Customize the plot
    ax.set_xlabel("X-axis (Chordwise) [m]")
    ax.set_ylabel("Y-axis (Thickness) [m]")
    ax.set_zlabel("Z-axis (Spanwise) [m]")
    ax.set_title("3D Airfoil Shapes Along Blade Span", pad=20)
    ax.set_zlim(0, 120)

    # Set aspect ratio and view angle
    ax.set_box_aspect([1, 1, 2])
    ax.view_init(elev=20, azim=45)  # Adjust view angle
    # Add a color bar to represent the span positions
    sm = plt.cm.ScalarMappable(cmap=plt.cm.rainbow, norm=plt.Normalize(
                                            vmin=r.min(), vmax=r.max())
                                                )
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.1, aspect=30)
    cbar.set_label("Blade Span Position (r) [m]")
    # Adjust layout to prevent overlap
    plt.tight_layout()
    return fig, ax


def sigma_calc(r, c):
    """
    Calculate the solidity of the blade.

    Parameters:
        r (array-like): Blade span positions.
        B (array-like): Number of blades.
        c (array-like): Chord length.

    Returns:
        sigma (function): Function to calculate solidity at a given radius.
    """

    # Calculate solidity
    r_safe = np.where(r == 0, 1e-6, r)
    sigma = 3*c / (2 * np.pi * r_safe)
    return sigma


# 10 computing a and a'
def interpolate_2d(alpha_values, polar_files_dir, r, data_type="cl"):
    """
    Interpolates Cl or Cd as a function of alpha and blade span.
    Returns data with shape (len(r), len(alpha_values))
    """

    # Initialize storage for data
    polar_data = []
    blspn_positions = []
    alpha_data = []
    polar_files = []  # To store the names of polar files

    # Loop through polar files and extract data
    for i, file_name in enumerate(sorted(os.listdir(polar_files_dir))):
        file_path = os.path.join(polar_files_dir, file_name)
        if os.path.isfile(file_path) and file_name.endswith(".dat"):
            alpha, cl, cd = load_airfoil_polar(file_path)
            data = cl if data_type == "cl" else cd

            # Store the data for this airfoil section
            polar_data.append(data)
            alpha_data.append(alpha)
            polar_files.append(file_name)  # Store the file name
            if i < len(r):
                blspn_positions.append(r[i])

    # Convert lists to numpy arrays and ensure same length as r
    blspn_positions = np.array(blspn_positions)

    # Initialize the interpolated data array with the correct shape (r, alpha)
    interpolated_data = np.zeros((len(blspn_positions), len(alpha_values)))

    # Interpolate for each blade section
    for i, (alpha_section, data_section) in enumerate(zip(alpha_data,
                                                          polar_data)):
        # Sort alpha and data to ensure proper interpolation
        sort_idx = np.argsort(alpha_section)
        alpha_section = alpha_section[sort_idx]
        data_section = data_section[sort_idx]
        # Reverse the data if necessary (e.g., if Cd is inverted)

        if data_type == "cd" and data_section[0] > data_section[-1]:
            data_section = data_section[::-1]
            alpha_section = alpha_section[::-1]

        # Remove any duplicate alpha values
        unique_idx = np.unique(alpha_section, return_index=True)[1]
        alpha_section = alpha_section[unique_idx]
        data_section = data_section[unique_idx]

        # Perform interpolation for this section
        interpolated_values = np.interp(
            alpha_values,
            alpha_section,
            data_section,
            left=data_section[0],    # Use first value for extrapolation
            right=data_section[-1]   # Use last value for extrapolation
        )

        # Store the interpolated values (notice we store in rows now)
        interpolated_data[i, :] = interpolated_values

    # Create meshgrid for output

    blspn_grid, alpha_grid = np.meshgrid(blspn_positions,
                                         alpha_values, indexing='ij')

    return interpolated_data, alpha_grid, blspn_grid


def compute_a_s(r, B, alpha_values, cl_data, cd_data, sigma, v, p, w,
                tolerance=1e-6, max_iter=10000):
    """
    Compute the axial induction factor (a) and the
    tangential induction factor (a').
    """
    # Initialize arrays
    an = np.full_like(r, 1/3)  # Initialize with theoretical optimal value
    an_prime = np.zeros_like(r)
    cl_new = np.zeros_like(r)
    cd_new = np.zeros_like(r)

    # Constants
    w_rpm = w[0]
    w_new = w_rpm * 2 * np.pi / 60
    u_new = v[0]
    A_new = p[0]

    for iteration in range(max_iter):
        # Compute flow angle phi
        phi = np.arctan2((1 - an) * u_new, ((1 + an_prime) * w_new * r))

        # Calculate angle of attack in degrees
        alpha_comp = np.degrees(phi) - A_new - B
        alpha_comp = np.where(r == 0, 0, alpha_comp)  # Handle root section

        # Clip alpha to valid range
        alpha_comp = np.clip(alpha_comp, alpha_values.min(),
                             alpha_values.max())

        # Create interpolator functions for cl and cd
        for i in range(len(r)):
            # Get cl and cd values directly from interpolated data
            cl_new[i] = np.interp(alpha_comp[i], alpha_values, cl_data[i, :])
            cd_new[i] = np.interp(alpha_comp[i], alpha_values, cd_data[i, :])
        # Compute normal and tangential coefficients
        Cn = cl_new * np.cos(phi) - cd_new * np.sin(phi)
        Ct = cl_new * np.sin(phi) + cd_new * np.cos(phi)

        # Compute new induction factors with Glauert correction
        a_new = np.zeros_like(r)
        for i in range(len(r)):
            if Cn[i] > 0:  # Avoid division by zero
                a_temp = 1 / (4 * np.sin(phi[i])**2 / (sigma[i] * Cn[i]) + 1)
                # Apply Glauert correction for a > 0.4
                if a_temp > 0.4:
                    a_new[i] = 0.4
                else:
                    a_new[i] = a_temp
            else:
                a_new[i] = 0

        # Compute tangential induction factor
        a_prime_new = np.zeros_like(r)
        for i in range(len(r)):
            if Ct[i] > 0:  # Avoid division by zero
                a_prime_new[i] = 1 / (4 * np.sin(phi[i]) * np.cos(phi[i]) /
                                      (sigma[i] * Ct[i]) - 1)

        # Apply relaxation factor
        relaxation = 0.25
        a_new = (1 - relaxation) * an + relaxation * a_new
        a_prime_new = (1 - relaxation) * an_prime + relaxation * a_prime_new

        # Check convergence
        if (np.all(np.abs(a_new - an) < tolerance) and
            np.all(np.abs(a_prime_new - an_prime) < tolerance)):
            break

        # Update values for next iteration
        an = a_new
        an_prime = a_prime_new

    # Debug: Print values for each blade element
    # print(f"Iteration {iteration + 1}:")
    # for i in range(len(r)):
    #    print(f"  Blade Element r = {r[i]:.2f} m:")
    #    print(f"    Cl = {cl_new[i]:.4f}, Cd = {cd_new[i]:.4f}")
    #    print(f"alpha_comp = {alpha_comp[i]:.4f}")
    #    print(f"    a = {an[i]:.4f}, a' = {an_prime[i]:.4f}")
    #    print(f"  phi = {np.degrees(phi[i]):.2f}°")

    return cl_new, cd_new, alpha_comp, an, an_prime, u_new


def save_blade_results(r, u_new, cl_new, cd_new, alpha_comp, an,
                       an_prime,
                       output_dir='./outputs/calculated_blade_parameters'):
    """
    Save blade element results to CSV files for each wind speed.

    Args:
        r (array): Blade span positions [m]
        u_new (float): Wind speed [m/s]
        cl_new (array): Lift coefficients
        cd_new (array): Drag coefficients
        alpha_comp (array): Angle of attack [deg]
        an (array): Axial induction factors
        an_prime (array): Tangential induction factors
        output_dir (str): Directory to save results (default: './outputs')
    """
    try:
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Create DataFrame with all results
        results_df = pd.DataFrame({
            'r [m]': r,
            'Cl [-]': cl_new,
            'Cd [-]': cd_new,
            'Alpha [deg]': alpha_comp,
            'a [-]': an,
            'a_prime [-]': an_prime
        })

        # Save to CSV with wind speed in filename
        filename = f'blade results V{u_new:.1f}.csv'
        filepath = Path(output_dir) / filename
        results_df.to_csv(filepath, index=False)

        print(f"Blade results saved successfully to {filepath}")
        return True
    except Exception as e:
        print(f"Error saving blade results: {e}")
        return False


def plot_3d_cl_cd_vs_r_alpha(r, alpha_values, cl_data, cd_data, alpha_comp):
    """
    Plot 3D graphs for cl and cd with respect to
    blade span (r) and angle of attack (alpha_comp).

    Args:
        r (array): Blade span positions [m].
        alpha_values (array): Angle of attack grid [deg].
        cl_data (2D array): Lift coefficient data for
        each blade span and angle of attack.
        cd_data (2D array): Drag coefficient data for
        each blade span and angle of attack.
        alpha_comp (array): Angle of attack values to interpolate [deg].
    """
    # Create a meshgrid for r and alpha_comp
    R, Al = np.meshgrid(r, alpha_comp, indexing='ij')

    # Interpolate cl and cd data to match the meshgrid
    cl_new = np.zeros_like(R)
    cd_new = np.zeros_like(R)

    for i in range(len(r)):
        # Interpolate cl and cd from alpha_values to alpha_comp
        cl_new[i, :] = np.interp(alpha_comp, alpha_values, cl_data[i, :])
        cd_new[i, :] = np.interp(alpha_comp, alpha_values, cd_data[i, :])

    fig = plt.figure(figsize=(16, 8))

    # First subplot: cl vs r and alpha_comp
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(R, Al, cl_new, cmap='viridis',
                     edgecolor='none', alpha=0.8)
    ax1.set_title("Lift Coefficient (Cl) vs Blade Span (r) and Angle of Attack"
                  )
    ax1.set_xlabel("Blade Span (r) [m]")
    ax1.set_ylabel("Angle of Attack (Alpha) [deg]")
    ax1.set_zlabel("Lift Coefficient (Cl)")
    ax1.view_init(elev=30, azim=45)  # Adjust the view angle

    # Second subplot: cd vs r and alpha_comp
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(R, Al, cd_new, cmap='plasma', edgecolor='none', alpha=0.8)
    ax2.set_title("Drag Coefficient (Cd) vs Blade Span (r) and Angle of Attack"
                  )
    ax2.set_xlabel("Blade Span (r) [m]")
    ax2.set_ylabel("Angle of Attack (Alpha) [deg]")
    ax2.set_zlabel("Drag Coefficient (Cd)")
    ax2.view_init(elev=30, azim=45)  # Adjust the view angle

    # Adjust layout
    plt.tight_layout()


def calculate_rotor_parameters(r, w, v, an, an_prime, rho=1.225):
    """Calculate rotor parameters for each blade element"""

    # Convert rotational speed from rpm to rad/s
    w_rpm = w[0]
    w_new = w_rpm * 2 * np.pi / 60
    u_new = v[0]

    # Calculate differentials for each element
    dr = np.diff(r)
    r_mid = (r[:-1] + r[1:]) / 2
    an_mid = (an[:-1] + an[1:]) / 2
    an_prime_mid = (an_prime[:-1] + an_prime[1:]) / 2

    # Initialize arrays for local values

    dT = np.zeros(len(r)-1)
    dM = np.zeros(len(r)-1)

    # Calculate thrust and torque for each element
    for i in range(len(r)-1):
        dT[i] = (4 * np.pi * r_mid[i] * rho * (u_new ** 2) *
                 an_mid[i] * (1 - an_mid[i]) * dr[i])
        dM[i] = (4 * np.pi * (r_mid[i] ** 3) * rho * u_new * w_new *
                 an_prime_mid[i] * (1 - an_mid[i]) * dr[i])

    # Calculate total values
    T = np.sum(dT)
    M = np.sum(dM)
    P = M * w_new

    # Calculate rotor coefficients
    R = r[-1]
    A = np.pi * R**2
    CT = T / (0.5 * rho * A * u_new**2)
    CP = P / (0.5 * rho * A * u_new**3)

    # Print local values for each element
    # for i in range(len(r)-1):
    # local_power = dM[i] * w_new
    # print(f"Blade Element r = {r_mid[i]:.2f} m:")
    # print(f"  Local Thrust = {dT[i]/1000:.4f} kN")
    # print(f"  Local Torque = {dM[i]/1000:.4f} kNm")
    # print(f"  Local Power = {local_power/1e6:.4f} MW")

    # Print total values
    # print("\nTotal Rotor Values:")
    # print(f"Total Thrust = {T/1000:.4f} kN")
    # print(f"Total Torque = {M/1000:.4f} kNm")
    # print(f"Total Power = {Power/1e6:.4f} MW")
    # print(f"CT = {CT:.4f}")
    # print(f"CP = {CP:.4f}")

    return {
        'thrust': T,
        'torque': M,
        'power': P,
        'thrust_coefficient': CT,
        'power_coefficient': CP,
        'rotor_area': A
    }


def save_rotor_parameters(v, rotor_parameters,
                          output_dir='./outputs/rotor_parameters'):
    """
    Save rotor parameters (thrust, torque, power, etc.) for
    each wind speed to a CSV file.

    Args:
        wind_speeds (array): Array of wind speeds [m/s].
        rotor_parameters (list of dict): List of
        dictionaries containing rotor parameters for each wind speed.
        output_dir (str): Directory to save the
        results (default: './outputs/rotor_parameters').
    """
    try:
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Prepare data for saving
        data = {
            'Wind Speed [m/s]': v,
            'Thrust [N]': [params['thrust'] for params in rotor_parameters],
            'Torque [Nm]': [params['torque'] for params in rotor_parameters],
            'Power [W]': [params['power'] for params in rotor_parameters],
            'Thrust Coefficient [-]': [params['thrust_coefficient']
                                       for params in rotor_parameters],
            'Power Coefficient [-]': [params['power_coefficient']
                                      for params in rotor_parameters],
            'Rotor Area [m^2]': [params['rotor_area']
                                 for params in rotor_parameters],
        }

        # Create a DataFrame
        results_df = pd.DataFrame(data)
        # Save to CSV
        output_file = Path(output_dir) / 'rotor_parameters.csv'
        results_df.to_csv(output_file, index=False)

        print(f"Rotor parameters saved successfully to {output_file}")
        return True

    except Exception as e:
        print(f"Error saving rotor parameters: {e}")

        return False


# 6 Compute optimal operational strategy
def compute_optimal_strategy(wind_speed, opt_file_path):
    """
    Compute optimal blade pitch angle (p) and rotational speed (ω)
    as a function of wind speed (u_new)
    based on the provided operational strategy.

    Parameters:
        wind_speed (float): Wind speed (u_new) in m/s.
        opt_file_path (str): Path to the `IEA_15MW_RWT_Onshore.opt` file.

    Returns:
        tuple: Interpolated blade pitch angle (p)
        in degrees and rotational speed (ω) in rpm.
    """
    # Check if the file exists
    if not os.path.exists(opt_file_path):
        raise FileNotFoundError(f"Operational strategy file not found: {opt_file_path}")

    # Load data from the .opt file
    try:
        data = np.loadtxt(opt_file_path, skiprows=1)
        u_new = data[:, 0]  # Wind speed [m/s]
        p = data[:, 1]  # Blade pitch angle [deg]
        ω = data[:, 2]    # Rotational speed [rpm]
    except Exception as e:
        raise ValueError(f"Error reading operational strategy file: {e}")

    # Validate input wind speed
    if wind_speed < 0:
        raise ValueError("Wind speed must be a positive number.")
    # Handle edge cases for wind speed
    if wind_speed < u_new.min():
        return 0.0, 0.0  # Default values for non-operational state
    elif wind_speed > u_new.max():
        return p[-1], ω[-1]  # Use the last available data point

    # Create interpolation functions
    p_interp = interp1d(u_new, p, kind='linear', fill_value="extrapolate")
    ω_interp = interp1d(u_new, ω, kind='linear', fill_value="extrapolate")

    # Interpolate for the given wind speed
    p_opt = p_interp(wind_speed)
    ω_opt = ω_interp(wind_speed)
    return p_opt, ω_opt

# 7 Compute and plot power curve and thrust curve

def compute_power_and_thrust_curves(wind_speeds, operational_strategy_path,
                                    geometry, polar_files_dir, path_geometry,
                                    rho=1.225):
    polar_files_dir = "./inputs/IEA-15-240-RWT/Airfoils/polar_files"
    path_geometry = "./inputs/IEA-15-240-RWT/IEA-15-240-RWT_AeroDyn15_blade.dat"

    """
    Compute power and thrust curves based on the optimal operational strategy.

    Parameters:
        wind_speeds (array): Array of wind speeds (V_0) in m/s.
        operational_strategy_path (str): Path to the operational strategy file.
        geometry (tuple): Geometry data (r, B, c, Ai).
        rho (float): Air density in kg/m^3 (default: 1.225).

    Returns:
        tuple: Arrays of power (P) in kW and thrust (T) in kN 
        for each wind speed.
    """
    r, B, c, Ai = geometry
    power = []
    thrust = []
    for u_new in wind_speeds:
        # Get optimal blade pitch angle (p) and rotational speed (ω)
        p, ω = compute_optimal_strategy(u_new, operational_strategy_path)

        # Compute rotor parameters (thrust, torque, power)
        sigma = sigma_calc(r, c)
        cl_data, _, _ = interpolate_2d(np.linspace(-180, 180, 100),
                                       polar_files_dir, path_geometry,
                                       data_type="cl")
        cd_data, _, _ = interpolate_2d(np.linspace(-180, 180, 100),
                                       polar_files_dir, path_geometry,
                                       data_type="cd")
        _, _, _, an, an_prime, _ = compute_a_s(r, B,
                                               np.linspace(-180, 180, 100),
                                               cl_data, cd_data, sigma,
                                               [u_new], [p], [ω])
        rotor_params = calculate_rotor_parameters(r, [ω], [u_new],
                                                  an, an_prime, rho)

        # Append results
        power.append(rotor_params['power'] / 1000)  # Convert to kW
        thrust.append(rotor_params['thrust'] / 1000)  # Convert to kN

    return np.array(power), np.array(thrust)


def plot_power_and_thrust_curves(wind_speeds, power, thrust,
                                 operational_strategy_path):
    """
    Plot power and thrust curves, overlapping real operational strategy data.

    Parameters:
        wind_speeds (array): Computed wind speeds (V_0) in m/s.
        power (array): Computed power values (P) in kW.
        thrust (array): Computed thrust values (T) in kN.
        operational_strategy_path (str): Path to real operational
        data (.opt file).
    """
    # Load real operational strategy data
    real_data = np.loadtxt(operational_strategy_path, skiprows=1)
    real_wind_speed = real_data[:, 0]  # Wind speed [m/s]
    real_power = real_data[:, 3]       # Aerodynamic power [kW]
    real_thrust = real_data[:, 4]      # Aerodynamic thrust [kN]

    plt.figure(figsize=(14, 6))

    # Power curve
    plt.subplot(1, 2, 1)
    plt.plot(wind_speeds, power, label="Computed Power Curve(Experiemental)",
             linestyle='-')
    plt.plot(real_wind_speed, real_power, label="Real Power Curve",
             linestyle='--')
    plt.xlabel("Wind Speed (m/s)")
    plt.ylabel("Power (kW)")
    plt.title("Power Curve: $P(V_0)$")
    plt.grid(True)
    plt.legend()

    # Thrust curve
    plt.subplot(1, 2, 2)
    plt.plot(wind_speeds, thrust, label="Computed Thrust Curve(Experiemental)",
             linestyle='-')
    plt.plot(real_wind_speed, real_thrust, label="Real Thrust Curve",
             linestyle='--')
    plt.xlabel("Wind Speed (m/s)")
    plt.ylabel("Thrust (kN)")
    plt.title("Thrust Curve: $T(V_0)$")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()


def compute_rotor_thrust_torque_power(u_new, p, ω, geometry, cl_data,
                                      cd_data, rho=1.225):
    """
    Compute thrust (T), torque (M), and power (P) of the rotor.

    Parameters:
        u_new (float): Inflow wind speed [m/s].
        p (float): Blade pitch angle [deg].
        ω (float): Rotational speed [rad/s].
        geometry (tuple): Geometry data (r, B, c, Ai).
        cl_data (array): Lift coefficient data.
        cd_data (array): Drag coefficient data.
        rho (float): Air density [kg/m^3] (default: 1.225).

    Returns:
        dict: Dictionary containing thrust (T), torque (M), and power (P).
    """
    r, B, c, Ai = geometry

    # Compute solidity
    sigma = sigma_calc(r, c)

    # Compute axial and tangential induction factors
    _, _, _, an, an_prime = compute_a_s(
        r, u_new, ω, p, B, np.linspace(-180, 180, 100), cl_data, cd_data, sigma
    )

    # Compute rotor parameters
    rotor_params = calculate_rotor_parameters(r, an, an_prime, rho)

    return {
        "thrust": rotor_params["thrust"],  # Thrust [N]
        "torque": rotor_params["torque"],  # Torque [Nm]
        "power": rotor_params["power"],    # Power [W]
    }
# final line