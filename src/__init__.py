import numpy as np
from pathlib import Path
from io import StringIO
import os
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
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
    

    # Loop through polar files and extract data
    for i, file_name in enumerate(sorted(os.listdir(polar_files_dir))):
        file_path = os.path.join(polar_files_dir, file_name)
        if os.path.isfile(file_path) and file_name.endswith(".dat"):
            alpha, cl, cd = load_airfoil_polar(file_path)
            data = cl if data_type == "cl" else cd
            
            # Store the data for this airfoil section
            polar_data.append(data)
            alpha_data.append(alpha)
            if i < len(r):
                blspn_positions.append(r[i])

    # Convert lists to numpy arrays and ensure same length as r
    blspn_positions = np.array(blspn_positions)
    
    # Initialize the interpolated data array with the correct shape (r, alpha)
    interpolated_data = np.zeros((len(blspn_positions), len(alpha_values)))

    # Interpolate for each blade section
    for i, (alpha_section, data_section) in enumerate(zip(alpha_data, polar_data)):
        # Sort alpha and data to ensure proper interpolation
        sort_idx = np.argsort(alpha_section)
        alpha_section = alpha_section[sort_idx]
        data_section = data_section[sort_idx]
        
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
    blspn_grid, alpha_grid = np.meshgrid( blspn_positions, alpha_values, indexing='ij')


    return interpolated_data, alpha_grid, blspn_grid



def interpolate_wind_speed_var_parameter(u, v, p, w, P, T, num_points=50):
    """
    Interpolates parameters (p, w, P, T) for each wind speed value in time series data.
    
    Parameters:
        u (ndarray): Time series data with shape (n,2) where:
                     u[:,0] = time values
                     u[:,1] = wind speed values
        v (ndarray): Reference wind speeds (the x-axis for interpolation)
        p (ndarray): Pitch angles corresponding to reference wind speeds
        w (ndarray): Rotational speeds corresponding to reference wind speeds
        P (ndarray): Power values corresponding to reference wind speeds
        T (ndarray): Torque values corresponding to reference wind speeds
    
    Returns:
        dict: Dictionary containing interpolated parameters for each time step with keys:
              'time', 'wind_speed', 'pitch', 'rotational_speed', 'power', 'torque'
    """
    # Extract wind speeds from time series data
    wind_speeds = u[:, 1]
    
    # Create interpolation functions for each parameter
    interp_p = interp1d(v, p, kind='linear', bounds_error=False, fill_value=(p[0], p[-1]))
    interp_w = interp1d(v, w, kind='linear', bounds_error=False, fill_value=(w[0], w[-1]))
    interp_P = interp1d(v, P, kind='linear', bounds_error=False, fill_value=(P[0], P[-1]))
    interp_T = interp1d(v, T, kind='linear', bounds_error=False, fill_value=(T[0], T[-1]))
    
    # Interpolate parameters for each wind speed
    interpolated_p = interp_p(wind_speeds)
    interpolated_w = interp_w(wind_speeds)
    interpolated_P = interp_P(wind_speeds)
    interpolated_T = interp_T(wind_speeds)

    indices = np.linspace(0, len(u) - 1, num_points, dtype=int)
    
    return {
        'time': u[indices, 0],
        'wind_speed': wind_speeds[indices],
        'pitch': interpolated_p[indices],
        'rotational_speed': interpolated_w[indices],
        'power': interpolated_P[indices],
        'torque': interpolated_T[indices]
    }


def compute_a_s(r,wind_speeds, interpolated_p, interpolated_w , B, alpha_values, cl_data, cd_data, sigma,rho = 1.225, tolerance=1e-6, max_iter=1000):
    """
    Compute the axial induction factor (a) and the tangential induction factor (a').
    """
    # Ensure r and alpha_values are strictly ascending and unique
    r = np.unique(r)
    alpha_values = np.unique(alpha_values)

    # Ensure B is a NumPy array and matches r's shape
    B = np.array(B)
    if len(B) != len(r):
        raise ValueError(f"B shape {B.shape} does not match r shape {r.shape}")

    # Initialize a and a' arrays
    an = np.zeros_like(r)
    an_prime = np.zeros_like(r)

    # Constants
    w_rpm = interpolated_w
    w_new = w_rpm * 2 * np.pi / 60
    u_new = wind_speeds
    A_new = interpolated_p
    # Create interpolation functions
    cl_interp_func = RegularGridInterpolator((r, alpha_values), cl_data, method='linear', 
                                            bounds_error=False, fill_value=None)
    cd_interp_func = RegularGridInterpolator((r, alpha_values), cd_data, method='linear', 
                                            bounds_error=False, fill_value=None)

    # Initialize storage for Cl and Cd
    cl_new = np.zeros((len(r), len(alpha_values)))
    cd_new = np.zeros((len(r), len(alpha_values)))
    alpha_comp = np.zeros((len(r), len(alpha_values)))
    
     # Create meshgrid for interpolation
    r_grid, alpha_grid = np.meshgrid(r, alpha_values, indexing='ij')
    
    for iteration in range(max_iter):


        # Compute flow angle phi for each r
        phi = np.arctan2((1 - an) * u_new , ((1 + an_prime) * w_new))
        A_new_rad = np.radians(A_new) if np.max(np.abs(A_new)) > 2*np.pi else A_new
        B_rad = np.radians(B) if np.max(np.abs(B)) > 2*np.pi else B
        # Create alpha_comp with correct broadcasting
        for i in range(len(r)):
            alpha_comp[:,i]  = phi[i] - (A_new_rad[i] + B_rad[i])
        
        


        # Ensure all arrays have correct shapes before stacking
        r_points = r_grid.flatten() 
        points = np.column_stack((r_points, np.clip(alpha_comp.flatten(), alpha_values.min(), alpha_values.max())))


        # Interpolate Cl and Cd
        cl_new = cl_interp_func(points).reshape(len(r), len(alpha_values))
        cd_new = cd_interp_func(points).reshape(len(r), len(alpha_values))
        

        # Compute normal and tangential coefficients
        Cn = cl_new * np.cos(phi[:, np.newaxis]) - cd_new * np.sin(phi[:, np.newaxis])
        Ct = cl_new * np.sin(phi[:, np.newaxis]) + cd_new * np.cos(phi[:, np.newaxis])

        # Compute new induction factors
        a_new = 1 / (4 * (np.sin(phi[:, np.newaxis]) ** 2) / (sigma[:, np.newaxis] * Cn) + 1)
        a_prime_new = 1 / (4 * np.sin(phi[:, np.newaxis]) * np.cos(phi[:, np.newaxis]) / 
                          (sigma[:, np.newaxis] * Ct) - 1)

        # Check convergence
        if (np.all(np.abs(a_new - an[:, np.newaxis]) < tolerance) and 
            np.all(np.abs(a_prime_new - an_prime[:, np.newaxis]) < tolerance)):
            break

        # Update values for next iteration
        an = np.clip(a_new[:, 0], 0, 0.4)  
        an_prime = np.clip(a_prime_new[:, 0], 0, None)  # Clip a' to be ≥ 0
        
        dr = np.diff(r)  # Distance between elements
        r_mid = (r[1:] + r[:-1]) / 2

        #shorten the shape of u_new and w_new to match the shape of r
        
        an_mid = (an[:-1] + an[1:]) / 2
        an_prime_mid = (an_prime[:-1] + an_prime[1:]) / 2
        u_new_mid = (u_new[:-1] + u_new[1:]) / 2
        w_new1 = (w_new[:-1] + w_new[1:]) / 2
        dT = 4 * np.pi * r_mid * rho * (u_new_mid ** 2) * an_mid * (1 - an_mid) * dr
        dM = 4 * np.pi * (r_mid ** 3) * rho * u_new_mid * w_new1 * an_prime_mid * (1 - an_mid) * dr
   
    return cl_new, cd_new, alpha_comp, an, an_prime, dT, dM, w_new, u_new



def calculate_rotor_parameters(r, interpolated_w, dT, dM, u_new, rho=1.225):

    """
    Calculate thrust, torque, power and their coefficients.

    Parameters:
        r (array): Blade span positions [m]
        an (array): Axial induction factors
        an_prime (array): Tangential induction factors
        u_new (float): Wind speed [m/s]
        w_rpm (float): Rotational speed [rpm]
        rho (float): Air density [kg/m^3]

    Returns:
        dict: Dictionary containing rotor parameters
    """

    
    # Integrate to get total thrust and torque
    T = np.sum(dT)  # Total thrust [N]
    M = np.sum(dM)  # Total torque [Nm]
    
    w_new = interpolated_w * 2 * np.pi / 60  # Convert to rad/s
    P = M * w_new
    # Calculate rotor area using actual radius
    R = r[-1]  # Use last r value as rotor radius
    A = np.pi * R**2  # Rotor area [m^2]
    
    # Calculate coefficients
    CT = T / (0.5 * rho * A * u_new**2)  # Thrust coefficient [-]
    CP = P / (0.5 * rho * A * u_new**3)  # Power coefficient [-]

   
    
    return {
        'thrust': T,
        'torque': M,
        'power': P,
        'thrust_coefficient': CT,
        'power_coefficient': CP,
        'rotor_area': A,
        
    }


# 6 Compute optimal operational strategy
def compute_optimal_strategy(wind_speed, opt_file_path):
    """
    Compute optimal blade pitch angle (θ_p) and rotational speed (ω)
    as a function of wind speed (V_0) based on the provided operational strategy.

    Parameters:
        wind_speed (float): Wind speed (V_0) in m/s.
        opt_file_path (str): Path to the `IEA_15MW_RWT_Onshore.opt` file.

    Returns:
        tuple: Interpolated blade pitch angle (θ_p) in degrees and rotational speed (ω) in rpm.
    """
    # Check if the file exists
    if not os.path.exists(opt_file_path):
        raise FileNotFoundError(f"Operational strategy file not found: {opt_file_path}")

    # Load data from the .opt file
    try:
        data = np.loadtxt(opt_file_path, skiprows=1)  # Adjust skiprows if needed
        V_0 = data[:, 0]  # Wind speed [m/s]
        θ_p = data[:, 1]  # Blade pitch angle [deg]
        ω = data[:, 2]    # Rotational speed [rpm]
    except Exception as e:
        raise ValueError(f"Error reading operational strategy file: {e}")

    # Validate input wind speed
    if wind_speed < 0:
        raise ValueError("Wind speed must be a positive number.")

    # Handle edge cases for wind speed
    if wind_speed < V_0.min():
        return 0.0, 0.0  # Default values for non-operational state
    elif wind_speed > V_0.max():
        return θ_p[-1], ω[-1]  # Use the last available data point

    # Create interpolation functions
    θ_p_interp = interp1d(V_0, θ_p, kind='linear', fill_value="extrapolate")
    ω_interp = interp1d(V_0, ω, kind='linear', fill_value="extrapolate")

    # Interpolate for the given wind speed
    θ_p_opt = θ_p_interp(wind_speed)
    ω_opt = ω_interp(wind_speed)

    return θ_p_opt, ω_opt


# 7 Compute and plot power curve and thrust curve
def compute_power_and_thrust_curves(wind_speeds, operational_strategy_path, geometry, rho=1.225):
    """
    Compute power and thrust curves based on the optimal operational strategy.

    Parameters:
        wind_speeds (array): Array of wind speeds (V_0) in m/s.
        operational_strategy_path (str): Path to the operational strategy file.
        geometry (tuple): Geometry data (r, B, c, Ai).
        rho (float): Air density in kg/m^3 (default: 1.225).

    Returns:
        tuple: Arrays of power (P) in kW and thrust (T) in kN for each wind speed.
    """
    r, B, c, Ai = geometry
    power = []
    thrust = []

    for V_0 in wind_speeds:
        # Get optimal blade pitch angle (θ_p) and rotational speed (ω)
        θ_p, ω = compute_optimal_strategy(V_0, operational_strategy_path)

        # Compute rotor parameters (thrust, torque, power)
        sigma = sigma_calc(r, c)
        cl_data, _, _ = interpolate_2d(np.linspace(-180, 180, 100), "./inputs/IEA-15-240-RWT/Airfoils/polar_files", "./inputs/IEA-15-240-RWT/IEA-15-240-RWT_AeroDyn15_blade.dat", data_type="cl")
        cd_data, _, _ = interpolate_2d(np.linspace(-180, 180, 100), "./inputs/IEA-15-240-RWT/Airfoils/polar_files", "./inputs/IEA-15-240-RWT/IEA-15-240-RWT_AeroDyn15_blade.dat", data_type="cd")
        _, _, _, an, an_prime = compute_a_s(r, V_0, ω, 0, B, np.linspace(-180, 180, 100), cl_data, cd_data, sigma)
        rotor_params = calculate_rotor_parameters(r, an, an_prime, rho)

        # Append results
        power.append(rotor_params['power'] / 1000)  # Convert to kW
        thrust.append(rotor_params['thrust'] / 1000)  # Convert to kN

    return np.array(power), np.array(thrust)


def plot_power_and_thrust_curves(wind_speeds, power, thrust):
    """
    Plot power and thrust curves.

    Parameters:
        wind_speeds (array): Array of wind speeds (V_0) in m/s.
        power (array): Array of power values (P) in kW.
        thrust (array): Array of thrust values (T) in kN.
    """
    plt.figure(figsize=(12, 6))

    # Power curve
    plt.subplot(1, 2, 1)
    plt.plot(wind_speeds, power, label="Power Curve", color="blue")
    plt.xlabel("Wind Speed (m/s)")
    plt.ylabel("Power (kW)")
    plt.title("Power Curve (P(V_0))")
    plt.grid(True)
    plt.legend()

    # Thrust curve
    plt.subplot(1, 2, 2)
    plt.plot(wind_speeds, thrust, label="Thrust Curve", color="green")
    plt.xlabel("Wind Speed (m/s)")
    plt.ylabel("Thrust (kN)")
    plt.title("Thrust Curve (T(V_0))")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()


def compute_rotor_thrust_torque_power(V_0, θ_p, ω, geometry, cl_data, cd_data, rho=1.225):
    """
    Compute thrust (T), torque (M), and power (P) of the rotor.

    Parameters:
        V_0 (float): Inflow wind speed [m/s].
        θ_p (float): Blade pitch angle [deg].
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
        r, V_0, ω, 0, B, np.linspace(-180, 180, 100), cl_data, cd_data, sigma
    )

    # Compute rotor parameters
    rotor_params = calculate_rotor_parameters(r, an, an_prime, rho)

    return {
        "thrust": rotor_params["thrust"],  # Thrust [N]
        "torque": rotor_params["torque"],  # Torque [Nm]
        "power": rotor_params["power"],    # Power [W]
    }

