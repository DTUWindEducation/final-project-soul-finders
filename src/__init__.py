import numpy as np
from pathlib import Path
from io import StringIO
import os
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


# 1. Load the geometry of the wind turbine blade
def load_geometry(path_geometry):
    """
    Load the geometry of the wind turbine blade
    :ath_geometry: path to the geometry file
    :return: r, B, c, Ai
    """
    path_geometry = Path(path_geometry)
    data = np.genfromtxt(path_geometry, delimiter='', skip_header=6)
    r, B, c, Ai = data[:, 0], data[:, 4], data[:, 5], data[:, 6]
   
    return r, B, c, Ai


# 2. Load the operational strategy of the wind turbine
def load_operational_strategy(path_operational_strategy):
    """
    Load the operational strategy of the wind turbine"
    """
    path_operational_strategy = Path(path_operational_strategy)
    data = np.genfromtxt(path_operational_strategy, delimiter='', skip_header=1)
    u, a, w, P, T = data[:, 0], data[:, 1], data[:, 2], data[:, 3], data[:, 4]
    
    return u, a, w, P, T


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
def interpolate_2d(alpha_values, polar_files_dir, path_geometry, data_type="cl"):
    """
    Interpolates Cl or Cd as a function of alpha and blade span.
    Returns data with shape (len(r), len(alpha_values))
    """
    # Load blade span data
    r, _, _, _ = load_geometry(path_geometry)

    # Initialize storage for data
    polar_data = []
    alpha_data = []
    blspn_positions = []

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
    alpha_grid, blspn_grid = np.meshgrid(alpha_values, blspn_positions)

    return interpolated_data, alpha_grid, blspn_grid

def compute_a_s(r, u, w, a, B, alpha_values, cl_data, cd_data, sigma, tolerance=1e-6, max_iter=100):
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
    w_new = 6.93 # rad/s
    u_new = 9
    A_new = 0.000535

    # Create interpolation functions
    cl_interp_func = RegularGridInterpolator((r, alpha_values), cl_data, method='linear', 
                                            bounds_error=False, fill_value=None)
    cd_interp_func = RegularGridInterpolator((r, alpha_values), cd_data, method='linear', 
                                            bounds_error=False, fill_value=None)

    # Initialize storage for Cl and Cd
    cl_new = np.zeros((len(r), len(alpha_values)))
    cd_new = np.zeros((len(r), len(alpha_values)))
    alpha_comp = np.zeros((len(r), len(alpha_values)))

    for iteration in range(max_iter):
        # Compute flow angle phi for each r
        phi = np.arctan((1 - an) * u_new / ((1 + an_prime) * w_new))
        phi_d = np.degrees(phi) 
        # Create alpha_comp with correct broadcasting
        alpha_comp = np.zeros((len(r), len(alpha_values)))
        for i in range(len(r)):
            alpha_comp[:,i]  = phi_d[i] - (A_new + B[i])

        # Create meshgrid for interpolation
        r_grid, alpha_grid = np.meshgrid(r, alpha_values, indexing='ij')
        
        # Ensure all arrays have correct shapes before stacking
        r_points = r_grid.flatten()
        alpha_points = alpha_comp.flatten()
        
        if len(r_points) != len(alpha_points):
            raise ValueError(f"Dimension mismatch: r_points ({len(r_points)}) != alpha_points ({len(alpha_points)})")
            
        points = np.column_stack((r_points, alpha_points))

        # Interpolate Cl and Cd
        cl_new = cl_interp_func(points).reshape(len(r), len(alpha_values))
        cd_new = cd_interp_func(points).reshape(len(r), len(alpha_values))

        # Compute normal and tangential coefficients
        Cn = cl_new * np.cos(phi_d[:, np.newaxis]) - cd_new * np.sin(phi_d[:, np.newaxis])
        Ct = cl_new * np.sin(phi_d[:, np.newaxis]) + cd_new * np.cos(phi_d[:, np.newaxis])

        # Compute new induction factors
        a_new = 1 / (4 * (np.sin(phi_d[:, np.newaxis]) ** 2) / (sigma[:, np.newaxis] * Cn) + 1)
        a_prime_new = 1 / (4 * np.sin(phi_d[:, np.newaxis]) * np.cos(phi_d[:, np.newaxis]) / 
                          (sigma[:, np.newaxis] * Ct) - 1)

        # Check convergence
        if (np.all(np.abs(a_new - an[:, np.newaxis]) < tolerance) and 
            np.all(np.abs(a_prime_new - an_prime[:, np.newaxis]) < tolerance)):
            break

        # Update values for next iteration
        an = a_new[:, 0]
        an_prime = a_prime_new[:, 0]

    return cl_new, cd_new, alpha_comp, an, an_prime

def create_2d_table(data, alpha_comp, r, data_type="Cl"):
    """
    Create a 2D pandas DataFrame for Cl or Cd values with segregated alpha ranges.
    
    Parameters:
        data (numpy.ndarray): 2D array of shape (n_span, n_alpha) with Cl or Cd values
        alpha_comp (numpy.ndarray): 2D array of shape (n_span, n_alpha) with angle of attack values
        r (numpy.ndarray): Array of blade span positions
        data_type (str): Type of data ("Cl" or "Cd")
    
    Returns:
        pandas.DataFrame: 2D table with r values as rows and computed alpha values as columns
    """
    # Verify shapes
    if data.shape != alpha_comp.shape:
        raise ValueError(f"Data shape {data.shape} doesn't match alpha_comp shape {alpha_comp.shape}")
    if len(r) != data.shape[0]:
        raise ValueError(f"Number of span positions {len(r)} doesn't match data rows {data.shape[0]}")
    
    # Analyze alpha distribution for each span position
    alpha_stats = pd.DataFrame(index=r, columns=['min', 'max', 'mean', 'std'])
    for i, r_val in enumerate(r):
        alpha_stats.loc[r_val] = {
            'min': alpha_comp[i,:].min(),
            'max': alpha_comp[i,:].max(),
            'mean': alpha_comp[i,:].mean(),
            'std': alpha_comp[i,:].std()
        }
    
    # Create adaptive bins based on alpha distribution
    overall_min = alpha_stats['min'].min()
    overall_max = alpha_stats['max'].max()
    std_mean = alpha_stats['std'].mean()
    
    # Create bins with finer resolution where alpha values are concentrated
    bin_width = min(0.5, std_mean / 2)  # Adaptive bin width
    alpha_bins = np.arange(overall_min - bin_width, 
                          overall_max + bin_width, 
                          bin_width)
    
    # Initialize DataFrame
    df = pd.DataFrame(index=r, columns=alpha_bins[:-1])
    
    # Fill the DataFrame using binning approach
    for i, r_val in enumerate(r):
        # Get alpha values and corresponding data for this span position
        alphas = alpha_comp[i, :]
        values = data[i, :]
        
        # Sort alpha values and corresponding data
        sort_idx = np.argsort(alphas)
        alphas = alphas[sort_idx]
        values = values[sort_idx]
        
        # Create bins and compute mean values in each bin
        indices = np.digitize(alphas, alpha_bins) - 1
        for j in range(len(alpha_bins) - 1):
            mask = indices == j
            if np.any(mask):
                df.loc[r_val, alpha_bins[j]] = np.mean(values[mask])
            else:
                df.loc[r_val, alpha_bins[j]] = np.nan
    
    # Remove columns with all NaN values
    df = df.dropna(axis=1, how='all')
    
    # Add alpha statistics as additional columns
    df = pd.concat([df, alpha_stats], axis=1)
    
    # Set index and column names
    df.index.name = 'Blade span [m]'
    df.columns.name = 'Angle of attack [deg]'
    

    return df, alpha_stats

def calculate_rotor_parameters(r, an, an_prime, rho=1.225):
    """
    Calculate thrust, torque, power and their coefficients.
    
    Parameters:
        r (array): Blade span positions [m]
        an (array): Axial induction factors
        an_prime (array): Tangential induction factors
        rho (float): Air density [kg/m^3]
    
    Returns:
        dict: Dictionary containing rotor parameters
    """
    w_new = 6.93  # rad/s
    u_new = 9.0   # m/s

    # Ensure positive induction factors and slice to match r_mid length
    #an = np.abs(an[:-1])  # Take absolute value and slice
    an = np.clip(an[:-1], 0, 1)  # Ensure 0 <= a < 1.
    #an_prime = an_prime[:-1]  # Slice to match length
    an_prime = np.clip(an_prime[:-1], 0, None)  # Ensure a' >= 0.
    
    # Calculate differential elements
    dr = np.diff(r)  # Distance between elements
    r_mid = (r[1:] + r[:-1]) / 2  # Midpoints for integration
    
   
    
    # Calculate differential thrust and torque for each element
    dT = 4 * np.pi * r_mid * rho * (u_new ** 2) * an * (1 - an) * dr
    dM = 4 * np.pi * (r_mid ** 3) * rho * u_new * w_new * an_prime * (1 - an) * dr
    
    # Integrate to get total thrust and torque
    T = np.sum(dT)  # Total thrust [N]
    M = np.sum(dM)  # Total torque [Nm]
    
    # Calculate power
    #P = M * w_new  # Power [W]
    P = max(0, M * w_new)  # Power [W], ensure non-negative.
    
    # Calculate rotor area
    R = 120  # Rotor radius [m]
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
        'dT': dT,
        'dM': dM,
        
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
