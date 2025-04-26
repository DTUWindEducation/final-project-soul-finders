import numpy as np
from pathlib import Path
from io import StringIO
import os
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import interp1d


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
    alpha_grid, blspn_grid = np.meshgrid(alpha_values, blspn_positions, indexing='ij')


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
        phi_d = np.degrees(phi) 
        # Create alpha_comp with correct broadcasting
        for i in range(len(r)):
            alpha_comp[:,i]  = phi_d[i] - (A_new[i] + B[i])
        alpha_comp = np.abs(alpha_comp)


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
    
    w_rad_s = interpolated_w * 2 * np.pi / 60  # convert rpm -> rad/s
    P = M * w_rad_s   # Power [W]
    
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
