import numpy as np
from pathlib import Path
from io import StringIO
import os
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

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
    Interpolates Cl or Cd as a function of alpha and blade span using 2D interpolation.

    Parameters:
        alpha_values (array-like): Array of alpha values to interpolate.
        polar_files_dir (str): Directory containing the polar files.
        path_geometry (str): Path to the blade geometry file.
        data_type (str): Type of data to interpolate ("cl" for lift coefficient, "cd" for drag coefficient).

    Returns:
        interpolated_data (2D array): Interpolated Cl or Cd values for the given alpha and blade span.
        alpha_grid (2D array): Grid of alpha values used for interpolation.
        blspn_grid (2D array): Grid of blade span values used for interpolation.
    """
    # Load blade span data
    r, _, _, _ = load_geometry(path_geometry)

    # Initialize storage for data
    data_list = []
    blspn_positions = []

    # Loop through polar files and extract data
    for i, file_name in enumerate(sorted(os.listdir(polar_files_dir))):
        file_path = os.path.join(polar_files_dir, file_name)
        if os.path.isfile(file_path) and file_name.endswith(".dat"):
            alpha, cl, cd = load_airfoil_polar(file_path)
            data = cl if data_type == "cl" else cd  # Select Cl or Cd based on data_type
            data_list.append(data)
            if i < len(r):  # Ensure `i` does not exceed the length of `r`
                blspn_positions.append(r[i])  # Associate polar file with corresponding blade span position
            else:
                print(f"Warning: No blade span position for polar file {file_name}")

    # Create a grid for alpha and blade span
    alpha_grid, blspn_grid = np.meshgrid(alpha_values, blspn_positions)

    # Initialize the interpolated_data array with the correct shape
    interpolated_data = np.zeros((len(blspn_positions), len(alpha_values)))

    # Interpolate data values
    for i, data in enumerate(data_list):
        interp_func = np.interp(alpha_values, alpha, data)  # 1D interpolation for each blade span
        interpolated_data[i, :] = interp_func

    return interpolated_data, alpha_grid, blspn_grid

def compute_a_s(r, u, w, a, B, alpha_values, cl_data, cd_data, sigma, tolerance=1e-6, max_iter=100):
    """
    Compute the axial induction factor (a) and the tangential induction factor (a').

    Parameters:
        r (array-like): Blade span positions.
        u (array-like): Wind speed.
        w (float): Rotational speed.
        a (float): Axial induction factor.
        B (float): Number of blades.
        alpha_values (array-like): Array of alpha values.
        cl_data (2D array): Interpolated lift coefficient data.
        cd_data (2D array): Interpolated drag coefficient data.
        sigma (array-like): Solidity function.
        tolerance (float): Convergence tolerance.
        max_iter (int): Maximum number of iterations.

    Returns:
        tuple: Final values of a, a', Cl, Cd, and alpha.
    """
    # Initialize a and a' arrays
    an = np.zeros_like(r)
    an_prime = np.zeros_like(r)

    # Create interpolation functions for Cl and Cd
    cl_interp_func = RegularGridInterpolator((r, alpha_values), cl_data, method='linear', bounds_error=False, fill_value=None)
    cd_interp_func = RegularGridInterpolator((r, alpha_values), cd_data, method='linear', bounds_error=False, fill_value=None)

    # Interpolate w and u to match the shape of r
    w_new = np.interp(r, np.linspace(r.min(), r.max(), len(w)), w)
    u_new = np.interp(r, np.linspace(r.min(), r.max(), len(u)), u)

    for iteration in range(max_iter):
        # Expand a and a' for vectorized calculations
        an_expanded = an[:, np.newaxis]
        an_prime_expanded = an_prime[:, np.newaxis]

        # Compute flow angle phi
        phi = np.arctan((1 - an) * u_new / ((1 + an_prime) * w_new))
        phi_n = phi[:, np.newaxis]  # Expand phi to shape (len(r), 1)

        alpha_comp = np.zeros((len(r), len(alpha_values)))

        for i in range(len(r)):
            A_new = np.interp(r[i], np.linspace(r.min(), r.max(), len(a)), a)
            # Compute new alpha (angle of attack) for each blade span position
            alpha_comp[i] = phi_n[i] - (A_new + B[i])
        
        # Interpolate Cl and Cd for the new alpha and r
        cl_new = np.zeros_like(alpha_comp)
        cd_new = np.zeros_like(alpha_comp)
        for i in range(len(r)):
            points = np.array([[r[i], alpha] for alpha in alpha_comp[i, :]])
            cl_new[i, :] = cl_interp_func(points)
            cd_new[i, :] = cd_interp_func(points)

        # Compute lift and drag coefficients
        Cn = cl_new * np.cos(phi_n) - cd_new * np.sin(phi_n)
        Ct = cl_new * np.sin(phi_n) + cd_new * np.cos(phi_n)

        # Expand sigma for vectorized calculations
        sigma_n = sigma[:, np.newaxis]

        # Compute new a and a'
        a_new = 1 / (4 * (np.sin(phi_n) ** 2) / (sigma_n * Cn) + 1)
        a_prime_new = 1 / (4 * np.sin(phi_n) * np.cos(phi_n) / (sigma_n * Ct) - 1)

        # Check for convergence
        if np.all(np.abs(a_new - an_expanded) < tolerance) and np.all(np.abs(a_prime_new - an_prime_expanded) < tolerance):
            break

        # Update a and a'
        an = a_new[:, 0]
        an_prime = a_prime_new[:, 0]

    return an, an_prime, cl_new, cd_new, alpha_comp, B, phi_n, A_new


# 8 & 9 create a 2D table for interpolated Cl or Cd values
def interpolated_table(interpolated_data, alpha_values, r, data_type="cl_new"):
    """
    Creates a 2D table for interpolated Cl or Cd values.
    """
    interpolated_data = np.array(interpolated_data)
    if interpolated_data.ndim != 2:
        raise ValueError("Interpolated data must be a 2D array.")

    if interpolated_data.shape != (len(r), len(alpha_values)):
        raise ValueError(
            "Dimensions of interpolated_data do not match blade span and alpha values."
        )

    table = pd.DataFrame(
        interpolated_data,
        index=r,
        columns=alpha_values
    )
    table.index.name = "Blade Span (r)"
    table.columns.name = f"Angle of Attack (α) - {data_type}"
    return table

