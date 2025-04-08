import numpy as np
from pathlib import Path
from io import StringIO
import os
import pandas as pd


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
    

# 5 . Plot the 3D airfoil shape for each chord in the shape_files


# 6. Interpolate Cl or Cd as a function of alpha and blade span using 2D interpolation
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


# 8 & 9 create a 2D table for interpolated Cl or Cd values
def interpolated_table(interpolated_data, alpha_values, blspn_positions, data_type="Cl"):
    """
    Creates a 2D table for interpolated Cl or Cd values.

    Parameters:
        interpolated_data (2D array): Interpolated Cl or Cd values.
        alpha_values (array-like): Array of alpha values.
        blspn_positions (array-like): Array of blade span positions.
        data_type (str): Type of data ("Cl" for lift coefficient, "Cd" for drag coefficient).

    Returns:
        pd.DataFrame: A DataFrame representing the 2D table.
    """
    # Create a DataFrame with blade span as rows and alpha as columns
    table = pd.DataFrame(
        interpolated_data,
        index=blspn_positions,
        columns=alpha_values
    )
    table.index.name = "Blade Span (r)"
    table.columns.name = f"Angle of Attack (α) - {data_type}"
    return table


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

def compute_a_s(r, u, w, interpolated_cl, interpolated_cd, sigma, tolerance=1e-6, max_iter=100):
    """
    Compute the axial induction factor (a) and the tangential induction factor (a').

    Parameters:
        r (array-like): Blade span positions.
        u (array-like): Wind speed.
        w (float): Rotational speed.
        B (float): Number of blades.ś
        interpolated_cl (array-like): Interpolated lift coefficient.
        interpolated_cd (array-like): Interpolated drag coefficient.
        sigma (function): Solidity function.
        tolerance (float): Convergence tolerance.
        max_iter (int): Maximum number of iterations.

    Returns:
        tuple: Updated values of a and a'.
    """
    # Initialize a and a' arrays
    an = np.zeros_like(r)
    an_prime = np.zeros_like(r)
    # Interpolate w to match the shape of r
    w_new = np.interp(r, np.linspace(r.min(), r.max(), len(w)), w)
    # Interpolate u to match the shape of r
    u_new = np.interp(r, np.linspace(r.min(), r.max(), len(u)), u)
    for iteration in range(max_iter):
        an_expanded = an[:, np.newaxis]
        an_prime_expanded = an_prime[:, np.newaxis]
        # Compute flow angle
        phi = np.arctan((1 - an) * u_new/ ((1 + an_prime) * w_new))
        phi_n= phi[:, np.newaxis]  # Expand phi to shape (50, 1)
        # Compute lift and drag coefficients
        Cn = interpolated_cl * np.cos(phi_n) - interpolated_cd * np.sin(phi_n)
        Ct = interpolated_cl * np.sin(phi_n) + interpolated_cd * np.cos(phi_n)
        sigma_n = sigma[:, np.newaxis]  # Expand sigma to shape (50, 1)
        # Compute axial induction factor (a) and tangential induction factor (a')
        a_new = 1 / (4 * (np.sin(phi_n) ** 2) / (sigma_n * Cn) + 1)
        a_prime_new = 1 / (4 * np.sin(phi_n) * np.cos(phi_n) / (sigma_n * Ct) - 1)

        # Check for convergence
        if np.all(np.abs(a_new - an_expanded) < tolerance) and np.all(np.abs(a_prime_new - an_prime_expanded) < tolerance):
            break

        # Update a and a'
        an = a_new[:, 0]
        an_prime = a_prime_new[:, 0]

    return an, an_prime


##### concerns ####
"""
1. where should I and why calculate the step 3 for alpha angle


2. in computeing the a and a' i have used interpolated w and u with retrospect to r is that alright 



"""