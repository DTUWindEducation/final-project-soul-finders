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
    print("Blade_span", r)
    print("twist_angle", B)
    print("chord_length", c)
    print("Airfoil index", Ai)
    return r, B, c, Ai


# 2. Load the operational strategy of the wind turbine
def load_operational_strategy(path_operational_strategy):
    """
    Load the operational strategy of the wind turbine"
    """
    path_operational_strategy = Path(path_operational_strategy)
    data = np.genfromtxt(path_operational_strategy, delimiter='', skip_header=1)
    u, a, w, P, T = data[:, 0], data[:, 1], data[:, 2], data[:, 3], data[:, 4]
    print("Wind speed", u)
    print("Blade pitch angle", a)
    print("Rotational speed", w)
    print("Power", P)
    print("Thrust", T)
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



# 6. interpolate 2D airfoil polar data CL
def interpolate_cl_2d(alpha_values, polar_files, path_geometry):
    """
    Interpolates Cl as a function of alpha and blade span using 2D interpolation.

    Parameters:
        alpha_values (array-like): Array of alpha values to interpolate.
        polar_files_dir (str): Directory containing the polar files.
        path_geometry (str): Path to the blade geometry file.

    Returns:
        interpolated_cl (2D array): Interpolated Cl values for the given alpha and blade span.
        alpha_grid (2D array): Grid of alpha values used for interpolation.
        blspn_grid (2D array): Grid of blade span values used for interpolation.
    """
    # Load blade span data
    r, _, _, _ = load_geometry(path_geometry)

    # Initialize storage for Cl data
    cl_data = []
    blspn_positions = []

    # Loop through polar files and extract Cl data
    for i, file_name in enumerate(sorted(os.listdir(polar_files))):
        file_path = os.path.join(polar_files, file_name)
        if os.path.isfile(file_path) and file_name.endswith(".dat"):
            alpha, cl, _ = load_airfoil_polar(file_path)
            cl_data.append(cl)
            if i < len(r):  # Ensure `i` does not exceed the length of `r`
                blspn_positions.append(r[i])  # Associate polar file with corresponding blade span position
            else:
                print(f"Warning: No blade span position for polar file {file_name}")

    # Create a grid for alpha and blade span
    alpha_grid, blspn_grid = np.meshgrid(alpha_values, blspn_positions)

    # Initialize the interpolated_cl array with the correct shape
    interpolated_cl = np.zeros((len(blspn_positions), len(alpha_values)))

    # Interpolate Cl values
    for i, cl in enumerate(cl_data):
        interp_func = np.interp(alpha_values, alpha, cl)  # 1D interpolation for each blade span
        interpolated_cl[i, :] = interp_func

    return interpolated_cl, alpha_grid, blspn_grid



# 7. interpolate 2D airfoil polar data CD

def interpolate_cd_2d(alpha_values, polar_files, path_geometry):
    """
    Interpolates Cl as a function of alpha and blade span using 2D interpolation.

    Parameters:
        alpha_values (array-like): Array of alpha values to interpolate.
        polar_files_dir (str): Directory containing the polar files.
        path_geometry (str): Path to the blade geometry file.

    Returns:
        interpolated_cl (2D array): Interpolated Cl values for the given alpha and blade span.
        alpha_grid (2D array): Grid of alpha values used for interpolation.
        blspn_grid (2D array): Grid of blade span values used for interpolation.
    """
    # Load blade span data
    r, _, _, _ = load_geometry(path_geometry)

    # Initialize storage for Cl data
    cd_data = []
    blspn_positions = []

    # Loop through polar files and extract Cl data
    for i, file_name in enumerate(sorted(os.listdir(polar_files))):
        file_path = os.path.join(polar_files, file_name)
        if os.path.isfile(file_path) and file_name.endswith(".dat"):
            alpha, _, cd = load_airfoil_polar(file_path)
            cd_data.append(cd)
            if i < len(r):  # Ensure `i` does not exceed the length of `r`
                blspn_positions.append(r[i])  # Associate polar file with corresponding blade span position
            else:
                print(f"Warning: No blade span position for polar file {file_name}")

    # Create a grid for alpha and blade span
    alpha_grid, blspn_grid = np.meshgrid(alpha_values, blspn_positions)

    # Initialize the interpolated_cl array with the correct shape
    interpolated_cd = np.zeros((len(blspn_positions), len(alpha_values)))

    # Interpolate Cl values
    for i, cd in enumerate(cd_data):
        interp_func = np.interp(alpha_values, alpha, cd)  # 1D interpolation for each blade span
        interpolated_cd[i, :] = interp_func

    return interpolated_cd, alpha_grid, blspn_grid


# 8 create a 2D table for interpolated Cl values
def create_cl_table(interpolated_cl, alpha_values, blspn_positions):
    """
    Creates a 2D table for interpolated Cl values.

    Parameters:
        interpolated_cl (2D array): Interpolated Cl values.
        alpha_values (array-like): Array of alpha values.
        blspn_positions (array-like): Array of blade span positions.

    Returns:
        pd.DataFrame: A DataFrame representing the 2D table.
    """
    # Create a DataFrame with blade span as rows and alpha as columns
    cl_table = pd.DataFrame(
        interpolated_cl,
        index=blspn_positions,
        columns=alpha_values
    )
    cl_table.index.name = "Blade Span (r)"
    cl_table.columns.name = "Angle of Attack (α)"
    return cl_table


# 9 create a 2D table for interpolated Cd values
def create_cd_table(interpolated_cd, alpha_values, blspn_positions):
    """
    Creates a 2D table for interpolated Cl values.

    Parameters:
        interpolated_cl (2D array): Interpolated Cl values.
        alpha_values (array-like): Array of alpha values.
        blspn_positions (array-like): Array of blade span positions.

    Returns:
        pd.DataFrame: A DataFrame representing the 2D table.
    """
    # Create a DataFrame with blade span as rows and alpha as columns
    cd_table = pd.DataFrame(
        interpolated_cd,
        index=blspn_positions,
        columns=alpha_values
    )
    cd_table.index.name = "Blade Span (r)"
    cd_table.columns.name = "Angle of Attack (α)"
    return cd_table



def sigma_calc(r, B, c):
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
    sigma = lambda r: B / (2 * np.pi * r) * c
    return sigma


# 10 computing a and a'

def compute_a_s(r, u, w, B, interpolated_cl, interpolated_cd, sigma, tolerance=1e-6, max_iter=100):
    """
    Compute the axial induction factor (a) and the tangential induction factor (a').

    Parameters:
        r (array-like): Blade span positions.
        u (array-like): Wind speed.
        w (float): Rotational speed.
        B (float): Number of blades.
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

    for iteration in range(max_iter):
        # Compute flow angle
        phi = np.arctan((1 - an) * u / ((1 + an_prime) * w))
        # Compute angle of attack
        alpha = phi - (w + B)
        # Compute lift and drag coefficients
        Cn = interpolated_cl * np.cos(phi) - interpolated_cd * np.sin(phi)
        Ct = interpolated_cl * np.sin(phi) + interpolated_cd * np.cos(phi)

        # Compute axial induction factor (a) and tangential induction factor (a')
        a_new = 1 / (4 * (np.sin(phi) ** 2) / (sigma * Cn) + 1)
        a_prime_new = 1 / (4 * np.sin(phi) * np.cos(phi) / (sigma * Ct) - 1)

        # Check for convergence
        if np.all(np.abs(a_new - an) < tolerance) and np.all(np.abs(a_prime_new - an_prime) < tolerance):
            break

        # Update a and a'
        an = a_new
        an_prime = a_prime_new

    return an, an_prime

