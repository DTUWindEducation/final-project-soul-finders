import numpy as np
from pathlib import Path


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
        from io import StringIO
        data = np.genfromtxt(StringIO("".join(data_lines)), delimiter=None)

        # Extract Alpha, Cl, and Cd columns
        alpha, cl, cd = data[:, 0], data[:, 1], data[:, 2]
        return alpha, cl, cd
    except Exception as e:
        print(f"Error loading airfoil polar data: {e}")
        return None, None, None
    

