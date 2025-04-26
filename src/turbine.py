from pathlib import Path
import numpy as np

class TurbineData:
    """
    A class to handle the loading of wind turbine geometry and operational strategy data.

    Attributes:
        geometry_path (str): Path to the geometry file.
        operational_strategy_path (str): Path to the operational strategy file.
        geometry_data (dict): Dictionary containing geometry data (r, B, c, Ai).
        operational_strategy_data (dict): Dictionary containing operational strategy data (u, a, w, P, T).
    """

    def __init__(self, geometry_path, operational_strategy_path):
        """
        Initialize the WindTurbineData object with paths to the geometry and operational strategy files.

        Parameters:
            geometry_path (str): Path to the geometry file.
            operational_strategy_path (str): Path to the operational strategy file.
        """
        self.geometry_path = Path(geometry_path)
        self.operational_strategy_path = Path(operational_strategy_path)
        self.geometry_data = None
        self.operational_strategy_data = None

    def load_geometry(self):
        """
        Load the geometry of the wind turbine blade.

        Returns:
            dict: A dictionary containing blade span positions (r), number of blades (B), chord lengths (c), and airfoil indices (Ai).
        """
        data = np.genfromtxt(self.geometry_path, delimiter='', skip_header=6)
        r, B, c, Ai = data[:, 0], data[:, 4], data[:, 5], data[:, 6]
        self.geometry_data = {"r": r, "B": B, "c": c, "Ai": Ai}
        return self.geometry_data

    def load_operational_strategy(self):
        """
        Load the operational strategy of the wind turbine.

        Returns:
            dict: A dictionary containing wind speed (u), aangle pitch (p), rotational speed (w), power (P), and torque (T).
        """
        data = np.genfromtxt(self.operational_strategy_path, delimiter='', skip_header=1)
        v, p, w, P, T = data[:, 0], data[:, 1], data[:, 2], data[:, 3], data[:, 4]
        self.operational_strategy_data = {"v": v, "p": p, "w": w, "P": P, "T": T}
        return self.operational_strategy_data
    
    def load_wind_speed(self):
        """
        Load the wind speed data.

        Returns:
            np.ndarray: Array of wind speed values.
        """
        data = np.genfromtxt(self.operational_strategy_path, delimiter='', skip_header=1)
        u = data[:, 0]
        return u