"""
This module contains the TurbineData class, which handles the loading
of wind turbine geometry and operational strategy data.
"""

from pathlib import Path
import numpy as np


class TurbineData:
    """
    A class to handle the loading of wind turbine
    geometry and operational strategy data.

    Attributes:
        geometry_path (str): Path to the geometry file.
        operational_strategy_path (str): Path to the operational strategy file.
        geometry_data (dict): Dictionary containing
        geometry data (blade_span_positions, twist,
        chord_lengths, airfoil_indices).
        operational_strategy_data (dict): Dictionary
        containing operational strategy data
        (wind_speed, angle_pitch, rotational_speed, power, torque).
    """

    def __init__(self, geometry_path, operational_strategy_path):
        """
        Initialize the WindTurbineData object with
        paths to the geometry and operational strategy files.

        Parameters:
            geometry_path (str): Path to the geometry file.
            operational_strategy_path (str):
            Path to the operational strategy file.
        """
        self.geometry_path = Path(geometry_path)
        self.operational_strategy_path = Path(operational_strategy_path)
        self.geometry_data = None
        self.operational_strategy_data = None

    def load_geometry(self):
        """
        Load the geometry of the wind turbine blade.

        Returns:
            dict: A dictionary containing blade span
            positions (blade_span_positions),
            number of blades (num_blades), chord lengths
            (chord_lengths), and airfoil indices (airfoil_indices).
        """
        data = np.genfromtxt(self.geometry_path, delimiter='', skip_header=6)
        blade_span_positions, twist_angle, chord_lengths, airfoil_indices = (
            data[:, 0], data[:, 4], data[:, 5], data[:, 6]
        )
        self.geometry_data = {
            "blade_span_positions": blade_span_positions,
            "twist_angle": twist_angle,
            "chord_lengths": chord_lengths,
            "airfoil_indices": airfoil_indices,
        }
        return self.geometry_data

    def load_operational_strategy(self):
        """
        Load the operational strategy of the wind turbine.

        Returns:
            dict: A dictionary containing wind speed (wind_speed),
            angle pitch (angle_pitch), rotational speed (rotational_speed),
            power (power), and torque (torque).
        """
        data = np.genfromtxt(self.operational_strategy_path,
                             delimiter='', skip_header=1)
        wind_speed, angle_pitch = data[:, 0], data[:, 1]
        rotational_speed, power, torque = data[:, 2], data[:, 3], data[:, 4]
        self.operational_strategy_data = {
            "wind_speed": wind_speed,
            "angle_pitch": angle_pitch,
            "rotational_speed": rotational_speed,
            "power": power,
            "torque": torque,
        }
        return self.operational_strategy_data