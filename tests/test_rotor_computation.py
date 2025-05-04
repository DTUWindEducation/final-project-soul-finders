import numpy as np
import pytest
from src.compute import (compute_rotor_thrust_torque_power, sigma_calc, compute_a_s)

def test_compute_rotor_thrust_torque_power():
    # Blade geometry
    r = np.linspace(5, 100, 10)  # 10 elements along the blade span
    B = 3
    c = np.ones_like(r) * 3.0  # constant chord
    Ai = np.zeros_like(r)  # dummy airfoil indices
    geometry = (r, B, c, Ai)

    # Alpha angle range
    alpha_range = np.linspace(-180, 180, 100)

    # Fake aerodynamic data for testing
    cl_data = np.tile(np.linspace(0.5, 1.0, 100), (10, 1))  # shape (10, 100)
    cd_data = np.tile(np.linspace(0.01, 0.05, 100), (10, 1))

    # Operation point
    u_new = 10.0  # m/s
    p = 2.0       # degrees
    ω = 12.0      # rpm

    # Run the function
    result = compute_rotor_thrust_torque_power(
    u_new=[u_new],
    p=[p],
    ω=[ω],
    geometry=geometry,
    cl_data=cl_data,
    cd_data=cd_data,
    rho=1.225
)

    # Assertions to ensure outputs make sense
    assert isinstance(result, dict)
    assert "thrust" in result
    assert "power" in result
    assert "torque" in result

    # Values should be greater than zero for this setup
    assert result["thrust"] > 0
    assert result["power"] > 0
    assert result["torque"] > 0
