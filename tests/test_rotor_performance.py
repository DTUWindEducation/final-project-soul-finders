import sys
import os
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src import (
    interpolate_2d,
    compute_a_s,
    calculate_rotor_parameters,
    sigma_calc  # <<<<< ADD THIS
)
from src.turbine import TurbineData

def test_rotor_performance():
    # Load real blade geometry
    turbine_data = TurbineData(
        geometry_path="inputs/IEA-15-240-RWT/IEA-15-240-RWT_AeroDyn15_blade.dat",
        operational_strategy_path="inputs/IEA-15-240-RWT/IEA_15MW_RWT_Onshore.opt"
    )
    geometry = turbine_data.load_geometry()
    r = geometry['r']
    c = geometry['c']
    B = 3  # 3 blades

    sigma = sigma_calc(r, c)
    
    alpha_range = np.linspace(-180, 180, 100)
    polar_dir = "inputs/IEA-15-240-RWT/Airfoils/polar_files"

    cl_data, _, _ = interpolate_2d(alpha_range, polar_dir, r, data_type="cl")
    cd_data, _, _ = interpolate_2d(alpha_range, polar_dir, r, data_type="cd")

    # Pick operating point
    v = [12.0]
    p = [2.0]
    w = [6.0]

    _, _, _, an, an_prime, _ = compute_a_s(r, B, alpha_range, cl_data, cd_data, sigma, v, p, w)
    rotor_output = calculate_rotor_parameters(r, w, v, an, an_prime)

    assert rotor_output["power"] > 0
    assert rotor_output["thrust"] > 0
    assert rotor_output["power_coefficient"] < 1
