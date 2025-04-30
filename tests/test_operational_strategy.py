import sys
import os
import numpy as np
import pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import compute_optimal_strategy

def test_compute_optimal_strategy():
    operational_strategy_path = "inputs/IEA-15-240-RWT/IEA_15MW_RWT_Onshore.opt"
    
    # Test within wind speed range
    pitch, omega = compute_optimal_strategy(10.0, operational_strategy_path)
    assert pitch >= 0
    assert omega >= 0

    # Test below minimum wind speed (expect ValueError)
    with pytest.raises(ValueError):
        pitch, omega = compute_optimal_strategy(-5.0, operational_strategy_path)

    # Test above maximum wind speed
    pitch, omega = compute_optimal_strategy(50.0, operational_strategy_path)
    assert pitch >= 0
    assert omega >= 0
