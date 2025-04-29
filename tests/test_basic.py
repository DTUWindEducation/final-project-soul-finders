import pytest
from src import load_airfoil_shape, load_airfoil_polar

def test_load_airfoil_shape():
    # A simple test to check if airfoil loading works (dummy file)
    try:
        x, y = load_airfoil_shape("inputs/IEA-15-240-RWT/Airfoils/cord_files/IEA-15-240-RWT_AF00_Coords.txt")
        assert len(x) > 0
        assert len(y) > 0
    except Exception as e:
        pytest.fail(f"Loading airfoil shape failed: {e}")

def test_load_airfoil_polar():
    try:
        alpha, cl, cd = load_airfoil_polar("inputs/IEA-15-240-RWT/Airfoils/polar_files/IEA-15-240-RWT_AeroDyn15_Polar_00.dat")
        assert len(alpha) > 0
        assert len(cl) > 0
        assert len(cd) > 0
    except Exception as e:
        pytest.fail(f"Loading airfoil polar failed: {e}")
