"""Test constant module."""

import numpy as np

from rocket_util.constant import G0, MU_E, GC, RADIUS_E, ConstantsProvider
from rocket_util.scaler import Scaler, ScalingInput


class ExampleConstants(ConstantsProvider):
    """Example constants for testing purposes."""

    def __init__(self, scaler: Scaler = None):
        """Initialize example constants with an optional scaler."""
        super().__init__(scaler=scaler)
        self._stored_constant = {
            "G0": G0,
            "MU_E": MU_E,
            "GC": GC,
            "RADIUS_E": RADIUS_E,
        }


class TestConstantsProvider:
    """Test class for ConstantsProvider."""

    def test_constants_no_scalar(self):
        """Test initialization of constants without scaling."""
        constants = ExampleConstants()
        assert np.isclose(constants.G0, G0.m)
        assert np.isclose(constants.MU_E, MU_E.m)
        assert np.isclose(constants.GC, GC.m)
        assert np.isclose(constants.RADIUS_E, RADIUS_E.m)

    def test_constants_with_scaler(self):
        """Test initialization of constants with a scaler."""
        scaler = Scaler(scaling_input=ScalingInput((G0, MU_E, GC)))
        scaled_constants = ExampleConstants(scaler)
        assert np.isclose(scaled_constants.G0, 0.1019716213)
        assert np.isclose(scaled_constants.MU_E, 1)
        assert np.isclose(scaled_constants.GC, 1)
        assert np.isclose(scaled_constants.RADIUS_E, 1.0004273)
