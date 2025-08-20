"""Test for scaler module."""

import numpy as np
import pytest

from rocket_util.constant import G0, MU_E, GC
from rocket_util.units import Q_
from rocket_util.scaler import Scaler, ScalingInput


class TestScalingInput:
    """Test the ScalingInput class."""

    def test_scaling_input_initialization(self):
        """Test initialization of ScalingInput with valid reference units."""
        reference_units = [G0, MU_E, GC]
        scaling_input = ScalingInput(reference_unit_conversions=reference_units)
        assert np.allclose(
            scaling_input.get_transformation_matrix(),
            np.asarray([[1, 0, -2], [3, 0, -2], [3, -1, -2]]),
        )


class TestScaler:
    """Test the Scaler class."""

    @pytest.fixture
    def default_scaler(self):
        """Fixture to provide a default scaler instance."""
        reference_units = [G0, MU_E, GC]
        scaling_input = ScalingInput(reference_unit_conversions=reference_units)
        yield Scaler(scaling_input=scaling_input)

    def test_scaler_initialization(self, default_scaler):
        """Test initialization of Scaler with valid ScalingInput."""
        expected_scale_factors = [6375412.79, 5.97216187e24, 806.294722]
        assert all(
            np.isclose(value, expected_scale_factors[i])
            for i, value in enumerate(default_scaler.unit_basis_scale_factor_mapping.values())
        )

    def test_scaler_scale(self, default_scaler):
        """Test scaling of a physical quantity."""
        # Scale a physical quantity
        physical_quantity = Q_(1, "km^2/s^2")
        assert np.isclose(default_scaler.scale(physical_quantity), 1.5994512774331827e-08)

    def test_scaler_unscale(self, default_scaler):
        """Test unscaling of a physical quantity."""
        # Unscale a physical quantity
        assert np.isclose(default_scaler.unscale(1.5994512774331827e-08, "km^2/s^2").m, 1.0)
