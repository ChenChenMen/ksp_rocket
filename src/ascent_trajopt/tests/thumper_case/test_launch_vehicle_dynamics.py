"""Test Thumper launch vehicle dynamics model - 3DOF translation only."""

import numpy as np
import pytest

from ascent_trajopt.dynamics.launch_vehicle import TranslationOnlyConstant
from rocket_util.constant import GC, MU_E, RADIUS_E
from rocket_util.scaler import Scaler, ScalingInput


class TestTranslationOnlyConstant:
    """Test suite for TranslationOnlyConstant class."""

    @pytest.fixture
    def scaler(self):
        """Fixture to provide a scaler."""
        yield Scaler(ScalingInput(reference_unit_conversions=(GC, MU_E, RADIUS_E)))

    @pytest.fixture
    def scaled_constant_provider(self, scaler):
        """Fixture to provide a scaled constant provider."""
        yield TranslationOnlyConstant(scaler=scaler)

    def test_atmo_density(self, scaled_constant_provider):
        """Test the initialization of TranslationOnlyConstant."""
        assert np.isclose(scaled_constant_provider.RHO_E(0), scaled_constant_provider.RHO_SEA_LEVEL_E)
        assert np.isclose(scaled_constant_provider.RHO_E(1), 0)
