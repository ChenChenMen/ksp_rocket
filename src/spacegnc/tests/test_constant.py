"""Test constant module."""

import numpy as np
import pytest

from spacegnc.constant import ScaledConstantProvider
from spacegnc.scaler import Scaler, ScalingInput
from spacegnc.units import Q_

# Standard gravitational acceleration (m/s^2)
G0 = Q_(1, "g0")
# Gravitational constant (m^3 kg^-1 s^-2)
GC = Q_(6.67430e-11, "m^3 kg^-1 s^-2")

## Earth Specific constants
# Standard gravitational parameter (m^3/s^2)
MU_E = Q_(3.986004418e14, "m^3/s^2")
# Earth's radius (m)
RADIUS_E = Q_(6378137, "m")


@pytest.fixture
def with_example_scaler():
    """Fixture that set an universal scaler."""
    scaler = Scaler(scaling_input=ScalingInput((G0, MU_E, GC)))
    ScaledConstantProvider.set_scaler(scaler)
    yield
    ScaledConstantProvider.set_scaler(None)


@pytest.fixture
def example_constant_provider():
    """Fixture to provide a ConstantsProvider instance."""
    yield ScaledConstantProvider(G0=G0, MU_E=MU_E, GC=GC, RADIUS_E=RADIUS_E)


def test_constants_no_scalar(example_constant_provider):
    """Test initialization of constants without scaling."""
    assert np.isclose(example_constant_provider.G0, G0.m)
    assert np.isclose(example_constant_provider.MU_E, MU_E.m)
    assert np.isclose(example_constant_provider.GC, GC.m)
    assert np.isclose(example_constant_provider.RADIUS_E, RADIUS_E.m)


def test_constants_with_scaler(with_example_scaler, example_constant_provider):
    """Test initialization of constants with a scaler."""
    assert np.isclose(example_constant_provider.G0, 0.1019716213)
    assert np.isclose(example_constant_provider.MU_E, 1)
    assert np.isclose(example_constant_provider.GC, 1)
    assert np.isclose(example_constant_provider.RADIUS_E, 1.0004273)
