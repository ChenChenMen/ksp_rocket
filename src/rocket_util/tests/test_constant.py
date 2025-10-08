"""Test constant module."""

import numpy as np
import pytest

from rocket_util.constant import G0, MU_E, GC, RADIUS_E, ScaledConstantProvider
from rocket_util.scaler import Scaler, ScalingInput


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
