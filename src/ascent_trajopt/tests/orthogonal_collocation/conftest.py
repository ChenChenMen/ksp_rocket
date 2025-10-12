"""Define common fixtures for orthogonal collocation tests."""

import numpy as np
import pytest

from ascent_trajopt.dynamics.array_store import DynamicVariablesArray, DynamicModelDimension
from ascent_trajopt.dynamics.pendulum import PendulumCartDynamicsModel
from ascent_trajopt.orthogonal_collocation.discretizer import HPDiscretizer, HPSegmentConfig
from ascent_trajopt.orthogonal_collocation.initial_guess import guess_from_linear_interpolation


@pytest.fixture
def dynamics_model():
    """Create a pendulum cart dynamic model."""
    yield PendulumCartDynamicsModel()


@pytest.fixture
def discretizer():
    """Create a discretizer instance."""
    segment_scheme = (HPSegmentConfig(5, 1.0),)
    yield HPDiscretizer(segment_scheme)


@pytest.fixture
def initial_condition(dynamics_model):
    """Create an initial condition."""
    dimension = DynamicModelDimension.from_dynamic_model(dynamics_model)
    yield DynamicVariablesArray([0.0, 0.0, 0.0, 0.0, 0.0], dimension)


@pytest.fixture
def final_condition(dynamics_model):
    """Create a final condition."""
    dimension = DynamicModelDimension.from_dynamic_model(dynamics_model)
    yield DynamicVariablesArray([np.pi, 0.0, 1.0, 0.0, 0.0], dimension)


@pytest.fixture
def optimization_array(discretizer, initial_condition, final_condition):
    """Create an optimization array with initial guess."""
    yield guess_from_linear_interpolation(discretizer, initial_condition, final_condition)
