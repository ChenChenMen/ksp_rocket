"""Define common fixtures for orthogonal collocation tests."""

import numpy as np
import pytest

from ascent_trajopt.dynamics.array_store import DynamicVariablesArray, DynamicModelDimension
from ascent_trajopt.dynamics.pendulum import PendulumCartDynamicsModel, SinglePendulumDynamicsModel
from ascent_trajopt.orthogonal_collocation.components import ProblemInputComponents
from ascent_trajopt.orthogonal_collocation.discretizer import HPDiscretizer, HPSegmentConfig
from ascent_trajopt.orthogonal_collocation.initial_guess import guess_from_linear_interpolation


"""
Below defines a standard test dynamic model with median complexity for testing.
A pendulum on a cart with 5 state variables and 1 control input.
Discretized with a single segment of 5 collocation points.
"""


@pytest.fixture
def input_components():
    """Create the problem input components."""
    segment_scheme = (HPSegmentConfig(5, 1.0),)
    dynamics_model = PendulumCartDynamicsModel()
    dimension = DynamicModelDimension.from_dynamic_model(dynamics_model)

    yield ProblemInputComponents(
        dynamics_model=dynamics_model,
        discretizer=HPDiscretizer(segment_scheme),
        initial_condition=DynamicVariablesArray([0.0, 0.0, 0.0, 0.0, 0.0], dimension),
        final_condition=DynamicVariablesArray([np.pi, 0.0, 1.0, 0.0, 0.0], dimension),
    )


@pytest.fixture
def optimization_array(input_components):
    """Create an optimization array with initial guess."""
    yield guess_from_linear_interpolation(problem_input=input_components)


"""
Below defines a simplified test dynamic model with low complexity for testing.
A single pendulum with 2 state variables and 1 control inputs.
Discretized with a single segment of 2 collocation points.
"""


@pytest.fixture
def simple_input_components():
    """Create the problem input components."""
    segment_scheme = (HPSegmentConfig(2, 1.0),)
    dynamics_model = SinglePendulumDynamicsModel()
    dimension = DynamicModelDimension.from_dynamic_model(dynamics_model)

    yield ProblemInputComponents(
        dynamics_model=dynamics_model,
        discretizer=HPDiscretizer(segment_scheme),
        initial_condition=DynamicVariablesArray([0.0, 0.0, 0.0], dimension),
        final_condition=DynamicVariablesArray([np.pi, 0.0, 0.0], dimension),
    )


@pytest.fixture
def simple_optimization_array(simple_input_components):
    """Create an optimization array with initial guess."""
    yield guess_from_linear_interpolation(problem_input=simple_input_components)
