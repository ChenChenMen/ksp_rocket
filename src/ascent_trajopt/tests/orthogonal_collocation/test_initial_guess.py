"""Test the initial guess generation for orthogonal collocation."""

import numpy as np
import pytest

from ascent_trajopt.dynamics.array_store import DynamicModelDimension
from ascent_trajopt.orthogonal_collocation.discretizer import HPDiscretizer, HPSegmentConfig
from ascent_trajopt.orthogonal_collocation.initial_guess import guess_from_linear_interpolation


@pytest.mark.parametrize(
    "num_points, end_times",
    [
        ([5], [1.0]),
        ([3, 4], [0.5, 1.0]),
        ([2, 3, 4], [0.3, 0.7, 1.0]),
    ],
)
def test_initial_guess_linear_interpolation(num_points, end_times, input_components):
    """Test the initial guess generation using linear interpolation."""
    # Create the test case discretizer and dimension
    dimension = DynamicModelDimension.from_dynamic_model(input_components.dynamics_model)
    discretizer = HPDiscretizer(
        segment_scheme=tuple(HPSegmentConfig(n_points, end_time) for n_points, end_time in zip(num_points, end_times))
    )
    # Replace the original discretizer in input components for the test case
    input_components.discretizer = discretizer

    # Generate the initial guess
    initial_guess = guess_from_linear_interpolation(problem_input=input_components)
    # Check the shape of the generated guess
    assert initial_guess.size == discretizer.total_num_points * dimension.total_dimension + 1

    initial_state, initial_control = input_components.initial_condition.state, input_components.initial_condition.control
    final_state, final_control = input_components.final_condition.state, input_components.final_condition.control

    # Check linearity of interpolation for a few intermediate points
    for point_idx, relative_time in enumerate(discretizer.discretized_point_array):
        expected_state = (1 - relative_time) * initial_state + relative_time * final_state
        expected_control = (1 - relative_time) * initial_control + relative_time * final_control

        point = initial_guess.point(point_idx)
        assert np.allclose(point.state, expected_state)
        assert np.allclose(point.control, expected_control)
