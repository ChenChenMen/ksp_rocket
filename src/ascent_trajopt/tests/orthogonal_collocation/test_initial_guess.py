"""Test the initial guess generation for orthogonal collocation."""

import numpy as np
import pytest

from ascent_trajopt.orthogonal_collocation.array_store import DynamicVariablesArray
from ascent_trajopt.orthogonal_collocation.discretizer import HPDiscretizer, HPSegmentConfig
from ascent_trajopt.orthogonal_collocation.initial_guess import guess_from_linear_interpolation
from ascent_trajopt.orthogonal_collocation.array_store import DynamicSystemDimension


@pytest.mark.parametrize(
    "num_points, end_times",
    [
        ([3, 4], [0.5, 1.0]),
        ([2, 3, 4], [0.3, 0.7, 1.0]),
    ],
)
def test_initial_guess_linear_interpolation(num_points, end_times):
    """Test the initial guess generation using linear interpolation."""
    # Create a sample discretizer and dimension
    dimension = DynamicSystemDimension(num_state=5, num_control=2)
    discretizer = HPDiscretizer(
        segment_scheme=tuple(HPSegmentConfig(n_points, end_time) for n_points, end_time in zip(num_points, end_times))
    )

    # Define initial and final conditions
    initial_condition = DynamicVariablesArray(
        np.array([1.0] * dimension.num_state + [0.0] * dimension.num_control), dimension
    )
    final_condition = DynamicVariablesArray(
        np.array([10.0] * dimension.num_state + [5.0] * dimension.num_control), dimension
    )

    # Generate the initial guess
    initial_guess = guess_from_linear_interpolation(discretizer, initial_condition, final_condition)

    # Check the shape of the generated guess
    assert initial_guess.size == discretizer.total_num_points * dimension.total_dimension + 1

    # Check linearity of interpolation for a few intermediate points
    for point_idx, relative_time in enumerate(discretizer.discretized_point_array):
        point = initial_guess.point(point_idx)
        expected_state = (1 - relative_time) * initial_condition.state + relative_time * final_condition.state
        expected_control = (1 - relative_time) * initial_condition.control + relative_time * final_condition.control

        assert np.allclose(point.state, expected_state)
        assert np.allclose(point.control, expected_control)
