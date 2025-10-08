"""Test the array store module."""

import numpy as np
import pytest

from ascent_trajopt.orthogonal_collocation.array_store import (
    OptimizationArray,
    DynamicVariablesArray,
    DynamicSystemDimension,
)
from ascent_trajopt.orthogonal_collocation.discretizer import HPDiscretizer, HPSegmentConfig


@pytest.mark.parametrize(
    "num_points, end_times",
    [
        ([3, 4], [0.5, 1.0]),
        ([2, 3, 4], [0.3, 0.7, 1.0]),
    ],
)
def test_optimization_array_creation(num_points, end_times):
    """Test the creation of an OptimizationArray."""
    # Create a sample discretizer and dimension
    dimension = DynamicSystemDimension(num_state=7, num_control=4)
    discretizer = HPDiscretizer(
        segment_scheme=tuple(HPSegmentConfig(n_points, end_time) for n_points, end_time in zip(num_points, end_times))
    )

    # Create a sample optimization vector
    optimization_vector = np.random.rand(sum(num_points) * (dimension.num_state + dimension.num_control) + 1)
    opt_array = OptimizationArray(optimization_vector, discretizer, dimension)

    # Check that the array is created correctly
    assert opt_array.size == sum(num_points) * (dimension.num_state + dimension.num_control) + 1

    # Check point retrieval
    for point_index in range(discretizer.total_num_points):
        point = opt_array.point(point_index)
        assert isinstance(point, DynamicVariablesArray)
        assert point.state.size == dimension.num_state
        assert point.control.size == dimension.num_control

        total_point_idx = point_index * (dimension.num_state + dimension.num_control)
        assert np.allclose(point.state, optimization_vector[total_point_idx : total_point_idx + dimension.num_state])
        assert np.allclose(
            point.control,
            optimization_vector[
                total_point_idx + dimension.num_state : total_point_idx + dimension.num_state + dimension.num_control
            ],
        )

    # Check iteration over segments
    for segment_index in range(discretizer.total_num_segments):
        segment_points = opt_array.segment(segment_index)
        expected_num_points = num_points[segment_index]
        assert len(segment_points) == expected_num_points
        for point in segment_points:
            assert isinstance(point, DynamicVariablesArray)
            assert point.state.size == dimension.num_state
            assert point.control.size == dimension.num_control
