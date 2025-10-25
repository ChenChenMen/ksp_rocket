"""Test the array store module."""

import numpy as np
import pytest

from ascent_trajopt.dynamics.array_store import DynamicModelDimension, DynamicVariablesArray
from ascent_trajopt.orthogonal_collocation.array_store import OptimizationArray
from ascent_trajopt.orthogonal_collocation.discretizer import HPDiscretizer, HPSegmentConfig


@pytest.mark.parametrize(
    "num_points, end_times",
    [
        ([5], [1.0]),
        ([3, 4], [0.5, 1.0]),
        ([2, 3, 4], [0.3, 0.7, 1.0]),
    ],
)
def test_optimization_array_creation(num_points, end_times, input_components):
    """Test the creation of an OptimizationArray."""
    # Create a sample discretizer and dimension
    dimension = DynamicModelDimension.from_dynamic_model(input_components.dynamics_model)
    discretizer = HPDiscretizer(
        segment_scheme=tuple(HPSegmentConfig(n_points, end_time) for n_points, end_time in zip(num_points, end_times))
    )

    # Create a sample optimization vector
    optimization_vector = np.random.rand(sum(num_points) * dimension.total_dimension + 1)
    optimization_array = OptimizationArray(optimization_vector, discretizer=discretizer, dimension=dimension)

    # Check that the array is created correctly
    assert optimization_array.size == sum(num_points) * dimension.total_dimension + 1

    # Check point retrieval
    for point_index in range(discretizer.total_num_points):
        point = optimization_array.point(point_index)
        assert isinstance(point, DynamicVariablesArray)
        assert point.state.size == dimension.num_state
        assert point.control.size == dimension.num_control

        total_point_idx = point_index * dimension.total_dimension
        assert np.allclose(point.state, optimization_vector[total_point_idx : total_point_idx + dimension.num_state])
        assert np.allclose(
            point.control,
            optimization_vector[total_point_idx + dimension.num_state : total_point_idx + dimension.total_dimension],
        )

    # Check iteration over segments
    for segment_index in range(discretizer.total_num_segments):
        segment_points = optimization_array.segment(segment_index)
        expected_num_points = num_points[segment_index]
        assert len(segment_points) == expected_num_points
        for point in segment_points:
            assert isinstance(point, DynamicVariablesArray)
            assert point.state.size == dimension.num_state
            assert point.control.size == dimension.num_control


def test_optimization_array_time_property(optimization_array):
    """Test time property retrieval from optimization array."""
    assert np.isclose(optimization_array.time, optimization_array[-1])


def test_optimization_array_segment_point_slice(optimization_array):
    """Test segment point index slice retrieval."""
    discretizer = optimization_array.discretizer
    dimension = optimization_array.dimension

    for segment_index in range(discretizer.total_num_segments):
        start_index, end_index = OptimizationArray.segment_point_index_slice(discretizer, dimension, segment_index)

        # Verify that the indices correspond to the correct points
        start_point_index, end_point_index = OptimizationArray.segment_index_slice(discretizer, segment_index)
        expected_start_index, _ = OptimizationArray.point_index_slice(discretizer, dimension, start_point_index)
        _, expected_end_index = OptimizationArray.point_index_slice(discretizer, dimension, end_point_index - 1)

        assert start_index == expected_start_index
        assert end_index == expected_end_index

        segment_array = optimization_array.segment_as_ndarray(segment_index)
        segment_points = optimization_array.segment(segment_index)
        reconstructed_segment_array = np.hstack([point.view(np.ndarray) for point in segment_points])
        assert np.allclose(segment_array, reconstructed_segment_array)
