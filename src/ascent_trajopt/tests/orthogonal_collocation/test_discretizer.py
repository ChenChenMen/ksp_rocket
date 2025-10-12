"""Test the discretizer module."""

import pytest
import numpy as np

from ascent_trajopt.orthogonal_collocation.discretizer import HPDiscretizer, HPSegmentConfig


def test_invalid_segment_end_time():
    """Test that invalid segment end times raise ValueError."""
    with pytest.raises(ValueError):  # End time decreases
        HPDiscretizer(segment_scheme=[(3, 0.5), (4, 0.4)])

    with pytest.raises(ValueError):  # End time exceeds 1
        HPDiscretizer(segment_scheme=[(3, 0.5), (4, 1.2)])


def test_valid_discretization():
    """Test valid discretization."""
    discretizer = HPDiscretizer(segment_scheme=(HPSegmentConfig(3, 0.5), HPSegmentConfig(4, 1.0)))
    expected_points = np.asarray([0, 0.17752551, 0.42247449, 0.5, 0.60617027, 0.79526657, 0.95570602])
    assert np.allclose(discretizer.discretized_point_array, expected_points)


def test_get_interpolator_for_segment():
    """Test the get_interpolator_for_segment method."""
    discretizer = HPDiscretizer(segment_scheme=(HPSegmentConfig(3, 0.5), HPSegmentConfig(4, 1.0)))
    interpolator = discretizer.get_interpolator_for_segment(0)
    # The interpolator should have interpolation points equal to the first segment's points
    expected_points = np.asarray(
        [[[-8, 9.715476, -1.715476], [-3.265986, 1.55051, 1.715476], [3.265986, -9.715476, 6.44949]]]
    )
    np.testing.assert_allclose(interpolator.differentiation_matricies, expected_points, rtol=1e-5)


def test_num_points_and_discretized_point_collection():
    """Test num_points and discretized_point_collection attributes."""
    segment_scheme = (HPSegmentConfig(2, 0.3), HPSegmentConfig(3, 0.7), HPSegmentConfig(4, 1.0))
    discretizer = HPDiscretizer(segment_scheme=segment_scheme)
    assert discretizer.total_num_points == sum(seg.n_points for seg in segment_scheme)
    assert len(discretizer.discretized_point_collection) == len(segment_scheme)
    for idx, seg in enumerate(segment_scheme):
        assert len(discretizer.discretized_point_collection[idx]) == seg.n_points


def test_get_end_time_for_segment():
    """Test the get_end_time_for_segment method."""
    discretizer = HPDiscretizer(segment_scheme=(HPSegmentConfig(3, 0.5), HPSegmentConfig(4, 1.0)))
    assert np.isclose(discretizer.get_end_time_for_segment(segment_index=0), 0.5)
    assert np.isclose(discretizer.get_end_time_for_segment(segment_index=1), 1.0)
