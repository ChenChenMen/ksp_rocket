"""Test the discretizer module."""

import pytest
import numpy as np

from ascent_trajopt.discretizer import HPDiscretizer, HPSegment


def test_invalid_segment_end_time():
    """Test that invalid segment end times raise ValueError."""
    with pytest.raises(ValueError):  # End time decreases
        HPDiscretizer(segments=[(3, 0.5), (4, 0.4)])

    with pytest.raises(ValueError):  # End time exceeds 1
        HPDiscretizer(segments=[(3, 0.5), (4, 1.2)])


def test_valid_discretization():
    """Test valid discretization."""
    discretizer = HPDiscretizer(segments=(HPSegment(3, 0.5), HPSegment(4, 1.0)))
    expected_points = np.asarray([0, 0.17752551, 0.42247449, 0.5, 0.60617027, 0.79526657, 0.95570602])
    assert np.allclose(discretizer.all_points, expected_points)
