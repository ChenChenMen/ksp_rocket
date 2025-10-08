"""Test orthogonal polynomial implementation."""

import numpy as np
import pytest

from ascent_trajopt.orthogonal_collocation.polynomials import get_legendre_gauss_radau_points


@pytest.mark.parametrize("num_points", [2, 3, 4, 5, 6])
def test_legendre_gauss_radau_points(num_points):
    """Test the Legendre-Gauss-Radau points generation."""
    points = get_legendre_gauss_radau_points(num_points)
    # Check that the points are within the interval [-1, 1]
    assert np.all(points >= -1) and np.all(points <= 1)
    # Check that the first point is -1
    assert np.isclose(points[0], -1)
    # Check that the points are sorted
    assert np.all(np.diff(points) > 0)
