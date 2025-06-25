"""Test the interpolator functionality."""

import matplotlib.pyplot as plt
import jax.numpy as np
import pytest

from ascent_trajopt.interpolator import BarycentricInterpolator


class TestBarycentricInterpolator:
    """Test class for BarycentricInterpolator."""

    def test_invalid_initialization(self):
        """Test initialization with mismatched points and values."""
        points, values = np.asarray([0.0, 1.0]), np.asarray([1.0, 2.0, 3.0])
        with pytest.raises(ValueError):
            BarycentricInterpolator(points, values)

    def _plot_interpolation(self, interpolated_point_value, sampled_point_value, show_plots):
        """Helper function to plot the interpolation results."""
        if not show_plots:
            return
        plt.scatter(sampled_point_value[0], sampled_point_value[1], color="blue", label="Interpolation Points")
        plt.scatter(interpolated_point_value[0], interpolated_point_value[1], color="red", label="Interpolated")
        plt.show()

    def test_value_at_with_unit_interval(self, show_plots):
        """Test value_at method with interval of [-1, 1]."""
        number_of_points = 20
        # Generate Chebyshev points and corresponding values for a test function
        chebyshev_points = np.cos(np.pi * np.arange(number_of_points + 1) / number_of_points)
        values = np.arctan(5 * chebyshev_points)
        interpolator = BarycentricInterpolator(chebyshev_points, values)

        # Test interpolation at the sample points
        sample_points = np.linspace(-1, 1, 100)
        self._plot_interpolation(
            (chebyshev_points, values), (sample_points, interpolator.value_at(sample_points)), show_plots
        )

    def test_value_at_with_custom_points(self, show_plots):
        """Test value_at method with custom points."""
        number_of_points = 20
        # Generate Chebyshev points and corresponding values for a test function
        chebyshev_points = -3 + 10 * np.cos(np.pi * np.arange(number_of_points + 1) / number_of_points)
        values = np.arctan(chebyshev_points)
        interpolator = BarycentricInterpolator(chebyshev_points, values)

        # Test interpolation at the sample points
        sample_points = np.linspace(-3, 7, 100)
        self._plot_interpolation(
            (chebyshev_points, values), (sample_points, interpolator.value_at(sample_points)), show_plots
        )

    def test_value_out_of_range(self):
        """Test value_at method with a point outside the interpolation range."""
        points, values = np.asarray([0.0, 1.0, 2.0]), np.asarray([1.0, 2.0, 3.0])
        interpolator = BarycentricInterpolator(points, values)
        with pytest.raises(ValueError):
            interpolator.value_at(-1.0)
