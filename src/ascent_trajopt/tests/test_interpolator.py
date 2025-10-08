"""Test the interpolator functionality."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from ascent_trajopt.interpolator import BarycentricInterpolator


class TestBarycentricInterpolator:
    """Test class for BarycentricInterpolator."""

    def test_invalid_initialization(self):
        """Test initialization with mismatched points and values."""
        points, values = np.asarray([0.0, 1.0]), np.asarray([1.0, 2.0, 3.0])
        with pytest.raises(ValueError):
            BarycentricInterpolator(points, values)

    def _plot_interpolation(self, interpolated_point_value, sampled_point_value, expected_value=None, show_plots=False):
        """Helper function to plot the interpolation results."""
        if not show_plots:
            return

        # Plot the interpolation results
        plt.scatter(*sampled_point_value, color="blue", label="Interpolation Points")
        plt.scatter(*interpolated_point_value, color="red", label="Interpolated")

        if expected_value is not None:
            plt.scatter(sampled_point_value[0], expected_value, color="green", label="Expected")
            plt.figure()  # Plot error subplot if expected values are provided
            plt.plot(sampled_point_value[0], np.abs(sampled_point_value[1] - expected_value))

        plt.show()

    def test_value_at_with_unit_interval(self, show_plots):
        """Test value_at method with interval of [-1, 1]."""

        def expected_func(array):
            return np.arctan(5 * array)

        number_of_points = 20
        # Generate Chebyshev points and corresponding values for a test function
        chebyshev_points = np.cos(np.pi * np.arange(number_of_points + 1) / number_of_points)
        values = expected_func(chebyshev_points)
        interpolator = BarycentricInterpolator(chebyshev_points, values)

        # Test interpolation at the sample points
        sample_points = np.linspace(-1, 1, 100)
        expected_values = expected_func(sample_points)
        # Interpolated values at points
        interpolated_values = interpolator.value_at(sample_points)

        interp_point_value = (chebyshev_points, values)
        sample_point_value = (sample_points, interpolated_values)
        self._plot_interpolation(interp_point_value, sample_point_value, expected_values, show_plots)
        assert np.allclose(interpolated_values, expected_values, atol=1e-2)

    def test_deriv_at_with_unit_interval(self, show_plots):
        """Test deriv_at method with interval of [-1, 1]."""
        number_of_points = 20
        # Generate Chebyshev points and corresponding values for a test function
        chebyshev_points = np.cos(np.pi * np.arange(number_of_points + 1) / number_of_points)
        values = np.sin(5 * chebyshev_points)
        interpolator = BarycentricInterpolator(chebyshev_points, values)

        # Test interpolation at the sample points
        sample_points = np.linspace(-1, 1, 100)
        expected_values = 5 * np.cos(5 * sample_points)
        # Interpolated values at points
        interpolated_derivs = interpolator.deriv_at(sample_points)

        interp_point_value = (chebyshev_points, values)
        sample_point_value = (sample_points, interpolated_derivs)
        self._plot_interpolation(interp_point_value, sample_point_value, expected_values, show_plots)
        assert np.allclose(interpolated_derivs, expected_values, atol=1e-8)

    def test_value_at_with_custom_points(self, show_plots):
        """Test value_at method with custom points."""
        expected_func = np.arctan
        number_of_points = 20
        # Generate Chebyshev points and corresponding values for a test function
        chebyshev_points = -3 + 10 * np.cos(np.pi * np.arange(number_of_points + 1) / number_of_points)
        values = expected_func(chebyshev_points)
        interpolator = BarycentricInterpolator(chebyshev_points, values)

        # Test interpolation at the sample points
        sample_points = np.linspace(-3, 7, 100)
        expected_values = expected_func(sample_points)
        # Interpolated values at points
        interpolated_values = interpolator.value_at(sample_points)

        interp_point_value = (chebyshev_points, values)
        sample_point_value = (sample_points, interpolated_values)
        self._plot_interpolation(interp_point_value, sample_point_value, expected_values, show_plots)
        assert np.allclose(interpolated_values, expected_values, atol=1e-1)

    def test_deriv_at_with_custom_points(self, show_plots):
        """Test value_at method with custom points."""
        number_of_points = 20
        # Generate Chebyshev points and corresponding values for a test function
        chebyshev_points = -3 + 10 * np.cos(np.pi * np.arange(number_of_points + 1) / number_of_points)
        values = np.sin(chebyshev_points / 2)
        interpolator = BarycentricInterpolator(chebyshev_points, values)

        # Test interpolation at the sample points
        sample_points = np.linspace(-3, 7, 100)
        expected_values = np.cos(sample_points / 2) / 2
        # Interpolated values at points
        interpolated_values = interpolator.deriv_at(sample_points)

        interp_point_value = (chebyshev_points, values)
        sample_point_value = (sample_points, interpolated_values)
        self._plot_interpolation(interp_point_value, sample_point_value, expected_values, show_plots)
        assert np.allclose(interpolated_values, expected_values, atol=1e-8)

    def test_value_out_of_range(self):
        """Test value_at method with a point outside the interpolation range."""
        points, values = np.asarray([0.0, 1.0, 2.0]), np.asarray([1.0, 2.0, 3.0])
        interpolator = BarycentricInterpolator(points, values)
        with pytest.raises(ValueError):
            interpolator.value_at(-1.0)
