"""Test the interpolator functionality."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from spacegnc.interpolator import BarycentricInterpolator, LinearInterpolator


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

        interpolated_x = np.atleast_2d(interpolated_point_value[0])
        interpolated_y = np.atleast_2d(interpolated_point_value[1])
        if interpolated_y.shape[0] > 1 and interpolated_x.shape[0] == 1:
            interpolated_x = np.repeat(interpolated_x, interpolated_y.shape[0], axis=0)
        for idx, (interpolated_xi, interpolated_yi) in enumerate(zip(interpolated_x, interpolated_y)):
            plt.scatter(interpolated_xi, interpolated_yi, color="red", label=f"Interpolation Points {idx}")

        # Plot the interpolation results
        sampled_x = np.atleast_2d(sampled_point_value[0])
        sampled_y = np.atleast_2d(sampled_point_value[1])
        if sampled_y.shape[0] > 1 and sampled_x.shape[0] == 1:
            sampled_x = np.repeat(sampled_x, sampled_y.shape[0], axis=0)
        for idx, (sampled_xi, sampled_yi) in enumerate(zip(sampled_x, sampled_y)):
            plt.plot(sampled_xi, sampled_yi, color="blue", label=f"Interpolated {idx}")

        if expected_value is not None:
            expected_value = np.atleast_2d(expected_value)
            for idx, (sampled_xi, expected_valuei) in enumerate(zip(sampled_x, expected_value)):
                plt.scatter(sampled_xi, expected_valuei, color="green", label=f"Expected {idx}")

        plt.legend()
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

    def test_value_at_interpolation_weight(self):
        """Test weights_for_value_at method with interval of [-1, 1]."""
        expected_func = np.arctan
        number_of_points = 20
        # Generate Chebyshev points and corresponding values for a test function
        chebyshev_points = np.cos(np.pi * np.arange(number_of_points + 1) / number_of_points)
        values = expected_func(chebyshev_points)
        interpolator = BarycentricInterpolator(chebyshev_points, values)

        weight_array = interpolator.weights_for_value_at(0.1)
        assert np.isclose(weight_array @ values, interpolator.value_at(0.1))
        weight_array = interpolator.weights_for_value_at(1.0)
        assert np.isclose(weight_array @ values, interpolator.value_at(1.0))

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

    def test_with_custom_min_max_bounds(self):
        """Test construction of an interpolator with custom min/max bounds."""
        points, values = np.asarray([0.0, 1.0, 2.0]), np.asarray([1.0, 4.0, 9.0])
        interpolator = BarycentricInterpolator(points, values, min_bound=-1, max_bound=5)
        assert np.isclose(interpolator.value_at(-1.0), 0)
        assert np.isclose(interpolator.value_at(-0.5), 0.25)
        assert np.isclose(interpolator.value_at(0.0), 1.0)
        assert np.isclose(interpolator.value_at(2.0), 9.0)
        assert np.isclose(interpolator.value_at(4.0), 25.0)
        assert np.isclose(interpolator.value_at(5.0), 36.0)

    def test_multiple_sample_points_and_values(self, show_plots):
        """Test value_at method with multiple sample points and values."""
        number_of_points = 20
        # Generate Chebyshev points and corresponding values for a test function
        chebyshev_points = 1 + 2 * np.cos(np.pi * np.arange(number_of_points + 1) / number_of_points)
        values = np.asarray([np.sin(chebyshev_points), np.cos(chebyshev_points)])
        interpolator = BarycentricInterpolator(chebyshev_points, values)

        # Test interpolation at the sample points
        sample_points = np.linspace(-1, 3, 100)
        expected_values = np.asarray([np.sin(sample_points), np.cos(sample_points)])
        # Interpolated values at points
        interpolated_values = interpolator.value_at(sample_points)
        interpolation_point_value = (chebyshev_points, values)
        sample_point_value = (sample_points, interpolated_values)
        self._plot_interpolation(interpolation_point_value, sample_point_value, expected_values, show_plots)
        assert np.allclose(interpolated_values, expected_values, atol=1e-1)

        # Interpolated derivative values at points
        expected_deriv_values = np.asarray([np.cos(sample_points), -np.sin(sample_points)])
        interpolated_deriv_values = interpolator.deriv_at(sample_points)
        assert np.allclose(interpolated_deriv_values, expected_deriv_values, atol=1e-8)


class TestLinearInterpolator:
    """Test class for LinearInterpolator."""

    def test_invalid_initialization(self):
        """Test initialization with mismatched points and values."""
        points, values = np.asarray([0.0, 1.0]), np.asarray([1.0, 2.0, 3.0])
        with pytest.raises(ValueError):
            LinearInterpolator(points, values)

    def test_value_at(self):
        """Test value_at method."""
        points, values = np.asarray([0.0, 1.0, 2.0]), np.asarray([1.0, 4.0, 9.0])
        interpolator = LinearInterpolator(points, values)
        assert np.isclose(interpolator.value_at(0.5), 2.5)
        assert np.isclose(interpolator.value_at(1.5), 6.5)
        assert np.isclose(interpolator.value_at(0.0), 1.0)
        assert np.isclose(interpolator.value_at(2.0), 9.0)

    def test_multiple_interpolation_values(self):
        """Test value_at method with multiple interpolation values."""
        points = np.asarray([0.0, 1.0, 2.0])
        values = np.asarray([[1.0, 2.0, 3.0], [1.0, 4.0, 9.0]])
        interpolator = LinearInterpolator(points, values)
        assert np.allclose(interpolator.value_at(0.5), [1.5, 2.5])
        assert np.allclose(interpolator.value_at(1.5), [2.5, 6.5])
        assert np.allclose(interpolator.value_at(0.0), [1.0, 1.0])
        assert np.allclose(interpolator.value_at(2.0), [3.0, 9.0])

    def test_multiple_sample_points(self):
        """Test value_at method with multiple sample points."""
        points = np.asarray([0.0, 1.0, 2.0])
        values = np.asarray([1.0, 4.0, 9.0])
        interpolator = LinearInterpolator(points, values)
        sample_points = np.asarray([0.5, 1.5])
        expected_values = np.asarray([2.5, 6.5])
        assert np.allclose(interpolator.value_at(sample_points), expected_values)

    def test_multiple_sample_points_and_values(self):
        """Test value_at method with multiple sample points and values."""
        points = np.asarray([0.0, 1.0, 2.0])
        values = np.asarray([[1.0, 2.0, 3.0], [1.0, 4.0, 9.0]])
        interpolator = LinearInterpolator(points, values)
        sample_points = np.asarray([0.5, 1.5])
        expected_values = np.asarray([[1.5, 2.5], [2.5, 6.5]])
        assert np.allclose(interpolator.value_at(sample_points), expected_values)

    def test_multiple_interpolation_points_and_values(self):
        """Test value_at method with multiple interpolation points and values."""
        points = np.asarray([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]])
        values = np.asarray([[1.0, 2.0, 3.0], [1.0, 4.0, 9.0]])
        interpolator = LinearInterpolator(points, values)
        sample_points = np.asarray([[0.5, 1.5], [0.5, 1.5]])
        expected_values = np.asarray([[1.5, 2.5], [2.5, 6.5]])
        assert np.allclose(interpolator.value_at(sample_points), expected_values)
