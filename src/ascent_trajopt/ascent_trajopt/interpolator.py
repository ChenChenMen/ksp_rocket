"""Define polynominal interepolator."""

import jax.numpy as np
from jax.numpy import ndarray


class BarycentricInterpolator:
    """Barycentric Interpolator for trajectory optimization."""

    # Interpolation point accuracy, used to determine if sample points overlapping with
    # interepolation points, if so, value is extracted directly from interpolation values.
    INTERPOLATION_ACCURACY = 1e-8

    def __init__(self, interpolation_points: ndarray, interpolation_values: ndarray):
        """Initialize the interpolator with given points and values."""
        # Check if the input points and values are valid
        if interpolation_points.shape != interpolation_values.shape:
            raise ValueError("Interpolation points and values must have the same shape.")

        # Record bounds of interpolation points for validation during evaluation
        self._min_bound = np.nanmin(interpolation_points)
        self._max_bound = np.nanmax(interpolation_points)

        # Normalize the interpolation points to the interval [-1, 1]
        self.interpolation_points = self._normalize_interval(interpolation_points)
        self.interpolation_values = interpolation_values

        # Compute the barycentric weights for the interpolation points
        self.weights = self._compute_weights()

    def _normalize_interval(self, points: ndarray) -> ndarray:
        """Normalize the interpolation points to the interval [-1, 1]."""
        return 2 * (points - self._min_bound) / (self._max_bound - self._min_bound) - 1

    def _compute_weights(self):
        """Compute the barycentric weights for the interpolation points."""
        diff_matrix = self.interpolation_points[:, None] - self.interpolation_points[None, :]
        diff_matrix = diff_matrix + np.eye(diff_matrix.shape[0])  # Avoid division by zero on diagonal
        weights = 1 / np.prod(diff_matrix, axis=1)
        return weights

    def value_at(self, sample_points: ndarray):
        """Evaluate the interpolator at given points."""
        if np.nanmin(sample_points) < self._min_bound or np.nanmax(sample_points) > self._max_bound:
            raise ValueError("Some given sample points are outside the interpolation range.")

        # Normalize the sample points to the interval [-1, 1]
        sample_points = self._normalize_interval(sample_points)

        # Compute the barycentric interpolation values
        diff_matrix = sample_points[:, None] - self.interpolation_points[None, :]
        terms = self.weights / diff_matrix
        numerator = np.sum(terms * self.interpolation_values, axis=1)
        denominator = np.sum(terms, axis=1)
        return numerator / denominator
