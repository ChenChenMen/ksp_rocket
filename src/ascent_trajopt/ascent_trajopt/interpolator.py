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

        # Organize the input data as 2D arrays, assuming interpolation direction is column-wise
        interpolation_points = np.atleast_2d(interpolation_points)
        interpolation_values = np.atleast_2d(interpolation_values)

        # Record bounds of interpolation points for validation during evaluation
        self._min_bound = np.nanmin(interpolation_points, axis=1)
        self._max_bound = np.nanmax(interpolation_points, axis=1)

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
        weights = []
        for i in range(self.interpolation_points.shape[0]):  # Iterate over rows (sets of data)
            diff_matrix = self.interpolation_points[i, :, None] - self.interpolation_points[i, None, :]
            diff_matrix = diff_matrix + np.eye(diff_matrix.shape[0])  # Avoid division by zero on diagonal
            row_weights = 1 / np.prod(diff_matrix, axis=1)
            weights.append(row_weights)
        return np.array(weights)

    def value_at(self, sample_points: ndarray):
        """Evaluate the interpolator at given points."""
        sample_points = np.atleast_2d(sample_points)
        if np.nanmin(sample_points, axis=1) < self._min_bound or np.nanmax(sample_points, axis=1) > self._max_bound:
            raise ValueError("Some given sample points are outside the interpolation range.")

        # Normalize the sample points to the interval [-1, 1]
        sample_points = self._normalize_interval(sample_points)

        interpolated_values = []
        for idx, sample_point in enumerate(sample_points):  # Iterate over rows (sets of data)
            # Compute the barycentric interpolation values
            terms = self.weights[idx] / (sample_point[:, None] - self.interpolation_points[idx, None, :])
            # Append the interpolated value for the current sample point
            interpolated_values.append(np.sum(terms * self.interpolation_values, axis=1) / np.sum(terms, axis=1))
        return np.squeeze(np.asarray(interpolated_values))
