"""Define polynominal interepolator."""

import numpy as np
from numpy import ndarray


class BarycentricInterpolator:
    """Barycentric Interpolator for trajectory optimization."""

    def __init__(
        self,
        interpolation_points: ndarray,
        interpolation_values: ndarray = None,
        min_bound: float = None,
        max_bound: float = None,
    ):
        """Initialize the interpolator with given points and values.

        Allow user to provide explicit min and max bounds for scaling and implicit extrapolation
        """
        # Organize the input data as 2D arrays, assuming interpolation direction is column-wise
        interpolation_points = np.atleast_2d(interpolation_points)

        # Record bounds of interpolation points for validation during evaluation
        self._min_bound = min_bound or np.nanmin(interpolation_points, axis=1)
        self._max_bound = max_bound or np.nanmax(interpolation_points, axis=1)
        self._interpolation_interval = self._max_bound - self._min_bound

        # Normalize the interpolation points to the interval [-1, 1]
        self.interpolation_points = self._normalize_interval(interpolation_points)
        self.interpolation_point_length = interpolation_points.shape[0]
        self.interpolation_point_count = interpolation_points.shape[1]
        self.interpolation_values = None

        # Reigster the interpolation derivatives at the interpolation points
        self._differentiation_matricies = None
        self._interpolation_derivatives = None

        # Register the barycentric weights for the interpolation points
        self._barycentric_weights = None

        # If interpolation values are provided, register them
        if interpolation_values is not None:
            self.register_interpolation_values(interpolation_values)

    @property
    def barycentric_weights(self):
        """Lazily compute the barycentric weights for the interpolation points."""
        if self._barycentric_weights is None:
            barycentric_weights = []
            for i in range(self.interpolation_point_length):  # Iterate over rows (sets of data)
                diff_matrix = self.interpolation_points[i, :, None] - self.interpolation_points[i, None, :]
                # Avoid division by zero on diagonal with indentity overwrite
                diff_matrix = diff_matrix + np.eye(diff_matrix.shape[0])
                row_weights = 1 / np.prod(diff_matrix, axis=1)
                barycentric_weights.append(row_weights)
            self._barycentric_weights = np.asarray(barycentric_weights)
        return self._barycentric_weights

    @property
    def differentiation_matricies(self):
        """Lazily compute the differentiation matrix for the interpolation points."""
        if self._differentiation_matricies is None:
            diff_matrix_collection = []
            for i in range(self.interpolation_point_length):  # Iterate over rows (sets of data)
                weight_ratio_matrix = self.barycentric_weights[i, None, :] / self.barycentric_weights[i, :, None]
                points_diff_matrix = self.interpolation_points[i, :, None] - self.interpolation_points[i, None, :]
                diff_matrix = weight_ratio_matrix / points_diff_matrix / self._interpolation_interval * 2
                # Handle the diagonal elements separately, first with zero entry to coordinate sum
                np.fill_diagonal(diff_matrix, 0)
                # then drop the sum into the diagonal term
                np.fill_diagonal(diff_matrix, -1 * np.sum(diff_matrix, axis=1))
                diff_matrix_collection.append(diff_matrix)
            self._differentiation_matricies = np.asarray(diff_matrix_collection)
        return self._differentiation_matricies

    @property
    def interpolation_derivatives(self):
        """Lazily compute the derivatives for the interpolation polynomial."""
        if self.interpolation_values is None:
            raise ValueError("Interpolation values must be registered before computing derivatives.")
        if self._interpolation_derivatives is None:
            derivatives = []
            for i in range(self.interpolation_point_length):  # Iterate over rows (sets of data)
                diff_matrix = self.differentiation_matricies[i]
                # Compute derivative from the differentiation matrix
                derivatives.append((diff_matrix @ self.interpolation_values[i, :, None]).squeeze())
            self._interpolation_derivatives = np.asarray(derivatives)
        return self._interpolation_derivatives

    def register_interpolation_values(self, interpolation_values: ndarray):
        """Register new interpolation values for the existing interpolation points."""
        interpolation_values = np.atleast_2d(interpolation_values)
        # Check if the input points and values are valid
        if self.interpolation_point_count != interpolation_values.shape[1]:
            raise ValueError("Interpolation points and values must have the same shape.")

        self.interpolation_values = interpolation_values
        # Reset the interpolation derivatives
        self._interpolation_derivatives = None

    def value_at(self, sample_points: ndarray):
        """Evaluate the interpolated value at sample points."""
        if self.interpolation_values is None:
            raise ValueError("Interpolation values must be registered before evaluation.")
        return self._interpolate(sample_points, interpolation_values=self.interpolation_values)

    def deriv_at(self, sample_points: ndarray):
        """Evaluate the interpolated derivative at sample points."""
        return self._interpolate(sample_points, interpolation_values=self.interpolation_derivatives)

    def weights_for_value_at(self, sample_points: ndarray):
        """Retrieve the interpolation matrix at given points."""
        sample_points = np.atleast_2d(sample_points)
        if np.nanmin(sample_points, axis=1) < self._min_bound or np.nanmax(sample_points, axis=1) > self._max_bound:
            raise ValueError("Some given sample points are outside the interpolation range.")

        # Normalize the sample points to the interval [-1, 1]
        sample_points = self._normalize_interval(sample_points)

        interpolated_weights = []
        for idx, sample_point in enumerate(sample_points):
            terms = self.barycentric_weights[idx] / (sample_point[:, None] - self.interpolation_points[idx, None, :])
            weight_array = terms / np.sum(terms, axis=1)[:, np.newaxis]
            # Replace nan with one as they are generated from coincide points between samples and interpolation points
            weight_array[np.isnan(weight_array)] = 1
            interpolated_weights.append(weight_array)
        return np.asarray(interpolated_weights)

    def _interpolate(self, sample_points: ndarray, interpolation_values: ndarray):
        """Evaluate the interpolator at given points."""
        # Append the interpolated value for the current sample point
        interpolated_values = [
            np.sum(interpolated_weight * interpolation_values, axis=1)
            for interpolated_weight in np.atleast_2d(self.weights_for_value_at(sample_points))
        ]
        return np.squeeze(np.asarray(interpolated_values))

    def _normalize_interval(self, points: ndarray) -> ndarray:
        """Normalize the interpolation points to the interval [-1, 1]."""
        return 2 * (points - self._min_bound) / (self._max_bound - self._min_bound) - 1
