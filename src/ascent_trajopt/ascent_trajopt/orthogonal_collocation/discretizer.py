"""Defines the orthogonal collocation discretization of the trajectory."""

import numpy as np
from typing import NamedTuple

from ascent_trajopt.interpolator import BarycentricInterpolator
from ascent_trajopt.orthogonal_collocation.polynomials import get_legendre_gauss_radau_points


class HPSegmentConfig(NamedTuple):
    """Defines a segment for discretization.

    Attributes:
        n_points: Number of collocation points in the segment.
        end_time: End time of the segment in the normalized time domain (0, 1].
    """

    n_points: int
    end_time: float


class HPDiscretizer:
    """Manages the discretization by hp adaptive collocation.

    Internally assumes a normalized time domain [0, 1]. Each segment is
    discretized for othrogonal collocation.
    """

    def __init__(self, segment_scheme: tuple[HPSegmentConfig]):
        """Provide segments to initialize the discretizer.

        A segment is provided with the number of points and the end unified time
        of the segment, the end time should be in (0, 1]. However, note that the
        dicretization scheme is Legendre-Gauss-Radau, which means the tau = 1 is
        not included in the discretization points, but tau = 0 is.
        """
        self.segment_scheme = segment_scheme

        discretized_point_collection, num_point_rolling_collection = [], []
        previous_end_time, total_num_points = 0, 0

        # Iteratively discretize each segment and append the points
        for idx, (n_points, end_time) in enumerate(self.segment_scheme):
            # Validate the segment end time
            if end_time <= previous_end_time or end_time > 1:
                raise ValueError(f"Invalid segment end time {end_time} for the {idx}th segment with {n_points} points.")

            # Discretize the segment and append the points
            interval_length = end_time - previous_end_time

            # Discretize the segment into normalized points
            discretized_segments = self._discretize_segment(n_points, interval_length)
            total_num_points += n_points

            # Append to the collections
            discretized_point_collection.append(discretized_segments + previous_end_time)
            num_point_rolling_collection.append(total_num_points)

            # Update the previous end time
            previous_end_time = end_time

        self.discretized_point_collection = discretized_point_collection
        self.discretized_point_array = np.concatenate(discretized_point_collection)
        self.num_point_rolling_collection = num_point_rolling_collection

        self.total_num_points = total_num_points
        self.total_num_segments = len(self.segment_scheme)

    def get_interpolator_for_segment(self, segment_index: int) -> BarycentricInterpolator:
        """Get the interpolator for a specific segment."""
        if segment_index < 0 or segment_index >= len(self.segment_scheme):
            raise IndexError(f"Segment index {segment_index} is out of range.")

        # Get the collocation points for the segment
        segment_points = self.discretized_point_collection[segment_index]
        return BarycentricInterpolator(interpolation_points=segment_points)

    def _discretize_segment(self, n_points: int, interval_length: float):
        """Discretize a single segment with LGR points."""
        lgr_points = get_legendre_gauss_radau_points(n_points)
        return (lgr_points + 1) * interval_length / 2

    def __iter__(self) -> np.ndarray:
        """Iterator over the segments."""
        return iter(self.discretized_point_collection)
