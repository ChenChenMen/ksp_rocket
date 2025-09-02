"""Defines the orthogonal collocation discretization of the trajectory."""

import numpy as np
from typing import NamedTuple

from ascent_trajopt.orthogonal_polynomials import get_legendre_gauss_radau_points


class HPSegment(NamedTuple):
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

    def __init__(self, segments: tuple[HPSegment]):
        """Provide segments to initialize the discretizer.

        A segment is provided with the number of points and the end unified time
        of the segment, the end time should be in (0, 1]. However, note that the
        dicretization scheme is Legendre-Gauss-Radau, which means the tau = 1 is
        not included in the discretization points, but tau = 0 is.
        """
        self.segments = segments

        all_points = []
        previous_end_time = 0
        for idx, (n_points, end_time) in enumerate(self.segments):
            # Validate the segment end time
            if end_time <= previous_end_time or end_time > 1:
                raise ValueError(f"Invalid segment end time {end_time} for the {idx}th segment with {n_points} points.")

            # Discretize the segment and append the points
            interval_length = end_time - previous_end_time
            all_points.append(self._discretize_segment(n_points, interval_length) + previous_end_time)
            previous_end_time = end_time

        # Create tuple of all points for immutability
        self.all_points = np.concatenate(all_points)

    def _discretize_segment(self, n_points: int, interval_length: float):
        """Discretize a single segment."""
        lgr_points = get_legendre_gauss_radau_points(n_points)
        return (lgr_points + 1) * interval_length / 2
