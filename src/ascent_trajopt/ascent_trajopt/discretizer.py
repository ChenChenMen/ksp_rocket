"""Defines the orthogonal collocation discretization of the trajectory."""

from ascent_trajopt.orthogonal_polynomials import get_legendre_gauss_radau_points


class HPDiscretizer:
    """Manages the discretization by hp adaptive collocation.

    Internally assumes a normalized time domain [0, 1]. Each segment is
    discretized for othrogonal collocation.
    """

    def __init__(self, segments: list[int, float]):
        """Provide segments to initialize the discretizer.
        
        A segment is provided with the number of points and the end unified time
        of the segment, the end time should be in (0, 1]. However, note that the
        dicretization scheme is Legendre-Gauss-Radau, which means the tau = 1 is
        not included in the discretization points, but tau = 0 is.
        """
        self.segments = tuple(segments)

        all_points = []
        previous_end_time = 0
        for idx, (n_points, end_time) in enumerate(self.segments):
            # Validate the segment end time
            if end_time <= previous_end_time or end_time > 1:
                raise ValueError(f"Invalid segment end time {end_time} for the {idx}th segment with {n_points} points.")

            # Discretize the segment and append the points
            interval_length = end_time - previous_end_time
            all_points.extend(self._discretize_segment(n_points, interval_length))
            previous_end_time = end_time

        # Create tuple of all points for immutability
        self.all_points = tuple(all_points)

    def _discretize_segment(self, n_points: int, interval_length: float):
        """Discretize a single segment."""
        lgr_points = get_legendre_gauss_radau_points(n_points)
        return (lgr_points + 1) * interval_length / 2
