"""Define array store for collocation optimization variables."""

import numpy as np

from ascent_trajopt.orthogonal_collocation.discretizer import HPDiscretizer
from ascent_trajopt.dynamics.array_store import DynamicModelDimension, DynamicVariablesArray


class OptimizationArray(np.ndarray):
    """A thin wrapper around ndarray to represent optimization variables segmented by HPDiscretizer."""

    def __new__(
        cls,
        input_array,
        discretizer: HPDiscretizer,
        dimension: DynamicModelDimension,
    ):
        """Subclass ndarray to include discretizer information."""
        casted_array = np.asarray(input_array).view(cls)
        casted_array.discretizer = discretizer
        casted_array.dimension = dimension

        expected_length = cls.get_expected_length(discretizer, dimension)
        # Ensure the input data has consistent points as suggested by the discretizer
        if casted_array.shape[0] != expected_length:
            raise ValueError(
                f"Input array length {casted_array.shape[0]} does not match expected length "
                f"{expected_length} from discretizer with {discretizer.total_num_points} points "
                f"for optimization and point dimension of {dimension.total_dimension}."
            )

        return casted_array

    def __array_finalize__(self, casted_array):
        """Ensure the discretizer attribute is preserved during ndarray operations."""
        if casted_array is None:
            return

        if len(self.shape) > 1:  # Only support 1D optimization array
            raise ValueError("OptimizationArray must be a 1D array.")

        self.discretizer: HPDiscretizer = getattr(casted_array, "discretizer", None)
        self.dimension: DynamicModelDimension = getattr(casted_array, "dimension", None)

    @staticmethod
    def get_expected_length(discretizer: HPDiscretizer, dimension: DynamicModelDimension) -> int:
        """Get the expected length from discretizer and each point's dimension."""
        # The additional 1 if for the total time duration at the end of the optimization vector
        return dimension.total_dimension * discretizer.total_num_points + 1

    @staticmethod
    def point_index_slice(discretizer: HPDiscretizer, dimension: DynamicModelDimension, point_index: int) -> tuple[int, int]:
        """Get the indices for a specific point."""
        if point_index < 0 or point_index >= discretizer.total_num_points:
            raise IndexError(f"Point index {point_index} is out of range.")

        # Obtain start and end indices for the point requested
        point_dimension = dimension.total_dimension
        start_index = point_index * point_dimension
        end_index = start_index + point_dimension
        return start_index, end_index

    @staticmethod
    def segment_index_slice(discretizer: HPDiscretizer, segment_index: int) -> tuple[int, int]:
        """Get the point indices for a specific segment."""
        if segment_index < 0 or segment_index >= discretizer.total_num_segments:
            raise IndexError(f"Segment index {segment_index} is out of range.")

        # Determine the start and end indices for the segment
        num_point_rolling_collection = discretizer.num_point_rolling_collection
        start_point_index = 0 if segment_index == 0 else num_point_rolling_collection[segment_index - 1]
        end_point_index = num_point_rolling_collection[segment_index]
        return start_point_index, end_point_index

    @staticmethod
    def segment_point_index_slice(
        discretizer: HPDiscretizer, dimension: DynamicModelDimension, segment_index: int
    ) -> tuple[int, int]:
        """Get the indices from start point to end point of a specific segment."""
        start_point_index, end_point_index = OptimizationArray.segment_index_slice(discretizer, segment_index)
        start_index, _ = OptimizationArray.point_index_slice(discretizer, dimension, start_point_index)
        _, end_index = OptimizationArray.point_index_slice(discretizer, dimension, end_point_index - 1)
        return start_index, end_index

    def point(self, point_index: int) -> DynamicVariablesArray:
        """Get the variables at a specific point."""
        # Obtain start and end indices for the point requested
        start_index, end_index = OptimizationArray.point_index_slice(self.discretizer, self.dimension, point_index)
        return DynamicVariablesArray(self[start_index:end_index], dimension=self.dimension)

    def segment(self, segment_index: int) -> list[DynamicVariablesArray]:
        """Get the state and control variables for a specific segment."""
        # Construct point collection for given segment
        start_point_index, end_point_index = OptimizationArray.segment_index_slice(self.discretizer, segment_index)
        return [self.point(idx) for idx in range(start_point_index, end_point_index)]

    def segment_as_ndarray(self, segment_index: int) -> np.ndarray:
        """Get the state and control variables for a specific segment."""
        # Construct point collection for given segment
        start_index, end_index = OptimizationArray.segment_point_index_slice(self.discretizer, self.dimension, segment_index)
        return np.array(self[start_index:end_index])

    @property
    def time(self) -> float:
        """Get the total time duration variable at the end of the optimization array."""
        return self[-1]

    @property
    def expected_length(self) -> int:
        """Get the expected length from discretizer and each point's dimension."""
        return OptimizationArray.get_expected_length(self.discretizer, self.dimension)
