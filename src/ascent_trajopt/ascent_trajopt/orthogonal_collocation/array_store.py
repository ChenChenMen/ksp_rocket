"""Define array store for collocation optimization variables."""

from dataclasses import dataclass

import numpy as np

from ascent_trajopt.orthogonal_collocation.discretizer import HPDiscretizer


@dataclass
class DynamicSystemDimension:
    """Define dimension for dynamic system."""

    # Define the number of state
    num_state: float
    # Define the number of control
    num_control: float

    @property
    def total_dimension(self) -> float:
        """Get the total dimension of the dynamic system."""
        return self.num_state + self.num_control


class DynamicVariablesArray(np.ndarray):
    """A thin wrapper around ndarray to represent state and control variables for a dynamic system."""

    def __new__(cls, input_array, dimension: DynamicSystemDimension):
        """Subclass ndarray to include discretizer information."""
        casted_array = np.asarray(input_array).view(cls)
        casted_array.dimension = dimension

        # Ensure the input data has consistent points as suggested by the dimension
        if casted_array.shape[0] != (expected_length := casted_array.dimension.total_dimension):
            raise ValueError(
                f"Input array length {casted_array.shape[0]} does not match expected length {expected_length} "
                f"from dynamic system's state and control dimensions, {casted_array.dimension.num_state} and"
                f"{casted_array.dimension.num_control} respectively."
            )
        return casted_array

    def __array_finalize__(self, casted_array):
        """Ensure the discretizer attribute is preserved during ndarray operations."""
        if casted_array is None:
            return

        if len(self.shape) > 1:  # Only support 1D array
            raise ValueError("DynamicVariablesArray must be a 1D array.")

        self.dimension: DynamicSystemDimension = getattr(casted_array, "dimension", None)

    @property
    def state(self):
        """Getter for the states in the variables"""
        return self[: self.dimension.num_state]

    @property
    def control(self):
        """Getter for the control in the variables"""
        return self[-self.dimension.num_control :]


class OptimizationArray(np.ndarray):
    """A thin wrapper around ndarray to represent optimization variables segmented by HPDiscretizer."""

    def __new__(cls, input_array, discretizer: HPDiscretizer, dimension: DynamicSystemDimension):
        """Subclass ndarray to include discretizer information."""
        casted_array = np.asarray(input_array).view(cls)
        casted_array.discretizer = discretizer
        casted_array.dimension = dimension

        # Ensure the input data has consistent points as suggested by the discretizer
        point_dimension = casted_array.dimension.total_dimension
        # The additional 1 if for the total time duration at the end of the optimization vector
        expected_length = point_dimension * casted_array.discretizer.total_num_points + 1
        if casted_array.shape[0] != expected_length:
            raise ValueError(
                f"Input array length {casted_array.shape[0]} does not match expected length {expected_length} "
                f"from discretizer with {casted_array.discretizer.total_num_points} points and point"
                f"dimension {point_dimension}."
            )

        return casted_array

    def __array_finalize__(self, casted_array):
        """Ensure the discretizer attribute is preserved during ndarray operations."""
        if casted_array is None:
            return

        if len(self.shape) > 1:  # Only support 1D optimization array
            raise ValueError("OptimizationArray must be a 1D array.")

        self.discretizer: HPDiscretizer = getattr(casted_array, "discretizer", None)
        self.dimension: DynamicSystemDimension = getattr(casted_array, "dimension", None)

    def point(self, point_index: int) -> DynamicVariablesArray:
        """Get the variables at a specific point."""
        if point_index < 0 or point_index >= self.discretizer.total_num_points:
            raise IndexError(f"Point index {point_index} is out of range.")

        # Obtain start and end indices for the point requested
        point_dimension = self.dimension.total_dimension
        start_index = point_index * point_dimension
        end_index = (point_index + 1) * point_dimension
        return DynamicVariablesArray(self[start_index:end_index], dimension=self.dimension)

    def segment(self, segment_index: int) -> list[DynamicVariablesArray]:
        """Get the state and control variables for a specific segment."""
        if segment_index < 0 or segment_index >= self.discretizer.total_num_segments:
            raise IndexError(f"Segment index {segment_index} is out of range.")

        # Determine the start and end indices for the segment
        num_point_rolling_collection = self.discretizer.num_point_rolling_collection
        start_point_index = 0 if segment_index == 0 else num_point_rolling_collection[segment_index - 1]
        end_point_index = num_point_rolling_collection[segment_index]

        # Construct point collection for given segment
        return [self.point(idx) for idx in range(start_point_index, end_point_index)]
