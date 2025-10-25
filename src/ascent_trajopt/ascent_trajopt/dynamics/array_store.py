"""Define array store for collocation optimization variables."""

from dataclasses import dataclass

import numpy as np

from ascent_trajopt.dynamics.base import DynamicsModel


@dataclass
class DynamicModelDimension:
    """Define dimension for dynamic system."""

    # Define the number of state
    num_state: float
    # Define the number of control
    num_control: float

    @property
    def total_dimension(self) -> float:
        """Get the total dimension of the dynamic system."""
        return self.num_state + self.num_control

    @classmethod
    def from_dynamic_model(cls, dynamics_model: DynamicsModel) -> "DynamicModelDimension":
        """Create a dimension instance from a dynamic model."""
        return cls(num_state=dynamics_model.REQUIRED_STATE_NUM, num_control=dynamics_model.REQUIRED_CTRL_NUM)


class DynamicVariablesArray(np.ndarray):
    """A thin wrapper around ndarray to represent state and control variables for a dynamic system."""

    def __new__(cls, input_array, dimension: DynamicModelDimension):
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

        self.dimension: DynamicModelDimension = getattr(casted_array, "dimension", None)

    @property
    def state(self):
        """Getter for the states in the variables"""
        return self[: self.dimension.num_state]

    @property
    def control(self):
        """Getter for the control in the variables"""
        return self[-self.dimension.num_control :]
