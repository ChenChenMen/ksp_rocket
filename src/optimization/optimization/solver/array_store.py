"""Define array store for collocation optimization variables."""

import numpy as np
from types import ModuleType


class SlackedOptimizationArray(np.ndarray):
    """A thin wrapper around ndarray to represent optimization variables with slack variables."""

    def __new__(cls, input_array, unslacked_length: int):
        """Subclass ndarray to create optimziation array internal to solver with slack."""
        casted_array = np.asarray(input_array).view(cls)
        casted_array.unslacked_length = unslacked_length
        return casted_array

    def __array_finalize__(self, casted_array):
        """Ensure the discretizer attribute is preserved during ndarray operations."""
        if casted_array is None:
            return

        if len(self.shape) > 1:  # Only support 1D optimization array
            raise ValueError("OptimizationArray must be a 1D array.")
        self.unslacked_length = getattr(casted_array, "unslacked_length", None)

    def get_unslacked(self, unslacked_cls: ModuleType = None) -> np.ndarray:
        """Get the unslacked optimization array."""
        return self.view(unslacked_cls or np.ndarray)
