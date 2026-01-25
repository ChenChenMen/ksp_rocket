"""Define common functionals used in optimization."""

from dataclasses import dataclass

import numpy as np

from optimization.differentiation import Jacobian


@dataclass
class LinearMap:
    """Define a linear vector map.

    The stored A matrix is in size of m x n' where m is number of constraints,
    n' is number of selected variables. And a sparse selection matrix to translate
    the full variable array to the downselected version.
    """

    # Define the constraint in the form of A @ x + b = 0
    matrix: np.ndarray
    bias: np.ndarray

    # Downselection index array in size of n with the n' element set to true
    selection_indices: list = None

    def __post_init__(self):
        """Initialize a private jacobian field."""
        self._jacobian = None

    def evaluate_with(self, full_variable_array: np.ndarray) -> np.ndarray:
        """Evaluate the linear constraint with the given optimization array."""
        return self.jacobian.multiply_by(full_variable_array) + self.bias

    @property
    def jacobian(self) -> Jacobian:
        """Getter for the jacobian instance."""
        if self._jacobian is None:
            self._jacobian = Jacobian(self.matrix, self.selection_indices)
        return self._jacobian

    @property
    def input_size(self) -> int:
        """Getter for the input size of the described linear mapping."""
        return self.jacobian.col_size

    @property
    def output_size(self) -> int:
        """Getter for the output size of the described linear mapping."""
        return self.jacobian.row_size

    @classmethod
    def from_slice(
        cls, matrix: np.ndarray, bias: np.ndarray, selection_index_collection: list[tuple[int, int] | int] = None
    ) -> "LinearMap":
        """Formulate the linear constraint from slice parameters."""
        return cls(matrix=matrix, bias=bias, selection_indices=selection_index_collection)

    @classmethod
    def from_jacobian_bias(cls, jacobian: Jacobian, bias: np.ndarray) -> "LinearMap":
        """Formulate the linear constraint from jacobian and bias."""
        return cls(matrix=jacobian.matrix, bias=bias, selection_indices=jacobian.selection_indices)
