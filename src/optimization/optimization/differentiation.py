"""Manage and define differentiation utilities."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np


class SingleSelectedGradientEntry(NamedTuple):
    """Used to define single selection from the gradient entry

    Evaluation in this case suggests the result of the single selection from the full variable
    array by index scaled by the scaling multiplier. at the meantime, the dimension of the full
    vector is recorded for completeness
    """

    scale: float
    dimension: int
    index: int


class IdentityMatrixEntry(NamedTuple):
    """Used to define the identity matrix entry

    Evaluation in this case suggests the final matrix is float * np.eye(self.dimension)
    """

    scale: float
    dimension: int


def _array_selection(full_variable_array: np.ndarray, selection_indices: list) -> np.ndarray:
    """Select elements from full array based on selection indices."""
    full_ndarray = full_variable_array.view(np.ndarray)
    if selection_indices is None:
        return full_ndarray
    return full_ndarray[selection_indices]


@dataclass
class Gradient:
    """Define a gradient vector in 1D representation."""

    # Gradient vector, shall be treated as a column vector
    vector: np.ndarray | SingleSelectedGradientEntry

    # Downselection index array in size of n with the n' element set to true
    selection_indices: list = None

    def multiply_by(self, full_variable_array: np.ndarray) -> float:
        """Evaluate the gradient  with the given full array."""
        operating_array = _array_selection(full_variable_array, self.selection_indices)
        # Handle indentity jacobian cases
        if isinstance(self.vector, SingleSelectedGradientEntry):
            return self.vector.scale * operating_array[self.vector.index]
        return self.vector @ operating_array

    @property
    def dimension(self) -> int:
        """Get the dimension of the full variable array."""
        if isinstance(self.vector, SingleSelectedGradientEntry):
            return self.vector.dimension
        return self.vector.size

    @classmethod
    def from_jacobian_row(cls, jacobian: "Jacobian", row_index: int) -> "Gradient":
        """Construct a gradient from one row of Jacobian matrix."""
        if jacobian.is_multiple_of_identity():
            vector_entry = SingleSelectedGradientEntry(jacobian.matrix, jacobian.col_size, row_index)
            return cls(vector_entry, jacobian.selection_indices)
        return cls(jacobian.matrix[row_index, :], jacobian.selection_indices)


@dataclass
class Hessian:
    """Define a Hessian matrix."""

    # When a float is given, meaning that the hessian is float * np.eye(self.dimension)
    # A tuple of float and int is accepted, suggesting the multiple (float) of identity
    # matrix of size (int)
    matrix: np.ndarray | IdentityMatrixEntry

    # Downselection index array in size of n with the n' element set to true
    selection_indices: list = None

    @property
    def dimension(self) -> int:
        """Get the dimension of the full variable array."""
        if isinstance(self.matrix, IdentityMatrixEntry):
            return self.matrix.dimension
        return self.matrix.shape[0]

    @classmethod
    def empty_from_gradient(cls, gradient: Gradient) -> "Hessian":
        """Construct an empty Hessian matrix from gradient vector's shape."""
        matrix_entrty = IdentityMatrixEntry(0, gradient.dimension)
        return cls(matrix_entrty, gradient.selection_indices)


@dataclass
class Jacobian:
    """Define a jacobian matrix with optional downselection matrix.

    The stored jacobian matrix is in size of m x n' where m is size of outputs,
    n is number of downselected optimization variables.
    """

    # Jacobian matrix in size of m x n' or a float for identity jacobians
    matrix: np.ndarray | IdentityMatrixEntry

    # Downselection index array in size of n with the n' element set to true
    selection_indices: list = None

    def multiply_by(self, full_variable_array: np.ndarray) -> np.ndarray:
        """Evaluate the matrix multiplication with the given full array."""
        operating_array = _array_selection(full_variable_array, self.selection_indices)
        # Handle indentity jacobian cases
        if isinstance(self.matrix, IdentityMatrixEntry):
            return self.matrix.scale * operating_array
        return self.matrix @ operating_array

    def is_multiple_of_identity(self) -> bool:
        """Check if the jacobian is a multiple of identity matrix."""
        return isinstance(self.matrix, IdentityMatrixEntry)

    @property
    def row_size(self):
        """Quick return of the number of rows."""
        if isinstance(self.matrix, IdentityMatrixEntry):
            return self.matrix.dimension
        return self.matrix.shape[0]

    @property
    def col_size(self):
        """Quick return of the number of columns."""
        if isinstance(self.matrix, IdentityMatrixEntry):
            return self.matrix.dimension
        return self.matrix.shape[1]

    @classmethod
    def from_slice(
        cls, matrix: np.ndarray | IdentityMatrixEntry, selection_index_collection: list[tuple[int, int] | int] = None
    ) -> "Jacobian":
        """Get the slice matrix for the segment from optimization matrix.

        The selection index collection defines how to downselect the optimization
        array to apply the stored jacobian matrix at evaluation. The index collection
        is given in a list of double integer tuple or an integer. The tuple will be
        interpreted as slice and the single integer will be interpreted as a single
        index, the collection shall be in order to how jacobian matrix rows.
        """
        # If no downselection is needed, return directly
        if selection_index_collection is None:
            return cls(matrix, selection_indices=None)

        # Resolve the indices from the collection
        selection_indices = []
        for element in selection_index_collection:
            if isinstance(element, int):
                selection_indices.append(element)
            elif isinstance(element, tuple):
                start_idx, end_idx = element
                selection_indices.extend(range(start_idx, end_idx))
        return cls(matrix, selection_indices)


@dataclass
class HessiansForJacobian:
    """Define a list of hessian matrices for a Jacobian instance.

    Stores a list of hessian matrices each with size of n' x n', where n' is number
    of downselected optimization variables. A single instance is associated with one
    Jacobian instance because for each row of a jacobian matrix, there is a square
    Hessian matrix.
    """

    # A list of Hessians, each for a row of Jacobian and in size of n' x n'
    row_matrices: list[Hessian]
    # Define dimension of the hessian matrix, which is also the column shape of the jacobian
    dimension: int

    # Downselection index array in size of n with the n' element set to true
    selection_indices: list = None

    @classmethod
    def empty_from_jacobian(cls, jacobian: Jacobian) -> "HessiansForJacobian":
        """Create empty hessians from a Jacobian instances."""
        matrix_collection = []
        dimension = jacobian.col_size
        for row_index in range(jacobian.row_size):
            gradient = Gradient.from_jacobian_row(jacobian, row_index)
            hessian = Hessian.empty_from_gradient(gradient)
            assert hessian.dimension == dimension, "Dimension mismatch in empty hessian"
            matrix_collection.append(hessian)
        return cls(matrix_collection, dimension, selection_indices=jacobian.selection_indices)


class HessianEstimator:
    """Implement a standalong Hessian estimator backed by BFGS.

    By design, each instance of this estimator is responsible for a single
    Hessian estimation. This means that multiple instances of the estimator
    may be present at the same time throughout an iterative solve process.
    """

    def __init__(self, initial_guess: Hessian):
        """Initialize a Hessian estimator with an initial guess."""
        self._est_fwd_hessian = initial_guess
        self._est_inv_hessian = np.eye(initial_guess.matrix.shape[0])


def get_jacobian_by_perturbation(
    function_handle: Callable[[np.ndarray], list[np.ndarray]],
    optimization_array: np.ndarray,
    perturbation_percentage: float = 1e-3,
    min_perturbation: float = 1e-5,
) -> list[Jacobian]:
    """Numerically compute the constraint jacobian via finite difference perturbation."""
    num_variables = optimization_array.size

    jacobian_partials: list[list[np.ndarray]] = []
    for var_index in range(num_variables):
        # Compute perturbation size
        perturbation = max(abs(optimization_array[var_index]) * perturbation_percentage, min_perturbation)

        # Perturb positively
        perturbed_array_pos = optimization_array.copy()
        perturbed_array_pos[var_index] += perturbation
        constraint_pos_list = function_handle(perturbed_array_pos)

        # Perturb negatively
        perturbed_array_neg = optimization_array.copy()
        perturbed_array_neg[var_index] -= perturbation
        constraint_neg_list = function_handle(perturbed_array_neg)

        for idx, (constraint_pos, constraint_neg) in enumerate(zip(constraint_pos_list, constraint_neg_list)):
            partial = (constraint_pos - constraint_neg) / (2 * perturbation)
            if len(jacobian_partials) == idx:
                jacobian_partials.append([])
            jacobian_partials[idx].append(np.atleast_2d(partial).T)

    return [Jacobian(matrix=np.concatenate(jacobian_partial, axis=1)) for jacobian_partial in jacobian_partials]
