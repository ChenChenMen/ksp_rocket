"""Test definition of functional."""

import numpy as np

from optimization.differentiation import Jacobian
from optimization.functional import LinearMap


def test_linear_equality_constraint_from_slice():
    """Test the slice matrix generation in LinearEqualityConstraint."""
    matrix = np.array([[1, 2, 3], [4, 5, 6]])
    bias = np.array([7, 8])
    linear_equality_constraint = LinearMap.from_slice(matrix, bias, selection_index_collection=[(1, 4)])
    # Sandwich the eye matrix into the larger zero matrix
    expected_selection_indices = np.array([1, 2, 3])
    assert np.array_equal(linear_equality_constraint.matrix, matrix)
    assert np.array_equal(linear_equality_constraint.bias, bias)
    assert np.array_equal(linear_equality_constraint.selection_indices, expected_selection_indices)

    linear_equality_constraint = LinearMap.from_slice(matrix, bias, selection_index_collection=[(1, 3), 5])
    # Inferred by the time being the last element of the optimization array
    expected_selection_indices = np.array([1, 2, 5])
    assert np.array_equal(linear_equality_constraint.matrix, matrix)
    assert np.array_equal(linear_equality_constraint.bias, bias)
    assert np.array_equal(linear_equality_constraint.selection_indices, expected_selection_indices)


def test_linear_equality_constraint_from_jacobian():
    """Test the conversion from ConstraintJacobian to LinearEqualityConstraint."""
    matrix = np.array([[1, 2, 3], [4, 5, 6]])
    constraint_jacobian = Jacobian(matrix, selection_indices=np.array([1, 2, 3]))
    bias = np.array([7, 8])
    linear_equality_constraint = LinearMap.from_jacobian_bias(constraint_jacobian, bias)
    assert np.array_equal(linear_equality_constraint.matrix, matrix)
    assert np.array_equal(linear_equality_constraint.selection_indices, constraint_jacobian.selection_indices)
    assert np.array_equal(linear_equality_constraint.bias, bias)
