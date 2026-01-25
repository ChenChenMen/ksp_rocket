"""Test base constraints for optimization problems."""

import numpy as np
import pytest

from optimization.constraints import Bounds, ConstraintKind, LinearConstraint
from optimization.differentiation import Jacobian


def test_bounds_constraint_construction():
    """Test constructing bounds constraint."""
    constraint = Bounds(np.asarray([4, 5, 6]), np.asarray([3, 2, 1]))
    evaluated_constraints = constraint.eval_constraints(np.asarray([3, 4, 5]))
    # There are two inequality constraints one for each bound
    assert len(evaluated_constraints) == 2
    assert np.allclose(evaluated_constraints[0], np.asarray([1, 1, 1]))
    assert np.allclose(evaluated_constraints[1], np.asarray([0, 2, 4]))
    assert constraint.kinds[0] is ConstraintKind.INEQUALITY_BELOW
    assert constraint.kinds[1] is ConstraintKind.INEQUALITY_ABOVE


def test_bounds_constraint_jacobian():
    """Verify bounds constraint jacobian extraction is identity."""
    constraint = Bounds(np.asarray([4, 5, 6]), np.asarray([3, 2, 1]))
    constraint_jacobians = constraint.eval_jacobians(np.asarray([3, 4, 5]))
    # There are two inequality constraints one for each bound
    assert len(constraint_jacobians) == 2
    # Ensure the jacobians are multiple of identity matrices
    assert constraint_jacobians[0].is_multiple_of_identity()
    assert constraint_jacobians[1].is_multiple_of_identity()
    arbitrary_vector = np.asarray([7, 8, 9])
    assert np.allclose(constraint_jacobians[0].multiply_by(arbitrary_vector), -arbitrary_vector)
    assert np.allclose(constraint_jacobians[1].multiply_by(arbitrary_vector), arbitrary_vector)


@pytest.mark.parametrize("kind", [ConstraintKind.EQUALITY, ConstraintKind.INEQUALITY_BELOW, ConstraintKind.INEQUALITY_ABOVE])
def test_linear_constraint_construction(kind):
    """Test constructing equality linear constraint."""
    jacobian = Jacobian(matrix=np.asarray([[1, 2, 3], [4, 5, 6]]), selection_indices=np.asarray([2, 3, 5]))
    constraint = LinearConstraint.from_single_jacobian_bias(jacobian, bias=np.asarray([7, 8]), constraint_kind=kind)
    evaluated_constraints = constraint.eval_constraints(np.asarray([1, 2, 3, 4, 5, 6]))
    assert len(evaluated_constraints) == 1
    assert np.allclose(evaluated_constraints[0], np.asarray([36, 76]))
    assert constraint.kinds[0] is kind


def test_linear_constraint_jacobian():
    """Verify linear constraint jacobian extraction."""
    jacobian = Jacobian(matrix=np.asarray([[1, 2, 3], [4, 5, 6]]), selection_indices=np.asarray([2, 3, 5]))
    constraint = LinearConstraint.from_single_jacobian_bias(
        jacobian, bias=np.asarray([7, 8]), constraint_kind=ConstraintKind.EQUALITY
    )
    constraint_jacobian = constraint.eval_jacobians(np.asarray([1, 2, 3, 4, 5, 6]))
    assert np.allclose(jacobian.matrix, constraint_jacobian[0].matrix)
    assert np.allclose(jacobian.selection_indices, constraint_jacobian[0].selection_indices)
