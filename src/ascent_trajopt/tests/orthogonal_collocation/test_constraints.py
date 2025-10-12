"""Test orthogonal collocation dynamic constraints formulation."""

import numpy as np

from ascent_trajopt.dynamics.array_store import DynamicModelDimension
from ascent_trajopt.orthogonal_collocation.constraints import OrthogonalCollocationConstraint


def test_predefined_linear_constraints(dynamics_model, discretizer, initial_condition, final_condition):
    """Test the formulation of collocation constraints."""
    dynamic_constraint = OrthogonalCollocationConstraint(dynamics_model, discretizer, initial_condition, final_condition)
    dimension = DynamicModelDimension.from_dynamic_model(dynamics_model)
    assert len(dynamic_constraint._boundary_condition_constraints) == 2

    # Final condition constraint
    final_condition_constraint = dynamic_constraint._boundary_condition_constraints[0]
    interpolator = discretizer.get_interpolator_for_segment(0)
    interpolated_weights = interpolator.weights_for_value_at(discretizer.get_end_time_for_segment(0)).squeeze()
    interpolated_matrix = np.kron(interpolated_weights, np.eye(dimension.total_dimension))
    assert np.allclose(final_condition_constraint.matrix, interpolated_matrix)
    assert np.allclose(final_condition_constraint.bias, -final_condition)
    assert final_condition_constraint.index_slice == (0, 24)

    # Initial condition constraint
    initial_condition_constraint = dynamic_constraint._boundary_condition_constraints[1]
    interpolator = discretizer.get_interpolator_for_segment(discretizer.total_num_segments - 1)
    assert np.allclose(initial_condition_constraint.matrix, np.eye(dimension.total_dimension))
    assert np.allclose(initial_condition_constraint.bias, -initial_condition)
    assert initial_condition_constraint.index_slice == (0, 4)


def test_linearized_collocation_constraints(
    dynamics_model, discretizer, optimization_array, initial_condition, final_condition
):
    """Test the formulation of collocation constraints."""
    dynamic_constraint = OrthogonalCollocationConstraint(dynamics_model, discretizer, initial_condition, final_condition)
    constraints = dynamic_constraint.get_linearized_collocation_constraints(optimization_array)
    assert len(constraints) == 3
