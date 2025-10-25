"""Test orthogonal collocation dynamic constraints formulation."""

import numpy as np
import pytest
from scipy.linalg import block_diag

from ascent_trajopt.dynamics.array_store import DynamicModelDimension
from ascent_trajopt.dynamics.pendulum import SinglePendulumDynamicsModel
from ascent_trajopt.orthogonal_collocation.constraints import (
    OrthogonalCollocationConstraint,
    ConstraintJacobian,
    LinearEqualityConstraint,
    get_constraint_jacobian_by_perturbation,
)


def test_constraint_jacobian_from_slice():
    """Test the slice matrix generation in ConstraintJacobian."""
    matrix = np.array([[1, 2, 3], [4, 5, 6]])
    constraint_jacobian = ConstraintJacobian.from_slice(
        matrix, expected_optimization_array_length=6, segment_state_slice=(1, 4)
    )
    # Sandwich the eye matrix into the larger zero matrix
    expected_selection_matrix = np.array([[0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0]])
    assert np.array_equal(constraint_jacobian.matrix, matrix)
    assert np.array_equal(constraint_jacobian.selection_matrix, expected_selection_matrix)

    constraint_jacobian = ConstraintJacobian.from_slice(
        matrix, expected_optimization_array_length=6, segment_state_slice=(1, 3), end_with_time_partial=True
    )
    # Inferred by the time being the last element of the optimization array
    expected_selection_matrix = np.array([[0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1]])
    assert np.array_equal(constraint_jacobian.matrix, matrix)
    assert np.array_equal(constraint_jacobian.selection_matrix, expected_selection_matrix)


def test_linear_equality_constraint_from_slice():
    """Test the slice matrix generation in LinearEqualityConstraint."""
    matrix = np.array([[1, 2, 3], [4, 5, 6]])
    bias = np.array([7, 8])
    linear_equality_constraint = LinearEqualityConstraint.from_slice(
        matrix, bias, expected_optimization_array_length=6, segment_state_slice=(1, 4)
    )
    # Sandwich the eye matrix into the larger zero matrix
    expected_selection_matrix = np.array([[0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0]])
    assert np.array_equal(linear_equality_constraint.matrix, matrix)
    assert np.array_equal(linear_equality_constraint.bias, bias)
    assert np.array_equal(linear_equality_constraint.selection_matrix, expected_selection_matrix)

    linear_equality_constraint = LinearEqualityConstraint.from_slice(
        matrix, bias, expected_optimization_array_length=6, segment_state_slice=(1, 3), end_with_time_partial=True
    )
    # Inferred by the time being the last element of the optimization array
    expected_selection_matrix = np.array([[0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1]])
    assert np.array_equal(linear_equality_constraint.matrix, matrix)
    assert np.array_equal(linear_equality_constraint.bias, bias)
    assert np.array_equal(linear_equality_constraint.selection_matrix, expected_selection_matrix)


def test_jacobian_from_linear_equality_constraint():
    """Test the conversion from LinearEqualityConstraint to ConstraintJacobian."""
    matrix = np.array([[1, 2, 3], [4, 5, 6]])
    bias = np.array([7, 8])
    linear_equality_constraint = LinearEqualityConstraint(
        matrix=matrix,
        bias=bias,
        selection_matrix=np.array([[0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0]]),
    )
    constraint_jacobian = ConstraintJacobian.from_linear_equality_constraint(linear_equality_constraint)
    assert np.array_equal(constraint_jacobian.matrix, matrix)
    assert np.array_equal(constraint_jacobian.selection_matrix, linear_equality_constraint.selection_matrix)


def test_linear_equality_constraint_from_jacobian():
    """Test the conversion from ConstraintJacobian to LinearEqualityConstraint."""
    matrix = np.array([[1, 2, 3], [4, 5, 6]])
    constraint_jacobian = ConstraintJacobian(
        matrix=matrix,
        selection_matrix=np.array([[0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0]]),
    )
    bias = np.array([7, 8])
    linear_equality_constraint = LinearEqualityConstraint.from_jacobian_bias(constraint_jacobian, bias)
    assert np.array_equal(linear_equality_constraint.matrix, matrix)
    assert np.array_equal(linear_equality_constraint.selection_matrix, constraint_jacobian.selection_matrix)
    assert np.array_equal(linear_equality_constraint.bias, bias)


def test_predefined_linear_constraints(input_components):
    """Test the formulation of collocation constraints."""
    dynamic_constraint = OrthogonalCollocationConstraint(problem_input=input_components)
    dimension = DynamicModelDimension.from_dynamic_model(input_components.dynamics_model)
    discretizer = input_components.discretizer
    assert len(dynamic_constraint._boundary_condition_constraints) == 2

    # Final condition constraint
    final_condition_constraint = dynamic_constraint._boundary_condition_constraints[0]
    interpolator = discretizer.get_interpolator_for_segment(0)
    interpolated_weights = interpolator.weights_for_value_at(discretizer.get_end_time_for_segment(0)).squeeze()
    interpolated_matrix = np.kron(interpolated_weights, np.eye(dimension.total_dimension))
    assert np.allclose(final_condition_constraint.matrix, interpolated_matrix)
    assert np.allclose(final_condition_constraint.bias, -input_components.final_condition)
    assert final_condition_constraint.selection_matrix.shape[0] == 25

    # Initial condition constraint
    initial_condition_constraint = dynamic_constraint._boundary_condition_constraints[1]
    interpolator = discretizer.get_interpolator_for_segment(discretizer.total_num_segments - 1)
    assert np.allclose(initial_condition_constraint.matrix, np.eye(dimension.total_dimension))
    assert np.allclose(initial_condition_constraint.bias, -input_components.initial_condition)
    assert initial_condition_constraint.selection_matrix.shape[0] == 5


def simple_pendulum_continuous_xdot(theta: float, theta_dot: float) -> np.ndarray:
    """Simple pendulum control matrix for verification."""
    grav_over_leng = -9.80655 / SinglePendulumDynamicsModel.LENG_PEND
    return np.array([theta_dot, grav_over_leng * np.sin(theta)])


def test_eval_collocation_constraints(simple_input_components, simple_optimization_array):
    """Test the evaluation of collocation constraints."""
    dynamic_constraint = OrthogonalCollocationConstraint(simple_input_components)
    constraints = dynamic_constraint.eval_collocation_constraints(simple_optimization_array)
    # There should be 3 constraints: 1 initial, 1 final, and 1 collocation
    assert len(constraints) == 3

    # Check the dynamic collocation constraint values
    expected_dynamic_eval = np.concatenate(
        [
            simple_pendulum_continuous_xdot(theta=0.0, theta_dot=0.0),
            simple_pendulum_continuous_xdot(theta=2 * np.pi / 3, theta_dot=0.0),
        ]
    )
    expected_differentiation_matrix = np.kron(2 * np.array([[-0.75, 0.75], [-0.75, 0.75]]), np.eye(2, 3))
    expected_differentiation_value = expected_differentiation_matrix @ np.array([0.0, 0.0, 0.0, 2 * np.pi / 3, 0.0, 0.0])
    expected_colloecation_value = expected_differentiation_value - expected_dynamic_eval / 2
    assert np.allclose(constraints[0], expected_colloecation_value, rtol=1e-3)

    # In this setup both intial and final conditions constraints evaluate to zero
    assert np.allclose(constraints[1], np.zeros(simple_input_components.dimension.total_dimension))
    assert np.allclose(constraints[2], np.zeros(simple_input_components.dimension.total_dimension))


def test_eval_collocation_jacobians(simple_input_components, simple_optimization_array):
    """Test the evaluation of collocation constraint jacobians."""
    dynamic_constraint = OrthogonalCollocationConstraint(simple_input_components)
    jacobians = dynamic_constraint.eval_collocation_jacobians(simple_optimization_array)
    # Compute Jacobians via finite difference perturbation for verification
    jacobians_by_perturbation = get_constraint_jacobian_by_perturbation(
        dynamic_constraint.eval_collocation_constraints, simple_optimization_array
    )

    # There should be 3 constraints: 1 initial, 1 final, and 1 collocation
    assert len(jacobians) == len(jacobians_by_perturbation) == 3
    for jacobian, jacobian_by_perturbation in zip(jacobians, jacobians_by_perturbation):
        actual_jacobian = jacobian.matrix @ jacobian.selection_matrix
        assert np.allclose(actual_jacobian, jacobian_by_perturbation.matrix, rtol=1e-2)


@pytest.mark.skip(reason="Disabled until get_linearized_collocation_constraints is developed.")
def test_linearized_collocation_constraints(
    simple_dynamics_model, simple_discretizer, simple_optimization_array, simple_initial_condition, simple_final_condition
):
    """Test the formulation of collocation constraints."""
    dynamic_constraint = OrthogonalCollocationConstraint(
        simple_dynamics_model, simple_discretizer, simple_initial_condition, simple_final_condition
    )
    constraints = dynamic_constraint.get_linearized_collocation_constraints(simple_optimization_array)
    collocation_constraint = constraints[0]

    # There should be 3 constraints: 1 initial, 1 final, and 1 collocation
    assert len(constraints) == 3

    interpolator = simple_discretizer.get_interpolator_for_segment(0)
    # Manual reformulation of the interpolator for verification - check barycentric weights
    expected_interpolated_weights = np.array([-0.75, 0.75])
    assert np.allclose(interpolator.barycentric_weights.squeeze(), expected_interpolated_weights)

    # Manual reformulation of the interpolator for verification - check differentiation matrix
    expected_differentiation_matrix = 2 * np.array([[-0.75, 0.75], [-0.75, 0.75]])
    assert np.allclose(interpolator.differentiation_matricies.squeeze(), expected_differentiation_matrix)

    expected_differentiation_term = np.kron(expected_differentiation_matrix, np.eye(2, 3))

    def simple_pendulum_state_matrix(theta: float) -> np.ndarray:
        """Simple pendulum state matrix for verification."""
        grav_over_leng = -9.80655 / simple_dynamics_model.LENG_PEND
        return np.array([[0, 1], [grav_over_leng * np.cos(theta), 0]])

    # Manual reformulation of the dynamic term for verification - compute state matrix
    expected_state_matrix_point_0 = simple_pendulum_state_matrix(0.0)
    expected_state_matrix_point_1 = simple_pendulum_state_matrix(2 * np.pi / 3)

    def simple_pendulum_control_matrix(theta: float) -> np.ndarray:
        """Simple pendulum control matrix for verification."""
        pen_moi = simple_dynamics_model.MASS_PEND * simple_dynamics_model.LENG_PEND**2
        return np.array([[0], [1 / pen_moi]])

    # Manual reformulation of the dynamic term for verification - compute control matrix
    expected_control_matrix_point_0 = simple_pendulum_control_matrix(0.0)
    expected_control_matrix_point_1 = simple_pendulum_control_matrix(2 * np.pi / 3)

    expected_dynamic_term_point_0 = np.hstack((expected_state_matrix_point_0, expected_control_matrix_point_0))
    expected_dynamic_term_point_1 = np.hstack((expected_state_matrix_point_1, expected_control_matrix_point_1))
    expected_dynamic_term = block_diag(expected_dynamic_term_point_0, expected_dynamic_term_point_1)

    # Manual reformulation of the collocation for verification - check time invariant partials
    time_invariant_partials = expected_differentiation_term - expected_dynamic_term / 2
    assert np.allclose(collocation_constraint.matrix[:, :-1], time_invariant_partials, rtol=1e-3)

    # Manual reformulation of the dynamic term for verification - compute continuous xdot
    expected_xdot_point_0 = simple_pendulum_continuous_xdot(0.0, 0.0)
    expected_xdot_point_1 = simple_pendulum_continuous_xdot(2 * np.pi / 3, 0.0)
    expected_xdot = np.vstack((expected_xdot_point_0, expected_xdot_point_1))

    # Manual reformulation of the collocation for verification - check time partials
    time_partials = -expected_xdot / 2
    assert np.allclose(collocation_constraint.matrix[:, -1:], time_partials, rtol=1e-3)
