"""Define a collocation constraints for trajectory optimization."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.linalg import block_diag

from ascent_trajopt.orthogonal_collocation.array_store import OptimizationArray
from ascent_trajopt.orthogonal_collocation.components import ProblemInputComponents
from rocket_util.common_decorators import under_development


@dataclass
class ConstraintJacobian:
    """Define a jacobian matrix for optimization constraint.

    The stored jacobian matrix is in size of m x n' where m is number of constraints,
    n is number of downselected optimization variables. The downselection is defined
    by segment_state_slice and end_with_time_partial from the expected optimization
    array size n.
    """

    # Jacobian matrix in size of m x n'
    matrix: np.ndarray
    # Downselection matrix in size of n' x n
    selection_matrix: np.ndarray = None

    @classmethod
    def from_slice(
        cls,
        matrix: np.ndarray,
        expected_optimization_array_length: int = None,
        segment_state_slice: tuple[int, int] = None,
        end_with_time_partial: bool = False,
    ) -> "ConstraintJacobian":
        """Get the slice matrix for the segment from optimization matrix.

        The slice matrix defines how to downselect the optimization array to apply
        the stored jacobian matrix. The slice matrix is in size of n' x n for the
        reason stated above. The return is very sparse with only one "1" per row.
        """
        # If no downselection is needed, return directly
        if segment_state_slice is None or expected_optimization_array_length is None:
            return cls(matrix, selection_matrix=None)

        downselected_length = segment_state_slice[1] - segment_state_slice[0]
        selection_matrix = np.hstack(
            (
                np.zeros((downselected_length, segment_state_slice[0])),
                np.eye(downselected_length),
                np.zeros((downselected_length, expected_optimization_array_length - segment_state_slice[1])),
            )
        )
        if end_with_time_partial:  # Append time partial selection row to the last
            time_selection_row = np.zeros((1, expected_optimization_array_length))
            time_selection_row[0, -1] = 1.0
            selection_matrix = np.vstack((selection_matrix, time_selection_row))

        return cls(matrix, selection_matrix)

    @classmethod
    def from_linear_equality_constraint(cls, linear_equality_constraint: "LinearEqualityConstraint") -> "ConstraintJacobian":
        """Formulate the constraint jacobian from linear equality constraint."""
        return cls(matrix=linear_equality_constraint.matrix, selection_matrix=linear_equality_constraint.selection_matrix)


@dataclass
class LinearEqualityConstraint:
    """Define an equality constraint for optimization.

    The stored A matrix is in size of m x n' where m is number of constraints,
    n is number of downselected optimization variables. The downselection is defined
    by segment_state_slice and end_with_time_partial from the expected optimization
    array size n.
    """

    # Define the constraint in the form of A @ x + b = 0
    matrix: np.ndarray
    bias: np.ndarray

    # Downselection matrix in size of n' x n, Nonetype assumes n' == n
    selection_matrix: np.ndarray = None

    def evaluate_with(self, optimization_array: OptimizationArray) -> np.ndarray:
        """Evaluate the linear equality constraint with the given optimization array."""
        optimization_ndarray = optimization_array.view(np.ndarray)
        if self.selection_matrix is None:
            return self.matrix @ optimization_ndarray + self.bias
        return self.matrix @ self.selection_matrix @ optimization_ndarray + self.bias

    @classmethod
    def from_slice(
        cls,
        matrix: np.ndarray,
        bias: np.ndarray,
        expected_optimization_array_length: int = None,
        segment_state_slice: tuple[int, int] = None,
        end_with_time_partial: bool = False,
    ) -> "LinearEqualityConstraint":
        """Formulate the linear equality constraint from slice parameters."""
        jacobian = ConstraintJacobian.from_slice(
            matrix, expected_optimization_array_length, segment_state_slice, end_with_time_partial
        )
        return cls.from_jacobian_bias(jacobian, bias)

    @classmethod
    def from_jacobian_bias(cls, jacobian: ConstraintJacobian, bias: np.ndarray) -> "LinearEqualityConstraint":
        """Formulate the linear equality constraint from jacobian and bias."""
        return cls(matrix=jacobian.matrix, bias=bias, selection_matrix=jacobian.selection_matrix)


class OrthogonalCollocationConstraint:
    """Formulate orthogonal collocation constraints with dynamic model.

    The constraints governed by this class is organized by an index-inferred list.
    Let the total number of segments be S, then the constraints are organized as below:
        1st patch: collocation constraints for all segments [nonlinear equality]
        2nd patch: connection constraints between segments [linear equality]
        3rd patch: final + initial condition constraints [linear equality]

    The input optimization array X is the double stacked state and control vector
    at all collocation points of a single segement and all segments of the problem
    while append the total time duration t to the end, following the order below

        optimization variable array
        X = x_00, x_01, ..., x_0n, u_00, u_01, ..., u_0m
            x_10, x_11, ..., x_1n, u_10, u_11, ..., u_1m +
            x_20, x_21, ..., x_2n, u_20, u_21, ..., u_2m +
            ... +
            x_k0, x_k1, ..., x_kn, u_k0, u_k1, ..., u_km + t

        initial condition
        X0 = x0_0, x0_1, ..., x0_n, u0_0, u0_1, ..., u0_m

        final condition
        Xf = xf_0, xf_1, ..., xf_n, uf_0, uf_1, ..., uf_m
    """

    def __init__(self, problem_input: ProblemInputComponents):
        """Initialize the optimization problem.

        Pre-compute the linear segment connection and boundary condition constraints,
        the 3rd patch, given the dynamics model, discretizer, and dimension without the
        actual optimization array due to linearality.
        """
        self.dynamics_model = problem_input.dynamics_model
        self.discretizer = problem_input.discretizer

        # Define dimension info for convenience
        self._dimension = problem_input.dimension
        self._optimization_array_length = OptimizationArray.get_expected_length(self.discretizer, self._dimension)

        # With discretizer and dynamics model, we can compute the collocation differentiation matrix
        self._differentiation_matrix_collection = []

        # Pre-determine the actually linear constraints
        self._segment_connection_constraints: list[LinearEqualityConstraint] = []
        self._boundary_condition_constraints: list[LinearEqualityConstraint] = []

        max_bound_interpolated_weights = None
        total_dimension_eye_matrix = np.eye(self._dimension.total_dimension)
        for segment_index in range(self.discretizer.total_num_segments):
            start_index, end_index = OptimizationArray.segment_point_index_slice(
                self.discretizer, self._dimension, segment_index
            )

            # Get the differentiation matrix for the segment
            interpolator = self.discretizer.get_interpolator_for_segment(segment_index)
            # The index of 0 here is because it's for a single segment
            differentiation_matrix = interpolator.differentiation_matricies[0]

            # Record differentiation matrix for the segment
            self._differentiation_matrix_collection.append(
                np.kron(differentiation_matrix, np.eye(self._dimension.num_state, self._dimension.total_dimension))
            )

            # Format the max bound interpolated weights
            max_bound_interpolated_weights = interpolator.weights_for_value_at(
                self.discretizer.get_end_time_for_segment(segment_index)
            ).squeeze()

            if segment_index < self.discretizer.total_num_segments - 1:
                interpolated_matrix = np.kron(max_bound_interpolated_weights, total_dimension_eye_matrix)
                segement_connection_matrix = np.hstack((interpolated_matrix, -total_dimension_eye_matrix))
                self._segment_connection_constraints.append(
                    LinearEqualityConstraint.from_slice(
                        matrix=segement_connection_matrix,
                        bias=np.zeros(self._dimension.total_dimension),
                        expected_optimization_array_length=self._optimization_array_length,
                        segment_state_slice=(start_index, end_index + self._dimension.total_dimension),
                    )
                )

        # Formulate final condition constraint
        final_start_index, final_end_index = start_index, end_index
        self._boundary_condition_constraints.append(
            LinearEqualityConstraint.from_slice(
                matrix=np.kron(max_bound_interpolated_weights, total_dimension_eye_matrix),
                bias=-problem_input.final_condition.view(np.ndarray),
                expected_optimization_array_length=self._optimization_array_length,
                segment_state_slice=(final_start_index, final_end_index),
            )
        )

        # Formulate initial condition constraint
        self._boundary_condition_constraints.append(
            LinearEqualityConstraint.from_slice(
                matrix=total_dimension_eye_matrix,
                bias=-problem_input.initial_condition.view(np.ndarray),
                expected_optimization_array_length=self._optimization_array_length,
                segment_state_slice=OptimizationArray.point_index_slice(self.discretizer, self._dimension, 0),
            )
        )

    def eval_collocation_constraints(self, optimization_array: OptimizationArray) -> list[np.ndarray]:
        """Evaluate collocation constraints given the current optimization array.

        === 1st patch: dynamic constraints ===
        Takes in the system dyanmic model, f(x, u) and the discretization's
        differentiation matrix D, considering the below equality collocation constraints

            c_col(x, u, t) = 0

        where c_col(x, u, t) = D' @ x - t / 2 * f(x, u)
            x is the state vector at a single collocation point
            u is the control vector at a single collocation point
            t is the total time duration of the entire trajectorty

        The caveat here is that the differentiation matrix D' is not direct obtained
        from the interpolator. Instead, the interpolator produces a differentiation
        matrix D that describes the coupling between the same state variable across
        all collocation points in a single segment.

        === 2nd patch and 3rd patch: pre-computed linear constraints ===
        These constraints are pre-computed during initialization, simply multiplying
        with the current optimization array.
        """
        evaluated_constraints = []
        time_scale_factor = optimization_array.time / 2

        for segment_index, segment_points in enumerate(self.discretizer):
            segment_dynamic_variable_arrays = optimization_array.segment(segment_index)

            # Collect the dynamic model evaluation result for the segment
            evaluated_xdot_collection = []
            for point_index, dynamic_variable_array in enumerate(segment_dynamic_variable_arrays):
                current_time = np.atleast_1d(segment_points[point_index] * optimization_array.time)
                current_state = dynamic_variable_array.state
                current_control = dynamic_variable_array.control

                # Evaluate dynamic function with the current state and control
                xdot = self.dynamics_model.xdot(current_time, current_state, current_control)
                evaluated_xdot_collection.append(xdot)

            # Compute the dynamics constraint
            differentiation_matrix = self._differentiation_matrix_collection[segment_index]
            segment_cumulated_array = optimization_array.segment_as_ndarray(segment_index)
            evaluated_xdot_array = np.concatenate(evaluated_xdot_collection)
            evaluated_constraints.append(
                differentiation_matrix @ segment_cumulated_array - time_scale_factor * evaluated_xdot_array
            )

        evaluated_constraints.extend(  # Append the segment connection constraints [linear]
            constraint.evaluate_with(optimization_array) for constraint in self._segment_connection_constraints
        )
        evaluated_constraints.extend(  # Append the boundary condition constraints [linear]
            constraint.evaluate_with(optimization_array) for constraint in self._boundary_condition_constraints
        )
        return evaluated_constraints

    def eval_collocation_jacobians(self, optimization_array: OptimizationArray) -> list[ConstraintJacobian]:
        """Evaluate the jacobian of collocation constraints given the current optimization array.

        The jacobian matrices are organized in the same order as eval_collocation_constraints.

        Since the collocation constraints are mostly linear except for the dynamic
        function evaluation f(x, u), the jacobian is only non-trivial for the dynamic
        function partials.

        === 1st patch: dynamic constraints ===
        From c_col definition in eval_collocation_constraint, the jacobian for,
        for each segment, the collocation constraints is given in the form of

            J = [d c_col(x, u, t) / dx, d c_col(x, u, t) / du, d c_col(x, u, t) / dt]

        === 2nd patch and 3rd patch: pre-computed linear constraints ===
        Simply return the pre-computed constraint matrix.
        """
        evaluated_jacobian = []
        time_scale_factor = optimization_array.time / 2

        for segment_index, segment_points in enumerate(self.discretizer):
            # Obtain the linearized dynamic system given by the A and B matrices
            segment_dynamic_variable_arrays = optimization_array.segment(segment_index)

            # Iterate each point in segment to construct the state matrices
            time_partial_collection, state_partial_collection = [], []
            for point_index, dynamic_variable_array in enumerate(segment_dynamic_variable_arrays):
                current_time = np.atleast_1d(segment_points[point_index] * optimization_array.time)
                current_state = dynamic_variable_array.state
                current_control = dynamic_variable_array.control

                # Compute continuous state and input matrix for linearized model
                state_matrix = self.dynamics_model.continuous_state_matrix(current_time, current_state, current_control)
                input_matrix = self.dynamics_model.continuous_input_matrix(current_time, current_state, current_control)
                # Construct time invariant dynamic model jacobian per point
                state_partial_collection.append(np.hstack((state_matrix, input_matrix)))

                # Evaluate dynamic function at the point
                xdot = self.dynamics_model.xdot(current_time, current_state, current_control)
                time_partial_collection.append(xdot)

            # Construct the segment state partial from the point jocobian collection
            segment_state_partial = time_scale_factor * block_diag(*state_partial_collection)
            segment_differentiation_matrix = self._differentiation_matrix_collection[segment_index]
            # Construct the segment time partial from the point dynamic function evaluations
            segment_time_partial = -0.5 * np.atleast_2d(np.concatenate(time_partial_collection)).T

            # Construct the segment jacobian
            segment_jacobian = np.hstack((segment_differentiation_matrix - segment_state_partial, segment_time_partial))

            # Format the collocation constraint jacobian
            evaluated_jacobian.append(
                ConstraintJacobian.from_slice(
                    matrix=segment_jacobian,
                    expected_optimization_array_length=self._optimization_array_length,
                    segment_state_slice=OptimizationArray.segment_point_index_slice(
                        self.discretizer, self._dimension, segment_index
                    ),
                    end_with_time_partial=True,
                )
            )

        # Append up the pre-definded linearized constraints
        evaluated_jacobian.extend(  # Append the segment connection constraints [linear]
            ConstraintJacobian.from_linear_equality_constraint(constraint)
            for constraint in self._segment_connection_constraints
        )
        evaluated_jacobian.extend(  # Append the boundary connection constraints [linear]
            ConstraintJacobian.from_linear_equality_constraint(constraint)
            for constraint in self._boundary_condition_constraints
        )
        return evaluated_jacobian

    @under_development("No need to linearizing collocation constraint until SQP solver implementation.")
    def get_linearized_collocation_constraints(
        self, optimization_array: OptimizationArray
    ) -> list[LinearEqualityConstraint]:
        """Formulate linearized collocation constraint defined in eval_collocation_constraint.

        === 1st patch: dynamic constraints ===
        From c_col definition in eval_collocation_constraint, the linearized constraint for the
        collocation constraints is given in the form of

            c_col_linear(X) = A @ X + b

        where A = D' - t / 2 * jacobian(f) - 1/2 * f(x, u) at X_current
            b = c_col(X_current) - A @ X_current
        """
        # Stand up the linear constraint matrix
        segment_linear_constraints = []
        segment_index = None

        # Compute time duration scaling factor
        time_scale_factor = optimization_array.time / 2

        for segment_index, segment_points in enumerate(self.discretizer):
            # Obtain the linearized dynamic system given by the A and B matrices
            segment_dynamic_variable_arrays = optimization_array.segment(segment_index)

            # Iterate each point in segment to construct the state matrices
            time_partial_collection, state_partial_collection = [], []
            for point_index, dynamic_variable_array in enumerate(segment_dynamic_variable_arrays):
                current_time = np.atleast_1d(segment_points[point_index] * optimization_array.time)
                current_state = dynamic_variable_array.state
                current_control = dynamic_variable_array.control

                # Compute continuous state and input matrix for linearized model
                state_matrix = self.dynamics_model.continuous_state_matrix(current_time, current_state, current_control)
                input_matrix = self.dynamics_model.continuous_input_matrix(current_time, current_state, current_control)
                # Construct time invariant dynamic model jacobian per point
                state_partial_collection.append(np.hstack((state_matrix, input_matrix)))

                # Evaluate dynamic function at the point
                xdot = self.dynamics_model.xdot(current_time, current_state, current_control)
                time_partial_collection.append(xdot)

            # Construct the segment state partial from the point jocobian collection
            segment_state_partial = time_scale_factor * block_diag(*state_partial_collection)
            segment_differentiation_matrix = self._differentiation_matrix_collection[segment_index]
            # Construct the segment time partial from the point dynamic function evaluations
            segment_time_partial = -0.5 * np.atleast_2d(np.concatenate(time_partial_collection)).T

            # Construct the segment jacobian
            segment_jacobian = np.hstack((segment_differentiation_matrix - segment_state_partial, segment_time_partial))

            # Format the collocation constraint
            start_index, end_index = OptimizationArray.segment_point_index_slice(
                self.discretizer, self._dimension, segment_index
            )
            segment_linear_constraints.append(
                LinearEqualityConstraint(
                    matrix=segment_jacobian,
                    bias=np.zeros(end_index - start_index),
                    end_with_time_partial=True,
                    segment_state_slice=(start_index, end_index),
                )
            )

        # Append up the pre-definded linearized constraints
        segment_linear_constraints.extend(self._segment_connection_constraints)
        segment_linear_constraints.extend(self._boundary_condition_constraints)
        return segment_linear_constraints


def get_constraint_jacobian_by_perturbation(
    constraint_function: Callable[[OptimizationArray], list[np.ndarray]],
    optimization_array: OptimizationArray,
    perturbation_percentage: float = 1e-3,
    min_perturbation: float = 1e-5,
) -> list[ConstraintJacobian]:
    """Numerically compute the constraint jacobian via finite difference perturbation."""
    num_variables = optimization_array.expected_length

    jacobian_partials: list[list[np.ndarray]] = []
    for var_index in range(num_variables):
        # Compute perturbation size
        perturbation = max(abs(optimization_array[var_index]) * perturbation_percentage, min_perturbation)

        # Perturb positively
        perturbed_array_pos = optimization_array.copy()
        perturbed_array_pos[var_index] += perturbation
        constraint_pos_list = constraint_function(perturbed_array_pos)

        # Perturb negatively
        perturbed_array_neg = optimization_array.copy()
        perturbed_array_neg[var_index] -= perturbation
        constraint_neg_list = constraint_function(perturbed_array_neg)

        for idx, (constraint_pos, constraint_neg) in enumerate(zip(constraint_pos_list, constraint_neg_list)):
            partial = (constraint_pos - constraint_neg) / (2 * perturbation)
            if len(jacobian_partials) == idx:
                jacobian_partials.append([])
            jacobian_partials[idx].append(np.atleast_2d(partial).T)

    return [ConstraintJacobian(matrix=np.concatenate(jacobian_partial, axis=1)) for jacobian_partial in jacobian_partials]
