"""Define a collocation constraints for trajectory optimization."""

import numpy as np
from scipy.linalg import block_diag

from optimization.constraints import BaseConstraint, ConstraintKind
from optimization.differentiation import Jacobian, HessiansForJacobian
from optimization.functional import LinearMap

from spacegnc.common_decorators import under_development
from spacegnc.orthogonal_collocation.array_store import OptimizationArray
from spacegnc.orthogonal_collocation.components import ProblemInputComponents


class OrthogonalCollocationConstraint(BaseConstraint):
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
        self._segment_connection_constraints: list[LinearMap] = []
        self._boundary_condition_constraints: list[LinearMap] = []

        self._collocation_constraint_kind: list[ConstraintKind] = []
        self._segment_connection_constraint_kind: list[ConstraintKind] = []

        self._collocation_constraint_dimension: list[int] = []
        self._segment_connection_constraint_dimension: list[int] = []

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
                    LinearMap.from_slice(
                        matrix=segement_connection_matrix,
                        bias=np.zeros(self._dimension.total_dimension),
                        selection_index_collection=[(start_index, end_index + self._dimension.total_dimension)],
                    )
                )
                self._segment_connection_constraint_kind.append(ConstraintKind.EQUALITY)
                self._segment_connection_constraint_dimension.append(self._dimension.total_dimension)

            # Register the constraint kind and theoratical dimensions
            self._collocation_constraint_kind.append(ConstraintKind.EQUALITY)
            self._collocation_constraint_dimension.append(self._dimension.num_state)

        # Formulate final condition constraint
        final_start_index, final_end_index = start_index, end_index
        self._boundary_condition_constraints.append(
            LinearMap.from_slice(
                matrix=np.kron(max_bound_interpolated_weights, total_dimension_eye_matrix),
                bias=-problem_input.final_condition.view(np.ndarray),
                selection_index_collection=[(final_start_index, final_end_index)],
            )
        )

        # Formulate initial condition constraint
        self._boundary_condition_constraints.append(
            LinearMap.from_slice(
                matrix=total_dimension_eye_matrix,
                bias=-problem_input.initial_condition.view(np.ndarray),
                selection_index_collection=[OptimizationArray.point_index_slice(self.discretizer, self._dimension, 0)],
            )
        )

        # Collect boundary condition constraint information
        self._boundary_condition_constraint_kind = [ConstraintKind.EQUALITY, ConstraintKind.EQUALITY]
        self._boundary_condition_constraint_dimension = [
            problem_input.final_condition.dimension.total_dimension,
            problem_input.initial_condition.dimension.total_dimension,
        ]

    @property
    def kinds(self) -> list[ConstraintKind]:
        """Collocation constraints are equality constraints."""
        return (
            self._collocation_constraint_kind
            + self._segment_connection_constraint_kind
            + self._boundary_condition_constraint_kind
        )

    @property
    def dimensions(self) -> list[int]:
        """Provide dimensions for each collocation constraint batch."""
        return (
            self._collocation_constraint_dimension
            + self._segment_connection_constraint_dimension
            + self._boundary_condition_constraint_dimension
        )

    def eval_constraints(self, optimization_array: OptimizationArray) -> list[np.ndarray]:
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

    def eval_jacobians(self, optimization_array: OptimizationArray) -> list[Jacobian]:
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
            segment_jacobian_matrix = np.hstack(
                (segment_differentiation_matrix - segment_state_partial, segment_time_partial)
            )

            # Format the collocation constraint jacobian
            selection_index_collection = [
                OptimizationArray.segment_point_index_slice(self.discretizer, self._dimension, segment_index),
                optimization_array.expected_length - 1,
            ]
            evaluated_jacobian.append(Jacobian.from_slice(segment_jacobian_matrix, selection_index_collection))

        # Append up the pre-definded linearized constraints
        evaluated_jacobian.extend(  # Append the segment connection constraints [linear]
            constraint.jacobian for constraint in self._segment_connection_constraints
        )
        evaluated_jacobian.extend(  # Append the boundary connection constraints [linear]
            constraint.jacobian for constraint in self._boundary_condition_constraints
        )
        return evaluated_jacobian

    def eval_hessians(self, optimization_array: OptimizationArray) -> list[HessiansForJacobian]:
        """Evaluate the hessians of collocation constraints given the current optimization array.

        This is used to compute Hessian of the Lagrangian with Lagrangian multiplier to squash the
        dimension. Therefore, this method returns a list ConstraintHessian, each represents a list
        of square matrices. The output size should match that of eval_jacobians.
        """

    @under_development("No need to linearizing collocation constraint until SQP solver implementation.")
    def get_linearized_collocation_constraints(self, optimization_array: OptimizationArray) -> list[LinearMap]:
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
                LinearMap(
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
