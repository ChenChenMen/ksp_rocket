"""Define a collocation constraints for trajectory optimization."""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import block_diag

from ascent_trajopt.dynamics.array_store import DynamicModelDimension, DynamicVariablesArray
from ascent_trajopt.dynamics.base import DynamicsModel
from ascent_trajopt.orthogonal_collocation.array_store import OptimizationArray
from ascent_trajopt.orthogonal_collocation.discretizer import HPDiscretizer


@dataclass
class LinearEqualityConstraint:
    """Define an equality constraint for optimization."""

    # Define the constraint in the form of A @ x + b = 0
    matrix: np.ndarray
    bias: np.ndarray

    # Track a slice to downselect variables from the optimization vector
    index_slice: tuple[int, int] = None


class OrthogonalCollocationConstraint:
    """Formulate orthogonal collocation constraints with dynamic model."""

    def __init__(
        self,
        dynamics_model: DynamicsModel,
        discretizer: HPDiscretizer,
        initial_condition: DynamicVariablesArray,
        final_condition: DynamicVariablesArray,
    ):
        """Initialize the optimization problem."""
        self.dynamics_model = dynamics_model
        self.discretizer = discretizer

        self._dimension = DynamicModelDimension.from_dynamic_model(dynamics_model)
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

            segment_differentiation_matrix = np.kron(
                differentiation_matrix,
                np.eye(self._dimension.num_state, self._dimension.total_dimension),
            )
            # Record differentiation matrix for the segment
            self._differentiation_matrix_collection.append(segment_differentiation_matrix)

            # Format the max bound interpolated weights
            max_bound_interpolated_weights = interpolator.weights_for_value_at(
                self.discretizer.get_end_time_for_segment(segment_index)
            ).squeeze()

            if segment_index < self.discretizer.total_num_segments - 1:
                interpolated_matrix = np.kron(max_bound_interpolated_weights, total_dimension_eye_matrix)
                segement_connection_matrix = np.hstack((interpolated_matrix, total_dimension_eye_matrix))
                self._segment_connection_constraints.append(
                    LinearEqualityConstraint(
                        matrix=segement_connection_matrix,
                        bias=np.zeros(self._dimension.total_dimension),
                        index_slice=(start_index, end_index + self._dimension.total_dimension),
                    )
                )

        # Formulate final condition constraint
        final_start_index, final_end_index = OptimizationArray.segment_point_index_slice(
            self.discretizer, self._dimension, self.discretizer.total_num_segments - 1
        )
        self._boundary_condition_constraints.append(
            LinearEqualityConstraint(
                matrix=np.kron(max_bound_interpolated_weights, total_dimension_eye_matrix),
                bias=-final_condition,
                index_slice=(final_start_index, final_end_index),
            )
        )

        # Formulate initial condition constraint
        initial_start_index, initial_end_index = OptimizationArray.point_index_slice(
            self.discretizer, self._dimension, 0
        )
        self._boundary_condition_constraints.append(
            LinearEqualityConstraint(
                matrix=total_dimension_eye_matrix,
                bias=-initial_condition,
                index_slice=(initial_start_index, initial_end_index),
            )
        )

    def get_linearized_collocation_constraints(self, optimization_array: OptimizationArray):
        """Formulate the collocation as a linear constraint.

        Takes in the system dyanmic model, f(x, u) and the discretization's
        differentiation matrix D, considering the below collocation constraints

            D' @ x - t / 2 * f(x, u) = 0

        for each segment where x is the state vector at a single collocation point,
        u is the control vector at a single collocation point, and t is the total
        time duration of the entire trajectorty.

        The caveat here is that the differentiation matrix D' is not direct obtained
        from the interpolator. Instead, the interpolator produces a differentiation
        matrix D that describes the coupling between the same state variable across
        all collocation points in a single segment.

        The goal is to formulate it into the linear constraint form below

            A @ X + b = 0

        where X is the double stacked state and control vector at all collocation
        points of a single segement and all segments of the problem while append
        the total time duration t to the end, following the order below

            X0 = x_00, x_01, ..., x_0n, u_00, u_01, ..., u_0m

            X = x_10, x_11, ..., x_1n, u_10, u_11, ..., u_1m +
                x_20, x_21, ..., x_2n, u_20, u_21, ..., u_2m +
                ... +
                x_k0, x_k1, ..., x_kn, u_k0, u_k1, ..., u_km + t
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
            jacobian_matrix_collection = []
            for point_index, dynamic_variable_array in enumerate(segment_dynamic_variable_arrays):
                state_matrix = self.dynamics_model.continuous_state_matrix(
                    np.atleast_1d(segment_points[point_index]),
                    dynamic_variable_array.state,
                    dynamic_variable_array.control,
                )
                input_matrix = self.dynamics_model.continuous_input_matrix(
                    np.atleast_1d(segment_points[point_index]),
                    dynamic_variable_array.state,
                    dynamic_variable_array.control,
                )
                jacobian_matrix_collection.append(np.hstack((state_matrix, input_matrix)))

            # Construct the segment jacobian matrix from the point jocobian matrices
            segment_jacobian_matrix = time_scale_factor * block_diag(*jacobian_matrix_collection)
            segment_differentiation_matrix = self._differentiation_matrix_collection[segment_index]

            # Format the collocation constraint
            start_index, end_index = OptimizationArray.segment_point_index_slice(
                self.discretizer, self._dimension, segment_index
            )
            segment_linear_constraints.append(
                LinearEqualityConstraint(
                    matrix=segment_differentiation_matrix - segment_jacobian_matrix,
                    bias=np.zeros(end_index - start_index),
                    index_slice=(start_index, end_index),
                )
            )

        # Append up the pre-definded linearized constraints
        segment_linear_constraints.extend(self._segment_connection_constraints)
        segment_linear_constraints.extend(self._boundary_condition_constraints)
        return segment_linear_constraints
