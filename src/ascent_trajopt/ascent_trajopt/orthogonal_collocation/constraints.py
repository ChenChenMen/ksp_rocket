"""Define a collocation constraints for trajectory optimization."""

import numpy as np

from ascent_trajopt.orthogonal_collocation.array_store import OptimizationArray
from ascent_trajopt.orthogonal_collocation.discretizer import HPDiscretizer
from ascent_trajopt.dynamics.base import DynamicsModel


class OrthogonalCollocationConstraint:
    """Formulate orthogonal collocation constraints with dynamic model."""

    def __init__(self, dynamics_model: DynamicsModel, discreitizer: HPDiscretizer):
        """Initialize the optimization problem."""
        self.dynamics_model = dynamics_model

        # Create a discretizer instance to track
        self.discretizer = discreitizer

    def formulate_collocation_constraints(self, optimization_array: OptimizationArray):
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

            X = x_00, x_01, ..., x_0n, u_00, u_01, ..., u_0m +
                x_10, x_11, ..., x_1n, u_10, u_11, ..., u_1m +
                ... +
                x_k0, x_k1, ..., x_kn, u_k0, u_k1, ..., u_km + t
        """
        # Stand up the linear constraint matrix
        segment_linear_constraint_matrices = []
        for segment_index, segment_points in enumerate(self.discretizer):
            # Get the differentiation matrix for the segment
            interpolator = self.discretizer.get_interpolator_for_segment(segment_index)
            # The index of 0 here is because it's for a single segment
            differentiation_matrix = interpolator.differentiation_matricies[0]

            # Add to the linear constraint matrix collection
            segment_linear_constraint_matrices.append(
                np.kron(differentiation_matrix, np.eye(self.dynamics_model.REQUIRED_STATE_NUM))
            )

            # Obtain the linearized dynamic system given by the A and B matrices
            segment_dynamic_variable_arrays = optimization_array.segment(segment_index)

            # Iterate each point in segment to construct the state matrices
            jacobian_matrix_collection = []
            for point_index, dynamic_variable_array in enumerate(segment_dynamic_variable_arrays):
                state_matrix = self.dynamics_model.continuous_state_matrix(
                    time=segment_points[point_index], state=dynamic_variable_array.state, control=dynamic_variable_array.control
                )
                input_matrix = self.dynamics_model.continuous_input_matrix(
                    time=segment_points[point_index], state=dynamic_variable_array.state, control=dynamic_variable_array.control
                )
                jacobian_matrix_collection.append(np.hstack((state_matrix, input_matrix)))

            # Construct the segment jacobian matrix from the point jocobian matrices
            state_dimension = optimization_array.dimension.num_state
            segment_jacobian_matrix = np.kron(np.eye(state_dimension), jacobian_matrix_collection)
