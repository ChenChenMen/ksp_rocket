"""Define the problem formulation for orthogonal collocation trajectory optimization."""

from ascent_trajopt.dynamics.base import DynamicsModel
from ascent_trajopt.dynamics.array_store import DynamicVariablesArray
from ascent_trajopt.orthogonal_collocation.array_store import OptimizationArray
from ascent_trajopt.orthogonal_collocation.constraints import OrthogonalCollocationConstraint
from ascent_trajopt.orthogonal_collocation.discretizer import HPDiscretizer
from ascent_trajopt.orthogonal_collocation.initial_guess import guess_from_linear_interpolation


class OrthogonalCollocationProblem:
    """Define an orthogonal collocation problem for trajectory optimization."""

    def __init__(
        self,
        dynamics_model: DynamicsModel,
        discretizer: HPDiscretizer,
        initial_condition: DynamicVariablesArray,
        final_condition: DynamicVariablesArray,
        initial_guess: OptimizationArray = None,
    ):
        """Initialize the optimization problem."""
        # Store the initial and final conditions to track
        self.initial_condition = initial_condition
        self.final_condition = final_condition

        # Store the dynamics model to track
        self.dynamics_model = dynamics_model
        # Create a discretizer instance to track
        self.discretizer = discretizer

        # Compute the initial guess for the optimization variables
        self.initial_guess = initial_guess or guess_from_linear_interpolation(
            discretizer, initial_condition, final_condition
        )

        # Create the constraint object
        self.collocation_constraints = OrthogonalCollocationConstraint(
            dynamics_model, discretizer, initial_condition, final_condition
        )
