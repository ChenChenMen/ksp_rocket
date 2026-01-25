"""Define the problem formulation for orthogonal collocation trajectory optimization."""

from optimization.functional import BaseProblem

from ascent_trajopt.orthogonal_collocation.array_store import OptimizationArray
from ascent_trajopt.orthogonal_collocation.components import ProblemInputComponents
from ascent_trajopt.orthogonal_collocation.constraints import OrthogonalCollocationConstraint
from ascent_trajopt.orthogonal_collocation.initial_guess import guess_from_linear_interpolation


class OrthogonalCollocationProblem(BaseProblem):
    """Define an orthogonal collocation problem for trajectory optimization."""

    def __init__(self, problem_input: ProblemInputComponents, initial_guess: OptimizationArray = None):
        """Initialize the optimization problem."""
        # Store the overall problem input
        self.problem_input = problem_input

        # Compute the initial guess for the optimization variables
        self.initial_guess = initial_guess or guess_from_linear_interpolation(problem_input)

        # Create the constraint object
        self.collocation_constraints = OrthogonalCollocationConstraint(problem_input)

    def eval_objective(self, optimization_array: OptimizationArray) -> float:
        """"""
