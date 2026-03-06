"""Implement an interior point solver."""

import numpy as np

from optimization.problems import LinearProgram


class LinearProgramInteriorPointSolver:
    """Define an interior point solver.

    The solver logic considers a standard formed LP problem in form of
        min c.T @ x  s.t. A @ x = b, x ≥ 0

    See `preconditioner` module for standardization logic

    Dual problem:
        max b.T @ y  s.t. A.T @ y + s = c, s ≥ 0

    stationarity:        A.T @ y + s - c = 0
    primal feasibility:        A @ x - b = 0
    dual feasibility:           x ≥ 0, s ≥ 0
    complementarity:               x * s = 0
    """

    def __init__(self, problem: LinearProgram):
        """Accept a linear programming problem and precondition."""
        # Assert the problem is in standard form
        assert self._problem.inequality_total_dimension == 0, (
            "The problem is not in standard form. Please precondition the problem first."
        )

        self._problem = problem

        # Initialize the slack variable for the non-negativity constraint and the
        # lagrangian multipliers for the equality constraints
        self._lagrangian_multipliers = None
        self._slack_variable = None

    def _use_default_initial_guess(self):
        """Use a default initial guess for the optimization array."""
        

    def solve(self, initial_guess: np.ndarray, max_iterations: int = 100, rtol: float = 1e-3):
        """Solve the linear programming problem."""
        # Compute the residual for the KKT system and solve for the search direction
        stationarlity_res = self._problem.get_objective_jacobian()
        primal_feasibility_res = self._problem.eval_equality_constraints(initial_guess)


def solve_linear_program_interior_point_method(problem: LinearProgram):
    """Implement interior point method to solve linear program.

    The solver logic considers a standard formed LP problem in form of
    min c.T @ x  s.t. A @ x = b, x ≥ 0

    See `preconditioner` module for standardization logic
    """
