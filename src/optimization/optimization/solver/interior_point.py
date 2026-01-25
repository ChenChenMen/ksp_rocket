"""Implement an interior point solver."""

from optimization.problems import LinearProgram


class LinearProgramInteriorPointSolver:
    """Define an interior point solver.

    The solver logic considers a standard formed LP problem in form of
        min c.T @ x  s.t. A @ x = b, x ≥ 0

    See `preconditioner` module for standardization logic
    """

    def __init__(self, problem: LinearProgram):
        """Accept a linear programming problem and precondition."""
        self._problem = problem
        self._A = problem.equality_constraints if problem.equality_total_dimension > 0 else None

    def solve(self):
        """Solve the linear programming problem."""


def solve_linear_program_interior_point_method(problem: LinearProgram):
    """Implement interior point method to solve linear program."""
