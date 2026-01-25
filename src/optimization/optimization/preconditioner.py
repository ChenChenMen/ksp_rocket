"""Precondition the optimization problem."""


class LinearProgramStandardizer:
    """Standardize linear program into the positive and equality constrained form.

    The standardizer accepts a more general formed LP problem
        min c.T @ x  s.t. A @ x = b, A_ia @ x ≥ b_ia, A_ib @ x ≤ b_ib, ub ≥ x ≥ lb

    and construct an alternative LP problem in the standard form while tracking
    conversion between the standardized problem and the original problem.
        min c.T @ x  s.t. A @ x = b, x ≥ 0
    """
