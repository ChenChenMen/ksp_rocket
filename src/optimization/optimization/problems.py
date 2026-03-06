"""Define basic optimization problem structures."""

from abc import ABCMeta, abstractmethod
from collections.abc import Generator

import numpy as np

from optimization.constraints import BaseConstraint, Bounds, ConstraintKind, LinearConstraint
from optimization.functional import LinearMap


class BaseProblem(metaclass=ABCMeta):
    """Define base class for optimization problem."""

    def __init__(self):
        """Process constraints and register lagrange multipliers."""
        # Create the constraint kind partitions
        self._equality_idx_dimension_map = {}
        self._inequality_idx_dimension_map = {}

        self.equality_total_dimension = None
        self.inequality_total_dimension = None

        # Process provided constraints
        self._process_constraints()

    @property
    @abstractmethod
    def constraints(self) -> list[BaseConstraint]:
        """Provide all constraints of the problem."""

    @abstractmethod
    def eval_objective(self, optimization_array: np.ndarray) -> float:
        """Evaluate the objective function given the current optimization array."""

    def eval_equality_constraints(self, optimization_array: np.ndarray) -> list[np.ndarray]:
        """Evaluate the equality constraints given the current optimization array."""
        eval_results = []
        for constraint in self.generate_equality_constraints():
            eval_results.extend(constraint.eval_constraints(optimization_array))
        return eval_results

    def eval_inequality_constraints(self, optimization_array: np.ndarray) -> list[np.ndarray]:
        """Evaluate the inequality constraints given the current optimization array."""
        eval_results = []
        for constraint in self.generate_inequality_constraints():
            eval_results.extend(constraint.eval_constraints(optimization_array))
        return eval_results

    def generate_equality_constraints(self) -> Generator[BaseConstraint, None, None]:
        """Generate a list of equality constraints."""
        return (self.constraints[idx] for idx in self._equality_idx_dimension_map)

    def generate_inequality_constraints(self) -> Generator[BaseConstraint, None, None]:
        """Generate a list of inequality constraints."""
        return (self.constraints[idx] for idx in self._inequality_idx_dimension_map)

    def _process_constraints(self):
        """Process the constraints by kinds and initialize lagrangian multipliers."""
        constraint_iter = (  # Construct the constraint iterator
            (idx, kind, dimension)
            for idx, constraint in enumerate(self.constraints)
            for kind, dimension in zip(constraint.kinds, constraint.dimensions)
        )

        # Accumulate the total dimension during iteration
        equality_dimension, inequality_dimension = 0, 0
        for idx, kind, dimension in constraint_iter:
            match kind:  # Populate idx and dimension mapping
                case ConstraintKind.EQUALITY:
                    self._equality_idx_dimension_map[idx] = dimension
                    equality_dimension += dimension
                case ConstraintKind.INEQUALITY_ABOVE | ConstraintKind.INEQUALITY_BELOW:
                    self._inequality_idx_dimension_map[idx] = dimension
                    inequality_dimension += dimension
                case _:
                    raise ValueError(f"Constraint kind {kind} is not supported.")

        self.equality_total_dimension = equality_dimension
        self.inequality_total_dimension = inequality_dimension


class LinearProgram(BaseProblem):
    """Define a generic linear program with constraints and bounds."""

    def __init__(self, objective: LinearMap, constraints: list[LinearConstraint] = None, bounds: Bounds = None):
        """Construct with predefined linear constraints and objective."""
        # Check the objective output to be size 1
        assert objective.output_size == 1, "Linear cost must be a scalar"

        self._objective = objective
        self._constraints = constraints or []
        self.bounds = bounds
        super().__init__()

    @property
    def constraints(self) -> list[BaseConstraint]:
        """Provide all constraints of the problem."""
        return self._constraints

    def eval_objective(self, optimization_array: np.ndarray) -> float:
        """Evaluate the objective function given the current optimization array."""
        return self._objective.evaluate_with(optimization_array)

    def get_objective_jacobian(self) -> LinearMap:
        """Get the jacobian of the linear objective."""
        return self._objective.jacobian
