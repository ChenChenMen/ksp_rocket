"""Define basic constraint types."""

from abc import ABCMeta, abstractmethod
from enum import StrEnum, auto

import numpy as np

from optimization.differentiation import HessiansForJacobian, IdentityMatrixEntry, Jacobian
from optimization.functional import LinearMap


class ConstraintKind(StrEnum):
    """Define the constraint type."""

    EQUALITY = auto()
    INEQUALITY_BELOW = auto()  # A @ x ≤ b
    INEQUALITY_ABOVE = auto()  # A @ x ≥ b


class BaseConstraint(metaclass=ABCMeta):
    """Define base class for optimization constraint."""

    @property
    @abstractmethod
    def kinds(self) -> list[ConstraintKind]:
        """Provide constraint kind for each constraint batch."""

    @property
    @abstractmethod
    def dimensions(self) -> list[int]:
        """Provide dimension for each constraint batch."""

    @abstractmethod
    def eval_constraints(self, optimization_array: np.ndarray) -> list[np.ndarray]:
        """Evaluate the constraints given the current optimization array.

        Each constraint batch corresponds to one entry in the returned list.
        For inequaity constraints, the returned value should be non-negative when
        satisfied, that is the form:
            INEQUALITY_BELOW: b - A @ x ≥ 0
            INEQUALITY_ABOVE: A @ x - b ≥ 0
        """

    @abstractmethod
    def eval_jacobians(self, optimization_array: np.ndarray) -> list[Jacobian]:
        """Evaluate the jacobians of the constraints given the current optimization array."""

    @abstractmethod
    def eval_hessians(self, optimization_array: np.ndarray) -> list[HessiansForJacobian]:
        """Evaluate the hessians of the constraints given the current optimization array."""


class Bounds(BaseConstraint):
    """Define optimization variable bounds."""

    def __init__(self, upper_bound: np.ndarray = None, lower_bound: np.ndarray = None):
        """Track the upper and/or lower bound for the variable."""
        if upper_bound is None and lower_bound is None:
            raise ValueError("At least one bound must be provided")
        if upper_bound is not None and lower_bound is not None and upper_bound.size != lower_bound.size:
            raise ValueError("Inconsistent bound sizes")

        self._kinds, self._dimensions = [], []
        self._linear_representation: list[LinearMap] = []
        # Evaluate constraint kind for upper bounds
        if upper_bound is not None:
            upper_bound = np.atleast_1d(upper_bound)
            self._kinds.append(ConstraintKind.INEQUALITY_BELOW)
            self._dimensions.append(upper_bound.size)
            upper_bound_jacobian = Jacobian.from_slice(matrix=IdentityMatrixEntry(-1.0, upper_bound.size))
            self._linear_representation.append(LinearMap.from_jacobian_bias(upper_bound_jacobian, bias=upper_bound))

        # Evaluate constraint kind for lower bounds
        if lower_bound is not None:
            lower_bound = np.atleast_1d(lower_bound)
            self._kinds.append(ConstraintKind.INEQUALITY_ABOVE)
            self._dimensions.append(lower_bound.size)
            lower_bound_jacobian = Jacobian.from_slice(matrix=IdentityMatrixEntry(1.0, lower_bound.size))
            self._linear_representation.append(LinearMap.from_jacobian_bias(lower_bound_jacobian, bias=-lower_bound))

        self.upper = upper_bound
        self.lower = lower_bound

    @property
    def kinds(self) -> list[ConstraintKind]:
        """Define constraint kinds."""
        return self._kinds

    @property
    def dimensions(self) -> list[int]:
        """Define constraint dimensions."""
        return self._dimensions

    def eval_constraints(self, optimization_array: np.ndarray) -> list[np.ndarray]:
        """Evaluate bounds constraints at optimization array."""
        return [linear_map.evaluate_with(optimization_array) for linear_map in self._linear_representation]

    def eval_jacobians(self, _: np.ndarray) -> list[Jacobian]:
        """Evaluate jacobians for all bounds constraints."""
        return [linear_map.jacobian for linear_map in self._linear_representation]

    def eval_hessians(self, _: np.ndarray) -> list[HessiansForJacobian]:
        """Bounds have zero Hessian matrix."""
        return [HessiansForJacobian.empty_from_jacobian(linear_map.jacobian) for linear_map in self._linear_representation]


class LinearConstraint(BaseConstraint):
    """Define a generic linear constraint."""

    def __init__(self, linear_maps: list[LinearMap], constraint_kinds: list[ConstraintKind]):
        """Create linear constraint."""
        self._linear_maps = linear_maps
        self._jacobians = [linear_map.jacobian for linear_map in linear_maps]
        self._constraint_kinds = constraint_kinds

    @property
    def kinds(self) -> list[ConstraintKind]:
        """Define constraint kinds."""
        return self._constraint_kinds

    @property
    def dimensions(self) -> list[int]:
        """Define constraint dimensions."""
        return [np.atleast_1d(linear_map.bias).size for linear_map in self._linear_maps]

    @classmethod
    def from_single_jacobian_bias(cls, jacobian: Jacobian, bias: np.ndarray, constraint_kind: ConstraintKind):
        """Factory from a single pair of jacobian and bias array."""
        linear_map = LinearMap.from_jacobian_bias(jacobian, bias)
        return cls([linear_map], [constraint_kind])

    def eval_constraints(self, optimization_array: np.ndarray) -> list[np.ndarray]:
        """Evaluate constraints at optimization array."""
        return [linear_map.evaluate_with(optimization_array) for linear_map in self._linear_maps]

    def eval_jacobians(self, _: np.ndarray) -> list[Jacobian]:
        """Evaluate jacobians for all constraints."""
        return self._jacobians

    def eval_hessians(self, _: np.ndarray) -> list[HessiansForJacobian]:
        """Linear constraints have zero Hessian matrix."""
        return [HessiansForJacobian.empty_from_jacobian(jacobian) for jacobian in self._jacobians]
