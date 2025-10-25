"""Define collection of components abstraction for the optimization problem."""

from dataclasses import dataclass

from ascent_trajopt.dynamics.base import DynamicsModel
from ascent_trajopt.dynamics.array_store import DynamicModelDimension, DynamicVariablesArray
from ascent_trajopt.orthogonal_collocation.discretizer import HPDiscretizer


@dataclass
class ProblemInputComponents:
    """Define the input to the orthogonal collocation problem."""

    dynamics_model: DynamicsModel
    discretizer: HPDiscretizer
    initial_condition: DynamicVariablesArray
    final_condition: DynamicVariablesArray

    # Internally inferred problem definition
    _dimension: DynamicModelDimension = None

    @property
    def dimension(self) -> DynamicModelDimension:
        """Get the dynamic model dimension."""
        return self._dimension

    def __post_init__(self):
        """Post-initialization to compute dimensions."""
        self._dimension = DynamicModelDimension.from_dynamic_model(self.dynamics_model)
        # Validation of initial and final conditions
        if self.initial_condition.dimension != self.dimension:
            raise ValueError(
                f"Initial condition dimension {self.initial_condition.dimension} does not "
                f"match dynamic model dimension {self.dimension}."
            )
        if self.final_condition.dimension != self.dimension:
            raise ValueError(
                f"Final condition dimension {self.final_condition.dimension} does not "
                f"match dynamic model dimension {self.dimension}."
            )
