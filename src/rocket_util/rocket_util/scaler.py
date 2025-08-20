"""Define physical quantities scaler for optimization."""

from dataclasses import dataclass

import numpy as np

from rocket_util.units import Q_


class UnknownScalingBasisError(ValueError):
    """Raised when the scaling basis is not known."""


@dataclass
class ScalingInput:
    """Defines the input format to instantiate a scaler."""

    reference_unit_conversions: tuple[Q_]

    # Configure what unit basis to cover, default to mechanical units
    required_unit_basis: tuple[str] = ("[length]", "[mass]", "[time]")

    # Store the reference to basis transformation matrix
    _transformation_matrix: np.ndarray = None

    def __post_init__(self):
        """Post initialization check to see if unit basis are covered by given conversions."""
        # Construct a transformation matrix based on reference units' diemensions
        basis_length = len(self.required_unit_basis)
        transformation_matrix = np.zeros((len(self.reference_unit_conversions), basis_length))
        for ref_index, reference_unit in enumerate(self.reference_unit_conversions):
            # Sort the dimensions of the reference unit by required unit basis order
            for base_unit_key, dimension in reference_unit.dimensionality.items():
                if base_unit_key not in self.required_unit_basis:
                    raise UnknownScalingBasisError(
                        f"Unknown base unit '{base_unit_key}' in reference unit {reference_unit}."
                    )
                index = self.required_unit_basis.index(base_unit_key)
                # Add the dimension to the transformation matrix at the correct index
                transformation_matrix[ref_index, index] = dimension
        self._transformation_matrix = transformation_matrix

        assert np.linalg.matrix_rank(transformation_matrix) == len(self.required_unit_basis), (
            "Provided reference units do not suffice all required unit basis."
        )

    def get_transformation_matrix(self) -> np.ndarray:
        """Get the transformation matrix from reference input to required basis."""
        return self._transformation_matrix


class Scaler:
    """A class to scale physical quantities for optimization.

    The idea is to provide a couple anchor scales that can be used to
    derive scaling factors for basis units. Any input parameters will
    be scaled by the basis scaling factors based on their dimensions.
    """

    def __init__(self, scaling_input: ScalingInput):
        """Initialize the scaler with a scale factor."""
        unit_basis_scale_factor_map = {}
        # Inverse of the transformation matrix describes power of the reference units to get unit basis
        reference_power_matrix = np.linalg.pinv(scaling_input.get_transformation_matrix())
        for power_scaling_factors, unit_basis in zip(reference_power_matrix, scaling_input.required_unit_basis):
            # Have the scale factor map also act as a cache for higher powers of the unit basis
            unit_basis_scale_factor_map[(unit_basis, 1)] = np.prod(
                [
                    # Calculate the scale factor for each unit basis
                    np.power(np.abs(unit_conversion.to_base_units().magnitude), power)
                    for unit_conversion, power in zip(scaling_input.reference_unit_conversions, power_scaling_factors)
                ]
            )
        self.unit_basis_scale_factor_mapping = unit_basis_scale_factor_map

    def _get_scaling_factor(self, to_unit: str):
        """Obtain the scaling factor per dimension of the value into a tuple."""
        scaling_factor = 1
        unit = Q_(1, to_unit)
        for base_unit_key, dimension in unit.dimensionality.items():
            if (base_unit_key, dimension) in self.unit_basis_scale_factor_mapping:
                scaling_factor /= self.unit_basis_scale_factor_mapping[(base_unit_key, dimension)]
            elif (base_unit_key, 1) in self.unit_basis_scale_factor_mapping:
                scaling_value = np.power(self.unit_basis_scale_factor_mapping[(base_unit_key, 1)], dimension)
                self.unit_basis_scale_factor_mapping[(base_unit_key, dimension)] = scaling_value
                scaling_factor /= scaling_value
            else:
                raise UnknownScalingBasisError(f"Unknown base unit '{base_unit_key}' in unit {value}.")
        return scaling_factor.tolist()

    def scale(self, value: Q_ | list[Q_]) -> float | list[float]:
        """Scale the given value by the scale factor."""
        if isinstance(value, list):  # Scale each value in the list
            return [self.scale(v) for v in value]
        return value.m * self._get_scaling_factor(value.units)

    def unscale(self, value: float | list[float], to_unit: str) -> Q_ | list[Q_]:
        """Unscale the given value by the scale factor."""
        if isinstance(value, list):  # Scale each value in the list
            return [self.scale(v) for v in value]
        return Q_(value / self._get_scaling_factor(to_unit), to_unit)
