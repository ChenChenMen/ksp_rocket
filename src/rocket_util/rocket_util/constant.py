"""Constants used in evaluations."""

from rocket_util.units import Q_
from rocket_util.scaler import Scaler

## Celestial body constants =======================================
# Standard gravitational acceleration (m/s^2)
G0 = Q_(1, "g0")
# Gravitational constant (m^3 kg^-1 s^-2)
GC = Q_(6.67430e-11, "m^3 kg^-1 s^-2")

## Earth Specific constants
# Standard gravitational parameter (m^3/s^2)
MU_E = Q_(3.986e14, "m^3/s^2")
# Earth's radius (m)
RADIUS_E = Q_(6378137, "m")
# Earth's rotation rate (rad/s)
OMEGA_E = Q_(7.2921159e-5, "rad/s")
# Sea level atmospheric density (kg/m^3)
RHO_SEA_LEVEL_E = Q_(1.225, "kg/m^3")


class ConstantNotDefinedError(KeyError):
    """Requested constant is not defined in constant provider."""


class ConstantsProvider:
    """A class to provide constants with possible one off scaling."""

    def __init__(self, scaler: Scaler = None):
        """Initialize constants with an optional scaling function."""
        self._stored_constant: dict = None

        # Handle scaler related data structure
        self._scaler = scaler
        self._scaled_flags = {}

    def _get_magnitude(self, value: Q_ | float):
        """Strip units from a quantity if given."""
        return value.m if isinstance(value, Q_) else value

    def __dir__(self):
        """List all defined constants."""
        return list(self._stored_constant.keys()) if self._stored_constant else []

    def __getattr__(self, name):
        """Get the attribute, applying scaling if a scaler is defined."""
        if not self._stored_constant or name not in self._stored_constant:
            raise ConstantNotDefinedError(f"{name} is not defined in {self.__class__}")

        value = self._stored_constant[name]
        if self._scaler is not None and isinstance(value, Q_) and not self._scaled_flags.get(name, False):
            value = self._scaler.scale(value)
            self._scaled_flags[name] = True
            self._stored_constant[name] = value

        return self._get_magnitude(value)
