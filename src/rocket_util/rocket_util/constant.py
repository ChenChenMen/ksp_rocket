"""Constants used in evaluations."""

from rocket_util.units import Q_
from rocket_util.scaler import Scaler

## Celestial body constants
# Standard gravitational acceleration (m/s^2)
G0 = Q_(1, "g0")
# Standard gravitational parameter (m^3/s^2)
MU = Q_(3.986e14, "m^3/s^2")
# Gravitational constant (m^3 kg^-1 s^-2)
GC = Q_(6.67430e-11, "m^3 kg^-1 s^-2")


class ConstantsProvider:
    """A class to provide constants with possible one off scaling."""

    def __init__(self, scaler: Scaler = None):
        """Initialize constants with an optional scaling function."""
        self._scaler = scaler
        self._scaled_flags = {}

    def __getattribute__(self, name):
        """Get the attribute, applying scaling if a scaler is defined."""
        value = super().__getattribute__(name)
        if self._scaler is not None and isinstance(value, Q_) and name not in self._scaled_flags:
            value = self._scaler.scale(value)
            setattr(self, name, value)
            self._scaled_flags[name] = True
        return value
