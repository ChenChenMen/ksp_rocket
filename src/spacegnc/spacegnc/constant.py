"""Constants used in evaluations."""

from spacegnc.units import Q_
from spacegnc.scaler import Scaler


class ConstantNotDefinedError(KeyError):
    """Requested constant is not defined in constant provider."""


class ScaledConstantProvider:
    """A class to provide constants with possible one off scaling."""

    # Define an universal constant scaler
    scaler: Scaler = None

    @classmethod
    def set_scaler(cls, scaler: Scaler):
        """Set a universal scaler for all instances of this class."""
        cls.scaler = scaler

    def __init__(self, **stored_constants):
        """Initialize constants with an optional scaling function."""
        self._stored_constant: dict = {
            key: (
                self._get_magnitude(value) if not isinstance(value, Q_) or self.scaler is None else self.scaler.scale(value)
            )
            for key, value in stored_constants.items()
        }

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
        return self._stored_constant[name]
