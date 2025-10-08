"""Define launch vehicle dynamics model."""

import jax.numpy as np
from rocket_util.constant import G0, MU_E, OMEGA_E, RADIUS_E, RHO_SEA_LEVEL_E, ScaledConstantProvider

from ascent_trajopt.dynamics.base import DynamicsModel


class LaunchVehicleConstantProvider(ScaledConstantProvider):
    """Physical constants for translation-only launch vehicle dynamics."""

    def __init__(self):
        """Initialize translation-only constants with optional scaling."""
        super().__init__(
            G0=G0,
            MU_E=MU_E,
            OMEGA_E=OMEGA_E,
            RADIUS_E=RADIUS_E,
            RHO_SEA_LEVEL_E=RHO_SEA_LEVEL_E,
            RHO_E=self._atmo_density,
        )

    def _atmo_density(self, altitude: float) -> float:
        """Compute atmospheric density at a given altitude."""
        unscaled_si_altitude = self.scaler.unscale(altitude, "m")
        # Simple exponential model for atmosphere density
        return self.RHO_SEA_LEVEL_E * np.exp(-unscaled_si_altitude.m / 8500)


class LaunchVehicle3DOF(DynamicsModel):
    """Launch vehicle dynamics model for trajectory optimization.

    Extreme simplification of a launch vehicle in 3D space as a point mass without
    a concept of orientation and angular velocity.
    """

    # Define required number of elements in the state/control vector
    # Position (3), velocity (3), mass (1)
    REQUIRED_STATE_NUM: int = 7

    # Control vector: throttle (1), thrust vector in ECI (3)
    REQUIRED_CTRL_NUM: int = 4

    def __init__(self, launch_vehicle_properties: ScaledConstantProvider):
        """Initialize the launch vehicle dynamics model."""
        super().__init__()
        self._physical_properties = LaunchVehicleConstantProvider()
        self._launch_vehicle_properties = launch_vehicle_properties

    def continuous_dynamics(self, time, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        """Compute the continuous dynamics of the launch vehicle."""
        # Extract state variables
        position = state[:3]
        velocity = state[3:6]
        mass = state[6]

        # Extract control variables
        throttle = control[0]
        thrust_vector = control[1:]

        # Extract from launch vehicle properties
        max_thrust = self._launch_vehicle_properties.max_thrust
        thrust_magnitude = max_thrust * throttle
        reference_area = self._launch_vehicle_properties.reference_area
        drag_coefficient = self._launch_vehicle_properties.drag_coefficient
        isp = self._launch_vehicle_properties.isp

        # Intermediate calculations
        position_magnitude = np.linalg.norm(position)
        # Drag related intermediate calculations
        inverse_ballstic_coefficient = reference_area * drag_coefficient / mass
        atmo_relative_velocity = velocity - np.cross(self._physical_properties.OMEGA_E, position)
        atmo_relative_velocity_magnitude = np.linalg.norm(atmo_relative_velocity)
        # Thrust related intermediate calculations
        exhaust_velocity = self._physical_properties.G0 * isp

        # Extract from physical constants
        rho = self._physical_properties.RHO_E(position_magnitude - self._physical_properties.RADIUS_E)

        # Compute acceleration
        gravitational_acceleration = -self._physical_properties.MU_E / (position_magnitude**3) * position
        drag_acceleration = (
            -(inverse_ballstic_coefficient * rho / 2) * atmo_relative_velocity_magnitude * atmo_relative_velocity
        )
        thrust_acceleration = -thrust_vector * thrust_magnitude / mass
        total_acceleration = gravitational_acceleration + drag_acceleration + thrust_acceleration

        # Return the time derivative of the state vector
        return np.concatenate((velocity, total_acceleration, -throttle * thrust_magnitude / exhaust_velocity))


class LaunchVehicle6DOF(DynamicsModel):
    """Launch vehicle dynamics model for trajectory optimization.

    Simulating a point mass launch vehicle in 3D space with rigid body dynamics.
    The model neglects the launch vehicle's shape and therefore CG and body frame
    shifts. MoI changes are only result of varying mass.
    """

    # Define required number of elements in the state/control vector
    # Position (3), velocity (3), orientation (4), angular velocity (3), mass (1)
    REQUIRED_STATE_NUM: int = 14

    # Control vector: throttle (1), thrust vector (2)
    REQUIRED_CTRL_NUM: int = 3

    def continuous_dynamics(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        """Compute the continuous dynamics of the launch vehicle."""
        # # Extract state variables
        # position = state[:3]
        # velocity = state[3:6]
        # orientation = state[6:10]
        # angular_velocity = state[10:13]
        # mass = state[13]

        # # Extract control variables
        # throttle = control[0]
        # thrust_vector = control[1:]
