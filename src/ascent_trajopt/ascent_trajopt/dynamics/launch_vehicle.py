"""Define launch vehicle dynamics model."""

import jax.numpy as np

from ascent_trajopt.dynamics.base import DynamicsModel


class LaunchVehicleDynamics(DynamicsModel):
    """Launch vehicle dynamics model for trajectory optimization."""

    # Define required number of elements in the state/control vector
    # Position (3), velocity (3), orientation (4), angular velocity (3), mass (1)
    REQUIRED_STATE_NUM: int = 14

    # Control vector: throttle (1), thrust vector (2)
    REQUIRED_CTRL_NUM: int = 3

    def continuous_dynamics(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        """Compute the continuous dynamics of the launch vehicle."""
        # Extract state variables
        position = state[:3]
        velocity = state[3:6]
        orientation = state[6:10]
        angular_velocity = state[10:13]
        mass = state[13]

        # Extract control variables
        throttle = control[0]
        thrust_vector = control[1:]
