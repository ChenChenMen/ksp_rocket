"""Test and visualize the dynamics of the pendulum systems."""

import numpy as np

from ascent_trajopt.dynamics.pendulum import PendulumCartDynamicsModel


def test_pendulum_cart_dynamics(uncontrolled_propagator):
    """Test the pendulum cart dynamics model."""
    pendulum_cart = PendulumCartDynamicsModel()
    # Initial state: [cart position, cart velocity, pendulum angle, pendulum angular velocity]
    initial_state = np.array([np.pi, 0.001, 0.0, 0.0])

    # Propagate and plot the uncontrolled dynamics for 10 seconds
    uncontrolled_propagator(
        dynamics_model=pendulum_cart,
        initial_state=initial_state,
        propagation_duration=5.0,
        state_labels=["theta", "theta_dot", "position", "velocity"],
    )
