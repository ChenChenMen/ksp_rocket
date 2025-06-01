"""Test dynamics model."""

import jax.numpy as np

from ascent_trajopt.dynamics.base import DynamicsModel
from rocket_util.constant import G0
from rocket_util.logconfig import create_logger

LOG = create_logger(__name__)


class SinglePendulumDynamicsModel(DynamicsModel):
    """Define a dynamics model for test usage."""
    # Define required number of elements in the state/control vector
    REQUIRED_STATE_NUM: int = 2
    REQUIRED_CTRL_NUM: int = 1

    # Define test model parameters
    MASS_PEND = 1
    LENG_PEND = 0.5

    def continuous_time_state_equation(self, time, state, control):
        """Define a continuous time state equation for tests.

        The system is time invariant

        Use the cart and inverted pendulum for example
        State:
            theta (pendulum position from stable equillibrium)
            theta_dot (pendulum counterclockwise rotational velocity)
        Control:
            force (external torque applied to the pendulum counterclockwise)
        """
        # Unpack state vector
        theta = state[0]
        theta_dot = state[1]

        # Unpack control vector
        force = control[0]

        theta_dotdot = force / (self.MASS_PEND * self.LENG_PEND ** 2) - G0 * np.sin(theta) / self.LENG_PEND
        # Formulate the state equation
        return np.asarray([theta_dot, theta_dotdot])


class PendulumCartDynamicsModel(DynamicsModel):
    """Define a dynamics model for test usage."""
    # Define required number of elements in the state/control vector
    REQUIRED_STATE_NUM: int = 4
    REQUIRED_CTRL_NUM: int = 1

    # Define test model parameters
    MASS_CART = 2
    MASS_PEND = 1
    LENG_PEND = 0.5

    def continuous_time_state_equation(self, time, state, control):
        """Define a continuous time state equation for tests.

        The system is time invariant

        Use the cart and inverted pendulum for example
        State:
            theta (pendulum position from stable equillibrium)
            theta_dot (pendulum counterclockwise rotational velocity)
            x (cart position from origin to the right)
            x_dot (cart velocity from origin to the right)
        Control:
            u (external force applied to the cart to the right)
        """
        # Unpack state vector
        theta = state[0]
        theta_dot = state[1]
        x_dot = state[3]

        # Unpack control vector
        force = control[0]

        # Define useful intermediate numeric result
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        sin_sq_theta = np.power(sin_theta, 2)
        sin_2theta = np.sin(2 * theta)

        total_mass = self.MASS_CART + self.MASS_PEND
        inv_total_mass = 1 / total_mass
        theta_dot_sq = np.power(theta_dot, 2)
        mass_leng_pend = self.MASS_PEND * self.LENG_PEND
        mass_leng_theta_dot_sq = mass_leng_pend * theta_dot_sq

        theta_dotdot = -1 / (self.MASS_CART + self.MASS_PEND * sin_sq_theta) / self.LENG_PEND * (
            0.5 * mass_leng_theta_dot_sq * sin_2theta + force * cos_theta + total_mass * G0 * sin_theta
        )
        x_dotdot = inv_total_mass * (force + mass_leng_theta_dot_sq * sin_theta + mass_leng_pend * theta_dotdot * cos_theta)

        # Formulate the state equation
        return np.asarray([theta_dot, theta_dotdot, x_dot, x_dotdot])


class TestDynamicsModel:
    """Test dynamics model functionalities with pendulum cart system."""

    def test_stable_equilibrium_perturbation_single_timestamp(self):
        """Verify linearlization close to stable equilibrium."""
        # Define testing initial condition
        time = np.array([0.0])
        state = np.array([0.0, 0.0])
        ctrl = np.array([0.0])

        pendulum_cart_dynamics = SinglePendulumDynamicsModel()
        linearized_state_matrix = pendulum_cart_dynamics.A(time, state, ctrl)
        assert np.allclose(linearized_state_matrix, np.array([[0.0, 1.0], [-G0 / pendulum_cart_dynamics.LENG_PEND, 0.0]]))

        linearized_input_matrix = pendulum_cart_dynamics.B(time, state, ctrl)
        assert np.allclose(linearized_input_matrix, np.array([[0.0], [1.0 / (pendulum_cart_dynamics.MASS_PEND * pendulum_cart_dynamics.LENG_PEND ** 2)]]))
