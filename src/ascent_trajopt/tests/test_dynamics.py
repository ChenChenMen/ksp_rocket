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

        theta_dotdot = force / (self.MASS_PEND * self.LENG_PEND**2) - G0.m * np.sin(theta) / self.LENG_PEND
        # Formulate the state equation
        return np.stack([theta_dot, theta_dotdot])


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
        theta_dot_sq = np.power(theta_dot, 2)
        mass_leng_pend = self.MASS_PEND * self.LENG_PEND
        mass_leng_theta_dot_sq = mass_leng_pend * theta_dot_sq
        effective_total_mass = self.MASS_CART + self.MASS_PEND * sin_sq_theta

        theta_dotdot = (
            -1
            / (effective_total_mass * self.LENG_PEND)
            * (0.5 * mass_leng_theta_dot_sq * sin_2theta + force * cos_theta + total_mass * G0.m * sin_theta)
        )
        x_dotdot = (
            1
            / effective_total_mass
            * (force + mass_leng_pend * theta_dot_sq * sin_theta + 0.5 * self.MASS_PEND * G0.m * sin_2theta)
        )

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
        single_pendulum_dynamics = SinglePendulumDynamicsModel()

        linearized_state_matrix = single_pendulum_dynamics.continuous_state_matrix(time, state, ctrl)
        expected_state_matrix = np.array([[0.0, 1.0], [-G0.m / single_pendulum_dynamics.LENG_PEND, 0.0]])
        assert np.allclose(linearized_state_matrix, expected_state_matrix), (
            "Linearized state matrix does not match expected value"
        )

        # Duplicated calls to make sure the result is repeatable
        assert np.allclose(linearized_state_matrix, single_pendulum_dynamics.continuous_state_matrix(time, state, ctrl))

        linearized_input_matrix = single_pendulum_dynamics.continuous_input_matrix(time, state, ctrl)
        expected_input_matrix = np.array(
            [[0.0], [1.0 / (single_pendulum_dynamics.MASS_PEND * single_pendulum_dynamics.LENG_PEND**2)]]
        )
        assert np.allclose(linearized_input_matrix, expected_input_matrix), (
            "Linearized input matrix does not match expected value"
        )

        # Duplicated calls to make sure the result is repeatable
        assert np.allclose(linearized_input_matrix, single_pendulum_dynamics.continuous_input_matrix(time, state, ctrl))
