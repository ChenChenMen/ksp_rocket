"""Test dynamics model."""

import jax.numpy as np

from ascent_trajopt.dynamics.base import DynamicsModel
from core.constant import G0
from core.util import create_logger

LOG = create_logger(__name__)


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
        
        Use the cart and inverted pendulum for example
        State:
            theta (pendulum position from stable equillibrium)
            theta_dot (pendulum counterclockwise rotational velocity)
            x (cart position from origin to the right)
            x_dot (cart velocity from origin to the right)
        Control:
            u (external force applied to the cart to the right)
        """
        # The system is time invariant
        time = time if time is not None else np.zeros(shape=(state.shape(1)))
        _, state, control = self.check_input_dimensions(time=time, state=state, control=control)

        # Unpack state vector 
        theta = state[0, :]
        theta_dot = state[1, :]
        x = state[2, :]
        x_dot = state[3, :]

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
            0.5 * mass_leng_theta_dot_sq * sin_2theta + control * cos_theta + total_mass * G0 * sin_theta
        )
        x_dotdot = inv_total_mass * (control + mass_leng_theta_dot_sq * sin_theta + mass_leng_pend * theta_dotdot * cos_theta)

        # Formulate the state equation
        return np.vstack([theta_dot, theta_dotdot, x_dot, x_dotdot])


time = np.array([0, 1, 2])
state = np.array([[0.01, 0.005, 0.001], [0, -0.001, 0.001], [0, 0.1, 0.2], [0.01, 0.01, 0.01]])
ctrl = np.array([0, 0.1, 0])
pendulum_cart_dynamics = PendulumCartDynamicsModel()
pendulum_cart_dynamics.A(time, state, ctrl)


class TestDynamicsModel:
    """Test dynamics model functionalities with pendulum cart system."""

    def test_stable_equilibrium_perturbation(self):
        """Verify linearlization close to stable equilibrium."""
        # Define testing initial condition
        time = np.array([0, 1, 2])
        state = np.array([[0.01, 0.005, 0.001], [0, -0.001, 0.001], [0, 0.1, 0.2], [0.01, 0.01, 0.01]])
        ctrl = np.array([0, 0.1, 0])
        pendulum_cart_dynamics = PendulumCartDynamicsModel()
        pendulum_cart_dynamics.A(time, state, ctrl)
