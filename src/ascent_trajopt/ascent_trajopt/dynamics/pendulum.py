"""Dynamics models of various pendulum systems."""

import jax.numpy as np

from ascent_trajopt.dynamics.base import DynamicsModel
from rocket_util.constant import G0


class SinglePendulumDynamicsModel(DynamicsModel):
    """Define a dynamics model of a single pendulum system."""

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

        gravity_accel = G0.m_as("m/s^2")
        theta_dotdot = force / (self.MASS_PEND * self.LENG_PEND**2) - gravity_accel * np.sin(theta) / self.LENG_PEND
        # Formulate the state equation
        return np.stack([theta_dot, theta_dotdot])


class PendulumCartDynamicsModel(DynamicsModel):
    """Define a dynamics model of a pendulum cart system."""

    # Define required number of elements in the state/control vector
    REQUIRED_STATE_NUM: int = 4
    REQUIRED_CTRL_NUM: int = 1

    # Define test model parameters
    MASS_CART = 1
    MASS_PEND = 1
    LENG_PEND = 1

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

        gravity_accel = G0.m_as("m/s^2")
        theta_dotdot = (
            -1
            / (effective_total_mass * self.LENG_PEND)
            * (0.5 * mass_leng_theta_dot_sq * sin_2theta + force * cos_theta + total_mass * gravity_accel * sin_theta)
        )
        x_dotdot = (
            1
            / effective_total_mass
            * (force + mass_leng_theta_dot_sq * sin_theta + 0.5 * self.MASS_PEND * gravity_accel * sin_2theta)
        )

        # Formulate the state equation
        return np.asarray([theta_dot, theta_dotdot, x_dot, x_dotdot])
