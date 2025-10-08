"""Test dynamics model."""

import jax.numpy as np

from ascent_trajopt.dynamics.pendulum import SinglePendulumDynamicsModel
from rocket_util.constant import G0
from rocket_util.logconfig import create_logger

LOG = create_logger(__name__)


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
        gravity_accel = G0.m_as("m/s^2")
        expected_state_matrix = np.array([[0.0, 1.0], [-gravity_accel / single_pendulum_dynamics.LENG_PEND, 0.0]])
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
