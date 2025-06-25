"""Tests for integrator."""

import jax.numpy as np
import pytest

from ascent_trajopt.integrator import RK_


class TestIntegrator:
    """Test class for integrator object."""

    @pytest.fixture
    def ode(self):
        """Create an ordinary differential equation to test."""
        # Define initial conditions
        indvar0 = np.array([0])
        dvar0 = np.array([[0.5], [0], [1]])

        def integrand_func(indvar: np.ndarray, dvar: np.ndarray):
            """Integrand func that has 1D independent and 2D dependent variabls."""
            # Break the dvar down into individual equations
            x = indvar[0]
            y1, y2, y3 = dvar[0, 0], dvar[1, 0], dvar[2, 0]
            return np.array(
                [
                    [y1 + np.sin(x)],
                    [np.exp(-1 * y2) * x],
                    [x * y3],
                ]
            )

        return integrand_func, indvar0, dvar0

    @pytest.fixture
    def solution(self):
        """The pairing solution for the above testing ode."""

        def solution_func(indvar: np.ndarray):
            """solution func that has an 1D independent variable."""
            x = indvar
            return np.array(
                [
                    -1 / 2 * (np.cos(x) + np.sin(x)) + np.exp(x),
                    np.log(np.power(x, 2) / 2 + 1),
                    np.exp(np.power(x, 2) / 2),
                ]
            )

        return solution_func

    def test_rk4(self, ode, solution):
        """Verify that the custom implemented RK4 integrator is correct."""
        integrand, init_indvar, init_dvar = ode
        # Define time step and end indvar
        step, end_indvar = 0.01, 1
        indvar_hist, dvar_hist = RK_.RK4(
            integrand=integrand,
            start_dvar=init_dvar,
            start_indvar=init_indvar,
            step=step,
            end_indvar=end_indvar,
            save_history=True,
        )
        solution_hist = solution(indvar_hist)
        assert np.allclose(dvar_hist, solution_hist, atol=1e-6)

    def test_rk45(self, ode, solution):
        """Verify that the custom implemented RK45 integrator is correct."""
        integrand, init_indvar, init_dvar = ode
        # Define time step and end indvar
        end_indvar = 1
        indvar_hist, dvar_hist, _ = RK_.RK45(
            integrand=integrand,
            start_dvar=init_dvar,
            start_indvar=init_indvar,
            end_indvar=end_indvar,
            save_history=True,
        )
        solution_hist = solution(indvar_hist)
        assert np.allclose(dvar_hist, solution_hist, atol=1e-6)
