"""Define integrator objects."""

import warnings

import numpy as np
from rocket_util.complexity import BaseMonitoredClass
from rocket_util.logconfig import create_logger

LOG = create_logger(__name__)


class InvalidButcherTableauError(ValueError):
    """Raise when input Butcher Tableau is invalid."""


class SingletonMeta(type):
    _instances = {}
    _instance_limit_for_warning = 10

    def __call__(cls, *args, **kwargs):
        """Call magic method to control __new__ method call at class construction."""
        if cls not in cls._instances:
            # Throw a memory warning if too much singletons are defined and created
            if len(cls._instances) >= cls._instance_limit_for_warning:
                warnings.warn(
                    f"{len(cls._instances)} singletons have been created. Watch out for high memory usage "
                    "since singleton metaclass has perminant record of all singletons constructed."
                )
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class RungeKuttaIntegrator(BaseMonitoredClass, metaclass=SingletonMeta):
    """Define runge kutta integrator singleton for ODEs that can plug in anywhere.

    Assume the integrand function is a function of one independent variable and
    a vector dependent variable, in the form of

        x_vec_dot = f(t, x_vec)

    where x_vec is a column vector of size (n, 1) and t is a scalar
    """

    RK4_BUTCHER_TABLEAU = [
        np.array([[1 / 2, 0, 0], [0, 1 / 2, 0], [0, 0, 1]]),
        np.array([[1 / 6, 1 / 3, 1 / 3, 1 / 6]]),
        np.array([0, 1 / 2, 1 / 2, 1]),
    ]

    RK45_EXTENDED_BUTCHER_TABLEAU = {
        "A": np.array(
            [
                [1 / 4, 0, 0, 0, 0],
                [3 / 32, 9 / 32, 0, 0, 0],
                [1932 / 2197, -7200 / 2197, 7296 / 2197, 0, 0],
                [439 / 216, -8, 3680 / 513, -845 / 4104, 0],
                [-8 / 27, 2, -3544 / 2565, 1859 / 4104, -11 / 40],
            ]
        ),
        "B": np.array([[25 / 216, 0, 1408 / 2565, 2197 / 4104, -1 / 5, 0]]),
        "BH": np.array([[16 / 135, 0, 6656 / 12825, 28561 / 56430, -9 / 50, 2 / 55]]),
        "TE": np.array(
            [[-1 / 360, 0, 128 / 4275, 2197 / 75240, -1 / 50, -2 / 55]],
        ),
        "C": np.array([0, 1 / 4, 3 / 8, 12 / 13, 1, 1 / 2]),
    }

    def _explicit_runge_kutta_single_step(
        self,
        integrand: callable,
        start_dvar,
        start_indvar,
        step,
        butcher_tableau: list,
    ):
        """General explicit runge kutta single step evaluator."""
        # Check Butcher Tableau for explicit RK integrator
        if len(butcher_tableau) != 3:
            raise InvalidButcherTableauError(
                "Butcher Tableau should contain exactly 3 components - coeff increment matrix A, final "
                f"weighting array B, and time step increment array C. Received {len(butcher_tableau)}"
            )
        # Unpack butcher tableau in order of A, B, and C
        coeff_increment, final_weighting, hstep_increment = butcher_tableau
        # Ensure the dimensions of each part of Butcher Tableau
        coeff_increment = np.atleast_2d(coeff_increment)
        final_weighting = np.atleast_2d(final_weighting)
        hstep_increment = np.atleast_1d(hstep_increment)

        # Check if coefficient increment matrix a lower triangular matrix
        if not np.all(np.triu(coeff_increment, k=1) == 0):
            raise InvalidButcherTableauError(
                "Input coefficient increment matrix A does not represent an explicit RK method."
            )
        # Record coefficient increment matrix shape which has to be a square matrix
        coeff_increment_shape = coeff_increment.shape
        if len(coeff_increment_shape) != 2 or coeff_increment_shape[0] != coeff_increment_shape[1]:
            raise InvalidButcherTableauError(
                f"Shape of coefficient increment matrix must be 2D and square, received {coeff_increment_shape}"
            )
        rk_order = coeff_increment_shape[0] + 1

        # Check dimensions for final weighting and time step increment arrays
        if final_weighting.size != rk_order:
            raise InvalidButcherTableauError(
                "Final weighting array B has to be one greater than the order of RK method. "
                f"Determined RK integrator order is {rk_order} received the final weighting "
                f"array of size {final_weighting.size} instead."
            )
        if hstep_increment.size != rk_order:
            raise InvalidButcherTableauError(
                "Time step increment array C has to be one greater than the order of RK method. "
                f"Determined RK integrator order is {rk_order} received the time step increment "
                f"array of size {final_weighting.size} instead."
            )

        # Iteratively resolve output with the valid Butcher Tableau
        slope_array = np.zeros(shape=(len(start_dvar), rk_order))
        coeff_increment = np.pad(
            coeff_increment,
            pad_width=((1, 0), (0, 1)),
            mode="constant",
            constant_values=0,
        )

        for idx in range(rk_order):
            # At this step the integrand should be accepting a single set of input
            slope_idx = integrand(
                start_indvar + step * hstep_increment[idx : idx + 1],
                start_dvar + step * slope_array @ coeff_increment[idx : idx + 1, :].T,
            )
            # Update slope array since it is immutable
            slope_array[:, idx] = slope_idx[:, 0]

        # Compute the delta result from final weighting array
        return step * slope_array @ final_weighting.T, slope_array

    def RK4(
        self,
        integrand: callable,
        start_dvar,
        start_indvar,
        step,
        end_indvar,
        save_history: bool = False,
    ):
        """Integrate with the classic Runge Kutta 4 method.

        The integrand function handle requires the following form - f(t, x_vec)
        where t is a 1D ndarray and x_vec is a 2D ndarray with t.size == x_vec.shape[1]
        """
        # Ensure dimensions of starting independent and dependent variables
        start_dvar = np.atleast_2d(start_dvar)
        start_indvar = np.atleast_1d(start_indvar)
        end_indvar = np.atleast_1d(end_indvar)

        # Create the output array
        output_array = [start_indvar, start_dvar]
        for curr_indvar in np.arange(start_indvar[0], end_indvar[0], step):
            # Mark the initial starting dependent variable
            curr_indvar = np.atleast_1d(curr_indvar)
            curr_dvar = output_array[1][:, -1:]

            # Compute delta with general explicit RK single step integrator
            step_delta, _ = self._explicit_runge_kutta_single_step(
                integrand=integrand,
                start_dvar=curr_dvar,
                start_indvar=curr_indvar,
                step=step,
                butcher_tableau=RungeKuttaIntegrator.RK4_BUTCHER_TABLEAU,
            )
            step_result = curr_dvar + step_delta

            # Save history if specified, otherwise keep replacing the first element
            if save_history:
                output_array[0] = np.hstack((output_array[0], curr_indvar + step))
                output_array[1] = np.hstack((output_array[1], step_result))
            else:
                output_array[0] = np.atleast_1d(curr_indvar + step)
                output_array[1] = np.atleast_2d(step_result)

        # Return the integration result along with the independent variable value achieved
        return output_array

    def RK45(
        self,
        integrand: callable,
        start_dvar,
        start_indvar,
        end_indvar,
        atol=1e-8,
        rtol=1e-5,
        save_history: bool = False,
    ):
        """Integrate with the Runge Kutta Fehlberg (RK45) with adaptive step size."""
        # Ensure dimensions of starting independent and dependent variables
        start_dvar = np.atleast_2d(start_dvar)
        start_indvar = np.atleast_1d(start_indvar)
        end_indvar = np.atleast_1d(end_indvar)

        # Select a default step size
        curr_step_size = 1
        # Create the output array with the adaptive step size if saving all history
        output_array = [start_indvar, start_dvar, np.array(np.nan)] if save_history else [start_indvar, start_dvar]

        # Outer loop that iterate through start to the end
        curr_indvar = start_indvar

        while not np.allclose(curr_indvar, end_indvar, atol=atol, rtol=0):
            # Compute the indvar to go
            indvar_to_go = end_indvar - curr_indvar
            # Mark the initial starting dependent variable
            curr_dvar = output_array[1][:, -1:]

            # Compute the estimated truncation error replacing final weighting
            truncation_error_butcher_tableau = [
                RungeKuttaIntegrator.RK45_EXTENDED_BUTCHER_TABLEAU["A"],
                RungeKuttaIntegrator.RK45_EXTENDED_BUTCHER_TABLEAU["TE"],
                RungeKuttaIntegrator.RK45_EXTENDED_BUTCHER_TABLEAU["C"],
            ]

            # Inner loop that determines the step size
            truncation_error, epsilon = 1e6, atol
            while truncation_error > epsilon:
                # Select min of the current step size and the indvar to go
                curr_step_size = np.minimum(curr_step_size, indvar_to_go[0])
                raw_truncation_error, slope_array = self._explicit_runge_kutta_single_step(
                    integrand=integrand,
                    start_dvar=curr_dvar,
                    start_indvar=curr_indvar,
                    step=curr_step_size,
                    butcher_tableau=truncation_error_butcher_tableau,
                )
                step_delta_5th = curr_step_size * slope_array @ RungeKuttaIntegrator.RK45_EXTENDED_BUTCHER_TABLEAU["BH"].T

                # Adaptive step size condition
                truncation_error = np.abs(raw_truncation_error[0, 0])
                epsilon = atol + rtol * np.abs(step_delta_5th[0, 0])

                # Update step size at the end and make memory of length 1
                prev_step_size = curr_step_size
                curr_step_size *= 0.9 * (epsilon / truncation_error) ** 0.2

            # Update the current anchor point
            step_result = curr_dvar + step_delta_5th
            curr_indvar = curr_indvar + prev_step_size

            # Save history if specified, otherwise keep replacing the first element
            if save_history:
                output_array[0] = np.hstack((output_array[0], curr_indvar))
                output_array[1] = np.hstack((output_array[1], step_result))
                output_array[2] = np.hstack((output_array[2], prev_step_size))
            else:
                output_array[0] = np.atleast_1d(curr_indvar)
                output_array[1] = np.atleast_2d(step_result)

        # Return the integration result along with the independent variable value achieved
        return output_array


# Create a Runge Kutta integrator singleton
RK_ = RungeKuttaIntegrator()
