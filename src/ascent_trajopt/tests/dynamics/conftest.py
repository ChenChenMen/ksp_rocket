"""Define dynamic model test fixtures.

Not only for testing but also create useful visualizations
"""

import numpy as np
import matplotlib.pyplot as plt
import pytest

from ascent_trajopt.dynamics.base import DynamicsModel
from rocket_util.integrator import RK_


@pytest.fixture
def uncontrolled_propagator():
    """Obtain a plotting propagator to visualize dynamics model with uncontrolled propagation."""

    total_steps = 1000

    def propagate_and_plot(
        dynamics_model: DynamicsModel,
        initial_state: np.ndarray,
        propagation_duration: float,
        state_labels: list[str] = None,
        suppress_plots: bool = False,
    ):
        """Propagate the dynamics model without control and plot the trajectory."""
        # Propagate the dynamics model using RK4 integrator
        time, states = RK_.RK4(
            integrand=lambda time, state: dynamics_model.continuous_time_state_equation(
                time, state, np.zeros((dynamics_model.REQUIRED_CTRL_NUM,))
            ),
            start_dvar=initial_state,
            start_indvar=0,
            step=propagation_duration / total_steps,
            end_indvar=propagation_duration,
            save_history=True,
        )

        if suppress_plots:
            return time, states

        fig = plt.figure()
        num_states = initial_state.size
        num_rows = int(np.floor(np.sqrt(num_states)))
        num_cols = int(np.ceil(num_states / num_rows))
        assert num_rows * num_cols >= num_states

        # Plot the trajectory of each state variable against time
        for state_idx in range(num_states):
            # Create a subplot for each state variable
            ax = fig.add_subplot(num_rows, num_cols, state_idx + 1)
            # Format the state label
            label = f"{state_idx + 1}" if state_labels is None else state_labels[state_idx]

            # Plot the trace of the state variable over time
            ax.plot(time, states[state_idx, :], label=f"State {label}")
            ax.scatter(time, states[state_idx, :], color="red")

            ax.set_xlabel("Time")
            ax.set_ylabel(f"State {label} Time Series")
            ax.grid()

        plt.tight_layout()
        plt.show()

        return time, states

    return propagate_and_plot
