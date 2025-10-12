"""Define launch vehicle dynamics at different phases of flight."""

import logging
import pickle
from functools import wraps

import jax.numpy as np
from jax.numpy import ndarray
from jax import jacrev

from rocket_util.integrator import RK_


LOG = logging.getLogger(__name__)


class RequiredDimensionNotSetError(SyntaxError):
    """Raise when a required dimension is not set for inherited class."""


class CacheNotSupportedError(SyntaxError):
    """Raise when unsupported function tries to access cache."""


class InconsistentInputDimensionError(ValueError):
    """Raise when input vector dimension is mismatched."""


class DynamicsModel:
    """Base dynamics model that carries operations."""

    # Cache size limit
    CACHE_SIZE_LIMIT: int = 200

    # Define required number of elements in the state/control vector
    REQUIRED_STATE_NUM: int = None
    REQUIRED_CTRL_NUM: int = None

    def with_cache_applied(method):
        """Decorator to apply cache query and update."""

        @wraps(method)
        def wrapper(self, *args, **kwargs):
            """Wrap with the input of the decorated method."""
            # Directly return at cache hit
            pickle_key = pickle.dumps((method.__name__, args, frozenset(kwargs.items())))
            if (query_result := self._cache.get(pickle_key)) is not None:
                return pickle.loads(query_result)

            # Otherwise execute the method
            update_result = method(self, *args, **kwargs)
            # Update cache with the new evaluation
            self._cache[pickle_key] = pickle.dumps(update_result)

            # If cache size exceeds the limit, remove the oldest entry
            if len(self._cache) > self.CACHE_SIZE_LIMIT:
                del self._cache[next(iter(self._cache))]
                LOG.warning(f"Cache size exceeded {self.CACHE_SIZE_LIMIT}, removed oldest entry.")
            return update_result

        return wrapper

    def with_single_timestamp_dimention_check(method):
        """Decorator to apply cache query and update."""

        @wraps(method)
        def wrapper(self, *args, **kwargs):
            """Wrap with the input of the decorated method."""
            assert isinstance(self, DynamicsModel), "Method decorator only works for DynamicsModel class."
            # Unpack the input arguments
            time, state, control = args

            # Check input dimensions
            if self.REQUIRED_STATE_NUM is None or self.REQUIRED_CTRL_NUM is None:
                raise RequiredDimensionNotSetError(f"Required input vector dimensions are not defined for {self.__class__}")

            # Ensure the input numpy dimensions are 1D array
            time = time or np.asarray([0])  # Default to zero if time is None
            state = np.atleast_1d(state)
            control = np.atleast_1d(control)

            # Check if the time, state, and control are 1D arrays
            assert time.ndim == 1, "Time vector must be a 1D array."
            assert state.ndim == 1, "State vector must be a 1D array."
            assert control.ndim == 1, "Control vector must be a 1D array."

            # Check if required dimensions of given state vectors
            if (num_state := state.size) != self.REQUIRED_STATE_NUM:
                raise InconsistentInputDimensionError(f"Input state has size {num_state}, expect {self.REQUIRED_STATE_NUM}.")
            # Check if required dimensions of given control vectors
            if (num_ctrl := control.size) != self.REQUIRED_CTRL_NUM:
                raise InconsistentInputDimensionError(f"Input control has size {num_ctrl}, expect {self.REQUIRED_CTRL_NUM}.")

            return method(self, time, state, control, **kwargs)

        return wrapper

    def __init__(self):
        """Construct with some internal used cache."""
        self._cache = {}
        # Create auto-differentiated linearization with stacked input. Guaranteed a wide Jacobian
        self._auto_diffed_linearization = jacrev(self._continuous_time_state_equation_stacked_input)

    @with_single_timestamp_dimention_check
    @with_cache_applied
    def continuous_state_matrix(self, time: ndarray, state: ndarray, control: ndarray, /):
        """Computed state matrix for state equation linearization with auto differentiation."""
        # Taking partial by constructing a temporary function that only depends on state or control
        stacked_jacobian = self._auto_diffed_linearization(np.hstack((time, state, control)))
        return stacked_jacobian[:, 1 : -self.REQUIRED_CTRL_NUM]

    @with_single_timestamp_dimention_check
    @with_cache_applied
    def continuous_input_matrix(self, time: ndarray, state: ndarray, control: ndarray, /):
        """Computed input matrix for state equation linearization with auto differentiation."""
        # Taking partial by constructing a temporary function that only depends on state or control
        stacked_jacobian = self._auto_diffed_linearization(np.hstack((time, state, control)))
        return stacked_jacobian[:, -self.REQUIRED_CTRL_NUM :]

    @with_single_timestamp_dimention_check
    def xdot(self, time: ndarray, state: ndarray, control: ndarray):
        """A dimension checked wrapper for continuous_time_state_equation."""
        return self.continuous_time_state_equation(time, state, control)

    @with_single_timestamp_dimention_check
    def deltax(self, kth_time: ndarray, kth_state: ndarray, kth_control: ndarray, /, *, time_step):
        """Discrete state equation with arbitrarily large time step, backed by RK45 integrator."""
        # Use the integrator to obtain the next step
        _, kp1th_state = RK_.RK45(
            # Set control to the zeroth order hold and time to the kth timestamp
            integrand=lambda x: self.continuous_time_state_equation(time=kth_time, state=x, control=kth_control),
            start_dvar=kth_state,
            start_indvar=kth_time,
            end_indvar=kth_time + time_step,
        )
        return kp1th_state - kth_state

    def _continuous_time_state_equation_stacked_input(self, stacked_input: ndarray):
        """Continuous time state equation with stacked input.

        Input is assumed to be stacked as [time, state, control] at a single timestamp.
        """
        time, state, control = (
            stacked_input[0],
            stacked_input[1 : -self.REQUIRED_CTRL_NUM],
            stacked_input[-self.REQUIRED_CTRL_NUM :],
        )
        # Call the continuous time state equation
        return self.continuous_time_state_equation(time, state, control)

    def continuous_time_state_equation(self, time: ndarray, state: ndarray, control: ndarray):
        """State equation implementation for a single timestamp.

        This function handle requires the following form - f(t, x_vec)
        where t is a scalar and x_vec is a 1D ndarray
        """
        raise NotImplementedError(
            f"Continuous time state equation is not implemented for {self.__class__.__name__}. "
            "Please implement this method in the derived class."
        )
