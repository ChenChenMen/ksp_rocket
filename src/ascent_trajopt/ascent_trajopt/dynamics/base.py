"""Define launch vehicle dynamics at different phases of flight."""

import logging
from functools import wraps

import jax.numpy as np
from jax.scipy.linalg import expm
from jax import jacfwd

from ascent_trajopt.integrator import RK_


LOG = logging.getLogger(__name__)


class RequiredDimensionNotSetError(SyntaxError):
    """Raise when a required dimension is not set for inherited class."""


class CacheNotSupportedError(SyntaxError):
    """Raise when unsupported function tries to access cache."""


class InconsistentInputDimensionError(ValueError):
    """Raise when input vector dimension is mismatched."""


class DynamicsModel:
    """Base dynamics model that carries operations."""

    # Define required number of elements in the state/control vector
    REQUIRED_STATE_NUM: int = None
    REQUIRED_CTRL_NUM: int = None

    def with_cache_applied(method):
        """Decorator to apply cache query and update."""

        @wraps(method)
        def wrapper(self, *args, **kwargs):
            """Wrap with the input of the decorated method."""
            query_result = self._query_cache(method.__name__, *args, **kwargs)
            # Directly return at cache hit
            if query_result is not None:
                return query_result

            # Otherwise execute the method
            update_result = method(self, *args, **kwargs)
            # Update cache with the new evaluation
            self._update_cache(method.__name__, *args, **kwargs, new_value=update_result)
            return update_result

        return wrapper

    def with_single_timestamp_dimention_check(method):
        """Decorator to apply cache query and update."""

        @wraps(method)
        def wrapper(self, *args, **kwargs):
            """Wrap with the input of the decorated method."""
            assert isinstance(self, DynamicsModel), 'Method decorator only works for DynamicsModel class.'
            # Unpack the input arguments
            time, state, control = args

            # Check input dimensions
            if self.REQUIRED_STATE_NUM is None or self.REQUIRED_CTRL_NUM is None:
                raise RequiredDimensionNotSetError(f'Required input vector dimensions are not defined for {self.__class__}')

            # Ensure the input numpy dimensions are 1D array
            time = time or np.asarray([0])  # Default to zero if time is None
            state = np.atleast_1d(state)
            control = np.atleast_1d(control)

            # Check if the time, state, and control are 1D arrays
            assert time.ndim == 1, 'Time vector must be a 1D array.'
            assert state.ndim == 1, 'State vector must be a 1D array.'
            assert control.ndim == 1, 'Control vector must be a 1D array.'

            # Check if required dimensions of given state vectors
            if (num_state := state.size) != self.REQUIRED_STATE_NUM:
                raise InconsistentInputDimensionError(f'Input state has size {num_state}, expect {self.REQUIRED_STATE_NUM}.')
            # Check if required dimensions of given control vectors
            if (num_ctrl := control.size) != self.REQUIRED_CTRL_NUM:
                raise InconsistentInputDimensionError(f'Input control has size {num_ctrl}, expect {self.REQUIRED_CTRL_NUM}.')

            return method(self, time, state, control, **kwargs)

        return wrapper

    def __init__(self):
        """Construct with some internal used cache."""
        # Define lru cache for computations memory of 1
        self._lru_cache = {
            'A': {'input_match': None, 'value_store': None},
            'B': {'input_match': None, 'value_store': None},
            'Ad': {'input_match': None, 'value_store': None},
            'Bd': {'input_match': None, 'value_store': None},
        }

    def _query_cache(self, func_name: str, time: np.ndarray, state: np.ndarray, control: np.ndarray):
        """Query cache helper function for internal cache."""
        LOG.debug(f'Querying internal cache for function {func_name}')
        # Check if the function output is supported by the internal cache
        if func_name not in self._lru_cache:
            raise CacheNotSupportedError(f'Internal cache not supporting function {func_name}')
        # Fetch from the internal cache
        input_match = self._lru_cache[func_name]['input_match']
        cache_hit = (
            input_match is not None and np.allclose(time, input_match[0]) and
            np.allclose(state, input_match[1]) and np.allclose(control, input_match[2])
        )
        return self._lru_cache[func_name]['value_store'] if cache_hit else None

    def _update_cache(self, func_name: str, time: np.ndarray, state: np.ndarray, control: np.ndarray, new_value: np.ndarray):
        """Query cache helper function for internal cache."""
        LOG.debug(f'Updating internal cache for function {func_name}')
        # Check if the function output is supported by the internal cache
        if func_name not in self._lru_cache:
            raise CacheNotSupportedError(f'Internal cache not supporting function {func_name}')
        # Update to the internal cache (cache size is 1)
        self._lru_cache[func_name]['input_match'] = (time, state, control)
        self._lru_cache[func_name]['value_store'] = new_value

    @with_single_timestamp_dimention_check
    def deltax(self, kth_time: np.ndarray, kth_state: np.ndarray, kth_control: np.ndarray, /, *, discrete_time_step):
        """Discrete time state equation wrapper."""
        # Set control to the zeroth order hold and time to the kth timestamp
        state_equation_integrand = lambda x: self.continuous_time_state_equation(time=kth_time, state=x, control=kth_control)
        # Use the integrator to obtain the next step
        _, kp1th_state = RK_.RK45(
            integrand=state_equation_integrand,
            start_dvar=kth_state,
            start_indvar=kth_time,
            end_indvar=kth_time + discrete_time_step,
        )
        return kp1th_state - kth_state

    @with_single_timestamp_dimention_check
    @with_cache_applied
    def A(self, time: np.ndarray, state: np.ndarray, control: np.ndarray, /):
        """Computed state matrix for state equation linearization with auto differentiation."""
        # Taking partial by constructing a temporary function that only depends on state or control
        state_only_func = lambda x: self.continuous_time_state_equation(time=time, state=x, control=control)
        return jacfwd(state_only_func)(state)

    @with_single_timestamp_dimention_check
    @with_cache_applied
    def B(self, time: np.ndarray, state: np.ndarray, control: np.ndarray, /):
        """Computed input matrix for state equation linearization with auto differentiation."""
        # Taking partial by constructing a temporary function that only depends on state or control
        control_only_func = lambda u: self.continuous_time_state_equation(time=time, state=state, control=u)
        return jacfwd(control_only_func)(control)

    @with_single_timestamp_dimention_check
    def Ad(self, kth_time: np.ndarray, kth_state: np.ndarray, kth_control: np.ndarray, /, *, discrete_time_step):
        """Computed discrete state matrix with auto differentiation."""
        # Formulate the ultra state control matrix
        state_size, ctrl_size = kth_state.shape[0], kth_control.shape[0]
        state_control_size = state_size + ctrl_size
        state_control_matrix = np.pad(
            np.hstack((kth_state, kth_control)), ((0, 0), (0, state_control_size - state_size)), 'constant', constant_values=0
        )
        exp_state_control_matrix = expm(state_control_matrix * discrete_time_step)

    def continuous_time_state_equation(self, time: np.ndarray, state: np.ndarray, control: np.ndarray):
        """State equation implementation for a single timestamp.

        This function handle requires the following form - f(t, x_vec)
        where t is a scalar and x_vec is a 1D ndarray
        """
        raise NotImplementedError(
            f'Continuous time state equation is not implemented for {self.__class__.__name__}. '
            'Please implement this method in the derived class.'
        )

    xdot = with_single_timestamp_dimention_check(continuous_time_state_equation)
