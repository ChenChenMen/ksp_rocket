"""Define launch vehicle dynamics at different phases of flight."""

import abc
import logging

import jax.numpy as np
from jax.scipy.linalg import expm
from jax import jacfwd

from ascent_trajopt.integrator import RK_


LOG = logging.getLogger(__name__)


class RequiredDimensionNotSetError(SyntaxError):
    """Raise when a required dimension is not set for inherited class."""


class CacheNotSupportedError(SyntaxError):
    """Raise when unsupported function tries to access cache."""


class InsufficientInputDimensionError(ValueError):
    """Raise when input vector dimension is mismatched."""


class DynamicsModel(metaclass=abc.ABCMeta):
    """Base dynamics model that carries operations."""

    # Define required number of elements in the state/control vector
    REQUIRED_STATE_NUM: int = None
    REQUIRED_CTRL_NUM: int = None
    
    def apply_cache(method):
        """Decorator to apply cache query and update."""
        def wrapper(self, *args, **kwargs):
            """Wrap with the input of the decorated method."""
            query_result = self._query_cache(method.__name__, *args, **kwargs)
            # Directly return at cache hit
            if query_result is not None:
                return query_result

            # Otherwise execute the method
            update_result = method(self, *args, **kwargs)
            yield update_result

            # Update cache with the new evaluation
            self._update_cache(method.__name__, *args, **kwargs, new_value=update_result)

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

    def check_input_dimensions(self, time: np.ndarray, state: np.ndarray, control: np.ndarray):
        """Check the input state and control vector dimensions."""
        # Check if required dimensions are defined
        if self.REQUIRED_STATE_NUM is None or self.REQUIRED_CTRL_NUM is None:
            raise RequiredDimensionNotSetError(f'Required input vector dimensions are not defined for {self.__class__}')
        # Ensure the input numpy dimensions is at least 2D
        time = np.atleast_1d(time)
        state = np.atleast_2d(state)
        control = np.atleast_2d(control)

        # Check if required dimensions of given state vectors
        if (num_state := state.shape[1]) != self.REQUIRED_STATE_NUM:
            raise InsufficientInputDimensionError(f'Input state has size {num_state}, expect {self.REQUIRED_STATE_NUM}.')
        # Check if required dimensions of given control vectors
        if (num_ctrl := control.shape[1]) != self.REQUIRED_CTRL_NUM:
            raise InsufficientInputDimensionError(f'Input control has size {num_ctrl}, expect {self.REQUIRED_CTRL_NUM}.')

        # Check if the time and state and control array have consistent shape
        if (time_size := time.size) != (state_size := state.shape[0]) != (ctrl_size := control.shape[0]):
            raise InsufficientInputDimensionError(
                f'Input arrays dimension don\'t agree. Provided {time_size} timestamps but {state_size} states '
                f'and {ctrl_size} controls.'
            )

        return time, state, control

    def xdot(self, time: np.ndarray, state: np.ndarray, control: np.ndarray):
        """Continuous time state equation wrapper."""
        time, state, control = self.check_input_dimensions(time=time, state=state, control=control)
        return self.continuous_time_state_equation(time=time, state=state, control=control)

    def deltax(self, kth_time: np.ndarray, kth_state: np.ndarray, kth_control: np.ndarray, discrete_time_step):
        """Discrete time state equation wrapper."""
        kth_time, kth_state, kth_control = self.check_input_dimensions(time=kth_time, state=kth_state, control=kth_control)
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

    @apply_cache
    def A(self, time: np.ndarray, state: np.ndarray, control: np.ndarray):
        """Computed state matrix for state equation linearization with auto differentiation."""
        time, state, control = self.check_input_dimensions(time=time, state=state, control=control)
        # Taking partial by constructing a temporary function that only depends on state or control
        state_only_func = lambda x: self.continuous_time_state_equation(time=time, state=x, control=control)
        return jacfwd(state_only_func)(state)

    @apply_cache
    def B(self, time: np.ndarray, state: np.ndarray, control: np.ndarray):
        """Computed input matrix for state equation linearization with auto differentiation."""
        time, state, control = self.check_input_dimensions(time=time, state=state, control=control)
        # Taking partial by constructing a temporary function that only depends on state or control
        control_only_func = lambda u: self.continuous_time_state_equation(time=time, state=state, control=u)
        return jacfwd(control_only_func)(control)

    def Ad(self, kth_time: np.ndarray, kth_state: np.ndarray, kth_control: np.ndarray, discrete_time_step):
        """Computed discrete state matrix with auto differentiation."""
        kth_time, kth_state, kth_control = self.check_input_dimensions(time=kth_time, state=kth_state, control=kth_control)
        # Formulate the ultra state control matrix
        state_size, ctrl_size = kth_state.shape[0], kth_control.shape[0]
        state_control_size = state_size + ctrl_size
        state_control_matrix = np.pad(
            np.hstack((kth_state, kth_control)), ((0, 0), (0, state_control_size - state_size)), 'constant', constant_values=0
        )
        exp_state_control_matrix = expm(state_control_matrix * discrete_time_step)

    @abc.abstractmethod
    def continuous_time_state_equation(self, time: np.ndarray, state: np.ndarray, control: np.ndarray):
        """State equation that in continuous time. Supports vector inputs.

        This function handle requires the following form - f(t, x_vec)
        where t is a 1D ndarray and x_vec is a 2D ndarray with t.size == x_vec.shape[1]
        """

from core.constant import G0

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
        theta = state[:, 0]
        theta_dot = state[:, 1]
        x_dot = state[:, 3]

        # Unpack control vector
        force = control[:, 0]

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
        return np.asarray([theta_dot[0], theta_dotdot[0], x_dot[0], x_dotdot[0]])


time = np.array([0, 1, 2])
state = np.array([[0.01, 0.005, 0.001], [0, -0.001, 0.001], [0, 0.1, 0.2], [0.01, 0.01, 0.01]]).T
ctrl = np.array([[0, 0.1, 0]]).T

time = np.array([0])
state = np.array([[0.01, 0, 0, 0]])
ctrl = np.array([[0]])
pendulum_cart_dynamics = PendulumCartDynamicsModel()
print(list(pendulum_cart_dynamics.A(time, state, ctrl)))
