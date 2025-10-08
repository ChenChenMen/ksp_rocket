"""Define the orthogonal polynomials and solver to compute the orthogonal collocation points."""

import pickle
from pathlib import Path

import numpy as np

_CACHE_PATH = Path(__file__).parent / ".cache"
_COEFFICIENT_CACHE_INDEX_FILENAME = "cached_index.pkl"


def _get_orthopoly_cache_content(filename: str):
    """Retrieve the cached value by key."""
    cache_file_path = _CACHE_PATH / filename
    return pickle.loads(cache_file_path.read_bytes()) if cache_file_path.exists() and cache_file_path.is_file() else {}


def with_orthopoly_cache(key: str = None):
    """Decorator to cache the results of a function."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            cache_key = key or f"{func.__name__}_{'_'.join(map(str, args))}"

            cache_index = _get_orthopoly_cache_content(filename=_COEFFICIENT_CACHE_INDEX_FILENAME).get(cache_key)
            if not key and cache_index:
                return _get_orthopoly_cache_content(filename=cache_index)

            result = func(*args, **kwargs)

            # Ensure the cache directory exists
            _CACHE_PATH.mkdir(parents=True, exist_ok=True)

            # Format the cache filename and update the index dictionary
            cache_filename = f"{cache_key}.pkl"
            cache_index_dict = _get_orthopoly_cache_content(filename=_COEFFICIENT_CACHE_INDEX_FILENAME)
            cache_index_dict[cache_key] = cache_filename

            # Save the updated index dictionary back to the cache
            cache_file_path = _CACHE_PATH / _COEFFICIENT_CACHE_INDEX_FILENAME
            pickle.dump(cache_index_dict, cache_file_path.open("wb"))

            # Save the latest value to the cache
            cache_file_path = _CACHE_PATH / cache_filename
            pickle.dump(result, cache_file_path.open("wb"))

            return result

        return wrapper

    return decorator


def get_legendre_polynomial_coefficients(nth: int):
    """Compute the n-th Legendre polynomial coefficients via the generating function."""
    _CACHE_KEY = "legendre_polynomial_coefficients"

    @with_orthopoly_cache(key=_CACHE_KEY)
    def _generate_polynomial_coefficients(computed_coefficients=None):
        """Continue generating coefficients from already computed coefficient sequence."""
        if computed_coefficients is None or len(computed_coefficients) < 2:
            # P_0(x) = 1, P_1(x) = x
            computed_coefficients = [[1], [0, 1]]

        for ith in range(len(computed_coefficients) - 1, nth):
            i_minus_1_term = [-ith / (ith + 1) * el for el in computed_coefficients[ith - 1]]
            i_term = [0] + [(2 * ith + 1) / (ith + 1) * el for el in computed_coefficients[ith]]
            # (n + 1)P_{n+1}(x) = (2n + 1)x P_n(x) - n P_{n-1}(x)
            len_diff = len(i_minus_1_term) - len(i_term)
            ith_coefficients = [a + b for a, b in zip(i_minus_1_term, i_term)] + i_term[len_diff:]
            computed_coefficients.append(ith_coefficients)

        return computed_coefficients

    # Retrieve the already computed coefficients from the cache.
    cache_file = _get_orthopoly_cache_content(_COEFFICIENT_CACHE_INDEX_FILENAME).get(_CACHE_KEY)
    polynomial_coefficients = _get_orthopoly_cache_content(cache_file) if cache_file else []

    # If the coefficients are not cached or not enough, generate them up to the nth polynomial.
    if not polynomial_coefficients or len(polynomial_coefficients) <= nth:
        polynomial_coefficients = _generate_polynomial_coefficients(polynomial_coefficients)

    return polynomial_coefficients[nth]


@with_orthopoly_cache()
def get_legendre_gauss_radau_points(nth: int) -> np.ndarray:
    """Compute the orthogonal collocation points for the n-th Legendre polynomial."""
    nth_coefficients = get_legendre_polynomial_coefficients(nth)
    n_minus_1_coefficients = get_legendre_polynomial_coefficients(nth - 1)
    len_diff = len(nth_coefficients) - len(n_minus_1_coefficients)

    coefficients = np.asarray(nth_coefficients) + np.asarray(n_minus_1_coefficients + [0] * len_diff)
    roots = np.roots(coefficients[::-1])
    real_roots = np.sort(roots.real[roots.imag == 0])

    # Make sure the first point is always -1
    real_roots[0] = -1
    return real_roots
