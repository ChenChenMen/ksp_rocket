"""Initial condition generator module for collocation."""

import numpy as np

from ascent_trajopt.orthogonal_collocation.array_store import OptimizationArray
from ascent_trajopt.orthogonal_collocation.components import ProblemInputComponents


def guess_from_linear_interpolation(problem_input: ProblemInputComponents, total_time: float = 1.0) -> OptimizationArray:
    """Generate initial conditions based on linear interpolation of the inital and final state."""
    if problem_input.initial_condition.dimension != problem_input.final_condition.dimension:
        raise ValueError("Initial and final conditions must have the same dimension.")

    dimension = problem_input.dimension
    initial_condition_array = problem_input.initial_condition.view(np.ndarray)
    final_condition_array = problem_input.final_condition.view(np.ndarray)

    # Create an empty array to hold the initial guess
    relative_time = np.kron(problem_input.discretizer.discretized_point_array, np.ones((dimension.total_dimension, 1))).T
    initial_guess_stack = (1 - relative_time) * initial_condition_array + relative_time * final_condition_array
    initial_guess = np.concat((initial_guess_stack.reshape((initial_guess_stack.size,)), np.atleast_1d(total_time)))
    return OptimizationArray(initial_guess, discretizer=problem_input.discretizer, dimension=dimension)
