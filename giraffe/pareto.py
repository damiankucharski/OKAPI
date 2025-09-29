from typing import Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np

from giraffe.utils import _euclidean_distances


def maximize(a, b):
    """
    Compare two values for maximization in Pareto optimization.

    This function determines if the first value (a) is at least as good as
    the second value (b) in the context of maximization (higher is better).

    Args:
        a: First value to compare
        b: Second value to compare

    Returns:
        True if a is greater than or equal to b, False otherwise
    """
    return a >= b


def minimize(a, b):
    """
    Compare two values for minimization in Pareto optimization.

    This function determines if the first value (a) is at least as good as
    the second value (b) in the context of minimization (lower is better).

    Args:
        a: First value to compare
        b: Second value to compare

    Returns:
        True if a is less than or equal to b, False otherwise
    """
    return a <= b


def paretoset(array: np.ndarray, objectives: Sequence[Callable[[float, float], bool]]):
    """
    Identify the Pareto-optimal set from a collection of points with multiple objectives.

    This function finds points that are not dominated by any other point, where dominance
    is determined based on the specified objective functions. A point dominates another
    if it is at least as good in all objectives and strictly better in at least one.

    Args:
        array: 2D array where each row is a point and each column represents a different objective
        objectives: Sequence of objective functions (maximize or minimize) for each column

    Returns:
        Boolean mask where True indicates a point belongs to the Pareto-optimal set

    Raises:
        AssertionError: If dimensions don't match or the array is not 2D
    """
    assert len(array.shape) == 2, "Array should be one dimensional, where first dimension is number of points, second dimension number of objectives"

    n_points, n_objectives = array.shape

    assert len(objectives) == n_objectives

    domination_mask = [True for _ in range(n_points)]

    for i in range(n_points):  # checking if ith point should be on the pareto front
        for j in range(n_points):
            if i == j:
                continue
            if np.array_equal(array[i], array[j]):
                continue

            point_domination_mask = [f(array[j, k], array[i, k]) for k, f in enumerate(objectives)]
            if all(point_domination_mask):  # j dominates i because at least as good at all objectives
                domination_mask[i] = False
                break
    return domination_mask


def plot_pareto_frontier(array: np.ndarray, objectives: Sequence[Callable[[float, float], bool]], figsize=(10, 6), title="Pareto Frontier"):
    """
    Visualize the Pareto frontier for a two-dimensional optimization problem.

    Parameters:
    -----------
    array : np.ndarray
        Array of points where the first dimension is the number of points and
        the second dimension must be exactly 2 (two criteria to optimize).
    objectives : Sequence[Callable]
        Sequence of two objective functions, each should be either maximize or minimize.
    figsize : tuple, optional
        Size of the figure (width, height) in inches. Default is (10, 6).
    title : str, optional
        Title of the plot. Default is "Pareto Frontier".

    Returns:
    --------
    fig, ax : tuple
        Matplotlib figure and axes objects.
    """
    assert len(array.shape) == 2, "Array should be two-dimensional"
    assert array.shape[1] == 2, "This function only works for two criteria (array.shape[1] must be 2)"
    assert len(objectives) == 2, "This function only works for two objectives"

    # Get the Pareto set (True for points on the Pareto frontier)
    pareto_mask = paretoset(array, objectives)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    # Extract Pareto and non-Pareto points
    pareto_points = array[pareto_mask]
    non_pareto_points = array[np.logical_not(pareto_mask)]

    # Plot non-Pareto points in blue
    if len(non_pareto_points) > 0:
        ax.scatter(non_pareto_points[:, 0], non_pareto_points[:, 1], color="blue", label="Non-Pareto points")

    # Plot Pareto points in red
    if len(pareto_points) > 0:
        ax.scatter(pareto_points[:, 0], pareto_points[:, 1], color="red", label="Pareto frontier points")

        # Sort points for line drawing based on the objectives
        # For two objectives, we typically want to sort by one coordinate
        # The sort direction depends on whether we're maximizing or minimizing
        sort_col = 0
        sort_ascending = isinstance(objectives[0], type(minimize))

        # Sort the Pareto points
        sorted_indices = np.argsort(pareto_points[:, sort_col])
        if not sort_ascending:
            sorted_indices = sorted_indices[::-1]

        sorted_pareto = pareto_points[sorted_indices]

        # Draw the line connecting Pareto points
        ax.plot(sorted_pareto[:, 0], sorted_pareto[:, 1], color="red", linestyle="-", linewidth=2)

    # Add labels and title
    ax.set_xlabel("Criterion 1")
    ax.set_ylabel("Criterion 2")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.7)

    return fig, ax


def _get_optimal_point_based_on_list_of_objective_functions(objectives: Sequence[Callable[[float, float], bool]]):
    opt_point = []
    for obj in objectives:
        if obj(1, 0):
            opt_point.append(1.0)
        else:
            opt_point.append(0.0)
    return np.array(opt_point)


def sort_by_optimal_point_proximity(array: np.ndarray, objectives: Sequence[Callable[[float, float], bool]]):
    assert np.max(array) <= 1, "all values in array should be in 0 -- 1 range, but there are values higher than 1"
    assert np.min(array) >= 0, "all values in array should be in 0 -- 1 range, but there are values lower than 0"

    opt_point = _get_optimal_point_based_on_list_of_objective_functions(objectives)
    assert len(opt_point) == array.shape[-1], "Number of objectives doesn't match the shape of the array"

    distances = _euclidean_distances(array, opt_point)
    indices = np.argsort(distances)
    return array[indices], indices
