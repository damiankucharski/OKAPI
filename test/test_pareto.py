import numpy as np
import pytest

# Import the functions to test
from giraffe.pareto import _get_optimal_point_based_on_list_of_objective_functions, maximize, minimize, paretoset, sort_by_optimal_point_proximity


def test_simple_maximize():
    """Test with single maximize objective"""
    points = np.array([[1], [2], [3]])
    result = paretoset(points, [maximize])
    assert result == [False, False, True]


def test_simple_minimize():
    """Test with single minimize objective"""
    points = np.array([[1], [2], [3]])
    result = paretoset(points, [minimize])
    assert result == [True, False, False]


def test_two_objectives_maximize():
    """Test with two maximize objectives"""
    points = np.array([[1, 1], [2, 2], [3, 1], [1, 3]])
    result = paretoset(points, [maximize, maximize])
    assert result == [False, True, True, True]


def test_mixed_objectives():
    """Test with mix of maximize and minimize objectives"""
    points = np.array(
        [
            [1, 4],  # Dominated by all others
            [2, 3],  # Dominated by [3,2] and [4,1]
            [3, 2],  # Dominated by [4,1]
            [4, 1],  # Not dominated by any point
        ]
    )
    result = paretoset(points, [maximize, minimize])
    assert result == [False, False, False, True]


@pytest.mark.parametrize(
    "points, objectives, expected_mask",
    [
        (
            np.array([[0.59, 0.937, 5.0], [0.597, 0.935, 5.0], [0.585, 0.94, 5.0], [0.586, 0.939, 5.0]]),
            [maximize, maximize, minimize],
            [True, True, True, True],
        ),
        (
            np.array(
                [
                    [4, 1],  # optimal, highest maximize
                    [3, 0],  # best minimize
                    [3, 3],  # dominated
                    [4, 1],  # the same as first
                ]
            ),
            [maximize, minimize],
            [True, True, False, True],
        ),
    ],
)
def test_mixed_objectives_multiple_optimal(points, objectives, expected_mask):
    """Test with mix of maximize and minimize objectives where multiple points are optimal"""
    result = paretoset(points, objectives)
    assert result == expected_mask


def test_identical_points():
    """Test handling of identical points"""
    points = np.array([[1, 1], [1, 1], [2, 2]])
    result = paretoset(points, [maximize, maximize])
    assert result == [False, False, True]


def test_empty_array():
    """Test with empty array"""
    points = np.array([]).reshape(0, 2)
    result = paretoset(points, [maximize, maximize])
    assert len(result) == 0


def test_single_point():
    """Test with single point"""
    points = np.array([[1, 1]])
    result = paretoset(points, [maximize, maximize])
    assert result == [True]


def test_array_dimension_error():
    """Test error handling for incorrect array dimensions"""
    points = np.array([1, 2, 3])  # 1D array
    with pytest.raises(AssertionError):
        paretoset(points, [maximize])


def test_objective_count_mismatch():
    """Test error handling for mismatched number of objectives"""
    points = np.array([[1, 2], [3, 4]])
    with pytest.raises(AssertionError):
        paretoset(points, [maximize])  # Only one objective for 2D points


def test_three_objectives():
    """Test with three objectives"""
    points = np.array([[1, 1, 1], [2, 2, 2], [3, 1, 2], [1, 3, 2]])
    result = paretoset(points, [maximize, maximize, maximize])
    assert result == [False, True, True, True]


def test_all_dominated():
    """Test case where all points except one are dominated"""
    points = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
    result = paretoset(points, [maximize, maximize])
    assert result == [False, False, False, True]


def test_none_dominated():
    """Test case where no points are dominated"""
    points = np.array([[4, 1], [3, 2], [2, 3], [1, 4]])
    result = paretoset(points, [maximize, maximize])
    assert result == [True, True, True, True]


## test helpers
@pytest.mark.parametrize(
    "objectives, optimal_point",
    [
        ([minimize, minimize, maximize], [0, 0, 1]),
    ],
)
def test_get_optimal_point_based_on_list_of_objective_functions(objectives, optimal_point):
    opt_point = np.array(optimal_point)
    returned = _get_optimal_point_based_on_list_of_objective_functions(objectives)
    assert np.array_equal(opt_point, returned)


@pytest.mark.parametrize(
    "points, objectives, expected_indices",
    [
        (
            np.array(
                [
                    [0.2, 0.3, 0.4],
                    [0.5, 0.5, 0.4],
                    [0.0, 0.1, 0.1],
                ]
            ),
            [minimize, minimize, minimize],
            [2, 0, 1],
        ),
        (
            np.array(
                [
                    [0.8, 0.7, 0.9],
                    [0.9, 0.9, 0.8],
                    [1.0, 0.8, 1.0],
                ]
            ),
            [maximize, maximize, maximize],
            [2, 1, 0],
        ),
        (
            np.array(
                [
                    [0.5, 0.5],
                    [0.1, 0.9],
                    [0.9, 0.1],
                ]
            ),
            [minimize, maximize],
            [1, 0, 2],
        ),
        # Edge cases
        (
            np.array([[0.5, 0.5]]),
            [minimize, minimize],
            [0],
        ),
        (
            np.array(
                [
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [1.0, 1.0],
                ]
            ),
            [minimize, minimize],
            [0, 1, 2],
        ),
    ],
)
def test_sort_by_optimal_point_proximity(points, objectives, expected_indices):
    _, ret_indices = sort_by_optimal_point_proximity(points, objectives)
    assert np.array_equal(ret_indices, expected_indices)


def test_sort_by_optimal_point_out_of_range():
    points = np.array([[1.2, 0.5], [0.5, 1.2]])
    with pytest.raises(AssertionError):
        sort_by_optimal_point_proximity(points, [minimize, minimize])


def test_sort_by_optimal_point_negative_values():
    points = np.array([[-0.2, 0.5], [0.5, -0.1]])
    with pytest.raises(AssertionError):
        sort_by_optimal_point_proximity(points, [minimize, minimize])


def test_sort_by_optimal_point_wrong_objectives():
    points = np.array([[0.2, 0.5]])
    with pytest.raises(AssertionError):
        sort_by_optimal_point_proximity(points, [minimize, minimize, minimize])
