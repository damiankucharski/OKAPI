import os

import matplotlib.pyplot as plt
import numpy as np

from okapi.pareto import maximize, minimize, plot_pareto_frontier

# Ensure the .test_dump/pareto directory exists in the project root
root_dir = os.path.dirname(os.path.dirname(__file__))
test_dump_dir = os.path.join(root_dir, ".test_dump")
pareto_test_dir = os.path.join(test_dump_dir, "pareto")
os.makedirs(pareto_test_dir, exist_ok=True)


def test_plot_pareto_maximize_maximize():
    """Test plotting Pareto frontier with two maximize objectives"""
    # Create sample data
    np.random.seed(42)  # For reproducibility
    n_points = 50
    points = np.random.rand(n_points, 2) * 10

    # Plot Pareto frontier for two maximize objectives
    fig, ax = plot_pareto_frontier(points, [maximize, maximize], title="Pareto Frontier (Maximize, Maximize)")

    # Save the figure
    filepath = os.path.join(pareto_test_dir, "pareto_maximize_maximize.png")
    fig.savefig(filepath)
    plt.close(fig)
    print(f"Created visualization: {filepath}")


def test_plot_pareto_maximize_minimize():
    """Test plotting Pareto frontier with mixed objectives"""
    # Create sample data
    np.random.seed(42)  # For reproducibility
    n_points = 50
    points = np.random.rand(n_points, 2) * 10

    # Plot Pareto frontier for mixed objectives
    fig, ax = plot_pareto_frontier(points, [maximize, minimize], title="Pareto Frontier (Maximize, Minimize)")

    # Add objective directions to the labels
    ax.set_xlabel("Criterion 1 (Maximize)")
    ax.set_ylabel("Criterion 2 (Minimize)")

    # Save the figure
    filepath = os.path.join(pareto_test_dir, "pareto_maximize_minimize.png")
    fig.savefig(filepath)
    plt.close(fig)
    print(f"Created visualization: {filepath}")


def test_special_case_pareto_all_points():
    """Test the case where all points are on the Pareto frontier"""
    # Create sample data where no point dominates another
    points = np.array([[1, 10], [2, 8], [3, 6], [4, 4], [5, 2], [10, 1]])

    # Plot Pareto frontier
    fig, ax = plot_pareto_frontier(points, [maximize, maximize], title="All Points on Pareto Frontier (Maximize, Maximize)")

    # Save the figure
    filepath = os.path.join(pareto_test_dir, "pareto_all_points.png")
    fig.savefig(filepath)
    plt.close(fig)
    print(f"Created visualization: {filepath}")


def test_special_case_single_point():
    """Test the case with a single dominant point"""
    # Create sample data with one dominant point
    points = np.array(
        [
            [1, 1],
            [2, 2],
            [3, 3],
            [4, 4],
            [5, 5],
            [10, 10],  # This point dominates all others
        ]
    )

    # Plot Pareto frontier
    fig, ax = plot_pareto_frontier(points, [maximize, maximize], title="Single Dominant Point (Maximize, Maximize)")

    # Save the figure
    filepath = os.path.join(pareto_test_dir, "pareto_single_point.png")
    fig.savefig(filepath)
    plt.close(fig)
    print(f"Created visualization: {filepath}")


if __name__ == "__main__":
    print("Testing Pareto frontier visualization...")
    test_plot_pareto_maximize_maximize()
    test_plot_pareto_maximize_minimize()
    test_special_case_pareto_all_points()
    test_special_case_single_point()
    print("All tests completed!")
