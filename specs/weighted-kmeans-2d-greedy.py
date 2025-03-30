"""I'm trying to find 2D-instances of the weighted k-means clustering problem for which the clustering found by the
complete-linkage-algorithm has a high cost relative to the optimal clustering.

So far, I have tried the following functions to generate such instances. Please write a similar one that doesn't use randomness and has the same signature, but improves on the objective by slightly changing some lines. Please only respond with code, no explanations.
"""

import numpy as np

import funsearch


@funsearch.run
def evaluate(n: int) -> float:
    """Returns the ratio of the found instance.

    The ratio is the maximum of the ratios for each k in {1,...,n}, where the ratio for a fixed k
    is the ratio between the cost of the complete-linkage-clustering and the cost of the optimal clustering for
    k clusters.
    """
    from exact_clustering import weighted_continuous_kmeans_price_of_greedy

    weighted_points = get_weighted_points(n)

    # Assert determinancy
    if not all(
        w1 == w2 and np.array_equal(v1, v2) for (w1, v1), (w2, v2) in zip(weighted_points, get_weighted_points(n))
    ):
        return 0.0

    # Merging identical points avoids floating-point-rounding-issues and improves performance
    merged_weighted_points: dict[np.ndarray, float] = {}
    for weight, v in weighted_points[:n]:
        point = tuple(v[:2])
        merged_weighted_points[point] = merged_weighted_points.get(point, 0.0) + weight
    # Sorting by largest weight first helps with performance
    points = sorted(((weight, list(v)) for v, weight in merged_weighted_points.items()), reverse=True)
    return max(0.0, weighted_continuous_kmeans_price_of_greedy(points))


@funsearch.evolve
def get_weighted_points(n: int) -> list[tuple[float, np.ndarray]]:
    """Return a new clustering-problem, specified by a list of n weighted points in 2D. The first
    element of each tuple is the weight of the point, the second the 2D-point itself."""
    points = []
    for i in range(n):
        weight = 1.0
        x = 0.0
        y = 0.0
        points.append((weight, np.array([x, y])))

    return points
