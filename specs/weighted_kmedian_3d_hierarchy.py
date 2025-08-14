"""I'm trying to find 3D-instances of the k-median clustering problem for which the best-possible hierarchical
(nested) clustering has a high cost. The cost of a hierarchical clustering is the maximum of its cost across each
of its levels. The cost of level `k` is the ratio between its cost and the optimal cost of a k-clustering.
Because optimal clusterings need not be nested, the cost of the best-possible hierarchical clustering
can exceed 1.0.

So far, I have tried the following functions to generate sets of points for which the best-possible hierarchical
clustering has a high cost. Please write a similar one that doesn't use randomness and has the same signature,
but improves on the objective by slightly changing some lines. Please only respond with code, no explanations.
"""

import numpy as np

import funsearch


@funsearch.run
def evaluate(n: int) -> float:
    """Returns the ratio of the found instance."""
    from exact_clustering import weighted_discrete_kmedian_price_of_hierarchy

    weighted_points = get_weighted_points(n)

    # Assert determinancy
    if not all(
        w1 == w2 and np.array_equal(v1, v2) for (w1, v1), (w2, v2) in zip(weighted_points, get_weighted_points(n))
    ):
        return 0.0

    # Merging identical points avoids floating-point-rounding-issues and improves performance
    merged_weighted_points: dict[np.ndarray, float] = {}
    for weight, v in weighted_points[:n]:
        point = tuple(v[:3])
        merged_weighted_points[point] = merged_weighted_points.get(point, 0.0) + weight
    # Sorting by largest weight first helps with performance
    points = sorted(((weight, list(v)) for v, weight in merged_weighted_points.items()), reverse=True)
    return max(0.0, weighted_discrete_kmedian_price_of_hierarchy(points))


@funsearch.evolve
def get_weighted_points(n: int) -> list[np.ndarray]:
    """Return a new clustering-problem, specified by a list of n weighted points in 3D. The first
    element of each tuple is the weight of the point, the second the 3D-point itself."""
    points = []
    for i in range(n):
        weight = 1.0
        x = 0.0
        y = 0.0
        z = 0.0
        points.append((weight, np.array([x, y, z])))

    return points
