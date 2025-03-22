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
    from clustering_rs import price_of_kmedian_hierarchy

    points = get_points(n)

    # Assert determinancy
    if not all(np.array_equal(v1, v2) for v1, v2 in zip(points, get_points(n))):
        return 0.0

    # TODO: Separate points more
    points_list = [v.tolist() for v in points[:n]]
    return max(0.0, price_of_kmedian_hierarchy(points_list))


@funsearch.evolve
def get_points(n: int) -> list[np.ndarray]:
    """Return a new clustering-problem, specified by a list of n points in 3D."""
    points = []
    for i in range(n):
        x = 0
        y = 0
        z = 0
        points.append(np.array([x, y, z]))

    return points
