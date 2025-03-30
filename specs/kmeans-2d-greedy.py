"""I'm trying to find 2D-instances of the k-means clustering problem for which the clustering found by the
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
    from exact_clustering import unweighted_continuous_kmeans_price_of_greedy

    points = get_points(n)

    # Assert determinancy
    if not all(np.array_equal(v1, v2) for v1, v2 in zip(points, get_points(n))):
        return 0.0

    # TODO: Separate points more
    points_list = [v.tolist() for v in points[:n]]
    return max(0.0, unweighted_continuous_kmeans_price_of_greedy(points_list))


@funsearch.evolve
def get_points(n: int) -> list[np.ndarray]:
    """Return a new clustering-problem, specified by a list of n points in 2D."""
    points = []
    for i in range(n):
        x = 0
        y = 0
        points.append(np.array([x, y]))

    return points
