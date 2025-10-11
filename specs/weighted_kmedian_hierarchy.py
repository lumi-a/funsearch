"""I'm trying to find instances of the k-median clustering problem for which the
best-possible hierarchical (nested) clustering has a high cost. The cost of a hierarchical clustering is
the maximum of its cost across each of its levels. The cost of level `k` is the ratio between its cost and
the optimal cost of a k-clustering.
Because optimal clusterings need not be nested, the cost of the best-possible hierarchical clustering
can exceed 1.0.

So far, I have tried the following functions to generate sets of points for which the best-possible hierarchical
clustering has a high cost. Please write a similar one that doesn't use randomness and has the same signature,
but improves on the objective by slightly changing some lines. Please only respond with code, no explanations.
"""

import numpy as np

import funsearch


@funsearch.run
def evaluate(_: int) -> float:
    """Returns the ratio of the found instance."""
    weighted_points = get_weighted_points()
    # Assert determinancy
    if not all(
        w1 == w2 and np.array_equal(v1, v2) for (w1, v1), (w2, v2) in zip(weighted_points, get_weighted_points())
    ):
        return 0.0

    return evaluate_instance(weighted_points[:32])


def evaluate_instance(weighted_points: list[tuple[float, np.ndarray]]) -> float:
    from exact_clustering import WeightedKMedianL1

    # Merging identical points avoids floating-point-rounding-issues and improves performance
    merged_weighted_points: dict[np.ndarray, float] = {}
    for weight, v in weighted_points:
        point = tuple(np.clip(v, -1e16, 1e16))
        clamped_weight = max(1.0, min(1e16, weight))
        merged_weighted_points[point] = merged_weighted_points.get(point, 0.0) + clamped_weight
    # Sorting by largest weight first helps with performance
    points = sorted(((weight, list(v)) for v, weight in merged_weighted_points.items()), reverse=True)
    return max(0.0, WeightedKMedianL1(points).price_of_hierarchy())


@funsearch.evolve
def get_weighted_points() -> list[tuple[float, np.ndarray]]:
    """Return a new weighted clustering-problem, specified by a list of weighted points.
    The returned tuple consists of the weight of the point, and the point itself.
    """
    weighted_points = []

    for i in range(1):
        weighted_points.append((1.0, np.array([0, 0, 0, 0])))

    for j in range(1):
        weighted_points.append((1e8, np.array([1, 0, 0, 0])))

    return weighted_points
