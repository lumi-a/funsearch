"""I'm trying to find 2D-instances of the k-means clustering problem for which the clustering found by the
complete-linkage-algorithm has a high cost relative to the optimal clustering.

So far, I have tried the following functions to generate such instances. Please write another one that is similar and has the same signature, but has some lines altered slightly.
"""

from typing import Literal
import funsearch
import numpy as np


@funsearch.run
def evaluate(n: int) -> float:
    """Returns the ratio of the found instance.

    The ratio is the maximum of the ratios for each k in {1,...,n}, where the ratio for a fixed k
    is the ratio between the cost of the complete-linkage-clustering and the cost of the optimal clustering for
    k clusters.
    """
    points = get_points(n)
    return 0


@funsearch.evolve
def get_points(n: int) -> list[np.ndarray]:
    """Return a new clustering-problem, specified by a list of n points in 2D."""
    return [np.array([0, 0]) for _ in range(n)]
