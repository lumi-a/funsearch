"""I'm trying to find instances of the gasoline-problem for which an iterative rounding algorithm has a poor (high) approximation-ratio in two dimensions.

The gasoline-problem looks for a permutation of the xs and ys (lists of points in ℕ^2) such that maximum of the differences of prefix-sums is as small as possible, i.e. maximum_(m,n) zs[n]-zs[m] is as as small as possible, where zs[n] = xs[0] - ys[0] + xs[1] - ys[1] + ... + xs[n//2] - (ys[n] if n is odd else 0).

To generate sets with poor approximation-ratios, I have tried the following functions so far. Please write another one that is similar and has the same signature, but has some lines altered slightly.
"""

import math
import numpy as np
import funsearch


@funsearch.run
def evaluate(n: int) -> float:
    """Returns the approximation-ratio of the gasoline problem."""
    from funsearch.gasoline.iterative_rounding import SlotOrdered
    from funsearch.memoize import memoize

    xs, ys = gasoline(n)

    # Assert determinancy
    xs1, ys1 = gasoline(n)
    if not (len(xs) == len(xs1) and len(ys) == len(ys1) and np.array_equal(xs, xs1) and np.array_equal(ys, ys1)):
        return 0.0

    length = min(len(xs), len(ys) + 1, n)  # ys will be one element shorter than xs
    # Clamp inputs to avoid overflows in gurobi
    xs = [np.clip(np.round(x[:2]), 0, 2**31 - 1) for x in xs[:length]]
    ys = [np.clip(np.round(y[:2]), 0, 2**31 - 1) for y in ys[: length - 1]]

    @memoize("gasoline-2d")
    def memoized_approximation_ratio(xs: list[np.ndarray], ys: list[np.ndarray]) -> float:
        return SlotOrdered().approximation_ratio(xs, ys)

    return memoized_approximation_ratio(xs, ys)


@funsearch.evolve
def gasoline(n: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Return a new gasoline-problem, specified by the two lists of 2d-non-negative-integer-points.
    Both lists should have length n and consist only of points in ℕ^2.
    """
    xs = []
    ys = []
    m = 1 + (n // 2)
    for i in range(2, m):
        rounded = int(m * (1 - 2 ** (-int(math.log2(i)))))
        xs.append(np.array([rounded, 0]))
        ys.append(np.array([rounded, 0]))

    xs.extend([np.array([m, 0]) for _ in range(m - 1)])
    xs.append(np.array([0, 0]))
    ys.extend([np.array([m - 1, 0]) for _ in range(m)])

    return xs, ys
