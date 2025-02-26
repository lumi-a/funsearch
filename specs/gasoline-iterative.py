"""Finds sets for which the iterative rounding algorithm on the gasoline-problem has a poor (high) approximation-ratio.

The gasoline-problem find a permutation of the xs and ys (lists of positive integers) such that maximum of the differences of prefix-sums is as small as possible, i.e. maximum_(m,n) zs[n]-zs[m] is as as small as possible, where zs[n] = xs[0] - ys[0] + xs[1] - ys[1] + ... + xs[n//2] - (ys[n] if n is odd else 0).

As such, the problem is invariant under a permutation of the xs and ys.

On every iteration, improve gasoline_v1 over the gasoline_vX methods from previous iterations. The methods must be deterministic.
Make only small code-changes.
"""

import funsearch
from funsearch.gasoline.iterative_rounding import SlotOrdered


@funsearch.run
def evaluate(n: int) -> float:
    """Returns the approximation-ratio of the gasoline problem."""
    xs, ys = [1], [1]
    for _ in range(n - 1):
        x, y = gasoline(xs, ys)

        # Assert determinancy
        if (x, y) != gasoline(xs, ys):
            return 0

        xs.append(x)
        ys.append(y)

    if any(x < 0 for x in xs) or any(y < 0 for y in ys):
        return 0

    return SlotOrdered().approximation_ratio(xs, ys)


@funsearch.evolve
def gasoline(xs: list[int], ys: list[int]) -> tuple[int, int]:
    """Given a gasoline-problem specified by the list of x-values and y-values,
    return a new gasoline-problem, with one additional x-value and y-value.
    The integers are always non-negative.
    """
    x = len(xs)
    y = len(ys) ** 2 + 1
    return x, y
