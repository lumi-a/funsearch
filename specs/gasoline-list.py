"""I'm trying to find sets for which an iterative rounding algorithm on the gasoline-problem has a poor (high) approximation-ratio.

The gasoline-problem looks for a permutation of the xs and ys (lists of positive integers) such that maximum of the differences of prefix-sums is as small as possible, i.e. maximum_(m,n) zs[n]-zs[m] is as as small as possible, where zs[n] = xs[0] - ys[0] + xs[1] - ys[1] + ... + xs[n//2] - (ys[n] if n is odd else 0).

To generate sets with poor approximation-ratios, I have tried the following functions so far. Please write another one that is similar and has the same signature, but has some lines altered slightly.
"""

import funsearch


@funsearch.run
def evaluate(n: int) -> float:
    """Returns the approximation-ratio of the gasoline problem."""
    from funsearch.gasoline.iterative_rounding import SlotOrdered

    xs, ys = gasoline(n)

    # Assert determinancy
    if (xs, ys) != gasoline(n):
        return 0

    xs = [max(0, min(2**31 - 1, int(x))) for x in xs]
    ys = [max(0, min(2**31 - 1, int(y))) for y in ys]

    return SlotOrdered().approximation_ratio(xs, ys)


@funsearch.evolve
def gasoline(n: int) -> tuple[list[int], list[int]]:
    """Return a new gasoline-problem, specified by the list of x-values and y-values.
    The integers will be clamped to [0, 2**31 - 1].
    """
    xs, ys = [], []
    for i in range(1, n):
        u = int(2**n * (1 - 2 ** (-i)))
        xs.extend([u for _ in range(2**i)])
        ys.extend([u for _ in range(2**i)])
    xs.extend([int(2**n) for _ in range(2**n)])
    u = int(2**n * (1 - 2 ** (-n)))
    ys.extend([u for _ in range(2**n)])
    return xs, ys
