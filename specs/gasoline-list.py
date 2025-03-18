"""I'm trying to find instances of the gasoline-problem for which an iterative rounding algorithm has a poor (high) approximation-ratio.

The gasoline-problem looks for a permutation of the xs and ys (lists of positive integers) such that maximum of the differences of prefix-sums is as small as possible, i.e. maximum_(m,n) zs[n]-zs[m] is as as small as possible, where zs[n] = xs[0] - ys[0] + xs[1] - ys[1] + ... + xs[n//2] - (ys[n] if n is odd else 0).

To generate sets with poor approximation-ratios, I have tried the following functions so far. Please write another one that is similar and has the same signature, but has some lines altered slightly.
"""

import math
import funsearch


@funsearch.run
def evaluate(n: int) -> float:
    """Returns the approximation-ratio of the gasoline problem."""
    from pathlib import Path

    from funsearch.gasoline.iterative_rounding import SlotOrdered

    xs, ys = gasoline(n)

    # Assert determinancy
    if (xs, ys) != gasoline(n):
        return 0.0

    length = min(len(xs), len(ys) + 1, n)  # ys will be one element shorter than xs
    # Clamp inputs to avoid overflows in gurobi
    xs = [max(0, min(2**31 - 1, int(x))) for x in xs[:length]]
    ys = [max(0, min(2**31 - 1, int(y))) for y in ys[: length - 1]]

    # Memoize the input. Use a separate file for every input, a single file wouldn't be thread-safe.
    memoization_path = Path.cwd() / ".memoization-cache" / "gasoline-0" / (str(xs) + "," + str(ys))
    if memoization_path.exists():
        return float(memoization_path.read_text())

    ratio = SlotOrdered().approximation_ratio(xs, ys)
    memoization_path.parent.mkdir(parents=True, exist_ok=True)
    memoization_path.write_text(str(ratio))
    return ratio


@funsearch.evolve
def gasoline(n: int) -> tuple[list[int], list[int]]:
    """Return a new gasoline-problem, specified by the list of x-values and y-values,
    each of which must have length at most `n`.
    """
    m = 1 + (n // 2)
    xs = [int(m * (1 - 2 ** (-int(math.log2(i))))) for i in range(2, m)]
    ys = [int(m * (1 - 2 ** (-int(math.log2(i))))) for i in range(2, m)]
    xs.extend([m for _ in range(m - 1)])
    xs.append(0)
    ys.extend([m - 1 for _ in range(m)])
    return xs, ys
